//! Pre-allocated scratch buffers for HIP forward pass.
//!
//! This module provides `HipScratch` for reusable intermediate buffers and
//! `HipRuntimeConfig` for runtime configuration. These enable zero-copy
//! forward passes by eliminating per-call allocations.
//!
//! # Memory Layout
//!
//! Buffers use web-rwkv conventions where shape[0] is the fastest-moving axis:
//! - Standard buffers: `[n_embd, max_seq_len, batch_size]`
//! - FFN hidden buffers: `[n_hidden, max_seq_len, batch_size]`
//! - LoRA buffers: `[lora_dim, max_seq_len, batch_size]`
//! - Output buffer: `[n_vocab, max_seq_len, batch_size]`

use super::blas::HipBlasContext;
use super::ffi::Result;
use super::model::Rwkv7ModelInfo;
use super::pinned::PinnedBuffer;
use super::tensor::{TensorHip, TensorShape};
use half::f16;

/// Default FLA chunk size (matches fla-org reference default for RWKV7).
pub const FLA_CHUNK_SIZE: usize = 16;

/// Runtime configuration for HIP inference.
///
/// Controls buffer sizing and batching behavior for the forward pass.
#[derive(Debug, Clone)]
pub struct HipRuntimeConfig {
    /// Maximum sequence length for prefill (tokens processed at once).
    /// Sequences longer than this are processed in chunks.
    /// Default: 256
    pub max_prefill_chunk: usize,

    /// Batch size (number of sequences processed in parallel).
    /// Default: 1
    pub batch_size: usize,

    /// FLA chunk size for chunked prefill. Default: 16.
    ///
    /// The FLA pipeline divides sequences into chunks of this size for
    /// parallel intra-chunk computation. Must be > 0. Smaller values
    /// reduce numerical error; larger values may improve throughput.
    pub fla_chunk_size: usize,
}

impl Default for HipRuntimeConfig {
    fn default() -> Self {
        Self {
            max_prefill_chunk: 256,
            batch_size: 1,
            fla_chunk_size: FLA_CHUNK_SIZE,
        }
    }
}

impl HipRuntimeConfig {
    /// Create a new config with specified max chunk size and batch size.
    pub fn new(max_prefill_chunk: usize, batch_size: usize) -> Self {
        Self {
            max_prefill_chunk,
            batch_size,
            fla_chunk_size: FLA_CHUNK_SIZE,
        }
    }

    /// Create config for single-token decode mode (batch_size=1, chunk=1).
    pub fn decode() -> Self {
        Self {
            max_prefill_chunk: 1,
            batch_size: 1,
            fla_chunk_size: FLA_CHUNK_SIZE,
        }
    }

    /// Create config for prefill with specified chunk size.
    pub fn prefill(max_chunk: usize) -> Self {
        Self {
            max_prefill_chunk: max_chunk,
            batch_size: 1,
            fla_chunk_size: FLA_CHUNK_SIZE,
        }
    }
}

/// LoRA dimension information extracted from model weights.
///
/// These dimensions vary by model and are needed to size LoRA scratch buffers.
#[derive(Debug, Clone, Copy)]
pub struct LoraDims {
    /// Decay LoRA rank (w1/w2 intermediate dimension)
    pub w_dim: usize,
    /// Adaptation LoRA rank (a1/a2 intermediate dimension)
    pub a_dim: usize,
    /// Gate LoRA rank (g1/g2 intermediate dimension)
    pub g_dim: usize,
    /// Value residual LoRA rank (v1/v2 intermediate dimension), if present
    pub v_dim: Option<usize>,
}

/// Pre-allocated scratch buffers for HIP inference.
///
/// All buffers are sized for `[dim, max_seq_len, batch_size]` to support
/// any sequence length up to `max_prefill_chunk`. Buffers are reused across
/// `step()` calls, eliminating allocation overhead.
///
/// # Buffer Categories
///
/// 1. **Standard buffers** `[n_embd, T, B]`: Main computation intermediates
/// 2. **FFN buffers** `[n_hidden, T, B]`: Feed-forward network intermediates
/// 3. **LoRA buffers** `[lora_dim, T, B]`: Low-rank adaptation intermediates
/// 4. **Output buffer** `[n_vocab, T, B]`: Final logits
///
/// # Usage
///
/// ```rust,ignore
/// let config = HipRuntimeConfig::new(256, 4);
/// let model = Rwkv7Hip::load("model.st")?.with_config(config)?;
///
/// // step() reuses scratch buffers internally
/// let (logits, state) = model.step(&[&tokens], None)?;
/// ```
#[derive(Debug)]
#[allow(non_snake_case)] // FLA buffer names (fla_A_qk, etc.) match paper notation
pub struct HipScratch {
    /// Configuration used to allocate these buffers
    pub config: HipRuntimeConfig,

    /// Reusable BLAS context (rocBLAS handle + null stream)
    pub blas_ctx: HipBlasContext,

    /// Model dimensions for validation
    pub n_embd: usize,
    pub n_hidden: usize,
    pub n_vocab: usize,

    /// LoRA dimensions
    pub lora_dims: LoraDims,

    /// Persistent GPU state (resident on device when enabled)
    pub att_shift_state_gpu: Vec<TensorHip<f16>>,
    pub ffn_state_gpu: Vec<TensorHip<f16>>,
    pub wkv_state_gpu: Vec<TensorHip<f32>>,

    // ========== Standard buffers [n_embd, T, B] ==========
    /// Main hidden state (persists across layers within forward pass)
    pub x: TensorHip<f16>,

    /// Layer norm output (reused for att_ln, ffn_ln, head_ln)
    pub x_ln: TensorHip<f16>,

    /// Token-shifted inputs for attention
    pub att_xr: TensorHip<f16>,
    pub att_xw: TensorHip<f16>,
    pub att_xk: TensorHip<f16>,
    pub att_xv: TensorHip<f16>,
    pub att_xa: TensorHip<f16>,
    pub att_xg: TensorHip<f16>,

    /// Linear projections
    pub att_r: TensorHip<f16>,
    pub att_k: TensorHip<f16>,
    pub att_v: TensorHip<f16>,

    /// Decay (after softplus transformation)
    pub att_w: TensorHip<f16>,

    /// Adaptation factor (after sigmoid)
    pub att_a: TensorHip<f16>,

    /// Gate (after LoRA projection)
    pub att_g: TensorHip<f16>,

    /// L2-normalized key
    pub att_kk: TensorHip<f16>,

    /// Controlled key (k * (1 + (a-1) * k_a))
    pub att_k_ctrl: TensorHip<f16>,

    /// WKV input: -kk
    pub wkv_a: TensorHip<f16>,

    /// WKV input: kk * a
    pub wkv_b: TensorHip<f16>,

    /// Decay for WKV: exp(w)
    pub w_decay: TensorHip<f16>,

    /// WKV7 output
    pub wkv_out: TensorHip<f16>,

    /// Group-normalized WKV output
    pub wkv_normed: TensorHip<f16>,

    /// WKV bonus (time_first contribution)
    pub wkv_bonus: TensorHip<f16>,

    /// Attention output (combined, gated, projected - can share storage)
    pub att_out: TensorHip<f16>,

    /// FFN token-shifted input
    pub ffn_xk: TensorHip<f16>,

    /// FFN output
    pub ffn_out: TensorHip<f16>,

    /// Value from first layer for residual (stored across layers)
    pub v_first: TensorHip<f16>,

    // ========== FFN hidden buffers [n_hidden, T, B] ==========
    /// FFN key projection output
    pub ffn_k: TensorHip<f16>,

    /// FFN squared ReLU output
    pub ffn_k_sq: TensorHip<f16>,

    // ========== LoRA buffers [lora_dim, T, B] ==========
    /// Decay LoRA intermediate
    pub lora_w: TensorHip<f16>,

    /// Adaptation LoRA intermediate
    pub lora_a: TensorHip<f16>,

    /// Gate LoRA intermediate
    pub lora_g: TensorHip<f16>,

    /// Value residual LoRA intermediate (may be empty if v_dim is None)
    pub lora_v: TensorHip<f16>,

    /// Decay LoRA tanh output
    pub lora_w_tanh: TensorHip<f16>,

    /// Adaptation LoRA projection output
    pub lora_a_proj: TensorHip<f16>,

    /// Gate LoRA sigmoid output
    pub lora_g_sig: TensorHip<f16>,

    /// Value residual projection output
    pub v_lora2: TensorHip<f16>,

    // ========== Temporary buffers ==========
    /// Temporary buffer for pointwise ops (std shape)
    pub temp1: TensorHip<f16>,

    /// Temporary buffer for pointwise ops (std shape)
    pub temp2: TensorHip<f16>,

    /// Temporary shift state output (state shape)
    pub new_att_shift: TensorHip<f16>,

    /// Temporary shift state output (state shape)
    pub new_ffn_shift: TensorHip<f16>,

    // ========== FLA chunked prefill buffers ==========
    // Pre-allocated for the 5-stage FLA pipeline. Shared across layers.
    // Sized for worst-case: max_total_chunks = batch_size * ceil_div(max_prefill_chunk, fla_chunk_size)
    // All buffers are f32 for FP32 precision (matching the plan's "FP32 state" requirement).

    /// Cumulative intra-chunk decay (Stage 1).
    /// Shape: `[head_size, n_head, max_prefill_chunk, batch_size]`
    pub fla_gi: TensorHip<f32>,

    /// Total chunk decay (Stage 1).
    /// Shape: `[head_size, n_head, max_prefill_chunk, batch_size]`
    pub fla_ge: TensorHip<f32>,

    /// Intra-chunk attention matrix: Q @ K^T (Stage 2).
    /// Shape: `[C, C, n_head, max_total_chunks]`
    pub fla_A_qk: TensorHip<f32>,

    /// Query-bias attention matrix: Q @ B^T (Stage 2).
    /// Shape: `[C, C, n_head, max_total_chunks]`
    pub fla_A_qb: TensorHip<f32>,

    /// Adapt-bias attention matrix: A @ B^T (Stage 2).
    /// Shape: `[C, C, n_head, max_total_chunks]`
    pub fla_A_ab: TensorHip<f32>,

    /// Adapt-key attention matrix: A @ K^T (Stage 2).
    /// Shape: `[C, C, n_head, max_total_chunks]`
    pub fla_A_ak: TensorHip<f32>,

    /// Inverse lower-triangular of A_ab (Stage 3).
    /// Shape: `[C, C, n_head, max_total_chunks]`
    pub fla_A_ab_inv: TensorHip<f32>,

    /// WY representation w output (Stage 3).
    /// Shape: `[head_size, n_head, max_prefill_chunk, batch_size]`
    pub fla_w_wy: TensorHip<f32>,

    /// WY representation u output (Stage 3).
    /// Shape: `[head_size, n_head, max_prefill_chunk, batch_size]`
    pub fla_u_wy: TensorHip<f32>,

    /// Per-chunk recurrent states (Stage 4).
    /// Shape: `[head_size, head_size, n_head, max_total_chunks]`
    pub fla_h: TensorHip<f32>,

    /// Corrected values after WY transform (Stage 4).
    /// Shape: `[head_size, n_head, max_prefill_chunk, batch_size]`
    pub fla_v_new: TensorHip<f32>,

    // ========== Output buffer [n_vocab, T, B] ==========
    /// Final logits output (f16)
    pub logits: TensorHip<f16>,

    /// Logits converted to f32 on GPU (for efficient download)
    pub logits_f32: TensorHip<f32>,

    /// Pinned host buffer for async logits download (avoids pageable memory allocation)
    pub logits_staging: PinnedBuffer<f32>,

    // ========== Token staging buffer [T, B] ==========
    /// GPU staging buffer for tokens (zero-copy chunking)
    pub token_staging: TensorHip<u32>,

    /// GPU buffer for sequence lengths (masked kernels)
    pub lens_gpu: TensorHip<i32>,

    // ========== Pinned host staging buffer [n_embd * T * B] ==========
    /// Pinned host buffer for async embedding upload (avoids sync on hipMemcpyAsync)
    pub emb_staging: PinnedBuffer<f16>,
}

impl HipScratch {
    /// Create new scratch buffers sized for the given model and config.
    ///
    /// # Arguments
    /// * `info` - Model dimensions (n_embd, n_hidden, n_vocab, etc.)
    /// * `lora_dims` - LoRA dimensions extracted from model weights
    /// * `config` - Runtime configuration (max_prefill_chunk, batch_size)
    ///
    /// # Errors
    /// Returns error if GPU memory allocation fails.
    pub fn new(
        info: &Rwkv7ModelInfo,
        lora_dims: LoraDims,
        config: HipRuntimeConfig,
    ) -> Result<Self> {
        let t = config.max_prefill_chunk;
        let b = config.batch_size;
        let c = info.n_embd;
        let h = info.n_hidden;
        let v = info.n_vocab;

        // Standard shape [n_embd, T, B]
        let std_shape = TensorShape::new(c, t, b, 1);

        // FFN hidden shape [n_hidden, T, B]
        let ffn_shape = TensorShape::new(h, t, b, 1);

        // Output shape [n_vocab, T, B]
        let out_shape = TensorShape::new(v, t, b, 1);
        let out_size = v * t * b;

        // LoRA shapes
        let lora_w_shape = TensorShape::new(lora_dims.w_dim, t, b, 1);
        let lora_a_shape = TensorShape::new(lora_dims.a_dim, t, b, 1);
        let lora_g_shape = TensorShape::new(lora_dims.g_dim, t, b, 1);
        let lora_v_shape = TensorShape::new(lora_dims.v_dim.unwrap_or(1), t, b, 1);
        let state_shape = TensorShape::new(c, b, 1, 1);
        let wkv_state_shape = TensorShape::new(info.head_size, info.head_size, info.n_head, b);

        // FLA shapes
        let fla_c = config.fla_chunk_size;
        let max_total_chunks = b * ((t + fla_c - 1) / fla_c); // ceil_div(t, fla_c) * b
        let head_size = info.head_size;
        let n_head = info.n_head;
        // Per-token buffers: [head_size, n_head, max_prefill_chunk, batch_size]
        let fla_per_token_shape = TensorShape::new(head_size, n_head, t, b);
        // Per-chunk attention matrices: [C, C, n_head, max_total_chunks]
        let fla_chunk_mat_shape = TensorShape::new(fla_c, fla_c, n_head, max_total_chunks);
        // Per-chunk state buffers: [head_size, head_size, n_head, max_total_chunks]
        let fla_chunk_state_shape =
            TensorShape::new(head_size, head_size, n_head, max_total_chunks);

        let mut att_shift_state_gpu = Vec::with_capacity(info.n_layer);
        let mut ffn_state_gpu = Vec::with_capacity(info.n_layer);
        let mut wkv_state_gpu = Vec::with_capacity(info.n_layer);

        for _ in 0..info.n_layer {
            att_shift_state_gpu.push(TensorHip::zeros(state_shape)?);
            ffn_state_gpu.push(TensorHip::zeros(state_shape)?);
            wkv_state_gpu.push(TensorHip::zeros(wkv_state_shape)?);
        }

        let blas_ctx = HipBlasContext::with_null_stream()?;

        Ok(Self {
            config,
            blas_ctx,
            n_embd: c,
            n_hidden: h,
            n_vocab: v,
            lora_dims,
            att_shift_state_gpu,
            ffn_state_gpu,
            wkv_state_gpu,

            // Standard buffers
            x: TensorHip::new(std_shape)?,
            x_ln: TensorHip::new(std_shape)?,
            att_xr: TensorHip::new(std_shape)?,
            att_xw: TensorHip::new(std_shape)?,
            att_xk: TensorHip::new(std_shape)?,
            att_xv: TensorHip::new(std_shape)?,
            att_xa: TensorHip::new(std_shape)?,
            att_xg: TensorHip::new(std_shape)?,
            att_r: TensorHip::new(std_shape)?,
            att_k: TensorHip::new(std_shape)?,
            att_v: TensorHip::new(std_shape)?,
            att_w: TensorHip::new(std_shape)?,
            att_a: TensorHip::new(std_shape)?,
            att_g: TensorHip::new(std_shape)?,
            att_kk: TensorHip::new(std_shape)?,
            att_k_ctrl: TensorHip::new(std_shape)?,
            wkv_a: TensorHip::new(std_shape)?,
            wkv_b: TensorHip::new(std_shape)?,
            w_decay: TensorHip::new(std_shape)?,
            wkv_out: TensorHip::new(std_shape)?,
            wkv_normed: TensorHip::new(std_shape)?,
            wkv_bonus: TensorHip::new(std_shape)?,
            att_out: TensorHip::new(std_shape)?,
            ffn_xk: TensorHip::new(std_shape)?,
            ffn_out: TensorHip::new(std_shape)?,
            v_first: TensorHip::new(std_shape)?,

            // FFN hidden buffers
            ffn_k: TensorHip::new(ffn_shape)?,
            ffn_k_sq: TensorHip::new(ffn_shape)?,

            // LoRA buffers
            lora_w: TensorHip::new(lora_w_shape)?,
            lora_a: TensorHip::new(lora_a_shape)?,
            lora_g: TensorHip::new(lora_g_shape)?,
            lora_v: TensorHip::new(lora_v_shape)?,
            lora_w_tanh: TensorHip::new(lora_w_shape)?,
            lora_a_proj: TensorHip::new(std_shape)?,
            lora_g_sig: TensorHip::new(lora_g_shape)?,
            v_lora2: TensorHip::new(std_shape)?,

            // Temporary buffers
            temp1: TensorHip::new(std_shape)?,
            temp2: TensorHip::new(std_shape)?,
            new_att_shift: TensorHip::new(state_shape)?,
            new_ffn_shift: TensorHip::new(state_shape)?,

            // FLA chunked prefill buffers (all f32)
            fla_gi: TensorHip::new(fla_per_token_shape)?,
            fla_ge: TensorHip::new(fla_per_token_shape)?,
            fla_A_qk: TensorHip::new(fla_chunk_mat_shape)?,
            fla_A_qb: TensorHip::new(fla_chunk_mat_shape)?,
            fla_A_ab: TensorHip::new(fla_chunk_mat_shape)?,
            fla_A_ak: TensorHip::new(fla_chunk_mat_shape)?,
            fla_A_ab_inv: TensorHip::new(fla_chunk_mat_shape)?,
            fla_w_wy: TensorHip::new(fla_per_token_shape)?,
            fla_u_wy: TensorHip::new(fla_per_token_shape)?,
            fla_h: TensorHip::new(fla_chunk_state_shape)?,
            fla_v_new: TensorHip::new(fla_per_token_shape)?,

            // Output buffer
            logits: TensorHip::new(out_shape)?,
            logits_f32: TensorHip::new(out_shape)?,
            logits_staging: PinnedBuffer::new(out_size)?,

            // Token staging buffer [T, B]
            token_staging: TensorHip::new(TensorShape::new(t, b, 1, 1))?,

            // Sequence lengths buffer [B]
            lens_gpu: TensorHip::new(TensorShape::new(b, 1, 1, 1))?,

            // Pinned host buffer for async embedding upload [n_embd * T * B]
            emb_staging: PinnedBuffer::new(c * t * b)?,
        })
    }

    /// Reset resident GPU state to zeros.
    pub fn reset_state_gpu(&mut self) -> Result<()> {
        for state in &mut self.att_shift_state_gpu {
            state.fill_zero()?;
        }
        for state in &mut self.ffn_state_gpu {
            state.fill_zero()?;
        }
        for state in &mut self.wkv_state_gpu {
            state.fill_zero()?;
        }
        Ok(())
    }

    /// Calculate total GPU memory used by scratch buffers in bytes.
    pub fn memory_bytes(&self) -> usize {
        let t = self.config.max_prefill_chunk;
        let b = self.config.batch_size;
        let c = self.n_embd;
        let h = self.n_hidden;
        let v = self.n_vocab;
        let ld = &self.lora_dims;

        let std_size = c * t * b;
        let ffn_size = h * t * b;
        let out_size = v * t * b;
        let token_size = t * b; // Token staging buffer

        let std_count = 26; // Number of standard buffers
        let ffn_count = 2; // Number of FFN hidden buffers

        let lora_size = (ld.w_dim + ld.a_dim + ld.g_dim + ld.v_dim.unwrap_or(0)) * t * b;

        // FLA buffer sizes (all f32)
        // Reconstruct dimensions from stored shapes
        let fla_c = self.config.fla_chunk_size;
        let n_head = self.fla_gi.shape().dim(1);
        let head_size = if n_head > 0 { c / n_head } else { 0 };
        let max_total_chunks = b * ((t + fla_c - 1) / fla_c);
        // 5 per-token buffers: fla_gi, fla_ge, fla_w_wy, fla_u_wy, fla_v_new
        // Each is [head_size, n_head, T, B] = head_size * n_head * T * B elements
        let fla_per_token_elements = 5 * head_size * n_head * t * b;
        // 5 chunk-matrix buffers: fla_A_qk, fla_A_qb, fla_A_ab, fla_A_ak, fla_A_ab_inv
        // Each is [C, C, n_head, max_total_chunks]
        let fla_chunk_mat_elements = 5 * fla_c * fla_c * n_head * max_total_chunks;
        // 1 chunk-state buffer: fla_h
        // Shape: [head_size, head_size, n_head, max_total_chunks]
        let fla_chunk_state_elements = head_size * head_size * n_head * max_total_chunks;
        let fla_f32_elements =
            fla_per_token_elements + fla_chunk_mat_elements + fla_chunk_state_elements;

        let f16_elements = std_count * std_size + ffn_count * ffn_size + lora_size + out_size;
        let f32_elements = out_size + fla_f32_elements; // logits_f32 + FLA buffers
        let u32_elements = token_size;
        f16_elements * std::mem::size_of::<f16>()
            + f32_elements * std::mem::size_of::<f32>()
            + u32_elements * std::mem::size_of::<u32>()
    }

    /// Check if buffers are large enough for given sequence length and batch size.
    pub fn supports(&self, seq_len: usize, batch_size: usize) -> bool {
        seq_len <= self.config.max_prefill_chunk && batch_size <= self.config.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HipRuntimeConfig::default();
        assert_eq!(config.max_prefill_chunk, 256);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_config_decode() {
        let config = HipRuntimeConfig::decode();
        assert_eq!(config.max_prefill_chunk, 1);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_config_prefill() {
        let config = HipRuntimeConfig::prefill(512);
        assert_eq!(config.max_prefill_chunk, 512);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_lora_dims() {
        let dims = LoraDims {
            w_dim: 32,
            a_dim: 64,
            g_dim: 128,
            v_dim: Some(32),
        };
        assert_eq!(dims.w_dim, 32);
        assert_eq!(dims.v_dim, Some(32));
    }
}
