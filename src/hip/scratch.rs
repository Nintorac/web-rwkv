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

/// WKV kernel selection for HIP backend.
///
/// The default (`Auto`) keeps existing behavior but allows overriding via
/// `WEB_RWKV_HIP_WKV_KERNEL`. Options:
/// - `auto`      : legacy register kernel unless heuristic chooses otherwise
/// - `register`  : original low-latency register-resident kernel
/// - `wave`      : high-occupancy wave-cooperative kernel (shared memory)
/// - `lds`       : LDS + atomics kernel for experimentation
/// - `wave_t1`   : wave-cooperative kernel specialized for decode (T=1)
/// - `colmajor_t1`: row-owned kernel — 1 thread/row, in-place state, no reductions (T=1)
/// - `batch_loop_t1`: embed-parallel, serial batch loop — mirrors WGPU strategy (T=1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WkvKernelKind {
    Auto,
    Register,
    Tiled,
    WaveReduceT1,
    WaveReduce,
    Lds,
    ColmajorT1,
    FusedT1,
    BatchLoopT1,
}

impl WkvKernelKind {
    /// Parse from environment variable `WEB_RWKV_HIP_WKV_KERNEL`.
    /// Returns Auto when unset or unrecognized.
    fn from_env() -> Self {
        match std::env::var("WEB_RWKV_HIP_WKV_KERNEL") {
            Ok(val) => match val.to_ascii_lowercase().as_str() {
                "register" | "reg" | "orig" => Self::Register,
                "tiled" | "global" => Self::Tiled,
                "wave_t1" | "wave-t1" | "wave32_t1" => Self::WaveReduceT1,
                "wave" | "wave32" | "wave-reduce" | "wave_reduce" => Self::WaveReduce,
                "lds" | "shared" => Self::Lds,
                "colmajor_t1" | "colmajor-t1" | "rowowned" | "row_owned" => Self::ColmajorT1,
                "fused_t1" | "fused-t1" | "fused" => Self::FusedT1,
                "batch_loop_t1" | "batch-loop-t1" | "batch_loop" => Self::BatchLoopT1,
                "auto" | "" => Self::Auto,
                _ => Self::Auto,
            },
            Err(_) => Self::Auto,
        }
    }
}

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

    /// Keep recurrent state resident on device and avoid per-call H2D/D2H transfers.
    /// Default: false
    pub resident_state: bool,

    /// Which WKV kernel implementation to use.
    /// Default: Auto (legacy kernel; override via WEB_RWKV_HIP_WKV_KERNEL).
    pub wkv_kernel: WkvKernelKind,
}

impl Default for HipRuntimeConfig {
    fn default() -> Self {
        Self {
            max_prefill_chunk: 256,
            batch_size: 1,
            resident_state: false,
            wkv_kernel: WkvKernelKind::from_env(),
        }
    }
}

impl HipRuntimeConfig {
    /// Create a new config with specified max chunk size and batch size.
    pub fn new(max_prefill_chunk: usize, batch_size: usize) -> Self {
        Self {
            max_prefill_chunk,
            batch_size,
            resident_state: false,
            wkv_kernel: WkvKernelKind::from_env(),
        }
    }

    /// Create config for single-token decode mode (batch_size=1, chunk=1).
    pub fn decode() -> Self {
        Self {
            max_prefill_chunk: 1,
            batch_size: 1,
            resident_state: false,
            wkv_kernel: WkvKernelKind::from_env(),
        }
    }

    /// Create config for prefill with specified chunk size.
    pub fn prefill(max_chunk: usize) -> Self {
        Self {
            max_prefill_chunk: max_chunk,
            batch_size: 1,
            resident_state: false,
            wkv_kernel: WkvKernelKind::from_env(),
        }
    }

    /// Explicitly set the WKV kernel implementation.
    pub fn with_wkv_kernel(mut self, kernel: WkvKernelKind) -> Self {
        self.wkv_kernel = kernel;
        self
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

/// Pre-allocated scratch buffers for HIP forward pass.
///
/// All buffers are sized for `[dim, max_seq_len, batch_size]` to support
/// any sequence length up to `max_prefill_chunk`. Buffers are reused across
/// forward calls, eliminating allocation overhead.
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
/// let config = HipRuntimeConfig::default();
/// let scratch = HipScratch::new(&model.info, &config)?;
///
/// // Forward pass reuses scratch buffers
/// let logits = model.forward_with_scratch(&tokens, &mut state, &mut scratch)?;
/// ```
#[derive(Debug)]
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

    /// Temporary WKV state output (wkv state shape)
    pub new_wkv_state: TensorHip<f32>,

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
            new_wkv_state: TensorHip::new(wkv_state_shape)?,

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

        let f16_elements = std_count * std_size + ffn_count * ffn_size + lora_size + out_size;
        let f32_elements = out_size; // logits_f32 buffer
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
