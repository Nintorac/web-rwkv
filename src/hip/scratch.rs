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

use super::ffi::Result;
use super::model::Rwkv7ModelInfo;
use super::pinned::PinnedBuffer;
use super::tensor::{TensorHip, TensorShape};

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
}

impl Default for HipRuntimeConfig {
    fn default() -> Self {
        Self {
            max_prefill_chunk: 256,
            batch_size: 1,
        }
    }
}

impl HipRuntimeConfig {
    /// Create a new config with specified max chunk size and batch size.
    pub fn new(max_prefill_chunk: usize, batch_size: usize) -> Self {
        Self {
            max_prefill_chunk,
            batch_size,
        }
    }

    /// Create config for single-token decode mode (batch_size=1, chunk=1).
    pub fn decode() -> Self {
        Self {
            max_prefill_chunk: 1,
            batch_size: 1,
        }
    }

    /// Create config for prefill with specified chunk size.
    pub fn prefill(max_chunk: usize) -> Self {
        Self {
            max_prefill_chunk: max_chunk,
            batch_size: 1,
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

    /// Model dimensions for validation
    pub n_embd: usize,
    pub n_hidden: usize,
    pub n_vocab: usize,

    /// LoRA dimensions
    pub lora_dims: LoraDims,

    // ========== Standard buffers [n_embd, T, B] ==========

    /// Main hidden state (persists across layers within forward pass)
    pub x: TensorHip<f32>,

    /// Layer norm output (reused for att_ln, ffn_ln, head_ln)
    pub x_ln: TensorHip<f32>,

    /// Token-shifted inputs for attention
    pub att_xr: TensorHip<f32>,
    pub att_xw: TensorHip<f32>,
    pub att_xk: TensorHip<f32>,
    pub att_xv: TensorHip<f32>,
    pub att_xa: TensorHip<f32>,
    pub att_xg: TensorHip<f32>,

    /// Linear projections
    pub att_r: TensorHip<f32>,
    pub att_k: TensorHip<f32>,
    pub att_v: TensorHip<f32>,

    /// Decay (after softplus transformation)
    pub att_w: TensorHip<f32>,

    /// Adaptation factor (after sigmoid)
    pub att_a: TensorHip<f32>,

    /// Gate (after LoRA projection)
    pub att_g: TensorHip<f32>,

    /// L2-normalized key
    pub att_kk: TensorHip<f32>,

    /// Controlled key (k * (1 + (a-1) * k_a))
    pub att_k_ctrl: TensorHip<f32>,

    /// WKV input: -kk
    pub wkv_a: TensorHip<f32>,

    /// WKV input: kk * a
    pub wkv_b: TensorHip<f32>,

    /// Decay for WKV: exp(w)
    pub w_decay: TensorHip<f32>,

    /// WKV7 output
    pub wkv_out: TensorHip<f32>,

    /// Group-normalized WKV output
    pub wkv_normed: TensorHip<f32>,

    /// WKV bonus (time_first contribution)
    pub wkv_bonus: TensorHip<f32>,

    /// Attention output (combined, gated, projected - can share storage)
    pub att_out: TensorHip<f32>,

    /// FFN token-shifted input
    pub ffn_xk: TensorHip<f32>,

    /// FFN output
    pub ffn_out: TensorHip<f32>,

    /// Value from first layer for residual (stored across layers)
    pub v_first: TensorHip<f32>,

    // ========== FFN hidden buffers [n_hidden, T, B] ==========

    /// FFN key projection output
    pub ffn_k: TensorHip<f32>,

    /// FFN squared ReLU output
    pub ffn_k_sq: TensorHip<f32>,

    // ========== LoRA buffers [lora_dim, T, B] ==========

    /// Decay LoRA intermediate
    pub lora_w: TensorHip<f32>,

    /// Adaptation LoRA intermediate
    pub lora_a: TensorHip<f32>,

    /// Gate LoRA intermediate
    pub lora_g: TensorHip<f32>,

    /// Value residual LoRA intermediate (may be empty if v_dim is None)
    pub lora_v: TensorHip<f32>,

    // ========== Output buffer [n_vocab, T, B] ==========

    /// Final logits output
    pub logits: TensorHip<f32>,

    // ========== Token staging buffer [T, B] ==========

    /// GPU staging buffer for tokens (zero-copy chunking)
    pub token_staging: TensorHip<u32>,

    // ========== Pinned host staging buffer [n_embd * T * B] ==========

    /// Pinned host buffer for async embedding upload (avoids sync on hipMemcpyAsync)
    pub emb_staging: PinnedBuffer<f32>,
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

        // LoRA shapes
        let lora_w_shape = TensorShape::new(lora_dims.w_dim, t, b, 1);
        let lora_a_shape = TensorShape::new(lora_dims.a_dim, t, b, 1);
        let lora_g_shape = TensorShape::new(lora_dims.g_dim, t, b, 1);
        let lora_v_shape = TensorShape::new(lora_dims.v_dim.unwrap_or(1), t, b, 1);

        Ok(Self {
            config,
            n_embd: c,
            n_hidden: h,
            n_vocab: v,
            lora_dims,

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

            // Output buffer
            logits: TensorHip::new(out_shape)?,

            // Token staging buffer [T, B]
            token_staging: TensorHip::new(TensorShape::new(t, b, 1, 1))?,

            // Pinned host buffer for async embedding upload [n_embd * T * B]
            emb_staging: PinnedBuffer::new(c * t * b)?,
        })
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
        let ffn_count = 2;  // Number of FFN hidden buffers

        let lora_size = (ld.w_dim + ld.a_dim + ld.g_dim + ld.v_dim.unwrap_or(0)) * t * b;

        let f32_elements = std_count * std_size + ffn_count * ffn_size + lora_size + out_size;
        let u32_elements = token_size;
        f32_elements * std::mem::size_of::<f32>() + u32_elements * std::mem::size_of::<u32>()
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
