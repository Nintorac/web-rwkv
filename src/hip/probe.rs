//! Feature-gated probe system for HIP backend validation.
//!
//! When `hip-probes` feature is enabled, allows registering hooks
//! to capture intermediate tensor values during forward pass.

use std::collections::HashMap;
use std::sync::Arc;

/// Hook points in the HIP forward pass.
///
/// Unlike WGPU's Hook enum, layer index is NOT embedded here.
/// Instead, layer is passed via ProbeContext.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HipHook {
    // === Embedding ===
    /// After embedding lookup, before ln0
    PostEmbed,
    /// After ln0 (layer 0 only)
    PostEmbedLayerNorm,

    // === Per-layer Attention ===
    /// After attention layer norm (x_ln1)
    PostAttLayerNorm,
    /// After token shift (xr, xw, xk, xv, xa, xg computed)
    PostAttTokenShift,
    /// After r, k, v linear projections
    PostAttLinear,
    /// After w LoRA + softplus_decay
    PostAttDecay,
    /// After a LoRA + sigmoid
    PostAttAdapt,
    /// After g LoRA
    PostAttGate,
    /// After value residual (v lerped with v_first, layers > 0)
    PostAttValueResidual,
    /// After L2 norm of k (kk)
    PostAttL2Norm,
    /// After control_k (k_ctrl)
    PostAttControlK,
    /// Before WKV7 kernel (w_decay, r, k_ctrl, v, wkv_a, wkv_b ready)
    PreWkv,
    /// WKV state before kernel (state.att_states[layer])
    PreWkvState,
    /// After WKV7 kernel (wkv_output)
    PostWkv,
    /// WKV state after kernel (updated state.att_states[layer])
    PostWkvState,
    /// After WKV bonus computation
    PostWkvBonus,
    /// After group norm on attention output
    PostAttGroupNorm,
    /// After gating (x_att_gated)
    PostAttGated,
    /// After output projection (x_att_out)
    PostAttOut,
    /// After attention residual (x += x_att_out)
    PostAtt,

    // === Per-layer FFN ===
    /// After FFN layer norm (x_ln2)
    PostFfnLayerNorm,
    /// After FFN token shift (xk_ffn)
    PostFfnTokenShift,
    /// After FFN key projection (k_ffn)
    PostFfnLinear,
    /// After squared ReLU (k_sq)
    PostFfnActivate,
    /// After FFN value projection (x_ffn_out)
    PostFfnOut,
    /// After FFN residual (x += x_ffn_out)
    PostFfn,

    // === Head ===
    /// After final layer norm
    PostHeadLayerNorm,
    /// After head projection (logits)
    PostHead,
}

/// Maximum dimensions for shape array.
pub const MAX_SHAPE_DIMS: usize = 6;

/// Context passed to probe callbacks.
#[derive(Clone)]
pub struct ProbeContext {
    /// Current layer index (None for embed/head hooks)
    pub layer: Option<usize>,
    /// Batch size (number of sequences)
    pub batch_size: usize,
    /// Sequence length (tokens per batch)
    pub seq_len: usize,
    /// Padded chunk size (GPU tensor T dimension)
    pub chunk_size: usize,
    /// Model configuration
    pub n_embd: usize,
    pub n_head: usize,
    pub head_size: usize,
    pub n_layer: usize,
    /// Shape hint for the data being probed [C, T, B] or similar (internal storage)
    pub shape_storage: [usize; MAX_SHAPE_DIMS],
    /// Length of valid entries in shape_storage
    pub shape_len: usize,
}

impl ProbeContext {
    /// Get the shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape_storage[..self.shape_len]
    }

    /// Set the shape from a slice.
    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape_len = shape.len().min(MAX_SHAPE_DIMS);
        self.shape_storage[..self.shape_len].copy_from_slice(&shape[..self.shape_len]);
    }
}

/// Trim downloaded GPU data to match the declared shape.
///
/// GPU tensors are allocated for the full padded chunk_size, but probes declare
/// shapes with actual_t. This function extracts just the real data, handling
/// the column-major stride correctly for batch>1.
///
/// The T dimension is always at index 1 in our shapes. For shapes without a T
/// dimension (like WKV state), no trimming is needed (data already matches shape).
pub fn trim_padded_data(data: &[f32], chunk_size: usize, shape: &[usize]) -> Vec<f32> {
    let expected: usize = shape.iter().product();

    // No trimming needed if data already matches, or shape has < 2 dims
    if data.len() <= expected || shape.len() < 2 {
        return data[..expected.min(data.len())].to_vec();
    }

    let actual_t = shape[1];

    // If T equals chunk_size, no padding to trim
    if actual_t == chunk_size {
        return data[..expected.min(data.len())].to_vec();
    }

    let inner = shape[0]; // contiguous elements per token
    let outer_count: usize = shape[2..].iter().product(); // batch * any extra dims
    let phys_block = inner * chunk_size; // physical stride per outer block
    let real_block = inner * actual_t; // real data per outer block

    let mut result = Vec::with_capacity(expected);
    for i in 0..outer_count {
        let src_start = i * phys_block;
        let src_end = src_start + real_block;
        if src_end <= data.len() {
            result.extend_from_slice(&data[src_start..src_end]);
        }
    }
    result
}

/// Probe callback function type.
///
/// Receives:
/// - `data`: The tensor data as a flat f32 slice
/// - `ctx`: Context with layer index, dimensions, etc.
pub type HipProbeFn = Box<dyn Fn(&[f32], &ProbeContext) + Send + Sync>;

/// Map from hook points to probe callbacks.
pub type HipProbeMap = HashMap<HipHook, HipProbeFn>;

/// Shared probe map wrapped in Arc for model storage.
pub type HipProbeMapRef = Arc<HipProbeMap>;

/// Probe macro - compiles to nothing without feature.
///
/// Usage: `hip_probe!(self, ctx, HipHook::PostAttLayerNorm, &x_ln1, [n_embd, t, b]);`
/// Note: Pass shape as `[a, b, c]` (array literal) NOT `&[a, b, c]`
#[macro_export]
macro_rules! hip_probe {
    ($model:expr, $ctx:expr, $hook:expr, $data:expr, [$($shape:expr),* $(,)?]) => {
        if let Some(ref probes) = $model.probes {
            if let Some(f) = probes.get(&$hook) {
                let shape = [$($shape),*];
                $ctx.set_shape(&shape);
                let expected_len: usize = shape.iter().product();
                if $data.len() == expected_len {
                    f($data, &$ctx);
                } else {
                    let trimmed = $crate::hip::probe::trim_padded_data($data, $ctx.chunk_size, &shape);
                    f(&trimmed, &$ctx);
                }
            }
        }
    };
}

/// Builder for probe maps with convenient registration.
#[derive(Default)]
pub struct HipProbeBuilder {
    probes: HipProbeMap,
}

impl HipProbeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a probe for a hook point.
    pub fn on(
        mut self,
        hook: HipHook,
        f: impl Fn(&[f32], &ProbeContext) + Send + Sync + 'static,
    ) -> Self {
        self.probes.insert(hook, Box::new(f));
        self
    }

    /// Register probes for multiple hooks with the same callback.
    pub fn on_many(
        mut self,
        hooks: &[HipHook],
        f: impl Fn(&[f32], &ProbeContext) + Send + Sync + Clone + 'static,
    ) -> Self {
        for &hook in hooks {
            self.probes.insert(hook, Box::new(f.clone()));
        }
        self
    }

    pub fn build(self) -> HipProbeMap {
        self.probes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_no_padding() {
        // Data already matches shape -- no trimming
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = trim_padded_data(&data, 1, &[3, 1, 2]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_trim_single_batch() {
        // shape=[2, 1, 1], chunk_size=4, data has 2*4*1=8 elements
        // Should extract first 2 elements (inner=2, actual_t=1)
        let data = vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = trim_padded_data(&data, 4, &[2, 1, 1]);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_trim_multi_batch() {
        // shape=[2, 1, 2], chunk_size=4, data has 2*4*2=16 elements
        // Batch 0: [1,2, pad,pad,pad,pad,pad,pad]
        // Batch 1: [3,4, pad,pad,pad,pad,pad,pad]
        let mut data = vec![0.0; 16];
        data[0] = 1.0;
        data[1] = 2.0;
        data[8] = 3.0;
        data[9] = 4.0;
        let result = trim_padded_data(&data, 4, &[2, 1, 2]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_trim_multi_batch_multi_token() {
        // shape=[2, 2, 2], chunk_size=4, data has 2*4*2=16 elements
        // inner=2, actual_t=2, outer_count=2
        // Batch 0: [1,2,3,4, pad,pad,pad,pad]
        // Batch 1: [5,6,7,8, pad,pad,pad,pad]
        let mut data = vec![0.0; 16];
        // batch 0: tokens at offsets 0..4
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;
        data[3] = 4.0;
        // batch 1: tokens at offsets 8..12
        data[8] = 5.0;
        data[9] = 6.0;
        data[10] = 7.0;
        data[11] = 8.0;
        let result = trim_padded_data(&data, 4, &[2, 2, 2]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_trim_stacked_tensor() {
        // shape=[2, 1, 1, 3], chunk_size=4
        // 3 stacked tensors, each [2, 1, 1]
        // Total physical: 2*4*1*3 = 24
        // outer_count = 1 * 3 = 3
        let mut data = vec![0.0; 24];
        data[0] = 1.0;
        data[1] = 2.0;
        data[8] = 3.0;
        data[9] = 4.0;
        data[16] = 5.0;
        data[17] = 6.0;
        let result = trim_padded_data(&data, 4, &[2, 1, 1, 3]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_trim_no_t_dimension() {
        // WKV state: shape=[2, 2, 3, 1], data matches exactly
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let result = trim_padded_data(&data, 4, &[2, 2, 3, 1]);
        assert_eq!(result, data);
    }

    #[test]
    fn test_trim_chunk_equals_actual() {
        // shape=[2, 4, 1], chunk_size=4 -- actual_t == chunk_size, no trimming
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let result = trim_padded_data(&data, 4, &[2, 4, 1]);
        assert_eq!(result, data);
    }
}
