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
                $ctx.set_shape(&[$($shape),*]);
                f($data, &$ctx);
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
