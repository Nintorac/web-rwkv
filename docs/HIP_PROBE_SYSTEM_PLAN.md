# HIP Probe System Plan

## Overview

Add a feature-gated probe system to the HIP backend that allows capturing intermediate tensor values during inference for validation against ground truth fixtures.

**Design principles:**
1. Probe the exact runtime code path (no separate instrumented implementation)
2. Zero overhead when disabled (compile-time feature gating)
3. Layer index in context, not hook enum (single registration covers all layers)
4. Simpler than WGPU's hook system (observe-only, no op injection)

## File Changes

### 1. `Cargo.toml` - Add feature flag

```toml
[features]
default = ["...]
hip-probes = []  # Enable HIP intermediate value probing
```

### 2. `src/hip/probe.rs` - New file

```rust
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

/// Context passed to probe callbacks.
pub struct ProbeContext<'a> {
    /// Current layer index (None for embed/head hooks)
    pub layer: Option<usize>,
    /// Current token step in sequence (for multi-token inference)
    pub batch_size: usize,
    /// Sequence length (tokens per batch)
    pub seq_len: usize,
    /// Model configuration
    pub n_embd: usize,
    pub n_head: usize,
    pub head_size: usize,
    pub n_layer: usize,
    /// Shape hint for the data being probed [C, T, B] or similar
    pub shape: &'a [usize],
}

/// Probe callback function type.
///
/// Receives:
/// - `data`: The tensor data as a flat f32 slice
/// - `ctx`: Context with layer index, dimensions, etc.
pub type HipProbeFn = Box<dyn Fn(&[f32], &ProbeContext) + Send + Sync>;

/// Map from hook points to probe callbacks.
pub type HipProbeMap = HashMap<HipHook, HipProbeFn>;

/// Probe macro - compiles to nothing without feature.
///
/// Usage: `probe!(self, ctx, HipHook::PostAttLayerNorm, &x_ln1, &[n_embd, t, b]);`
#[cfg(feature = "hip-probes")]
#[macro_export]
macro_rules! hip_probe {
    ($model:expr, $ctx:expr, $hook:expr, $data:expr, $shape:expr) => {
        if let Some(ref probes) = $model.probes {
            if let Some(f) = probes.get(&$hook) {
                $ctx.shape = $shape;
                f($data, &$ctx);
            }
        }
    };
}

#[cfg(not(feature = "hip-probes"))]
#[macro_export]
macro_rules! hip_probe {
    ($model:expr, $ctx:expr, $hook:expr, $data:expr, $shape:expr) => {};
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
    pub fn on(mut self, hook: HipHook, f: impl Fn(&[f32], &ProbeContext) + Send + Sync + 'static) -> Self {
        self.probes.insert(hook, Box::new(f));
        self
    }

    /// Register probes for multiple hooks with the same callback.
    pub fn on_many(mut self, hooks: &[HipHook], f: impl Fn(&[f32], &ProbeContext) + Send + Sync + Clone + 'static) -> Self {
        for &hook in hooks {
            self.probes.insert(hook, Box::new(f.clone()));
        }
        self
    }

    pub fn build(self) -> HipProbeMap {
        self.probes
    }
}
```

### 3. `src/hip/mod.rs` - Modifications

Add module and re-export:
```rust
#[cfg(feature = "hip-probes")]
pub mod probe;

#[cfg(feature = "hip-probes")]
pub use probe::{HipHook, HipProbeBuilder, HipProbeContext, HipProbeFn, HipProbeMap};
```

Add field to `Rwkv7Hip`:
```rust
pub struct Rwkv7Hip {
    pub info: Rwkv7ModelInfo,
    pub embed: EmbedHip,
    pub layers: Vec<LayerHip>,
    pub head: HeadHip,

    #[cfg(feature = "hip-probes")]
    probes: Option<Arc<HipProbeMap>>,
}
```

Add builder method:
```rust
impl Rwkv7Hip {
    /// Attach probes for capturing intermediate values.
    /// Only available with `hip-probes` feature.
    #[cfg(feature = "hip-probes")]
    pub fn with_probes(mut self, probes: HipProbeMap) -> Self {
        self.probes = Some(Arc::new(probes));
        self
    }
}
```

Update `load()` to initialize probes field:
```rust
// In Rwkv7Hip::load():
Ok(Self {
    info,
    embed,
    layers,
    head,
    #[cfg(feature = "hip-probes")]
    probes: None,
})
```

### 4. Instrument `forward_with_state()`

Insert probe points at each stage. Example (abbreviated):

```rust
pub fn forward_with_state(&self, tokens: &[&[u32]], state: &mut HipState) -> Result<Vec<f32>> {
    // ... setup code ...

    // Initialize probe context (compiles out without feature)
    #[cfg(feature = "hip-probes")]
    let mut probe_ctx = probe::ProbeContext {
        layer: None,
        batch_size: b,
        seq_len: t,
        n_embd,
        n_head,
        head_size,
        n_layer,
        shape: &[],
    };

    // Embedding
    // ... embedding lookup into x ...
    hip_probe!(self, probe_ctx, HipHook::PostEmbed, &x, &[n_embd, t, b]);

    // Layer 0 ln0
    if layer_idx == 0 {
        x = hip_layer_norm(&x, &ln0_w, &ln0_b, n_embd, t * b, LN_EPS)?;
        hip_probe!(self, probe_ctx, HipHook::PostEmbedLayerNorm, &x, &[n_embd, t, b]);
    }

    for layer_idx in 0..n_layer {
        #[cfg(feature = "hip-probes")]
        { probe_ctx.layer = Some(layer_idx); }

        // Attention layer norm
        let x_ln1 = hip_layer_norm(&x, &ln1_w, &ln1_b, n_embd, t * b, LN_EPS)?;
        hip_probe!(self, probe_ctx, HipHook::PostAttLayerNorm, &x_ln1, &[n_embd, t, b]);

        // Token shifts
        let (xr, new_att_shift) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_r, n_embd, t, b)?;
        // ... other shifts ...
        hip_probe!(self, probe_ctx, HipHook::PostAttTokenShift, &xr, &[n_embd, t, b]);

        // Linear projections
        let r = hip_sgemm(&w_r, &xr, n_embd, n_embd, t * b)?;
        let k = hip_sgemm(&w_k, &xk, n_embd, n_embd, t * b)?;
        let v = hip_sgemm(&w_v, &xv, n_embd, n_embd, t * b)?;
        hip_probe!(self, probe_ctx, HipHook::PostAttLinear, &r, &[n_embd, t, b]);

        // ... w, a, g LoRAs with probes ...

        // WKV
        hip_probe!(self, probe_ctx, HipHook::PreWkv, &r, &[n_embd, t, b]);
        let (wkv_output, new_att_state) = hip_wkv7(...)?;
        hip_probe!(self, probe_ctx, HipHook::PostWkv, &wkv_output, &[n_embd, t, b]);

        // ... continue with probes at each stage ...
    }

    #[cfg(feature = "hip-probes")]
    { probe_ctx.layer = None; }

    // Head
    let x_ln_out = hip_layer_norm(&x, &ln_out_w, &ln_out_b, n_embd, t * b, LN_EPS)?;
    hip_probe!(self, probe_ctx, HipHook::PostHeadLayerNorm, &x_ln_out, &[n_embd, t, b]);

    let logits = hip_sgemm(&head_w, &x_ln_out, n_vocab, n_embd, t * b)?;
    hip_probe!(self, probe_ctx, HipHook::PostHead, &logits, &[n_vocab, t, b]);

    Ok(logits)
}
```

### 5. `tests/hip_probes.rs` - New test file

```rust
//! Tests for HIP probe system.
//!
//! Run with: cargo test --features hip-probes hip_probes

#![cfg(feature = "hip-probes")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use web_rwkv::hip::{HipHook, HipProbeBuilder, HipProbeMap, Rwkv7Hip};

#[test]
fn test_probe_captures_intermediates() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    // Storage for captured values
    let captured: Arc<Mutex<HashMap<(HipHook, Option<usize>), Vec<f32>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let captured_clone = captured.clone();
    let probes = HipProbeBuilder::new()
        .on(HipHook::PostAttLayerNorm, move |data, ctx| {
            let key = (HipHook::PostAttLayerNorm, ctx.layer);
            captured_clone.lock().unwrap()
                .insert(key, data[..100].to_vec());  // first 100 values
        })
        .on(HipHook::PostWkv, {
            let captured = captured.clone();
            move |data, ctx| {
                let key = (HipHook::PostWkv, ctx.layer);
                captured.lock().unwrap()
                    .insert(key, data[..100].to_vec());
            }
        })
        .build();

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model")
        .with_probes(probes);

    let _logits = model.forward(&[0, 1, 2]).expect("Forward failed");

    let captured = captured.lock().unwrap();

    // Should have captured layer 0 PostAttLayerNorm
    assert!(captured.contains_key(&(HipHook::PostAttLayerNorm, Some(0))));

    // Should have captured all layers' PostWkv
    for layer in 0..12 {
        assert!(captured.contains_key(&(HipHook::PostWkv, Some(layer))),
            "Missing PostWkv for layer {}", layer);
    }

    println!("Captured {} probe points", captured.len());
}

#[test]
fn test_probe_validates_against_fixture() {
    // Load fixture
    let fixture_path = "tests/fixtures/ground_truth/step_0.npz";
    if !std::path::Path::new(fixture_path).exists() {
        eprintln!("Skipping: fixtures not found");
        return;
    }

    // ... load fixture, compare captured values to expected ...
}
```

## Probe Points Summary

| Hook | Location | Tensor | Shape |
|------|----------|--------|-------|
| `PostEmbed` | After embedding lookup | x | [C, T, B] |
| `PostEmbedLayerNorm` | After ln0 | x | [C, T, B] |
| `PostAttLayerNorm` | After ln1 | x_ln1 | [C, T, B] |
| `PostAttTokenShift` | After all token shifts | xr (representative) | [C, T, B] |
| `PostAttLinear` | After r,k,v projections | r (representative) | [C, T, B] |
| `PostAttDecay` | After w LoRA + softplus | w | [C, T, B] |
| `PostAttAdapt` | After a LoRA + sigmoid | a | [C, T, B] |
| `PostAttGate` | After g LoRA | g | [C, T, B] |
| `PostAttValueResidual` | After v lerp (layer>0) | v | [C, T, B] |
| `PostAttL2Norm` | After L2 norm | kk | [C, T, B] |
| `PostAttControlK` | After control_k | k_ctrl | [C, T, B] |
| `PreWkv` | Before WKV7 kernel | w_decay, r, k_ctrl, v, wkv_a, wkv_b (stacked) | [C, T, B, 6] |
| `PreWkvState` | WKV state before kernel | state.att_states[layer] | [N, N, H, B] |
| `PostWkv` | After WKV7 kernel | wkv_output | [C, T, B] |
| `PostWkvState` | WKV state after kernel | updated state | [N, N, H, B] |
| `PostWkvBonus` | After bonus | wkv_bonus | [C, T, B] |
| `PostAttGroupNorm` | After group norm | x_att_gn | [C, T, B] |
| `PostAttGated` | After gating | x_att_gated | [C, T, B] |
| `PostAttOut` | After output proj | x_att_out | [C, T, B] |
| `PostAtt` | After att residual | x | [C, T, B] |
| `PostFfnLayerNorm` | After ln2 | x_ln2 | [C, T, B] |
| `PostFfnTokenShift` | After FFN shift | xk_ffn | [C, T, B] |
| `PostFfnLinear` | After FFN key proj | k_ffn | [H, T, B] |
| `PostFfnActivate` | After squared relu | k_sq | [H, T, B] |
| `PostFfnOut` | After FFN value proj | x_ffn_out | [C, T, B] |
| `PostFfn` | After FFN residual | x | [C, T, B] |
| `PostHeadLayerNorm` | After final LN | x_ln_out | [C, T, B] |
| `PostHead` | After head proj | logits | [V, T, B] |

## Comparison with WGPU Hooks

| Aspect | WGPU | HIP (this design) |
|--------|------|-------------------|
| Layer index | In Hook enum: `PostAtt(usize)` | In context: `ctx.layer` |
| Registration | Per-layer | Single for all layers |
| Return type | `Result<TensorOp, ...>` | `()` (observe only) |
| Feature gate | Always compiled | `#[cfg(feature = "hip-probes")]` |
| Data access | Async GPU readback | Direct `&[f32]` slice |

## Implementation Order

1. Add `hip-probes` feature to `Cargo.toml`
2. Create `src/hip/probe.rs` with types and macro
3. Update `src/hip/mod.rs`:
   - Conditionally include probe module
   - Add `probes` field to `Rwkv7Hip`
   - Add `with_probes()` method
   - Initialize field in `load()`
4. Instrument `forward_with_state()` with `hip_probe!()` calls
5. Also instrument `forward_with_state_masked()` (same probes)
6. Add `tests/hip_probes.rs` with basic capture test
7. Add fixture validation test

## Design Decisions

### Multi-tensor hooks: Stack into single slice

Some hooks have multiple related tensors. Stack them and indicate count in shape:

```rust
// PostAttTokenShift: 6 tensors stacked
let stacked: Vec<f32> = [&xr, &xw, &xk, &xv, &xa, &xg]
    .into_iter()
    .flatten()
    .copied()
    .collect();
hip_probe!(self, ctx, HipHook::PostAttTokenShift, &stacked, &[n_embd, t, b, 6]);
```

Callback unpacks using shape:
```rust
.on(HipHook::PostAttTokenShift, |data, ctx| {
    let &[c, t, b, n] = ctx.shape else { return };
    let stride = c * t * b;
    let xr = &data[0*stride .. 1*stride];
    let xw = &data[1*stride .. 2*stride];
    // ...
})
```

Hooks with stacked tensors:
| Hook | Tensors | Stack order |
|------|---------|-------------|
| `PostAttTokenShift` | 6 | xr, xw, xk, xv, xa, xg |
| `PostAttLinear` | 3 | r, k, v |
| `PreWkv` | 6 | w_decay, r, k_ctrl, v, wkv_a, wkv_b |

### WKV state probing: Add dedicated hooks

Add `PreWkvState` and `PostWkvState` for capturing the recurrent state `[N, N, H, B]`:
- Separate from `PostWkv` which captures output activation
- Important for validating state evolution

### Epsilon constants: Separate ticket

Tracked in `bd-1xm` (P3) - extract hardcoded epsilon values to named constants.
