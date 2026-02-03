# Plan: Separate Prefill and Decode Modules (bd-2sh.8.17)

## Problem

FLA (prefill) stores WKV state as `state[k*64+v]` (K-rows, V-cols).
FusedT1Wkv (decode) reads `state[v*64+k]` (V-rows, K-cols).
Currently both share the same `HipScratch` buffers, so transitioning
from FLA prefill to FusedT1Wkv decode produces garbled output
(spearman 0.48 vs 0.999 expected).

## Solution

Two standalone module types, each with own scratch buffers and state.
Shared model weights via `Arc`. Explicit state handover via
`get_state()` / `load_state()` with transpose at the boundary.

## Architecture

```
Arc<Rwkv7Model> (shared model weights: layers, embed, head)
  │
  ├── HipPrefill (standalone, owns PrefillScratch)
  │     ├── Arc<Rwkv7Model>
  │     ├── PrefillScratch (FLA buffers + intermediates, T=max_chunk)
  │     ├── wkv_state in FLA layout [K_row, V_col]
  │     ├── prefill(tokens) → logits
  │     └── get_state() → HipState
  │
  └── HipDecode (standalone, owns DecodeScratch)
        ├── Arc<Rwkv7Model>
        ├── DecodeScratch (intermediates only, T=1, no FLA buffers)
        ├── wkv_state in decode layout [V_row, K_col]
        ├── load_state(HipState) — transpose + load, once
        └── decode(token) → logits
```

Usage:
```rust
let model = Arc::new(Rwkv7Model::load("model.st")?);

// Can create N of either independently
let prefill = HipPrefill::new(model.clone(), prefill_config)?;
let decode  = HipDecode::new(model.clone(), decode_config)?;

// Prefill, hand off, decode
let logits = prefill.prefill(&tokens)?;
let state = prefill.get_state();
decode.load_state(&state)?;
loop {
    let logits = decode.decode(&[next_token])?;
}
```

Future scenarios this enables:
- Multiple HipDecode on one GPU, no HipPrefill (serving cached states)
- HipPrefill only (state export for remote decode)
- N:M prefill:decode ratios

## HipState

GPU-resident, self-contained, transferable between modules.

```rust
pub struct HipState {
    pub att_shift: Vec<TensorHip<f16>>,   // [n_embd, B] per layer
    pub ffn_state: Vec<TensorHip<f16>>,   // [n_embd, B] per layer
    pub wkv_state: Vec<TensorHip<f32>>,   // [K, K, H, B] per layer
    pub layout: StateLayout,
}

pub enum StateLayout { Fla, Decode }
```

`load_state()` checks the layout tag. If source is Fla and target
is Decode (or vice versa), it transposes each K×K wkv block.
att_shift and ffn_state are always direct copies (no layout diff).

## Rwkv7Model (shared weights)

Extract model weights from current Rwkv7Hip into a standalone struct:

```rust
pub struct Rwkv7Model {
    pub info: Rwkv7ModelInfo,
    pub embed: EmbedHip,
    pub head: HeadHip,
    pub layers: Vec<LayerHip>,
}
```

`Rwkv7Hip` becomes a convenience wrapper (or is replaced entirely):
```rust
// Option A: Rwkv7Hip wraps both for backward compat
pub struct Rwkv7Hip {
    prefill: HipPrefill,
    decode: HipDecode,
}

// Option B: Remove Rwkv7Hip, callers use HipPrefill/HipDecode directly
// HipRuntime coordinates for the Runtime<Rnn> trait
```

## Implementation Phases

### Phase 1: Extract Rwkv7Model + transpose kernel
- Extract shared weights into `Rwkv7Model` in `src/hip/model/mod.rs`
- Wrap in `Arc` wherever referenced
- Add `kernel_state_transpose` to `fla.hip` (out-of-place: src→dst)
- FFI + Rust wrapper
- Define `HipState` and `StateLayout`

### Phase 2: DecodeScratch + HipDecode
- `DecodeScratch` in `src/hip/scratch.rs`: T=1 buffers, no FLA
- `HipDecode` struct: holds `Arc<Rwkv7Model>` + `DecodeScratch`
- `load_state()`: transpose wkv if needed, copy att_shift/ffn
- `decode(tokens) -> logits`: dispatch with FusedT1Wkv only

### Phase 3: Rename HipScratch → PrefillScratch + HipPrefill
- Rename existing scratch (already has all prefill buffers)
- `HipPrefill` struct: holds `Arc<Rwkv7Model>` + `PrefillScratch`
- `get_state() -> HipState`: clone state tensors, tag Fla layout
- `prefill(tokens) -> logits`: dispatch with FLA only

### Phase 4: Factor dispatch into helpers
- Extract from `step.rs::dispatch()`:
  - `embed_lookup(...)` — embedding + ln0
  - `attention_block(...)` — att LN, shift, LoRA, WKV, gating, proj
  - `ffn_block(...)` — FFN LN, shift, key/value, residual
  - `output_head(...)` — head LN, projection, logits staging
- These take explicit tensor refs, usable by both HipPrefill and HipDecode
- `HipPrefill::dispatch()` calls helpers with FLA for WKV
- `HipDecode::dispatch()` calls helpers with FusedT1Wkv for WKV
- Delete old unified `dispatch()`

### Phase 5: Update HipRuntime for backward compat
- `HipRuntime` holds `HipPrefill` + `HipDecode`
- `step(tokens, state)`: T>1 → prefill, then get_state/load_state, T=1 → decode
- `Runtime<Rnn>` trait: same coordination
- Existing callers work unchanged

### Phase 6: Validate
- test_mixed_fla_prefill_then_fused_decode: spearman >= 0.999
- All existing functional_metrics tests pass
- Pure prefill and pure decode tests unchanged

## Key Files

| File | Change |
|------|--------|
| `src/hip/model/mod.rs` | Extract Rwkv7Model, define HipState, HipPrefill, HipDecode |
| `src/hip/scratch.rs` | Split into PrefillScratch + DecodeScratch |
| `src/hip/model/step.rs` | Factor dispatch into reusable helpers |
| `src/hip/runtime.rs` | Update HipRuntime to use HipPrefill + HipDecode |
| `src/hip/kernels/fla.hip` | Out-of-place state transpose kernel |
| `src/hip/kernels/fla.rs` | FFI wrapper for transpose |
| `tests/functional_metrics.rs` | Validate mixed-path test |

## Memory Impact

- PrefillScratch: ~95 MB (same as current)
- DecodeScratch: ~12 MB (T=1 only, no FLA buffers)
- HipState (during handoff): ~5 MB (cloned state tensors)
- Model weights (shared via Arc): ~200 MB (loaded once)

## Verification

```bash
cargo test --features hip test_mixed_fla_prefill_then_fused_decode -- --nocapture
cargo test --features hip --test functional_metrics -- --nocapture
cargo test --features hip --test ground_truth -- --nocapture
```
