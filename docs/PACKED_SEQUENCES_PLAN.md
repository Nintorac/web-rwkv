# Switch prefill to packed/concatenated sequences (no rectangular padding)

## Problem

`prefill()` pads all sequences to `max_len`, creating a `B × max_len` rectangular grid. With `[125, 1, 1, 1]` tokens, this becomes `4 × 125 = 500` padded tokens, exceeding the 256-token buffer. The reference Triton FLA concatenates sequences flat: `[tok0..tok124, tok125, tok126, tok127]` = 128 total tokens, tracked via `cu_seqlens = [0, 125, 126, 127, 128]`.

## What stays the same

- **All element-wise ops** (layernorm, sigmoid, tanh, mul, GEMM, group_norm, wkv_bonus, control_k, decay_exp, v_first): just see `[dim, T_total, 1, 1]` instead of `[dim, T_max, B, 1]` — same total elements, no code changes
- **WKV state**: `[head_size, head_size, n_head, batch_size]` — one per sequence, indexed by seq_id, unchanged
- **Shift state buffers**: `[n_embd, batch_size]` — one per sequence, unchanged
- **FLA kernel internals**: already use `cu_seqlens` and `chunk_indices` for bounds

## What changes

### 1. `hip_prefill.rs` — `prefill()`
- Remove rectangular padding (lines 350-372, the `v.resize(max_len, 0)` block)
- Validation: `sum(lens) <= chunk_size` instead of `batch_size * max_len <= chunk_size`
- Always use the packed logits extraction branch (the existing `else` at line 394)

### 2. `hip_prefill.rs` — `dispatch_fla()`
- `t` changes from `tokens[0].len()` (max_len) to `lens.iter().sum()` (total tokens)
- All tensor shapes: `[dim, t_total, 1, 1]` instead of `[dim, t_max, b, 1]`
- Upload `batch_offsets` (cu_seqlens) to GPU alongside `lens_gpu`
- FLA kernel shape: `[head_size, n_head, t_total, 1]` instead of `[head_size, n_head, t_max, b]`
- Pass `batch_offsets_gpu` to shift state calls

### 3. `dispatch_helpers.rs` — `embed_lookup()`
- Pack sequences contiguously using cu_seqlens offsets
- Currently: `dst_offset = batch_idx * t * n_embd + time_idx * n_embd`
- Becomes: `dst_offset = (cu_seqlens[batch_idx] + time_idx) * n_embd`

### 4. `dispatch_helpers.rs` — `attention_block()` / `ffn_block()` shift state calls
- Pass `batch_offsets_gpu` to the shift kernels
- All 6 att shifts + 1 ffn shift use the same updated kernel

### 5. `rwkv_ops.hip` — shift state kernels
Both `kernel_channel_mix_state_f16` and `kernel_channel_mix_state_f16_masked` need `batch_offsets`:
```c
// Before (rectangular):
int offset = b * T * C + t * C + c;

// After (packed):
int bos = batch_offsets[b];
int offset = (bos + t) * C + c;
```
Can unify masked/non-masked into one kernel since both now need `lengths` + `batch_offsets`. The f32 variants too if still used.

### 6. `rwkv_ops.rs` + `ffi.rs` — Rust wrappers / FFI
- Update signatures to accept `batch_offsets: &TensorHip<i32>`
- Update launch calls

### 7. `fla.rs` — `batch_offsets` computation
The one-line fix:
```rust
// Before (rectangular stride):
let batch_offsets_host: Vec<i32> = (0..bb).map(|b| (b * t) as i32).collect();

// After (packed = cu_seqlens):
let batch_offsets_host: Vec<i32> = cu_seqlens_host[..bb].iter().map(|&x| x as i32).collect();
```
Already computed correctly from `lengths` on the line above.

### 8. `scratch.rs` — buffer sizing
- Standard buffers: `[dim, max_prefill_chunk, 1, 1]` instead of `[dim, max_prefill_chunk, batch_size, 1]` — uses less memory
- FLA per-token: `[head_size, n_head, max_prefill_chunk, 1]` instead of `[..., batch_size]`
- Add `batch_offsets_gpu: TensorHip<i32>` buffer `[batch_size + 1, 1, 1, 1]`
- Chunk matrix/state buffers: keep current sizing (overestimate is safe)

### 9. `decode.rs` — `dispatch_decode()` unified kernel interface
- Compute identity `batch_offsets[b] = b` (T=1, so offset = batch index)
- Upload to GPU, pass through to updated dispatch helpers
- No behavioral change: `(b + 0) * C + c = b * 1 * C + c`, same as rectangular with T=1

### 10. `runtime.rs` — remove empty-batch padding
- Delete the `infer_rnn` loop that pads empty batches with token 0 (lines 533-537)
- With packed format, zero-length batches are safe: shift kernels copy state and return (`lengths[b] <= 0` guard), FLA chunk_h returns early (`T <= 0`), element-wise ops see 0 contributed tokens
- Also remove debug `eprintln`

## Verification

```
cargo test --package hip-rwkv test_hip_runtime_multichunk_prefill_with_empty_batches -- --nocapture
```
Expects: 2+ chunks, top-10 overlap >= 8/10 vs direct inference.

Also run existing tests to check no regressions:
```
cargo test --package hip-rwkv test_hip_runtime_infer_proof_of_life -- --nocapture
cargo test --package hip-rwkv test_hip_runtime_stateful_inference -- --nocapture
```
