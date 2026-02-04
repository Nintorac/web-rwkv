# FLA (Flash Linear Attention) Implementation Plan for RWKV7 HIP Backend

## Overview

**Goal**: Implement efficient prefill for RWKV7 on AMD GPUs (HIP) using Flash Linear Attention (FLA) chunked parallel processing.

**Problem**: The current HIP WKV7 kernel processes tokens sequentially (recurrent mode), which is O(T) for prefill. For long prompts (2048+ tokens), this is a major bottleneck.

**Solution**: FLA chunked parallel processing divides the sequence into chunks, processes within-chunk in parallel using matrix operations, and transfers state between chunks sequentially. This reduces sequential depth from O(T) to O(T/C) where C is chunk size (64-128).

---

## Implementation Options Analyzed

### Option A: Native HIP Port (Recommended)
Port the FLA algorithm to HIP C++, using fla-org Triton code as reference.
- **Pros**: Clean integration with existing web-rwkv, full control, optimized for gfx1151
- **Cons**: More implementation effort

### Option B: Triton FFI (Experimental)
AOT compile Triton kernels to `.hsaco`, load via HIP runtime.
- **How it would work**:
  1. AOT compile with `triton.compile(target=GPUTarget("hip", 'gfx1151', 32))`
  2. Load `.hsaco` via `hipModuleLoad()`
  3. Launch via `hipModuleLaunchKernel()`
- **Pros**: Reuses existing FLA Triton code directly
- **Cons**: No official tooling, manual metadata management, version coupling, debugging difficulty

### Option C: Python Subprocess (Fallback)
Use rwkv-fla from Python, call from Rust via subprocess/IPC.
- **Pros**: Zero porting, uses existing code
- **Cons**: Performance overhead, adds Python dependency, complex integration

**Selected Approach**: Option A (Native HIP Port)

**Rationale**:
- Clean integration with existing web-rwkv Rust/HIP architecture
- Existing recurrent kernel provides exact reference for correctness testing
- Full control over precision (FP32 state, pairwise summation patterns already established)
- Chunk size configurable at runtime (default 64, adjustable based on precision/performance needs)
- No external dependencies or version coupling issues

---

## Appendix: Triton FFI Research (Not Pursuing)

*Documented for reference. Decided against due to lack of official tooling.*

If pursuing the Triton FFI route, here's what would be needed:

### Step 1: Environment Setup
```bash
# Install ROCm Triton
pip install triton

# Verify HIP linker path
export TRITON_HIP_LLD_PATH="$(rocm-sdk path --root)/llvm/bin/ld.lld"
```

### Step 2: AOT Compile FLA Kernel
```python
import triton
from triton.backends.compiler import GPUTarget
from fla.ops.rwkv7.chunk import chunk_rwkv7

# Compile for gfx1151 (RDNA 3.5, wave32)
target = GPUTarget("hip", "gfx1151", 32)
kernel_src = open("path/to/chunk_kernel.py").read()
output = triton.compile(kernel_src, target=target)
# Output: kernel.hsaco, kernel.json (metadata)
```

### Step 3: Load from Rust/HIP
```rust
extern "C" {
    fn hipModuleLoad(module: *mut hipModule_t, fname: *const c_char) -> hipError_t;
    fn hipModuleGetFunction(func: *mut hipFunction_t, module: hipModule_t, name: *const c_char) -> hipError_t;
    fn hipModuleLaunchKernel(
        func: hipFunction_t,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
}

fn load_triton_kernel(hsaco_path: &str) -> Result<hipFunction_t> {
    let mut module: hipModule_t = std::ptr::null_mut();
    let mut func: hipFunction_t = std::ptr::null_mut();

    unsafe {
        hipModuleLoad(&mut module, hsaco_path.as_ptr());
        hipModuleGetFunction(&mut func, module, b"chunk_rwkv7_fwd\0".as_ptr());
    }
    Ok(func)
}
```

### Challenges to Address
1. **Argument layout**: Parse `kernel.json` for parameter types, offsets
2. **Grid/block sizes**: Extract from metadata or compute at runtime
3. **Version management**: hsaco files are ROCm version-specific
4. **Build pipeline**: Need Python step in build.rs to pre-compile kernels

### Proof-of-Concept Milestones
1. [ ] Successfully AOT compile a simple Triton kernel for gfx1151
2. [ ] Load and run the hsaco from HIP C++ code
3. [ ] Load and run from Rust via FFI
4. [ ] Apply to actual FLA chunk kernel

### Assessment
This approach is **high-risk, high-reward**. If it works, we get the FLA kernels with minimal porting. But it's uncharted territory with no official support.

---

## Background Research Summary

### What is FLA?

Flash Linear Attention is an efficient algorithm for linear attention models that:
1. **Chunks the sequence** into segments of size C (typically 64-128)
2. **Intra-chunk**: Uses parallel matrix operations (GEMM) within each chunk
3. **Inter-chunk**: Sequentially propagates state between chunks

**Key References**:
- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) - Reference Triton implementation
- [Tiled Flash Linear Attention (TFLA)](https://arxiv.org/abs/2503.14376) - NeurIPS 2025 paper on efficient kernels
- [RWKV-FLA](https://wiki.rwkv.com/advance/RWKV-FLA.html) - RWKV-specific FLA implementation

### RWKV7 WKV State Update

The core computation for RWKV7 attention:
```
state_t = state_{t-1} * G_t + v_t^T · k̃_t

where G_t = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)  [transition matrix]
```

This can be chunked by:
1. Computing cumulative decay products within each chunk
2. Computing local state contributions within each chunk (parallel GEMM)
3. Propagating state across chunks sequentially

---

## Current HIP Backend State

**Files**:
- `/workspace/web-rwkv/src/hip/mod.rs` - Main HIP backend (63k lines)
- `/workspace/web-rwkv/src/hip/runtime.rs` - HipRuntime wrapper
- `/workspace/web-rwkv/src/hip/kernels/copy.hip` - HIP C++ kernels

**Existing WKV7 Kernel** (`kernel_wkv7_f32` at copy.hip:897-990):
- Sequential processing: loops over T tokens
- Grid: (H, B) blocks, Block: N=64 threads
- Each thread maintains one row of NxN state matrix in registers
- Uses pairwise summation for FP32 precision

**Target Hardware**: AMD Radeon 8060S (gfx1151, RDNA 3.5, Wave32)

---

## Implementation Approach: Hybrid (rocBLAS + Custom HIP)

| Operation | Implementation | Rationale |
|-----------|---------------|-----------|
| Intra-chunk GEMM | rocBLAS `sgemm_strided_batched` | Tuned for AMD GPUs |
| Cumulative decay scan | Custom HIP kernel | Parallel prefix sum |
| Inter-chunk state propagation | Custom HIP kernel | Sequential dependency |
| Output finalization | Custom HIP kernel | Combine local + propagated |

---

## Algorithm: Chunked WKV7

### Step 1: Divide sequence into chunks
```
num_chunks = ceil(T / C)  // C = 64 or 128
```

### Step 2: Intra-chunk cumulative decay (parallel within chunk)
For each chunk c, compute:
```
W_cum[c, i] = product(w[c*C + j] for j = 0 to i)  // cumulative decay
W_total[c] = product(w[c*C + j] for j = 0 to C-1)  // total chunk decay
```

### Step 3: Intra-chunk local contributions (GEMM)
For each chunk c, compute local state contribution:
```
S_local[c] = sum over i in [0, C): W_rev[i] * outer(v[i], k[i])
```
This can be expressed as batched GEMM.

### Step 4: Inter-chunk state propagation (sequential)
```
state[0] = initial_state
for c = 0 to num_chunks-1:
    state[c+1] = W_total[c] * state[c] + S_local[c]
```

### Step 5: Intra-chunk output computation (GEMM + correction)
```
O_local[c, i] = q[i] · state_local[c, i]^T  // local contribution
O_from_state[c, i] = q[i] · (W_cum[i] * state[c])^T  // from propagated state
O[c, i] = O_local[c, i] + O_from_state[c, i]
```

---

## Kernel Designs

### Kernel 1: `kernel_wkv7_chunk_scan`
**Purpose**: Compute cumulative decay products and prepare for GEMM

**Grid**: `dim3(num_chunks, H, B)`
**Block**: `dim3(64)` (2 wave32 wavefronts)

### Kernel 2: `kernel_wkv7_chunk_gemm_local`
**Purpose**: Compute local state contributions using rocBLAS batched GEMM

**Wraps**: `rocblas_sgemm_strided_batched`

### Kernel 3: `kernel_wkv7_chunk_propagate`
**Purpose**: Sequential state propagation across chunks

**Grid**: `dim3(H, B)`
**Block**: `dim3(64)`

### Kernel 4: `kernel_wkv7_chunk_output`
**Purpose**: Combine local and propagated contributions for final output

**Grid**: `dim3(num_chunks, H, B)`
**Block**: `dim3(C)` or `dim3(64)`

---

## File Structure

Follows the existing HIP backend pattern: `.hip`/`.rs` kernel pairs, model-level logic in `model/`, scratch buffers in `scratch.rs`.

### New Files

| File | Contents |
|------|----------|
| `src/hip/kernels/fla.hip` | All 5 FLA stage kernels: `kernel_fla_cumsum`, `kernel_fla_intra`, `kernel_fla_wy_repr`, `kernel_fla_chunk_h`, `kernel_fla_chunk_o` |
| `src/hip/kernels/fla.rs` | Rust FFI wrappers for each FLA kernel (shape validation, stream launch) |
| `src/hip/model/fla.rs` | `FlaChunkedWkv` struct (implements `WkvKernel` trait), chunk index precomputation (`prepare_chunk_indices`, `prepare_chunk_offsets`), intermediate buffer pool management, 5-stage pipeline orchestration |

### Files to Modify

| File | Change |
|------|--------|
| `src/hip/kernels/mod.rs` | Add `pub mod fla;` |
| `src/hip/model/mod.rs` | Add `pub mod fla;` |
| `src/hip/model/step.rs` | 3-tier dispatch: T=1 → FusedT1, 1<T<threshold → WaveReduce, T≥threshold → FlaChunkedWkv |
| `src/hip/scratch.rs` | Add FLA intermediate buffer allocations to `HipScratch` (fixed-size at init) |
| `build.rs` | Add `fla.hip` to hipcc compilation |

---

## Memory Layout

**Existing** (column-major, shape[0]-fastest):
- Inputs: `[K, H, T, B]` (K=head_size)
- State: `[K, K, H, B]`

**New FLA intermediate buffers** (per forward pass, shared across layers):

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `gi`, `ge` | `[K, H, T, 1]` each | Cumulative decays (Stage 1) |
| `A_qk`, `A_qb`, `A_ab`, `A_ak` | `[C, C, H, total_chunks]` each | Intra-chunk attention matrices (Stage 2) |
| `A_ab_inv` | `[C, C, H, total_chunks]` | Lower-triangular inverse (Stage 3) |
| `qg`, `kg`, `ag`, `bg` | `[K, H, T, 1]` each | Decay-scaled inputs (Stage 2) |
| `w_wy`, `u_wy` | `[K, H, T, 1]`, `[V, H, T, 1]` | WY representation outputs (Stage 3) |
| `h` | `[K, V, H, total_chunks]` | Per-chunk states (Stage 4) |
| `v_new` | `[V, H, T, 1]` | Corrected values (Stage 4) |

**Sizing:** Fixed at model load using `max_total_chunks = B * ceil_div(max_T, C)`. For equal-length batches, `total_chunks = B * ceil_div(T, C)` which is bounded by this max. For varlen, packed sequences are strictly denser, so the bound holds. No hipMalloc during inference.

**Memory budget** (768-dim model, C=16, T=4096, B=4, H=12):
- Additional working memory: ~48 MB

---

## API Design

```rust
/// Chunked WKV7 for efficient prefill
pub fn wkv7_chunked_f32(
    w_decay: &TensorHip<f32>,  // [N, H, T, B]
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,   // [N, N, H, B]
    output: &mut TensorHip<f32>,  // [N, H, T, B]
    state_out: &mut TensorHip<f32>,
    chunk_size: usize,  // 64 or 128
    stream: &Stream,
    rocblas_handle: RocblasHandle,
) -> Result<()>;

/// Auto-dispatch between recurrent and chunked
pub fn wkv7_dispatch_f32(...) -> Result<()> {
    if t >= 128 {
        wkv7_chunked_f32(...)  // Use chunked for prefill
    } else {
        wkv7_f32(...)  // Use recurrent for streaming
    }
}
```

---

## Implementation Phases

### Phase 0: Precision Testing Infrastructure (First)
**Goal**: Establish correctness baseline before writing chunked kernels.

1. Create precision comparison test: recurrent vs chunked output
2. Define tolerance thresholds (start strict: 1e-5 relative error)
3. Add fixtures for various sequence lengths (T=64, 128, 256, 512, 1024)
4. Document existing recurrent kernel precision characteristics

**Acceptance**: Test infrastructure can detect numerical drift.

### Phase 1: Minimal Chunked Proof-of-Concept
**Goal**: Validate the chunked algorithm math with smallest possible chunk.

1. Implement chunk size C=2 (two tokens per chunk) as a test case
2. This removes most parallelism complexity, focuses on state transfer correctness
3. Verify output matches recurrent kernel exactly
4. If C=2 works, the algorithm is correct; scaling to C=64 is optimization

### Phase 2: Cumulative Decay Kernel
1. Create `wkv7_chunk.hip` with `kernel_wkv7_chunk_scan`
2. Implement parallel prefix product for decay values
3. Unit tests against sequential reference

### Phase 3: Core Chunked Computation
1. Implement `kernel_wkv7_chunk_propagate` for inter-chunk state
2. Integrate rocBLAS for intra-chunk GEMM operations
3. Implement `kernel_wkv7_chunk_output` for final combination
4. Integration tests: chunked output == recurrent output

### Phase 4: Optimization
1. Profile with rocprof, identify bottlenecks
2. Tune block sizes for gfx1151 wave32
3. Add LDS optimizations for shared memory
4. Variable-length batch support (masked version)

### Phase 5: Integration
1. Add auto-dispatch based on sequence length
2. Update HipRuntime to use new kernels
3. Documentation and examples
4. Performance benchmarks

---

## Testing Strategy

### Correctness Tests
1. Chunk scan vs sequential reference (tolerance < 1e-6)
2. Chunked output vs recurrent output (tolerance < 1e-5)
3. State propagation matches recurrent state
4. Variable chunk sizes (16, 32, 64, 128)

### Performance Benchmarks
| Metric | Target |
|--------|--------|
| Prefill T=2048 | 5x faster than recurrent |
| Prefill T=4096 | 10x faster than recurrent |
| Streaming T=1 | Same as recurrent (no regression) |

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical precision | FP32 intermediates, pairwise summation |
| rocBLAS overhead | Custom kernel for small chunks < 64 |
| Memory bandwidth | Buffer reuse, stream overlap |
| gfx1151 wave32 | Block sizes multiple of 32 |

---

## Key Files to Modify

1. `/workspace/web-rwkv/src/hip/kernels/copy.hip` - Reference for kernel patterns
2. `/workspace/web-rwkv/src/hip/mod.rs` - Add FFI wrappers, dispatch logic
3. `/workspace/web-rwkv/build.rs` - Add new .hip file to compilation

---

## Verification

1. Run existing tests: `cargo test --features hip`
2. Compare against Python reference: `scripts/generate_test_fixtures.py`
3. Benchmark: `cargo bench --features hip`
4. Profile: `rocprof --hip-trace ./target/release/examples/hip_gen`

---

## Gap Analysis (Addendum)

*Analysis based on review of the [fla-org reference implementation](https://github.com/fla-org/flash-linear-attention) (cloned at `repos/flash-linear-attention/`) and the existing web-rwkv API surface.*

### Motivating Constraint: API Compatibility with WebGPU Backend

The FLA chunked kernel must integrate without changing the caller-facing API. Both the WebGPU and HIP backends expose the same `Runtime<Rnn>` trait with `RnnInput`/`RnnOutput`. The HIP backend additionally has:

- `WkvKernel` trait (`src/hip/model/prefill.rs`) with `compute()` and `supports_multi_token()`
- Pluggable kernel selection: `FusedT1Wkv` (T=1 streaming) vs `WaveReduceWkv` (T>1 prefill)
- State managed as GPU-resident buffers with explicit `read`/`write` per batch slot

FLA must plug in as a new `WkvKernel` implementation (e.g., `FlaChunkedWkv`) that is auto-dispatched when T exceeds a threshold. The external `infer()` / `infer_one()` API, `RnnInput`/`RnnOutput` shapes, and state semantics must remain unchanged.

---

### Gap 1: Variable-Length Sequences — Architectural, Not Phase 4

The plan defers variable-length batch support to Phase 4. This is a fundamental design decision that affects every kernel signature and buffer layout.

**What FLA actually does:** Packed/varlen (no padding). All sequences concatenated along the time axis. A `cu_seqlens: LongTensor[N+1]` tracks boundaries.

- Example: 3 sequences of lengths 100, 130, 50 → `cu_seqlens = [0, 100, 230, 280]`
- B=1; the "batch" is implicit in the packing

**Current HIP backend:** Requires all batches to have the same length. The WebGPU backend handles variable-length via chunking at `RnnInput` level (min 32-token chunks, aligned).

**Recommendation:** Design kernel signatures to accept `cu_seqlens` from day one, even if the initial implementation only supports equal-length batches (where `cu_seqlens` is trivially `[0, T, 2T, ..., B*T]`). Retrofitting varlen onto fixed-T kernels is painful.

---

### Gap 2: Missing Chunk Index Precomputation

The plan describes grids as `dim3(num_chunks, H, B)` but doesn't address how kernels know which sequence a chunk belongs to when sequences have different lengths.

**What FLA does:** A `prepare_chunk_indices()` function (in `fla/ops/utils/index.py`) precomputes a flat `[total_chunks, 2]` tensor mapping each chunk to `(sequence_id, chunk_id_within_sequence)`:

```
cu_seqlens = [0, 100, 230, 280], chunk_size = 64
chunks_per_seq = ceil_div([100, 130, 50], 64) = [2, 3, 1]
chunk_indices = [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0)]
```

A companion `prepare_chunk_offsets()` gives cumulative chunk counts per sequence: `[0, 2, 5, 6]` — used for indexing into intermediate state storage.

**For HIP:** This precomputation happens on the CPU (in Rust), uploaded as a small device buffer. The grid becomes `dim3(total_chunks, H, 1)` instead of `dim3(num_chunks, H, B)`.

---

### Gap 3: Partial Chunk Masking

When a sequence length isn't divisible by chunk size (common case), the last chunk is partial. The plan doesn't discuss this.

**FLA uses three mechanisms:**
1. **Block pointer `boundary_check`** — zeros out-of-bounds elements on tile loads
2. **Explicit scalar masks:** `mask=(offset < T)` for gate/decay loads
3. **Combined causal + boundary mask** for intra-chunk attention: `m_A = (causal) & (valid_query & valid_key)`

**For HIP (no Triton boundary_check):**
- Every load in every kernel needs a bounds check: `val = (idx < T) ? buf[idx] : 0.0f`
- The intra-chunk attention matrix needs a mask combining causality with boundary validity
- Last-element decay uses clamped index: `last_idx = min((i_t + 1) * BT, T) - 1`

This is correctness-critical — not an optimization.

---

### Gap 4: Missing WY Representation Stage

The plan has 4 kernels (scan, gemm_local, propagate, output). FLA actually has **5 stages**:

| Stage | FLA Function | Plan Kernel | Notes |
|-------|-------------|-------------|-------|
| 1. Cumulative decay | `chunk_rwkv6_fwd_cumsum` | `kernel_wkv7_chunk_scan` | Covered |
| 2. Intra-chunk attention matrices | `chunk_dplr_fwd_intra` | `kernel_wkv7_chunk_gemm_local` | Partially covered |
| **3. WY representation** | **`prepare_wy_repr_fwd`** | **Missing** | **Gap** |
| 4. Inter-chunk state recurrence | `chunk_dplr_fwd_h` | `kernel_wkv7_chunk_propagate` | Covered |
| 5. Output combination | `chunk_dplr_fwd_o` | `kernel_wkv7_chunk_output` | Covered |

**The WY representation step:**
- Computes `A_ab_inv` — inverse of the lower-triangular `A @ B^T` matrix within each chunk
- Uses block-wise inversion: `[A11, 0; A21, A22]^{-1} = [A11^{-1}, 0; -A22^{-1}*A21*A11^{-1}, A22^{-1}]`
- Produces `w = A_ab_inv @ a` and `u = A_ab_inv @ A_ak @ v`
- These compress chunk-internal recurrence into a low-rank form for the state propagation stage

This is a non-trivial kernel. It requires triangular matrix inversion within each chunk and cannot be skipped.

---

### Gap 5: Default Chunk Size

The plan proposes chunk size 64–128. FLA's RWKV7 defaults to **16**.

At chunk_size=16, the intra-chunk attention matrix is 16×16 (fits in registers, manageable FP32 error). At 64, it's 64×64 — significantly more memory and precision pressure.

**Recommendation:** Start with chunk_size=16 to match the reference. Validate correctness and precision. Then experiment with larger sizes.

---

### Gap 6: Step Function Changes for Prefill

The current HIP step function (`step_inner` in `src/hip/model/step.rs`) processes one layer at a time, calling the WKV kernel per layer. For FLA, the step function flow needs to accommodate:

1. **Buffer allocation for intermediate tensors** — the 5-stage pipeline produces several intermediates per layer:
   - `gi`, `ge` (cumulative decays) — `[T, H, K]` each
   - `A_ab`, `A_qk`, `A_qb`, `A_ak` (attention matrices) — `[num_chunks, H, C, C]` each
   - `A_ab_inv` (inverse) — `[num_chunks, H, C, C]`
   - `w_wy`, `u_wy` (WY outputs) — `[T, H, K]` and `[T, H, V]`
   - `h` (per-chunk states) — `[num_chunks, H, K, V]`

   These should be allocated once per forward pass (or pooled), not per-layer.

2. **Kernel dispatch sequence per layer:**
   ```
   For each layer:
     1. Token shift + layer norm (existing)
     2. Compute r, w, k, v, a, b (existing matmuls)
     3. FLA 5-stage pipeline:
        a. Cumulative decay scan
        b. Intra-chunk attention matrices
        c. WY representation
        d. Inter-chunk state propagation (reads/writes state)
        e. Output combination
     4. Output projection + feed-forward (existing)
   ```

3. **State read/write semantics:** The FLA pipeline reads `initial_state` at stage 4 start and writes `final_state` at stage 4 end, identically to the recurrent kernel's state update. The `WkvKernel::compute()` signature already takes `state: &mut TensorHip<f32>` — the FLA implementation reads it as `initial_state`, writes it as `final_state`.

4. **Dispatch threshold:** Add to `step_inner` or the `WkvKernel` selector:
   ```rust
   let wkv_kernel: &dyn WkvKernel = if num_tokens >= CHUNK_THRESHOLD {
       &FlaChunkedWkv { chunk_size: 16 }
   } else if num_tokens == 1 {
       &FusedT1Wkv
   } else {
       &WaveReduceWkv
   };
   ```

---

### Gap 7: Intermediate Buffer Management

The plan's memory budget section estimates ~96 MB but doesn't address *lifecycle*. The FLA pipeline needs several intermediate buffers that are:

- **Per-forward-pass** (shared across layers): The attention matrices, WY outputs, and per-chunk states have the same shape regardless of layer, so they can be allocated once and reused.
- **Sized by total_chunks** (not `num_chunks * B`): For varlen, `total_chunks = sum(ceil_div(seq_len_i, C))` across all sequences in the batch.

The existing HIP backend uses a buffer pool (`checkout_buffer` pattern in WebGPU, scratch allocations in HIP). FLA intermediates should follow the same pattern.

---

### Gap 8: Host-Side Precomputation

Before launching FLA kernels, the host (Rust) must compute:
- `chunk_indices: [total_chunks, 2]` — maps flat chunk ID to (seq_id, local_chunk_id)
- `chunk_offsets: [N+1]` — cumulative chunk counts per sequence
- `cu_seqlens: [N+1]` — already available from input batching

These are small tensors computed on CPU and uploaded once per forward pass. The plan doesn't include this step in the kernel launch sequence.

---

### Reference Implementation

The FLA reference is cloned at `repos/flash-linear-attention/`. Key files:

| File | Contents |
|------|----------|
| `fla/ops/rwkv7/chunk.py` | Entry point, delegates to DPLR |
| `fla/ops/generalized_delta_rule/dplr/chunk.py` | Forward pass orchestration (5 stages) |
| `fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py` | Stage 2: intra-chunk attention matrices |
| `fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py` | Stage 3: WY representation |
| `fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py` | Stage 4: inter-chunk state recurrence |
| `fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py` | Stage 5: output combination |
| `fla/ops/utils/index.py` | `prepare_chunk_indices`, `prepare_chunk_offsets` |
| `fla/ops/rwkv6/chunk.py` | Stage 1: `chunk_rwkv6_fwd_cumsum` (shared with RWKV6) |
