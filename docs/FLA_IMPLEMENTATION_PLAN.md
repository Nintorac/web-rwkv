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

## New Files to Create

1. **`/workspace/web-rwkv/src/hip/kernels/wkv7_chunk.hip`**
   - All chunked kernel implementations

2. **Additions to `/workspace/web-rwkv/src/hip/mod.rs`**
   - Rust FFI wrappers for chunked kernels
   - Auto-dispatch based on sequence length

---

## Memory Layout

**Existing** (column-major, N-fastest):
- Inputs: `[N, H, T, B]`
- State: `[N, N, H, B]`

**New buffers** (per forward pass):
- Chunk cumulative decay: `[N, H, num_chunks, B]`
- Chunk local states: `[N, N, H, num_chunks, B]`
- Chunk local outputs: `[N, H, C, num_chunks, B]`

**Memory budget** (768-dim model, C=64, T=4096, B=4, H=12):
- Additional working memory: ~96 MB

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
