# HIP Backend Optimization Findings

This document summarizes the optimization work and findings for the HIP (AMD GPU) backend of web-rwkv, conducted in January 2025.

## Executive Summary

We achieved **28-86% throughput improvements** on the HIP backend through GPU-side f16→f32 conversion for logits. Key findings:

1. **WKV kernel is well-optimized** - achieving 90% of theoretical memory bandwidth (181 GB/s)
2. **The remaining bottleneck is PCIe transfer** - logits download takes 7+ ms at batch=256
3. **gfx1151 (Strix Halo) lacks optimized Tensile kernels** - documented 2x regression vs gfx1100
4. **rocBLAS mixed-precision GEMM is slower than pure f16 GEMM + conversion**

## Hardware Context

- **GPU**: AMD Radeon 8060S (gfx1151 / RDNA 3.5 / Strix Halo)
- **ROCm Version**: 7.2.0
- **Known Issues**:
  - [gfx1151 rocBLAS/hipBLAS performance regression](https://github.com/ROCm/ROCm/issues/4748)
  - [rocblas_gemm_ex slow for m==1 cases](https://github.com/ROCm/rocBLAS/issues/1425)

## Performance Results

### 0.1b Model (decode-only)

| Batch | HIP (tok/s) | WGPU (tok/s) | HIP vs WGPU |
|-------|-------------|--------------|-------------|
| 32    | 1,676       | 1,263        | **1.33x faster** |
| 64    | 4,262       | 2,595        | **1.64x faster** |
| 128   | 5,651       | 5,085        | **1.11x faster** |
| 256   | 8,011       | 9,315        | 0.86x (14% slower) |

### 2.9b Model (decode-only)

| Batch | HIP (tok/s) | WGPU (tok/s) | HIP vs WGPU |
|-------|-------------|--------------|-------------|
| 32    | 492         | 170          | **2.89x faster** |
| 64    | 859         | 291          | **2.95x faster** |
| 128   | 1,251       | 644          | **1.94x faster** |
| 256   | 1,509       | 1,268        | **1.19x faster** |

HIP now **exceeds WGPU performance at all batch sizes** for the 2.9b model, and at batch sizes ≤128 for the 0.1b model.

## Optimization Attempts

### 1. GPU f16→f32 Conversion for Logits ✅ SUCCESS

**Problem**: CPU-side f16→f32 conversion was taking 17.6ms at batch=256, even with SIMD.

**Solution**: Added a HIP kernel to convert logits on GPU before download.

**Result**: 17.6ms → 7.1ms (2.5x faster)

**Files**: `src/hip/kernels/copy.hip`, `src/hip/kernels.rs`, `src/hip/model.rs`

### 2. WKV FMA/ILP Optimizations ❌ NO IMPROVEMENT

**Attempted**:
- Replaced pairwise summation with 4-way ILP accumulators
- Used explicit `__fmaf_rn` intrinsics
- Tried moving state from registers to shared memory (3.4x slower!)

**Result**: No measurable improvement - kernel is already well-optimized.

### 3. Mixed-Precision Head GEMM (rocblas_gemm_ex) ❌ SLOWER

**Problem**: Wanted to output f32 directly from head GEMM, eliminating conversion step.

**Attempted**: Used `rocblas_gemm_ex()` with f16 inputs, f32 output, f32 compute.

**Result**: 3.7ms vs 0.9ms for pure f16 GEMM - **4x slower!**

**Root Cause**: rocblas_gemm_ex has known performance issues on gfx1151.

### 4. hipBLASLt as Alternative ❌ SLOWER THAN ROCBLAS

**Attempted**: Implemented hipBLASLt as alternative BLAS backend.

**Result**:
- rocBLAS (gemm_ex): 3.727 ms
- hipBLASLt: 4.278 ms
- rocBLAS is 1.15x faster

**Conclusion**: Both libraries lack optimized kernels for gfx1151.

### 5. Multi-Batch WKV Kernel ❌ NO IMPROVEMENT

**Attempted**: Process 4 batch elements per block to share input loads.

**Result**: No improvement - each batch has different inputs, nothing to share.

### 6. Batched GEMV for WKV ❌ SLOWER

**Attempted**: Express WKV as rocBLAS batched GEMV operations (`rocblas_sgemv_strided_batched`).

**Result**:
| Batch | Original | rocBLAS GEMV | Speedup |
|-------|----------|--------------|---------|
| 256   | 16.1 ms  | 20.7 ms      | 0.78x   |
| 4096  | 208 ms   | 258 ms       | 0.80x   |

The custom kernel wins because:
- 3 kernel launches vs 1 fused kernel
- GEMV reads/writes state to memory; custom kernel keeps state in registers
- 64×64 matrices are too small for batched BLAS efficiency

### 7. Coalesced State Access via Shared Memory ❌ NO IMPROVEMENT

**Problem**: State access appeared non-coalesced (thread i reads stride-64 addresses).

**Attempted**: Stage state through shared memory for coalesced global memory access.

**Result**: No improvement - shared memory staging (17.9 KB) killed occupancy (3 blocks/CU vs 51).

### 8. Row-Owned Column-Major Kernel (colmajor_t1) ❌ REGRESSION

**Problem**: The wave_reduce_t1 kernel uses row-major decomposition requiring `__shfl_down` cross-lane reductions + `__syncthreads()` barriers, and separate state_in/state_out buffers doubling L2 pressure. WGPU uses a column-major decomposition that avoids these overheads.

**Attempted**: Row-owned decomposition with in-place state (bd-2eg.1 through bd-2eg.3):
- 64 threads (2 waves), each thread owns one row, iterates all 64 columns
- In-place state update (single buffer, halves L2 footprint)
- No `__shfl_down`, no `__syncthreads()` between sa and update passes
- Params loaded to LDS as f32 (6 × 64 × 4 = 1536 bytes)
- `__launch_bounds__(64, 8)` targeting 8 blocks/CU

**rocprofv3 Results (0.1b, B=256, T=1 decode):**

| Metric | wave_reduce_t1 (baseline) | colmajor_t1 | Delta |
|--------|--------------------------|-------------|-------|
| Avg kernel time | **441 μs** | **775 μs** | **1.76x slower** |
| Total (420 launches) | 185.4 ms | 325.4 ms | +140 ms |
| VGPRs | 40 | 56 | +40% |
| SGPRs | 128 | 128 | same |
| LDS | 17,792 B | 1,536 B | -91% |
| Workgroup size | 256 | 64 | -75% |

**Correctness**: All tests pass (Spearman ρ=0.9998, Top-1=100%, Top-5=95.7%, Top-10=99.3%).

**Root cause analysis**:
1. **Insufficient memory-level parallelism**: 64 threads issue 64 concurrent memory ops vs 256 for the baseline. WKV is memory-bound (reading 64×64 state matrix = 16KB per head per batch), so 4x fewer concurrent memory requests severely limits throughput.
2. **56 VGPRs** (higher than expected): The f32 accumulators for sa, y, plus loop variables and state values consume more registers than anticipated. Target was ≤24 VGPRs for 8 blocks/CU.
3. **Two full state passes**: Each thread reads all 64 state columns twice (sa pass + y pass). The baseline's wave-cooperative approach distributes state reads across 256 threads, requiring fewer reads per thread despite the shuffle overhead.
4. **Grid underutilization**: Grid is (H=32, B=1_per_head_group) = 32 blocks on 20 CUs. With 64 threads/block, only 2048 threads active. The 256-thread baseline puts 8192 threads on the same grid.

**Key insight**: Eliminating `__shfl_down` and `__syncthreads()` saves ~5% of ALU time, but the 4x reduction in memory concurrency costs ~76% more wall time. For a memory-bound 64×64 kernel, memory-level parallelism dominates over ALU efficiency.

**Decision gate**: Phase B (WMMA) should NOT be layered on top of this decomposition — the row-owned approach is fundamentally memory-starved. Alternative approaches needed.

## WKV Kernel Analysis

### Corrected Performance Analysis

Initial analysis suggested 7% bandwidth utilization, but this was **incorrect** - the test harness was measuring malloc/copy/free overhead, not kernel time.

**rocprof analysis revealed:**
| Operation | Time | % of Total |
|-----------|------|------------|
| hipMemcpyAsync | 124.4 ms | 61% |
| hipMalloc | 51.8 ms | 25% |
| hipFree | 25.5 ms | 13% |
| **hipLaunchKernel** | **1.8 ms** | **0.9%** |

The actual kernel is fast - memory management dominates in test harness.

### Real WKV Performance (Production Path)

At batch=256, 12 layers:
- **Total state R/W**: 1.24 GB
- **WKV time**: 6.86 ms (aggregated across 12 layers)
- **Achieved bandwidth**: 181 GB/s
- **Utilization**: **90.5%** of theoretical 200 GB/s

### rocprofv3 WKV Kernel Details

| Attribute | Value |
|-----------|-------|
| Kernel name | `kernel_wkv7_f16_masked` |
| Total GPU time | 220 ms (over 420 launches) |
| Per-launch time | **524 μs** |
| VGPR count | **144** (high register pressure) |
| SGPR count | 128 |
| Workgroup size | 64 (1 wave per workgroup) |
| Grid size | 768 × 256 = 196,608 workgroups |
| Shared memory | 1.28 KB |

**Register pressure limits occupancy**: With 144 VGPRs per thread on RDNA3:
- Max waves per SIMD: 1536 / 144 = 10 waves
- Max waves per CU: 10 × 4 SIMDs = 40 waves
- 12 CUs → ~480 concurrent waves (vs 196,608 total → 409 batches)

**⚠️ This 90% claim is dubious.** The hardware has ~256 GB/s theoretical bandwidth (LPDDR5X-8000, 256-bit bus), with ~212 GB/s measured in practice. So 181 GB/s would be ~85% utilization - plausible but suspicious given:

- WGPU's WKV is 1.7x faster (4.0ms vs 6.9ms)
- WGPU does MORE memory traffic (state in global memory vs HIP's register-based approach)
- If HIP achieves 85% bandwidth, WGPU would need ~145% to be 1.7x faster (impossible)

Either the bandwidth calculation is wrong, or WGPU's algorithm is fundamentally more efficient.

The WKV kernel **needs investigation** - it's a key optimization target.

### Per-Operation Breakdown (batch=256, 0.1b model)

From hip-prof:
| Operation | Time | Notes |
|-----------|------|-------|
| wkv | 6.86 ms | 90% bandwidth efficient |
| **logits_dl** | **7.2 ms** | **PCIe bottleneck** |
| att_proj | 1.72 ms | GEMM |
| ffn_v | 1.49 ms | GEMM |
| att_out | 1.04 ms | GEMM |
| ffn_k | 0.91 ms | GEMM |
| head | 0.87 ms | GEMM |

Total step: ~69 ms. The **logits download is 10% of step time**, limited by PCIe bandwidth.

## Current Bottleneck: Logits Download

At batch=256 with 50k vocab:
- Logits size: 256 × 50,277 × 4 bytes = **51.5 MB**
- Download time: 7.2 ms
- Achieved PCIe bandwidth: 7.2 GB/s (reasonable for PCIe Gen4 x8)

This explains why HIP is 18% slower than WGPU at batch=256 for 0.1b.

### rocprofv3 Deep Dive: Memory Pinning Overhead (January 2026)

Used `rocprofv3 --hip-trace --hsa-trace --kernel-trace --memory-copy-trace` to profile batch=256 decode-only workload.

**Key finding: GPU is only 20% utilized!**

| Metric | Value |
|--------|-------|
| Total runtime | 2.7 seconds |
| Kernel execution | 532 ms (20%) |
| Memory copy (GPU time) | 68 ms (2.5%) |
| **CPU overhead** | **~80% of time** |

**Root cause identified: Memory pinning on every logits transfer**

The `to_vec()` method in `src/hip/tensor.rs:384` allocates a fresh `Vec` (pageable memory) for every D2H transfer. HIP must pin/unpin this memory for DMA:

| HSA Operation | Count | Total Time | Per-Call |
|---------------|-------|------------|----------|
| `hsa_amd_memory_lock_to_pool` | 70 | 192 ms | 2.75 ms |
| `hsa_amd_memory_unlock` | 70 | 38 ms | 0.54 ms |

**Per-step overhead: ~6.5ms just for pinning!**

This accounts for most of the 7.2ms "logits_dl" time. The actual DMA transfer is only ~1ms.

**Fix**: Replace `to_vec()` with transfer to a pre-allocated `PinnedBuffer`. The async path (`forward_inner_async`) already does this correctly using `copy_to_slice_async()` with a pinned destination.

**Implementation path**:
1. Add `logits_staging: PinnedBuffer<f32>` to scratch (already exists for f16)
2. Change `forward_inner()` to use `copy_to_slice_async()` into the pinned buffer
3. Copy from pinned buffer to final Vec (fast CPU memcpy, no pinning)

Expected improvement: ~6ms saved per step → **~18% faster** at batch=256.

### Fix Applied (January 2026)

Changed `runtime.infer()` to use `forward_async().wait()` instead of `forward()`:
- `forward_async()` uses pre-allocated `PinnedBuffer<f32>` for logits
- GPU does f16→f32 conversion before download (same as sync path)
- No memory pinning overhead since buffer is already pinned

**Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Throughput (batch=256) | 3,700 tok/s | 4,346 tok/s | **+17%** |
| Step latency | 70 ms | 59 ms | **-11 ms** |
| Memory lock operations | 70 calls, 230ms | 0 calls | **-100%** |

Files modified:
- `src/hip/model.rs`: Changed `ForwardCompletion.logits_buffer` to `PinnedBuffer<f32>`, added GPU f16→f32 conversion in async path
- `src/hip/runtime.rs`: Changed `infer()` to use `forward_async().wait()`

## WGPU vs HIP Per-Operation Comparison

Added GPU timestamp-based profiling to WGPU backend (`wgpu-prof` feature) for direct comparison.

### Per-Step GPU Time (batch=256, 0.1b model, 32 steps)

| Operation | WGPU (ms) | HIP (ms) | Notes |
|-----------|-----------|----------|-------|
| wkv | 4.0 | 6.9 | HIP at 58% of WGPU speed |
| ffn_v | 3.6 | 1.5 | HIP 2.4x faster |
| att_adapt | 3.3 | 0.9 | HIP 3.7x faster |
| att_proj | 1.2 | 1.7 | HIP 1.4x slower |
| head | 1.2 | 0.9 | HIP 1.3x faster |
| ffn_k | 1.2 | 0.9 | HIP 1.3x faster |
| att_out | 0.8 | 1.0 | HIP 1.2x slower |
| att_vres | 0.8 | 0.2 | HIP 4x faster |
| readback | 20.0 | 7.2 | HIP 2.8x faster |

**Key Findings:**

1. **WKV is slower on HIP (6.9ms vs 4.0ms)** - HIP runs at 58% of WGPU's speed despite the "90% bandwidth" claim. This is a key optimization target.

2. **GEMM operations are mixed** - Some (ffn_v, head, ffn_k) are faster on HIP, others (att_proj, att_out) are slower. This is likely due to gfx1151's limited Tensile kernel coverage.

3. **Logits transfer is faster on HIP but still a bottleneck** - HIP's direct memcpy (7.2ms) beats WGPU's mapped buffer (20ms), but 7.2ms for 51MB is still significant. This is an optimization target.

**Execution Model (both backends):**
- Both HIP and WGPU are **async for the full forward pass** - kernels are queued without blocking
- Both have a **sync point after logits** where the sampler must read results
- The performance difference comes from kernel efficiency, not async behavior

### End-to-End Throughput (0.1b model)

| Batch | HIP (tok/s) | WGPU (tok/s) | Winner |
|-------|-------------|--------------|--------|
| 32    | 1,743       | 1,323        | HIP +32% |
| 64    | 4,328       | 2,442        | HIP +77% |
| 128   | 5,736       | 4,992        | HIP +15% |
| 256   | 7,882       | 9,636        | WGPU +22% |

HIP's advantage diminishes at higher batch sizes because:
- WKV kernel time dominates (6.9ms HIP vs 4.0ms WGPU)
- Logits transfer becomes a larger fraction of step time

### HIP Optimization Targets

1. **WKV kernel** - Currently 6.9ms vs WGPU's 4.0ms (HIP at 58% of WGPU speed).
   - HIP uses registers for state (should be faster), WGPU uses global memory (more traffic)
   - The "90% bandwidth" claim doesn't explain the gap
   - Need to investigate: kernel launch overhead? occupancy? memory access patterns?
   - Consider porting WGPU's algorithm to HIP to compare directly

2. **Logits transfer** - Currently 7.2ms for 51MB (~7 GB/s).
   - Options: fuse sampling on GPU, async pipelining, or investigate PCIe utilization

## WKV Kernel Deep Dive (January 2026)

### Root Cause: Register Pressure Killing Occupancy

The paradox: HIP WKV is 1.7x slower than WGPU (6.9ms vs 4.0ms) despite HIP keeping state in registers (minimal memory traffic) while WGPU keeps state in global memory (more traffic).

**rocprofv3 kernel stats reveal the problem:**

| Attribute | Value |
|-----------|-------|
| Total GPU time | 220 ms (over 420 launches) |
| Per-launch | 524 μs |
| **VGPRs** | **144** (very high) |
| SGPRs | 128 |
| Workgroup size | 64 (1 wave) |
| Grid size | 768 × 256 = 196,608 workgroups |
| Shared memory | 1.28 KB |

**Why 144 VGPRs is catastrophic for RDNA3:**

```
RDNA3 gfx1151 has:
- 192 KB VGPRs per SIMD (6144 32-bit registers)
- 4 SIMDs per CU
- Wave32 mode

With 64 threads/block using 144 VGPRs each:
- Per-block VGPRs = 64 × 144 = 9216 registers
- But SIMD only has 6144 registers!
- Must spill to memory OR limit to 1 partial wave

Result: Starved for parallelism, poor latency hiding
```

### Implementation Comparison

| Aspect | HIP (`kernel_wkv7_f16_masked`) | WGPU (`time_mix_v7.wgsl`) |
|--------|-------------------------------|--------------------------|
| **State storage** | Registers (64 floats/thread) | Global memory |
| **Grid** | (H, B) = heads × batch | Different decomposition |
| **Block size** | 64 threads (N = head_size) | BLOCK_SIZE (configurable) |
| **VGPRs** | 144 (from rocprofv3) | N/A |
| **Shared mem** | 1.28KB (5×64 floats) | ~1.5KB (6×BLOCK_SIZE vec4s) |
| **Per-thread state** | 64 floats (256 bytes) | 0 (in global memory) |

**Why WGPU wins despite more memory traffic:**

1. **Higher occupancy** - more waves in flight hiding latency
2. **Coalesced access** - threads access consecutive addresses
3. **L2 cache reuse** - state fits in L2 (256×64×64×4 = 4MB at batch=256)

### WKV Computation Structure

The WKV7 computation per timestep is:

```
For each thread i (row of state), for each column j:
1. sa[i] = Σⱼ a[j] * state[i,j]      // Dot product: 64 FMAs
2. state[i,j] = w[j]*state[i,j] + b[j]*sa[i] + k[j]*v[i]  // Element-wise update
3. y[i] = Σⱼ q[j] * state[i,j]       // Dot product: 64 FMAs
```

This is NOT a standard GEMM - it's two matrix-vector products plus element-wise updates. WMMA doesn't directly apply, but we can restructure for better parallelism.

### Optimization Strategies Implemented

#### Strategy 1: Global State with High Occupancy (`kernel_wkv7_tiled`)

Match WGPU's approach: keep state in global memory, use 256 threads per block.

```cpp
__launch_bounds__(256, 8)  // 256 threads, aim for 8 waves/CU
void kernel_wkv7_tiled(...)
```

- **State**: Global memory (in-place update)
- **Parameters**: Shared memory (cooperative load)
- **Reductions**: Atomic adds to shared memory
- **Benefits**: Much higher occupancy, coalesced memory access

#### Strategy 2: LDS State with Atomics (`kernel_wkv7_lds`)

State in Local Data Share (shared memory) - 16KB fits 64×64 floats.

```cpp
__shared__ float sh_state[64][64 + 1];  // +1 padding for bank conflicts
```

- **State**: LDS (fast, but limited capacity)
- **Load once, update in-place** per timestep
- **Trade-off**: LDS bandwidth vs global memory bandwidth

#### Strategy 3: Wave-Cooperative Reductions (`kernel_wkv7_wave_reduce`)

Avoid slow atomics by using wave shuffle reductions.

```cpp
// Each wave (32 threads) handles one row of state
for (int offset = 16; offset > 0; offset >>= 1) {
    partial_sa += __shfl_down(partial_sa, offset, 32);
}
```

- **8 waves process 8 rows in parallel** (64 rows / 8 iterations)
- **Explicit wave reductions** using `__shfl_down`
- **No atomics** - deterministic, faster on RDNA3

### WMMA Intrinsics Analysis

For reference, AMD RDNA3 WMMA intrinsics:

```cpp
// F16 inputs, F32 accumulator, wave32
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag, false)

// Fragment types
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
```

**WMMA requires 16×16×16 GEMM** - our matrix-vector products (64×64 @ 64×1) don't fit naturally. However, we could:

1. Batch 16 timesteps and process SA = state @ A (64×64 @ 64×16)
2. Use WMMA for tiled 16×16 blocks

**Challenge**: State changes between timesteps, can't trivially batch.

### Environment Variables to Try

```bash
# Enable hipBLASLt preference (for GEMM ops)
export ROCBLAS_USE_HIPBLASLT=1

# PyTorch TunableOp (if using PyTorch benchmarks)
export PYTORCH_TUNABLEOP_ENABLED=1

# hipBLASLt tuning
export HIPBLASLT_TUNING_FILE=/tmp/hipblaslt_tuning.txt
```

### Build Flags Considered

Current: `-O3 --offload-arch=gfx1151`

Additional options:
```bash
"-ffast-math"                    # Aggressive FP optimization
"-mwavefrontsize64=false"        # Force wave32 (default on RDNA3)
"-Rpass-analysis=kernel-resource-usage"  # Show register usage
"-save-temps"                    # Keep intermediate files
```

### New Kernel FFI Bindings

Added to `src/hip/kernels/copy.hip`:

| Kernel | Strategy | Threads | State Location |
|--------|----------|---------|----------------|
| `kernel_wkv7_f16_masked` | Original | 64 | Registers |
| `kernel_wkv7_tiled` | Global state | 256 | Global memory |
| `kernel_wkv7_lds` | LDS + atomics | 256 | Shared memory |
| `kernel_wkv7_wave_reduce` | Wave shuffle | 256 | Shared memory |
| `kernel_wkv7_wave_reduce_t1` | Wave shuffle (T=1 specialized) | 256 | Shared memory |
| `kernel_wkv7_colmajor_t1` | Row-owned, in-place state (T=1) | 64 | Global (in-place) |

**Runtime toggle:** `WEB_RWKV_HIP_WKV_KERNEL` selects the implementation (`register`, `wave`, `wave_t1`, `colmajor_t1`, `tiled`, `lds`, `auto`). `auto` now prefers the T=1 specialized kernel for decode; otherwise wave-reduce for `batch >= 32` or `T <= 2`, else register.

### Expected Results

| Kernel | Expected Occupancy | Expected Time |
|--------|-------------------|---------------|
| Original (register state) | ~1 wave/CU | 6.9ms (baseline) |
| Tiled (global state) | ~8 waves/CU | TBD |
| LDS + atomics | ~4 waves/CU | TBD |
| Wave reduce | ~4 waves/CU | TBD (likely fastest) |
| Wave reduce T=1 | ~4 waves/CU | TBD (decode-only) |

### January 2026 Update (gfx1151 Strix Halo, T=1 decode, B=256, 0.1b)

All runs use the hip-prof harness with three warmup iterations excluded from the metrics.

| Kernel (strategy) | Mean step (ms) | Throughput (tok/s) | WKV per-step (ms) | Notes |
|-------------------|----------------|--------------------|-------------------|-------|
| Register (baseline) | 61.36 | 4,172 | ~6.9–7.2 | Original register-resident state |
| Wave reduce | 61.31 | 4,176 | ~6.1–6.4 | Wave shuffle reductions |
| **Wave reduce T=1** | **61.22** | **4,182** | ~6.1–6.3 | Decode-specialized path; now stores params in LDS as f16 to cut LDS bandwidth (no perf gain) |
| Wave reduce T=1 (global state, params in LDS) | **60.35** | **4,242** | ~6.1–6.2 | State stays in global; LDS only for params; avoids LDS size pressure, slight gain |
| Tiled / global state | 71.19 | 3,596 | ~16.0 | High occupancy but global-state traffic dominates |
| LDS (state in shared) | 73.79 | 3,470 | ~18.4 | LDS bandwidth + atomics remain bottlenecks |
| **Row-owned colmajor_t1** | **63.9** | **4,005** | **~9.3** | In-place state, 64 threads; 1.76x slower WKV kernel due to low memory concurrency |

WGPU reference (decode, same point): **~4.0 ms WKV**, still ~1.5× faster than our best HIP kernel.

**Notes on recent experiments**
- Parameter vectors (`q,k,w,a,b`) are now staged into LDS as **f16** (not f4) to halve LDS bandwidth in the T=1 kernel; converting to f32 on use preserved correctness but delivered no measurable speedup.
- A “persistent per-layer launch” was considered (one long-lived grid that loops timesteps), but decode uses **T=1** so there is no inner loop to amortize; it would just idle the grid between steps without benefit.
- Warmup: the profiling harness performs 3 warmup iterations that are **not included** in the reported means.
- Removing LDS state entirely (streaming state from global in two 32-column passes) reduced LDS to ~1 KB but regressed WKV to ~6.5 ms and ~4,155 tok/s; the extra global reads outweighed the occupancy gain, so this variant was reverted.

Target: Match or exceed WGPU's 4.0ms.

### Why WGPU's WKV Is Faster: Parallelization Strategy Analysis

SQTT hardware thread traces (via RADV RGP capture) and rocprofv3 kernel-level profiling
reveal a **fundamental architectural difference** between HIP and WGPU's WKV implementations.

#### Dispatch Structure Comparison

| | WGPU `time_mix_v7` | HIP `wave_reduce_t1` |
|---|---|---|
| Workgroup size | 32 threads (1 wave32) | 256 threads (8 waves) |
| Grid (0.1b, B=256) | **6 workgroups** | **3072 blocks** |
| Total waves | **6** | **24,576** |
| Parallelism axis | Embedding dimension | (head × batch) |
| Batch handling | Serial loop in each WG | One block per batch |

WGPU partitions across the **embedding dimension** — `ceil(H×S/128)` workgroups, each
containing 32 threads that handle 32 vec4 elements. Each workgroup loops over **all 256
batches** sequentially (`for t in 0..shape[2]`), processing the full cursor array.

HIP partitions across **(head × batch)** — `dim3(H, B)` = 3072 blocks, each with 256
threads processing a single (head, batch) pair.

The wave count ratio is **4096×** (24,576 vs 6), independent of model size. This holds
because HIP launches `H × B × 8` waves while WGPU launches `H × S / 128` waves.

#### Why Fewer Waves Wins at Batch=256 (0.1b Model)

**1. Active memory working set**

WGPU processes batches one at a time. At any instant, only **1 batch's state** (~192KB
for 12 heads × 64 × 64 × 4B) is being accessed across all workgroups. This fits in L2
cache (~4MB on RDNA3). The second state read (for the update loop) hits L2.

HIP has ~160 concurrent blocks (4 blocks/CU × 40 CUs), each accessing a **different**
(head, batch) pair's state (16KB each). Per-CU working set: 4 × 16KB = 64KB, which
**exceeds the 16KB L0 cache**. The second state read (loop 2) misses L0.

**2. Zero-cost barriers**

WGPU workgroups have 1 wave (32 threads). `workgroupBarrier()` is free — the wave is
already in SIMD lockstep.

HIP blocks have 8 waves. `__syncthreads()` between the sa-compute and state-update loops
forces all 8 waves to drain. During the stall, the SQ schedules other blocks' waves on
the same CU, which access different state data, evicting our state from L1.

**3. In-place state update**

WGPU writes state back to the **same buffer** (`state: array<vec4<f32>>` is read-write).
The write address was just loaded, so it's likely still in cache.

HIP writes to a **separate `state_out` buffer** (`state_in` and `state_out` are different
pointers). Every state write is a cold write to a new cache line.

#### Profiling Evidence

| Metric | HIP (rocprofv3) | WGPU (SQTT) |
|--------|-----------------|-------------|
| Per-dispatch time | **441 μs** avg | — |
| Per-wave duration | — | **343 μs** avg (CU4) |
| Instructions/wave | ~40 (est.) | **30,920** (measured) |
| VGPRs | 48 | ~72 (PAL ELF metadata) |
| LDS | 1,152 B/block | 2,560–3,072 B/WG |
| Concurrent waves | ~160 blocks (1,280 waves) | 6–8 total |
| State reads per kernel | 2× (miss L0 on 2nd) | 2× (hit L2 on 2nd) |

WGPU's 30,920 instructions per wave confirms the serial batch loop — each wave does work
equivalent to 256 HIP blocks' worth of computation.

#### Why the Advantage Reverses at Larger Models (2.9b)

At batch=256, HIP is **1.19× faster** for the 2.9b model but **0.86×** for 0.1b. The
crossover occurs because the bottleneck shifts from **cache efficiency** to **memory
bandwidth saturation**.

| | 0.1b (12 heads, 768 dim) | 2.9b (~48 heads, 3072 dim) |
|---|---|---|
| Total state I/O | ~150 MB | ~600 MB |
| WGPU workgroups | 6 | 24 |
| HIP concurrent waves | ~1,280 | ~1,280 |
| Bottleneck | Cache efficiency | Memory bandwidth |

To sustain Strix Halo's ~120 GB/s bandwidth, the GPU needs hundreds of outstanding memory
requests to fill the pipeline (~200–400 cycle latency). HIP's ~1,280 concurrent waves
generate tens of thousands of outstanding loads, easily saturating bandwidth. WGPU's 24
waves have far less latency-hiding capacity.

At 0.1b the total traffic (150 MB) is small enough that cache behavior matters more than
raw bandwidth — WGPU's clean L2 hits win. At 2.9b the total traffic (600 MB) makes
bandwidth the bottleneck — HIP's massive parallelism wins.

Additionally, GEMMs dominate at larger model sizes, and HIP's rocBLAS leverages hardware
WMMA units while WGPU uses generic compute shaders.

#### Implication for HIP Optimization

The fused_t1 kernel addresses HIP-specific overhead (double state read, barriers, separate
output buffer) but retains the (head × batch) grid structure. A hypothetical
"embed-parallel" HIP kernel adopting WGPU's strategy would:

- Launch `ceil(H×S/128)` workgroups (6–32 depending on model)
- Loop over batches within each workgroup
- Eliminate L0/L1 cache thrashing at the cost of reduced bandwidth saturation

This would likely win at small models / high batch, and lose at large models where
bandwidth saturation is critical. A hybrid approach with a batch-size threshold could
select the optimal strategy at runtime.

**Batch size note**: WGPU's cursor encoding packs the batch index into 8 bits
(`cursor.batch = x & 0xff`), limiting batch to 256. This is a packing format choice, not
an architectural limit — repacking to 10+ bits would support larger batches.

## Recommendations

### Short-term

1. **Investigate logits download**: Compare WGPU's approach, consider async pipelining
2. **Fuse logits + sampling**: Compute argmax/top-k on GPU instead of downloading 51 MB
3. **Profile WGPU path**: Understand why it's faster at high batch sizes

### Medium-term

1. **Port FLA kernels**: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) has optimized RWKV7 kernels for prefill
2. **Chunk-based processing**: Process multiple timesteps together (for prefill)

### Long-term

1. **Wait for ROCm improvements**: gfx1151 support is still maturing
2. **Test on MI series GPUs**: hipBLASLt and rocBLAS are better optimized for datacenter GPUs

## Files Modified

| File | Changes |
|------|---------|
| `src/hip/kernels/copy.hip` | GPU f16→f32 kernel, WKV variants, hipBLASLt wrappers |
| `src/hip/ffi.rs` | FFI bindings for new kernels |
| `src/hip/kernels.rs` | Rust wrappers including batched GEMV, coalesced WKV |
| `src/hip/blas.rs` | Mixed-precision GEMM (gemm_ex) |
| `src/hip/blaslt.rs` | hipBLASLt integration (new) |
| `src/hip/scratch.rs` | logits_f32 buffer |
| `src/hip/model.rs` | Use GPU conversion for logits |

## References

- [gfx1151 Performance Regression Issue](https://github.com/ROCm/ROCm/issues/4748)
- [rocBLAS gemm_ex m==1 Issue](https://github.com/ROCm/rocBLAS/issues/1425)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [FlashAttention Paper](https://arxiv.org/pdf/2205.14135) - IO-awareness insights
- [Strix Halo ROCm Status](https://llm-tracker.info/_TOORG/Strix-Halo)
- [GEMM Optimization Blog](https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html)

## Appendix: Benchmark Commands

### HIP Profiling

```bash
# Run decode batch sweep
WEB_RWKV_BENCH_PROFILE=decode_batch_sweep cargo test --release --features hip \
    --test benchmarks -- --ignored --nocapture bench_smoke

# Run HIP profiling (per-operation timing)
WEB_RWKV_HIP_PROF=1 cargo test --release --features hip,hip-prof \
    --test hip_profiling -- --ignored --nocapture profile_decode_batch_256

# Profile with rocprofv3 (includes kernel dispatch timing)
rocprofv3 --hip-trace --hsa-trace --kernel-trace --memory-copy-trace \
    -o /tmp/rocprof_results -- ./target/release/deps/hip_profiling-* \
    profile_decode_batch_256 --ignored --nocapture

# Query rocprofv3 results (SQLite database)
sqlite3 /tmp/rocprof_results_results.db "
SELECT ks.display_name, COUNT(*) as n,
       printf('%.1f', AVG(d.end-d.start)/1000.0) as avg_us,
       printf('%.1f', SUM(d.end-d.start)/1e6) as total_ms,
       ks.arch_vgpr_count as vgprs, ks.group_segment_size as lds
FROM rocpd_kernel_dispatch d
JOIN kernel_symbols ks ON d.kernel_id = ks.kernel_id
GROUP BY ks.display_name ORDER BY total_ms DESC LIMIT 15"

# Compare rocBLAS vs hipBLASLt
cargo test --release --features hip --test hipblaslt_benchmark -- --ignored --nocapture
```

### WGPU / Vulkan Profiling

WGPU runs on RADV (Mesa's Vulkan driver). rocprofv3 only works for HIP, so WGPU
profiling requires different tools.

#### GPU Timestamp Queries (per-operation timing)

The `wgpu-prof` feature adds GPU timestamp queries around each operation. This
introduces some overhead but gives per-operation wall-clock GPU timing.

```bash
# Per-operation breakdown (wkv, att_proj, ffn_v, etc.)
cargo test --release --features wgpu-prof \
    --test wgpu_profiling -- --ignored --nocapture profile_decode_detailed

# Batch size sweep
cargo test --release --features wgpu-prof \
    --test wgpu_profiling -- --ignored --nocapture profile_decode_sweep
```

#### RADV RGP Trace Capture (hardware thread traces)

RADV can capture RGP (Radeon GPU Profiler) traces containing SQTT (SQ Thread
Trace) data — per-wavefront execution timing on the instruction-traced CU.
This has zero overhead on non-traced CUs and gives cycle-accurate wave timing.

```bash
# Capture one .rgp file per VkQueueSubmit
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=1 \
RADV_THREAD_TRACE_BUFFER_SIZE=32768 \
    cargo test --release --features wgpu-prof \
    --test wgpu_profiling -- --ignored --nocapture profile_decode_batch_256

# Traces are written to /tmp/<binary>_<timestamp>.rgp
ls -lhS /tmp/wgpu_profiling*.rgp
```

Environment variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| `MESA_VK_TRACE` | `rgp` | Enable RGP capture (must be `rgp`, not `1`) |
| `MESA_VK_TRACE_PER_SUBMIT` | `1` | One `.rgp` per VkQueueSubmit (required for compute-only workloads — frame-based capture only works for graphics) |
| `RADV_THREAD_TRACE_BUFFER_SIZE` | `32768` | SQTT buffer size in bytes per SE (larger = more data, bigger files) |

#### Decoding RGP / SQTT Traces

The `.rgp` files from RADV use the SQTT binary format (magic `0x50303042` / `B00P`),
not the newer RDF container format. AMD's `rocprof-trace-decoder` library can
parse the SQTT token stream into per-wave execution events.

```bash
# Decode SQTT tokens → SQLite (wave executions + occupancy events)
python3 scripts/rgp_decode_sqtt.py /tmp/wgpu_profiling-*.rgp

# Output: <input>_sqtt.db with tables:
#   wave_executions(wave_id, cu, simd, begin_time, end_time, duration_ticks, duration_ns, duration_us, instructions)
#   occupancy_events(wave_id, cu, simd, time, start)
```

**Dependencies**: The decoder script requires `tinygrad` (for ctypes bindings to
the AMD decoder library) and `librocprof-trace-decoder.so`:

```bash
# Install tinygrad (provides rocprof autogen bindings)
uv pip install -e repos/tinygrad

# Symlink the decoder library (ships with rocm-sdk-core pip package)
sudo ln -sf /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/librocprof-trace-decoder.so \
    /usr/local/lib/rocprof-trace-decoder.so
sudo ldconfig
```

#### Querying SQTT Results

```bash
# Top kernels by CU-time (instruction count is a kernel fingerprint)
sqlite3 /tmp/wgpu_sqtt_main.db "
SELECT instructions as insts, COUNT(*) as waves,
       printf('%.1f', SUM(duration_us)) as total_cu_us,
       printf('%.1f%%', SUM(duration_us)*100.0/
         (SELECT SUM(duration_us) FROM wave_executions WHERE duration_us>0)) as pct,
       printf('%.1f', AVG(duration_us)) as avg_us
FROM wave_executions WHERE duration_us > 0
GROUP BY instructions ORDER BY SUM(duration_us) DESC LIMIT 10"

# Wave launch pattern for a specific kernel (e.g. 30920-instruction WKV)
sqlite3 /tmp/wgpu_sqtt_main.db "
SELECT wave_id, simd, begin_time, end_time,
       printf('%.1f', (end_time-begin_time)/2900.0) as dur_us
FROM wave_executions WHERE instructions = 30920
ORDER BY begin_time LIMIT 20"
```

#### Extracting Shader Metadata from RGP

The `.rgp` files contain PAL-format ELFs with AMDGPU metadata (VGPRs, SGPRs,
wavefront size, LDS size) in msgpack-encoded `.note` sections. The
`rgp2sqlite.py` script extracts these:

```bash
python3 scripts/rgp2sqlite.py /tmp/wgpu_profiling-*.rgp
```

#### SQTT Limitations

- SQTT only traces **one CU per shader engine** (the instruction-traced CU).
  Wave counts and timing are samples, not totals.
- No dispatch-level correlation — waves are identified by instruction count
  (a fingerprint), not by shader name or dispatch ID.
- The `wgpu-prof` timestamp queries give more directly useful per-operation
  timing but add overhead. SQTT is zero-overhead on non-traced CUs.
