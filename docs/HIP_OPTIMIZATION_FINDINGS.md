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

```bash
# Run decode batch sweep
WEB_RWKV_BENCH_PROFILE=decode_batch_sweep cargo test --release --features hip \
    --test benchmarks -- --ignored --nocapture bench_smoke

# Run HIP profiling (per-operation timing)
WEB_RWKV_HIP_PROF=1 cargo test --release --features hip,hip-prof \
    --test hip_profiling -- --ignored --nocapture profile_decode_batch_256

# Run WGPU profiling (GPU timestamp-based per-operation timing)
cargo test --release --features wgpu-prof \
    --test wgpu_profiling -- --ignored --nocapture profile_decode_detailed

# Profile with rocprof (legacy)
rocprof --hip-trace ./target/release/examples/wkv_profile

# Profile with rocprofv3 (includes kernel dispatch timing)
rocprofv3 --hip-trace --hsa-trace --kernel-trace --memory-copy-trace \
    -o /tmp/rocprof_results -- ./target/release/deps/hip_profiling-* \
    profile_decode_batch_256 --ignored --nocapture

# Query rocprofv3 results (SQLite database)
sqlite3 /tmp/rocprof_results_results.db "
SELECT ks.display_name, COUNT(*) as n, SUM(d.end-d.start)/1e6 as total_ms
FROM rocpd_kernel_dispatch d
JOIN rocpd_info_kernel_symbol ks ON d.kernel_id = ks.id
GROUP BY ks.display_name ORDER BY total_ms DESC LIMIT 10"

# Compare rocBLAS vs hipBLASLt
cargo test --release --features hip --test hipblaslt_benchmark -- --ignored --nocapture
```
