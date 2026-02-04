# HIP Kernel Optimization Research for RWKV7 Backend

This document captures research on building high-throughput HIP kernels for the RWKV7 model, specifically targeting AMD RDNA 3.5 (gfx1151) architecture on Strix Halo platforms.

## Table of Contents

1. [Target Hardware](#1-target-hardware)
2. [HIP Programming Model](#2-hip-programming-model)
3. [Wavefront and SIMD Optimization](#3-wavefront-and-simd-optimization)
4. [Memory Hierarchy Optimization](#4-memory-hierarchy-optimization)
5. [Grid API and Parallelization](#5-grid-api-and-parallelization)
6. [BLAS Library Selection](#6-blas-library-selection)
7. [RWKV7 Kernel Requirements](#7-rwkv7-kernel-requirements)
8. [Unified Memory for APU](#8-unified-memory-for-apu)
9. [Debugging and Profiling](#9-debugging-and-profiling)
10. [Reference Implementations](#10-reference-implementations)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Target Hardware

### AMD Radeon 8060S (gfx1151) - Strix Halo

| Specification | Value | Notes |
|---------------|-------|-------|
| Architecture | RDNA 3.5 | gfx1151 ISA |
| Compute Units | 40 | Full Strix Halo configuration |
| SIMDs per CU | 2 | RDNA dual-SIMD design |
| **Wavefront Size** | **32** | Critical: differs from CDNA (64) |
| Max Workgroup Size | 1024 | Threads per workgroup |
| Max Waves per CU | 32 | Occupancy ceiling |
| LDS (Shared Memory) | 64 KB | Per workgroup limit |
| L1 Cache | 32 KB | Per CU |
| L2 Cache | 2048 KB | Shared across CUs |
| L3 Cache | 32768 KB | Infinity Cache |
| Max Clock | 2900 MHz | Boost frequency |
| Fast F16 | TRUE | Native half precision |
| Memory Type | APU (Shared) | Unified memory with CPU |

### Key Architectural Differences from CDNA

| Feature | RDNA 3.5 (gfx1151) | CDNA3 (MI300X) |
|---------|-------------------|----------------|
| Wavefront Size | 32 | 64 |
| Primary Market | Consumer/APU | Datacenter |
| Matrix Cores (HIP "matrix cores" feature) | No | Yes (MFMA) |
| Memory | Unified (APU) | HBM3 (192GB) |
| LDS | 64 KB | 64 KB |

**Implication**: Kernels must be designed for wave32 operations, and block sizes should be multiples of 32 rather than 64.

---

## 2. HIP Programming Model

### Thread Hierarchy

```
Grid (Kernel Launch)
├── Block 0 (Workgroup)
│   ├── Wavefront 0 (32 threads)
│   ├── Wavefront 1 (32 threads)
│   └── ...
├── Block 1
└── ...
```

### Key Concepts

1. **Grid**: Collection of all blocks for a kernel launch
2. **Block (Workgroup)**: Threads that can synchronize and share LDS
3. **Wavefront (Warp)**: 32 threads executing in SIMD lockstep on RDNA
4. **Thread**: Single execution unit

### HIP Kernel Launch

```cpp
// Grid dimensions
dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);
dim3 block(threads_per_block_x, threads_per_block_y, threads_per_block_z);

hipLaunchKernelGGL(kernel_function, grid, block, shared_mem_bytes, stream, args...);
```

### Built-in Variables

```cpp
__device__ int blockIdx.x, blockIdx.y, blockIdx.z;   // Block index in grid
__device__ int threadIdx.x, threadIdx.y, threadIdx.z; // Thread index in block
__device__ int blockDim.x, blockDim.y, blockDim.z;   // Block dimensions
__device__ int gridDim.x, gridDim.y, gridDim.z;      // Grid dimensions
```

---

## 3. Wavefront and SIMD Optimization

### Wave32 Specific Considerations

RDNA (gfx10+) under HIP uses 32-thread wavefronts (wave32). Do **not** assume wave64 is available: HIP does not support `warpSize == 64` on gfx10+ (even if the ISA can run in wave64 mode). Always query `warpSize` / `hipDeviceAttributeWarpSize` at runtime for portability, but for gfx1151 you can rely on **32**.

**Block Size Recommendations for gfx1151:**
- `dim3 block(32)` - Minimum efficient size
- `dim3 block(64)` - Two wavefronts, good balance
- `dim3 block(128)` - Common choice for high occupancy
- `dim3 block(256)` - Maximum occupancy scenarios

### Thread Divergence

```cpp
// BAD: Thread divergence wastes SIMD lanes
if (threadIdx.x % 2 == 0) {
    // Half the wavefront masked
}

// GOOD: Divergence at wavefront boundaries
if (threadIdx.x < 32) {
    // Entire wavefront 0 executes
} else {
    // Entire wavefront 1 executes
}
```

### Wavefront-Level Intrinsics

```cpp
// Warp/wave shuffle (exchange data within wavefront)
//
// NOTE: HIP's shuffle intrinsics do not support half-float directly.
// Use float/int (or pack two FP16 into a 32-bit int) if you need shuffles.
__shfl(value, lane);           // Read value from specific lane
__shfl_xor(value, mask);       // XOR-based shuffle
__shfl_up(value, delta);       // Shift up by delta lanes
__shfl_down(value, delta);     // Shift down by delta lanes

// Wavefront vote functions
__any(predicate);              // True if any thread in wavefront has true
__all(predicate);              // True if all threads in wavefront have true
__ballot(predicate);           // Bitmask of predicate across wavefront

// Wavefront reduction
// HIP does not provide CUDA-style __reduce_* intrinsics consistently across targets.
// Prefer rocPRIM warp_reduce (or implement reduction via shuffles).
```

### Occupancy Optimization

Occupancy = Active Wavefronts / Maximum Wavefronts per CU

Factors limiting occupancy:
1. **Registers per thread**: High usage reduces waves per CU
2. **LDS per block**: Large LDS allocation limits concurrent blocks
3. **Block size**: Too small wastes CU resources

```cpp
// Query occupancy
int numBlocks;
hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, sharedMem);
```

---

## 4. Memory Hierarchy Optimization

### Memory Latency (Approximate Cycles)

| Memory Type | Latency | Bandwidth | Size |
|-------------|---------|-----------|------|
| Registers | 0 | Highest | ~256 per thread |
| LDS | ~20-30 | ~12 TB/s | 64 KB per CU |
| L1 Cache | ~30-50 | ~6 TB/s | 32 KB per CU |
| L2 Cache | ~150-200 | ~2 TB/s | 2 MB shared |
| L3/Infinity | ~300-400 | ~1.5 TB/s | 32 MB |
| Global (APU) | ~400-600 | ~500 GB/s | System RAM |

**Caveat**: These values are rough heuristics and can be misleading across architectures, clocks, and cache states. Use them only for intuition; use `rocprof` / ROCm tooling to measure actual bottlenecks.

### LDS (Local Data Share) Optimization

```cpp
// Declare shared memory
__shared__ float shared_data[256];

// Bank conflict-free access (32 banks on RDNA)
// Sequential threads access sequential addresses
shared_data[threadIdx.x] = global_data[threadIdx.x];

// Bank conflict (wave32: all lanes hit the same bank with a 32-word stride)
shared_data[threadIdx.x * 32] = value;  // BAD: severe bank conflicts
```

**Bank Conflict Rules (32 banks):**
- Bank = (address / 4) % 32 for 32-bit values
- Consecutive 32-bit addresses map to consecutive banks
- Broadcasting (same address) is handled efficiently

### Global Memory Coalescing

```cpp
// COALESCED: Adjacent threads access adjacent memory
float val = global_array[blockIdx.x * blockDim.x + threadIdx.x];

// UNCOALESCED: Strided access pattern
float val = global_array[threadIdx.x * stride];  // BAD if stride > 1
```

**Coalescing Requirements:**
- Global memory is serviced with naturally-aligned 32/64/128-byte transactions; alignment and access pattern matter.
- Best performance: consecutive lanes access consecutive addresses (unit-stride), avoiding strided/scattered patterns.
- Treat the "128-byte per wave" intuition as a rule-of-thumb; validate with profiling on gfx1151.

### Memory Transaction Optimization

```cpp
// Use vector types for efficient loads when data is properly aligned
// (e.g. 128-bit = 4x f32). Misalignment can negate the benefit.
float4 val = *reinterpret_cast<float4*>(&global_array[idx]);

// LDS should use 128-bit loads when possible
#pragma unroll
for (int i = 0; i < 4; i++) {
    // Process val.x, val.y, val.z, val.w
}
```

### Cache Utilization

```cpp
// NOTE: CUDA-specific cache-hint intrinsics like __ldg / __stcg are not portable.
// On AMD/HIP, __ldg is treated as a no-op (loads behave like normal global loads).
// Prefer structuring access patterns (coalescing, alignment, reuse) and rely on profiling.
```

---

## 5. Grid API and Parallelization

### Cooperative Groups (Grid-Level Synchronization)

```cpp
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void kernel_with_grid_sync() {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: All blocks compute
    compute_phase1();

    // Grid-wide synchronization
    grid.sync();

    // Phase 2: Process aggregated results
    compute_phase2();
}

// Launch with cooperative kernel API
void* args[] = { ... };
hipLaunchCooperativeKernel((void*)kernel_with_grid_sync,
                            gridDim, blockDim, args, sharedMem, stream);
```

**Note**: Grid sync requires checking device capability:
```cpp
int supports_coop;
hipDeviceGetAttribute(&supports_coop, hipDeviceAttributeCooperativeLaunch, device);
```

**Also note**: Cooperative launches can impose grid-size/occupancy limits (the runtime may return an error if the grid is too large for a cooperative launch). Treat grid-sync kernels as an advanced tool: validate feasibility on gfx1151 early, and keep a non-cooperative fallback design.

### Multi-Stream Parallelism

```cpp
hipStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    hipStreamCreate(&streams[i]);
}

// Launch kernels on different streams (concurrent execution)
for (int i = 0; i < NUM_STREAMS; i++) {
    hipLaunchKernelGGL(kernel, grid, block, 0, streams[i], args[i]...);
}

// Synchronize all streams
for (int i = 0; i < NUM_STREAMS; i++) {
    hipStreamSynchronize(streams[i]);
}
```

### Graph-Based Execution

```cpp
hipGraph_t graph;
hipGraphExec_t graphExec;

// Capture kernel launches into a graph
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
// ... launch kernels ...
hipStreamEndCapture(stream, &graph);

// Instantiate and execute graph (lower launch overhead)
hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
hipGraphLaunch(graphExec, stream);
```

---

## 6. BLAS Library Selection

### Research Summary (January 2026)

For gfx1151 (RDNA 3.5), the choice of BLAS library significantly impacts both performance and reliability.

### hipBLASLt Status: NOT RECOMMENDED for gfx1151

**Observed Issues (ROCm 7.1 era; re-validate on your exact ROCm build):**

Per [ROCm Issue #5643](https://github.com/ROCm/ROCm/issues/5643):
- hipBLASLt falls back to hipBLAS with "unsupported architecture!" error on gfx1151
- The fastpath kernels are not working
- Performance can degrade substantially vs. what you'd expect from tuned GEMM kernels on GFX11
- Some required code objects / Tensile libraries may be missing (depends on packaging)

**Root Cause:**
hipBLASLt primarily targets datacenter GPUs (MI2xx, MI3xx) and high-end consumer (7800/7900 series). Full RDNA 3.5 consumer/APU support remains limited.

### rocBLAS: RECOMMENDED for gfx1151

rocBLAS is the safest default for GEMM on gfx1151 today. ROCm release notes indicate gfx1151 enablement for rocBLAS/hipBLAS in recent ROCm releases (still verify on your specific distro + driver stack):
- ROCm 7.0.2 release notes: rocBLAS + hipBLAS enable gfx1150/gfx1151
- ROCm 7.1.0 release notes: hipBLAS enable gfx1151 / rocBLAS enable gfx1150/gfx1151

**Important**: rocBLAS uses **column-major** storage by default. For the HIP backend we will store GEMM inputs/weights in rocBLAS-native column-major layout so callsites stay simple (no row/col mapping at every GEMM). Do not copy/paste GEMM dimension/lda/ldb/ldc examples without reconciling the chosen layout.

```cpp
#include <rocblas/rocblas.h>

// Create handle
rocblas_handle handle;
rocblas_create_handle(&handle);

// GEMM: C = alpha * A * B + beta * C
// NOTE: Adjust transposes/leading dimensions for column-major layout.
rocblas_gemm_ex(
    handle,
    rocblas_operation_none,  // transA
    rocblas_operation_none,  // transB (example only; pick based on layout)
    M, N, K,  // dimensions
    &alpha,
    A, rocblas_datatype_f16_r, lda,  // input: FP16
    B, rocblas_datatype_f16_r, ldb,  // weights: FP16
    &beta,
    C, rocblas_datatype_f16_r, ldc,  // output: FP16
    C, rocblas_datatype_f16_r, ldc,
    rocblas_datatype_f32_r,  // compute: FP32
    rocblas_gemm_algo_standard,
    0, 0  // solution index, flags
);
```

### Performance Expectations

| Operation | Size | rocBLAS Expected | Custom Kernel |
|-----------|------|------------------|---------------|
| GEMV (B=1) | 768→768 | ~0.1ms | ~0.05ms (optimized) |
| GEMM (B=32) | 768→768 | ~0.3ms | ~0.5ms |
| GEMM (B=32) | 768→3072 | ~0.8ms | ~1.2ms |

**Note**: The numbers above are placeholders. Do not treat them as targets until benchmarked on the actual gfx1151 system/ROCm build.

**Recommendation**: Use rocBLAS for large GEMMs; consider custom GEMV for B=1 streaming if profiling shows GEMV dominates.

### Integration with TheRock

```bash
# TheRock includes rocBLAS
source /opt/venv/bin/activate
export ROCM_SDK_ROOT=$(/opt/venv/bin/rocm-sdk path --root)

# Link flags for Rust build.rs
# -L${ROCM_SDK_ROOT}/lib -lrocblas
```

### Fallback Strategy

```rust
// In Rust wrapper
pub fn matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if rocblas_available() {
        rocblas_gemm(a, b, out)
    } else {
        custom_gemm_kernel(a, b, out)
    }
}
```

---

## 7. RWKV7 Kernel Requirements

### Core Operations

Based on the RWKV7 architecture, the following kernels are required:

#### 1. WKV7 State Update Kernel (Primary)

**Mathematical Operation:**
```
wkv_t = wkv_{t-1} * G_t + v_t^T · k̃_t

Where G_t = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)
```

**Kernel Interface:**
```cpp
__global__ void wkv7_forward(
    int T,              // Sequence length
    int H,              // Number of heads
    half* w,            // Decay weights [B, T, H, C]
    half* q,            // Query/receptance [B, T, H, C]
    half* k,            // Replacement key [B, T, H, C]
    half* v,            // Value [B, T, H, C]
    half* a,            // -κ̂ (negated normalized removal key) [B, T, H, C]
    half* b,            // κ̂ * a_t (removal key scaled by learning rate) [B, T, H, C]
    half* y,            // Output [B, T, H, C]
    float* state,       // State checkpoints [B, H, T/CHUNK, C, C]
    float* sa           // State-alpha intermediate [B, T, H, C]
);
```

**Grid Configuration:**
- Blocks: `dim3(H, B)` - One block per (head, batch)
- Threads: `dim3(C)` where C = head_size (64)
- For wave32: Consider `dim3(32)` or `dim3(64)` for RDNA

**State Layout:**
```
state[batch][head][v_dim][k_dim]  // [B, H, C, C] matrix per head
```

#### 2. Token Shift Kernel

```cpp
__global__ void token_shift(
    int T, int C,
    half* x,            // Input [B, T, C]
    half* x_prev,       // Previous token state [B, C]
    half* mix,          // Mix parameters [C]
    half* out           // Output [B, T, C]
);
```

#### 3. Linear Projection Kernels (GEMV/GEMM)

For attention: receptance, key, value, output projections
For FFN: expand (4x) and contract projections

Consider using rocBLAS for matrix operations:
```cpp
#include <rocblas/rocblas.h>
rocblas_gemm_ex(...);  // For batched matrix multiply
```

#### 4. Activation Kernels

```cpp
// ReLU^2 for channel mixing
__global__ void squared_relu(half* x, int n);

// Sigmoid for gates
__global__ void sigmoid(half* x, int n);

// Double exponential for decay: exp(-exp(w))
__global__ void decay_activation(half* w, half* out, int n);
```

#### 5. Normalization Kernels

```cpp
// LayerNorm
__global__ void layer_norm(half* x, half* weight, half* bias, int C, float eps);

// GroupNorm (per-head normalization)
__global__ void group_norm(half* x, half* weight, half* bias, int H, int C, float eps);
```

### Precision Considerations

| Tensor | Storage | Computation | Reason |
|--------|---------|-------------|--------|
| Weights | FP16/BF16 | FP32 | Memory bandwidth |
| State | FP32 | FP32 | Accumulation precision |
| Activations | FP16 | FP16/FP32 | Balance speed/accuracy |
| Output | FP16 | FP32 → FP16 | Match model format |

**Note**: gfx1151 has native F16 support (`Fast F16: TRUE`).

#### 6. Additional Required Kernels (Gap Analysis)

Based on analysis of the Vulkan reference implementation, these additional kernels are needed:

```cpp
// WKV Bonus (time_first_v7) - reduction for bonus term
// u_t = (r · (ρ ⊙ k̃)^T) v
__global__ void wkv_bonus(
    int T, int H, int C,
    half* r_k,          // Per-head bonus weight [H, C]
    half* r,            // Receptance [B, T, H, C]
    half* k,            // Replacement key [B, T, H, C]
    half* v,            // Value [B, T, H, C]
    half* out           // Output to add [B, T, H, C]
);

// Control K (replacement key computation)
// k̃ = k * (1 + (a - 1) * k_a)
__global__ void control_k(
    int n,
    half* k_a,          // Replacement rate booster [C]
    half* a,            // In-context learning rate [B, T, C]
    half* k             // Key (modified in-place) [B, T, C]
);

// Tanh activation (for decay LoRA)
__global__ void tanh_activation(half* x, int n);

// Softplus activation (for decay computation)
// out = -log(1 + exp(-x)) - 0.5
__global__ void softplus_decay(half* x, half* out, int n);

// Channel mix state update (FFN state management)
__global__ void channel_mix_state(
    int T, int C,
    uint32_t* cursors,  // Batch cursors
    float* state,       // FFN state [B, C]
    half* x,            // Input [B, T, C]
    half* out           // Output [B, T, C]
);
```

#### 7. Kernel Fusion Opportunities

For performance, consider fusing:

1. **Decay LoRA + Activation**: `tanh(xw @ w1) @ w2 + w0` → `-softplus(-result) - 0.5`
2. **Control K + L2 Norm**: Compute k̃ and κ̂ in single kernel
3. **Add + Activation**: Common pattern for LoRA outputs with sigmoid/tanh
4. **MatMul + Activation**: rocBLAS doesn't support fusion; custom kernel may be faster for small sizes

---

## 8. Unified Memory for APU

### APU Memory Architecture

gfx1151 (Strix Halo) is an APU with unified memory architecture. CPU and GPU share the same physical memory, enabling zero-copy data access.

### HIP Unified Memory API

```cpp
// Allocate managed memory (accessible by both CPU and GPU)
void* ptr;
hipMallocManaged(&ptr, size, hipMemAttachGlobal);

// Prefetch to GPU (optional optimization)
hipMemPrefetchAsync(ptr, size, deviceId, stream);

// Advise memory usage pattern
hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, deviceId);  // For weights
hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, deviceId);  // Keep on GPU
```

### Recommended Memory Strategy for RWKV7

| Data Type | Allocation | Access Pattern | Rationale |
|-----------|------------|----------------|-----------|
| **Model Weights** | `hipMallocManaged` + `SetReadMostly` | Read-only, all layers | Avoid copy, cache-friendly |
| **State Tensors** | `hipMalloc` (device) | Read-write, per-token | Frequent updates, keep on GPU |
| **Activations** | `hipMalloc` (device) | Temporary, per-layer | High bandwidth needed |
| **Input/Output** | `hipMallocManaged` | Occasional CPU access | Easy CPU↔GPU interop |

### Implementation Pattern

```cpp
class UnifiedTensor {
    void* data;
    bool managed;

public:
    static UnifiedTensor weights(size_t size) {
        void* ptr;
        hipMallocManaged(&ptr, size);
        hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, 0);
        return UnifiedTensor{ptr, true};
    }

    static UnifiedTensor activations(size_t size) {
        void* ptr;
        hipMalloc(&ptr, size);  // Device-only for performance
        return UnifiedTensor{ptr, false};
    }
};
```

### Performance Considerations

1. **First Access Latency**: Unified memory has higher first-access latency due to page faults
2. **Prefetching**: Use `hipMemPrefetchAsync` before kernel launch for predictable access
3. **Coherence Overhead**: Avoid simultaneous CPU+GPU access to same pages
4. **Page Migration**: Large allocations may trigger page migration; consider chunking

### Rust FFI Integration

```rust
// In build.rs or hip wrapper
extern "C" {
    fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn hipMemAdvise(ptr: *mut c_void, size: usize, advice: i32, device: i32) -> i32;
}

pub struct ManagedBuffer {
    ptr: *mut u8,
    size: usize,
}

impl ManagedBuffer {
    pub fn new_readonly(size: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            check_hip(hipMallocManaged(&mut ptr, size, 0))?;
            check_hip(hipMemAdvise(ptr, size, HIP_MEM_ADVISE_SET_READ_MOSTLY, 0))?;
        }
        Ok(Self { ptr: ptr as *mut u8, size })
    }
}
```

---

## 9. Debugging and Profiling

### Environment Variables for Debugging

```bash
# Serialize kernel execution (find faulty kernel)
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

# Disable SDMA (fixes artifacts on APUs)
export HSA_ENABLE_SDMA=0

# Verbose HIP runtime
export AMD_LOG_LEVEL=4

# Compiler debug info
export HIPCC_COMPILE_FLAGS_APPEND="-g -O0"
```

### ROCm Profiling Tools

#### rocprof - Command Line Profiler

```bash
# Basic kernel profiling
rocprof --stats ./my_app

# Hardware counter collection
rocprof -i counters.txt -o output.csv ./my_app

# HIP API tracing
rocprof --hip-trace ./my_app

# HSA API tracing
rocprof --hsa-trace ./my_app
```

#### Example Counter Configuration (counters.txt)

```
pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_LDS
pmc: TCC_HIT, TCC_MISS
pmc: FETCH_SIZE, WRITE_SIZE
```

### ROCgdb - Source-Level Debugger

```bash
# Launch debugger
rocgdb ./my_app

# Common commands
(gdb) info rocm
(gdb) break kernel_function
(gdb) info threads
(gdb) thread <id>
(gdb) print variable
```

### ltrace - Library Call Tracing

```bash
# Trace HIP runtime calls
ltrace -e 'hip*' ./my_app 2>&1 | head -100
```

### Memory Checking

```bash
# Check for memory leaks and errors
rocprof --stats ./my_app

# AMD-specific memory sanitizer
export HSA_TOOLS_LIB=librocr_sanitizer64.so
./my_app
```

### Performance Metrics to Monitor

1. **Occupancy**: Active wavefronts / Max wavefronts
2. **Memory Bandwidth**: GB/s achieved vs. theoretical peak
3. **Compute Utilization**: VALU instruction throughput
4. **LDS Bank Conflicts**: Serialized LDS accesses
5. **Cache Hit Rate**: L1/L2 efficiency

---

## 10. Reference Implementations

### HIP Reference (RWKV-LM-V7)

Location: `/workspace/web-rwkv/repos/RWKV-LM-V7/cuda/`

Files:
- `wkv7_hip.hip` - HIP kernel (bfloat16)
- `wkv7_hip_fp32.hip` - HIP kernel (float32)
- `wkv7_op.hip` - PyTorch binding

Key Implementation Details:
- Uses `dim3(H, B)` blocks with `dim3(C)` threads (C=64)
- BF16 I/O with FP32 state accumulation
- Checkpoints state every CHUNK_LEN=16 tokens
- Shared memory for input vectors (q, w, k, v, a, b)

**Forward Kernel Algorithm:**
```cpp
float state[C] = {0};  // Per-thread state row

for (int t = 0; t < T; t++) {
    // 1. Load inputs to shared memory
    __syncthreads();
    w[i] = exp(-exp(w_input[i]));  // Decay activation

    // 2. Compute sa = dot(state, a)
    float sa = 0;
    for (int j = 0; j < C; j++) {
        sa += a[j] * state[j];
    }

    // 3. State update + output
    float y = 0;
    for (int j = 0; j < C; j++) {
        state[j] = state[j] * w[j] + sa * b[j] + k[j] * v;
        y += state[j] * q[j];
    }
    output[t] = y;

    // 4. Checkpoint every CHUNK_LEN
    if ((t + 1) % CHUNK_LEN == 0) save_state();
}
```

### Vulkan/WGSL Reference (web-rwkv)

Location: `/workspace/web-rwkv/src/`

Key Files:
- `runtime/v7.rs` - RWKV7 model implementation
- `shaders/time_mix_v7.wgsl` - Time mixing shader
- `shaders/control_k_v7.wgsl` - Key control shader

Architecture Insights:
- Workgroup-shared memory for synchronization
- Per-head processing with vectorized operations
- Hook system for debugging intermediate values
- Macro-based shader specialization (FP16/FP32)

---

## 11. Testing Strategy

### Layer Parity Testing Approach

To ensure HIP implementation matches reference:

1. **Extract Reference Values from Python:**
```python
# In RWKV_Tmix_x070.forward()
def forward_with_debug(self, x, v_first):
    # ... existing code ...

    # Before CUDA kernel
    debug_inputs = {
        'r': r.clone(), 'w': w.clone(), 'k': k.clone(),
        'v': v.clone(), 'kk': kk.clone(), 'a': a.clone()
    }

    # After CUDA kernel
    x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
    debug_outputs = {'wkv_out': x.clone()}

    torch.save({'inputs': debug_inputs, 'outputs': debug_outputs},
               f'layer_{self.layer_id}_debug.pt')
    return x, v_first
```

2. **Test Individual Kernels:**
- Test `exp(-exp(w))` decay activation separately
- Test state update with known inputs
- Test normalization against PyTorch reference

3. **Numerical Tolerance:**
- FP32 state: expect exact match or < 1e-6 relative error
- FP16 output: allow 1e-3 relative error due to precision limits

4. **Edge Cases to Test:**
- Sequence length = 1 (no time dependency)
- Maximum sequence length (memory limits)
- Batch size variations
- Zero-initialized vs. pre-loaded state

### Test Fixture Generation

```python
def generate_test_fixtures(model_path, output_dir):
    """Generate test data from pre-trained model."""
    model = load_model(model_path)

    # Sample inputs
    test_sequences = [
        torch.randint(0, vocab_size, (1, 16)),   # Short sequence
        torch.randint(0, vocab_size, (1, 128)),  # Medium sequence
        torch.randint(0, vocab_size, (4, 64)),   # Batched
    ]

    for i, seq in enumerate(test_sequences):
        with torch.no_grad():
            # Hook to capture intermediates
            captured = {}
            def hook(module, input, output):
                captured['input'] = input
                captured['output'] = output

            model.blocks[0].att.register_forward_hook(hook)
            output = model(seq)

            torch.save({
                'input_tokens': seq,
                'layer_0_att_input': captured['input'],
                'layer_0_att_output': captured['output'],
                'final_output': output
            }, f'{output_dir}/test_case_{i}.pt')
```

---

## References

### Official Documentation
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)
- [HIP C++ Language Extensions (warpSize, shuffles)](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html)
- [HIP Hardware Features (occupancy limits, packed math)](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html)
- [HIP Hardware Implementation](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/hardware_implementation.html)
- [HIP Debugging Guide](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/debugging.html)
- [HIP Porting Guide (CUDA intrinsics like __ldg)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP Unified Memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/unified_memory.html)
- [ROCProfiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html)
- [HIP Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html)

### RDNA 3.5 / Strix Halo Specific
- [AMD Strix Halo ROCm Benchmarks](https://www.phoronix.com/review/amd-strix-halo-rocm-benchmarks)
- [llama.cpp ROCm gfx1151 Setup](https://medium.com/@GenerationAI/ultralytics-yolo-sam-with-rocm-7-0-on-amd-ryzen-ai-max-395-strix-halo-radeon-8060s-gfx1151-6f48bb9bcbf9)
- [ROCm RDNA4 Linux Guide](https://gist.github.com/apollo-mg/ecba6a0c29323325a7ac3babf08e53be)

### Architecture and Optimization
- [GEAK HIP Optimizations Blog](https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html)
- [ROCm 7.1 Improvements](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.1/README.html)
- [AMD MI300X Architecture (for comparison)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [ROCm 7.0.2 Release Notes (gfx1151 enablement claims)](https://rocm.docs.amd.com/projects/rocm/en/docs-7.0.2/about/release-notes.html)
- [ROCm 7.1.0 Release Notes (gfx1151 enablement claims)](https://rocm.docs.amd.com/projects/rocm/en/docs-7.1.0/about/release-notes.html)
- [ROCm 7.2 Release Notes (Radeon/Ryzen support updates)](https://www.amd.com/en/resources/support-articles/release-notes/ROCm-Release-Notes.html)

---

## Appendix: gfx1151 Specific Optimizations

### Known Issues and Workarounds

1. **Debugging SDMA / copies**: `export HSA_ENABLE_SDMA=0` can help isolate SDMA-related issues (debugging knob; validate perf impact).
2. **hipBLASLt Support**: Limited; prefer rocBLAS fallback
3. **GFX version overrides**: Some projects use `HSA_OVERRIDE_GFX_VERSION` as a workaround for support/perf issues. Treat this as a last-resort/debug tool only (it can be unsafe and may break at any time).

### Recommended Compile Flags

```bash
AMDGPU_TARGETS=gfx1151
HIPCC_FLAGS="-O3 -ffast-math --offload-arch=gfx1151"
```

### Wave32 Adaptation from Wave64 Kernels

When adapting CDNA (wave64) kernels to RDNA (wave32):
1. Change block sizes from multiples of 64 to multiples of 32
2. Update wavefront reduction operations
3. Adjust LDS bank conflict calculations (32 banks)
4. Verify occupancy calculations
