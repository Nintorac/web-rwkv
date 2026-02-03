// HIP kernels for RWKV inference backend
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>


extern "C" {

// ============================================================================
// WKV7 Core Kernel
// Implements RWKV7's time mixing attention mechanism.
//
// The kernel processes T tokens sequentially, maintaining an NxN state matrix
// per head per batch. Each thread handles one row of the state matrix.
//
// Input/Output shapes (web-rwkv convention with N as fastest axis):
//   w_decay: [N, H, T, B] - pre-computed decay = exp(-exp(w_raw))
//   q, k, v, a, b: [N, H, T, B] - query, key, value, attention components
//   state_in/out: [N, N, H, B] - NxN state matrix per head per batch
//   output: [N, H, T, B]
//
// Algorithm per timestep:
//   1. sa = sum(a[j] * state[i,j]) for all j  // attention over state
//   2. state[i,j] = state[i,j] * w[j] + sa * b[j] + k[j] * v[i]  // update
//   3. y[i] = sum(state[i,j] * q[j]) for all j  // output
// ============================================================================

// Fixed head size for RWKV7 - matches the reference implementation
#define WKV7_HEAD_SIZE 64

// Pairwise summation helper: reduces a 64-element array to a single sum.
// Uses tree-based reduction: error is O(epsilon * log(N)) vs O(epsilon * N).
// This improves numerical precision for FP32 accumulation.
__device__ __forceinline__ float pairwise_sum_64(float* __restrict__ tmp) {
    // Tree reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // This matches the Python pairwise_sum: x = x[..., :half] + x[..., half:]
    #pragma unroll
    for (int stride = 32; stride > 0; stride >>= 1) {
        #pragma unroll
        for (int j = 0; j < stride; j++) {
            tmp[j] = tmp[j] + tmp[j + stride];
        }
    }
    return tmp[0];
}

// ============================================================================
// WAVE-COOPERATIVE WKV7 KERNEL (No atomics, explicit reductions)
//
// Avoids atomicAdd by using explicit parallel reductions within waves.
// This should be faster on RDNA3 where atomics can be slow.
// ============================================================================

__global__ __launch_bounds__(256, 4)
void kernel_wkv7_wave_reduce(
    const __half* __restrict__ w_decay,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const float* __restrict__ state_in,
    __half* __restrict__ output,
    float* __restrict__ state_out,
    const int* __restrict__ lengths,
    int N, int H, int T, int B
) {
    // Grid: (H, B), Block: 256 threads (8 waves of 32)
    // Strategy: Each wave handles one row of state (64 elements)
    //           8 waves = 8 rows processed in parallel
    //           Loop over 64/8 = 8 iterations to cover all rows

    const int hh = blockIdx.x;
    const int bb = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;      // 0-7
    const int lane_id = tid % 32;      // 0-31

    if (hh >= H || bb >= B) return;

    const int real_len = lengths[bb];

    // State in LDS: 64x64 with padding
    __shared__ float sh_state[64][65];  // +1 for bank conflict avoidance

    // Parameters
    __shared__ float sh_q[64];
    __shared__ float sh_k[64];
    __shared__ float sh_w[64];
    __shared__ float sh_a[64];
    __shared__ float sh_b[64];

    // Per-row reduction results
    __shared__ float sh_sa[64];
    __shared__ float sh_y[64];

    // Load initial state
    const int state_base = hh * N * N + bb * N * N * H;
    #pragma unroll 4
    for (int elem = tid; elem < 64 * 64; elem += 256) {
        int i = elem / 64;
        int j = elem % 64;
        sh_state[i][j] = state_in[state_base + elem];
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        const int param_base = hh * N + t * H * N + bb * T * H * N;

        // Cooperative parameter load
        if (tid < 64) {
            int idx = param_base + tid;
            sh_q[tid] = __half2float(q[idx]);
            sh_w[tid] = __half2float(w_decay[idx]);
            sh_k[tid] = __half2float(k[idx]);
            sh_a[tid] = __half2float(a[idx]);
            sh_b[tid] = __half2float(b[idx]);
        }
        __syncthreads();

        if (t < real_len) {
            // Process 8 rows at a time (one per wave)
            for (int row_base = 0; row_base < 64; row_base += 8) {
                int i = row_base + wave_id;  // Row this wave handles

                // Step 1: Compute sa[i] = sum_j(a[j] * state[i,j])
                // Each wave (32 threads) reduces 64 elements
                // First: lanes 0-31 handle j=0-31, then j=32-63
                float partial_sa = 0.0f;

                #pragma unroll 2
                for (int jj = 0; jj < 64; jj += 32) {
                    int j = jj + lane_id;
                    partial_sa += sh_a[j] * sh_state[i][j];
                }

                // Wave-level reduction using DPP (or shuffle)
                #pragma unroll 5
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial_sa += __shfl_down(partial_sa, offset, 32);
                }

                // Lane 0 has the sum
                if (lane_id == 0) {
                    sh_sa[i] = partial_sa;
                }
            }
            __syncthreads();

            // Step 2 & 3: Update state and compute y
            for (int row_base = 0; row_base < 64; row_base += 8) {
                int i = row_base + wave_id;
                float sa_i = sh_sa[i];
                float v_i = __half2float(v[param_base + i]);

                float partial_y = 0.0f;

                #pragma unroll 2
                for (int jj = 0; jj < 64; jj += 32) {
                    int j = jj + lane_id;

                    float s = sh_state[i][j];
                    float new_s = sh_w[j] * s + sh_b[j] * sa_i + sh_k[j] * v_i;
                    sh_state[i][j] = new_s;

                    partial_y += sh_q[j] * new_s;
                }

                // Wave reduction for y
                #pragma unroll 5
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial_y += __shfl_down(partial_y, offset, 32);
                }

                if (lane_id == 0) {
                    sh_y[i] = partial_y;
                }
            }
            __syncthreads();

            // Write output
            if (tid < 64) {
                output[param_base + tid] = __float2half(sh_y[tid]);
            }
        }
        __syncthreads();
    }

    // Write final state
    #pragma unroll 4
    for (int elem = tid; elem < 64 * 64; elem += 256) {
        state_out[state_base + elem] = sh_state[elem / 64][elem % 64];
    }
}

hipError_t launch_wkv7_wave_reduce(
    const __half* w_decay,
    const __half* q,
    const __half* k,
    const __half* v,
    const __half* a,
    const __half* b,
    const float* state_in,
    __half* output,
    float* state_out,
    const int* lengths,
    int N, int H, int T, int B,
    hipStream_t stream
) {
    if (N <= 0 || H <= 0 || T <= 0 || B <= 0) {
        return hipSuccess;
    }
    if (N != 64) {
        return hipErrorInvalidValue;
    }
    dim3 grid(H, B);
    dim3 block(256);
    hipLaunchKernelGGL(kernel_wkv7_wave_reduce, grid, block, 0, stream,
                       w_decay, q, k, v, a, b, state_in, output, state_out, lengths,
                       N, H, T, B);
    return hipGetLastError();
}

// ============================================================================
// FUSED WKV7 KERNEL (T=1 decode specialization)
//
// Same 256-thread wave-reduce architecture as wave_reduce_t1 but with:
//
// 1. FUSED sa + update + y in a single per-row pass:
//    - State is read ONCE into 2 VGPRs (state_cache[2]), then reused for both
//      the sa dot product and the update+y computation.
//    - Eliminates the second global state read (saves 16KB/block at high batch).
//
// 2. ZERO barriers between sa and update/y passes:
//    - sa is reduced via __shfl_down then broadcast via __shfl to all lanes.
//    - No shared memory needed for sa or y results.
//    - Only barrier is the initial __syncthreads() after param loading.
//
// 3. IN-PLACE state update:
//    - Single state buffer (not separate state_in / state_out).
//    - Halves L2 write allocation pressure.
//    - Safe because each (head, batch) block has exclusive ownership.
//
// 4. Params as f32 in LDS (no per-use __half2float conversion):
//    - 5 × 64 × 4 = 1280 bytes LDS (vs 640 f16 + 512 for sh_sa/sh_y = 1152)
//
// Grid: (H, B), Block: 256 threads (8 waves). Lengths[b] == 0 → early return.
// ============================================================================

__global__ __launch_bounds__(256, 4)
void kernel_wkv7_fused_t1(
    const __half* __restrict__ w_decay,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ state,       // in-place: [N, N, H, B]
    __half* __restrict__ output,
    const int* __restrict__ lengths,
    int N, int H, int B
) {
    const int hh = blockIdx.x;
    const int bb = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;

    if (hh >= H || bb >= B) return;

    const int real_len = lengths[bb];
    if (real_len <= 0) {
        // In-place state: nothing to copy. Just return.
        return;
    }

    // Store parameters as f32 in LDS — eliminates per-use __half2float.
    __shared__ float sh_q[64];
    __shared__ float sh_k[64];
    __shared__ float sh_w[64];
    __shared__ float sh_a[64];
    __shared__ float sh_b[64];

    const int state_base = hh * N * N + bb * N * N * H;
    const int param_base = hh * N + bb * H * N;  // t = 0 only

    // 256 threads load 5 × 64 = 320 values (threads 0–63 load first array, etc.)
    if (tid < 64) {
        int idx = param_base + tid;
        sh_q[tid] = __half2float(q[idx]);
        sh_w[tid] = __half2float(w_decay[idx]);
        sh_k[tid] = __half2float(k[idx]);
        sh_a[tid] = __half2float(a[idx]);
        sh_b[tid] = __half2float(b[idx]);
    }
    __syncthreads();
    // --- No more barriers after this point ---

    // Fused loop: each wave processes one row per iteration.
    // 8 waves × 8 iterations = 64 rows.
    for (int row_base = 0; row_base < 64; row_base += 8) {
        int i = row_base + wave_id;

        // --- Phase 1: Read state into registers + compute sa ---
        float state_cache[2];  // 2 chunks × 1 lane = 2 VGPRs
        float partial_sa = 0.0f;

        #pragma unroll 2
        for (int jj = 0; jj < 64; jj += 32) {
            int j = jj + lane_id;
            float s = state[state_base + i * 64 + j];
            state_cache[jj >> 5] = s;          // cache for reuse
            partial_sa += sh_a[j] * s;
        }

        // Wave-level reduction for sa
        #pragma unroll 5
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_sa += __shfl_down(partial_sa, offset, 32);
        }
        // Broadcast sa from lane 0 to all lanes
        float sa_i = __shfl(partial_sa, 0, 32);

        // --- Phase 2: Update state in-place + compute y (from cached state) ---
        float v_i = __half2float(v[param_base + i]);
        float partial_y = 0.0f;

        #pragma unroll 2
        for (int jj = 0; jj < 64; jj += 32) {
            int j = jj + lane_id;
            float s = state_cache[jj >> 5];    // reuse cached value
            float new_s = sh_w[j] * s + sh_b[j] * sa_i + sh_k[j] * v_i;
            state[state_base + i * 64 + j] = new_s;  // in-place write
            partial_y += sh_q[j] * new_s;
        }

        // Wave-level reduction for y
        #pragma unroll 5
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_y += __shfl_down(partial_y, offset, 32);
        }

        // Direct output write — no shared memory needed
        if (lane_id == 0) {
            output[param_base + i] = __float2half(partial_y);
        }
    }
}

hipError_t launch_wkv7_fused_t1(
    const __half* w_decay,
    const __half* q,
    const __half* k,
    const __half* v,
    const __half* a,
    const __half* b,
    float* state,
    __half* output,
    const int* lengths,
    int N, int H, int B,
    hipStream_t stream
) {
    if (N != 64 || H <= 0 || B <= 0) {
        return hipErrorInvalidValue;
    }
    dim3 grid(H, B);
    dim3 block(256);
    hipLaunchKernelGGL(kernel_wkv7_fused_t1, grid, block, 0, stream,
                       w_decay, q, k, v, a, b, state, output, lengths,
                       N, H, B);
    return hipGetLastError();
}

} // extern "C"
