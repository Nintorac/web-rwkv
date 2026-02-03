// HIP softmax kernel for RWKV inference backend
// Three-pass numerically-stable softmax using the same reduction primitives as norm.hip.
// Input layout: [V, T, 1, 1] -- V contiguous elements per token, T tokens.
// One block per token, SOFTMAX_BLOCK_SIZE=256.
#include <hip/hip_runtime.h>

extern "C" {

#define SOFTMAX_BLOCK_SIZE 256

// Warp-level max reduction using shuffle instructions (same pattern as norm.hip)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Block-level max reduction using shared memory (same pattern as norm.hip)
__device__ float block_reduce_max(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    return val;
}

// Warp-level sum reduction using shuffle instructions (same pattern as norm.hip)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level sum reduction using shared memory (same pattern as norm.hip)
__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Per-token softmax over the vocab dimension.
// Input layout: [V, T, 1, 1] -- V contiguous elements per token, T tokens.
// One block per token.
__global__ void kernel_softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int V    // vocab size (elements per token)
) {
    int token = blockIdx.x;
    int base = token * V;

    __shared__ float smem[SOFTMAX_BLOCK_SIZE / 32];
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Pass 1: find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        local_max = fmaxf(local_max, input[base + i]);
    }
    float global_max = block_reduce_max(local_max, smem);
    if (threadIdx.x == 0) shared_max = global_max;
    __syncthreads();
    float max_val = shared_max;

    // Pass 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        local_sum += expf(input[base + i] - max_val);
    }
    float global_sum = block_reduce_sum(local_sum, smem);
    if (threadIdx.x == 0) shared_sum = global_sum;
    __syncthreads();
    float inv_sum = 1.0f / shared_sum;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        output[base + i] = expf(input[base + i] - max_val) * inv_sum;
    }
}

hipError_t launch_softmax_f32(
    const float* input,
    float* output,
    int V,          // vocab size
    int T,          // number of tokens
    hipStream_t stream
) {
    if (V <= 0 || T <= 0) return hipSuccess;
    hipLaunchKernelGGL(kernel_softmax_f32,
        dim3(T), dim3(SOFTMAX_BLOCK_SIZE), 0, stream,
        input, output, V);
    return hipGetLastError();
}

} // extern "C"
