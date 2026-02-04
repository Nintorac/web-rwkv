// HIP kernels for RWKV inference backend
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>


extern "C" {

// ============================================================================
// Layer Normalization Kernel
// Normalizes input tensor along the channel dimension (shape[0] in web-rwkv convention)
// Uses FP32 accumulation for numerical stability (Welford's algorithm)
// Input shape: [C, N, 1, 1] - normalizes each of N vectors of length C
// Weight/bias shape: [C, 1, 1, 1]
// ============================================================================

// Block size for layer norm - must be a power of 2 for efficient reduction
#define LAYERNORM_BLOCK_SIZE 256

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // First, reduce within each warp
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // Only the first warp does the final reduction
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// Layer normalization kernel
// Each block handles one vector of length C
// Input: [C, N, 1, 1] - flattened as C*N elements with C being contiguous
// For vector i (0 <= i < N), elements are at indices [i*C, (i+1)*C)
//
// Wait, actually in web-rwkv convention, shape[0] is fastest axis.
// So for shape [C, N, 1, 1], the memory layout is:
// element(c, n) is at index c + n*C
// To normalize vector n, we need elements at indices n*C, n*C+1, ..., n*C+C-1
__global__ void kernel_layer_norm_f32(
    const float* __restrict__ input,      // [C, N, 1, 1] - C is fastest axis
    const float* __restrict__ weight,     // [C, 1, 1, 1]
    const float* __restrict__ bias,       // [C, 1, 1, 1]
    float* __restrict__ output,           // [C, N, 1, 1]
    int C,                                 // Channel dimension (normalize over this)
    int N,                                 // Number of vectors to normalize
    float eps                              // Epsilon for numerical stability
) {
    // Each block handles one vector
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Shared memory for block reduction
    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float mean_shared;
    __shared__ float rstd_shared;  // 1 / sqrt(variance + eps)

    // Base offset for this vector in memory
    int base_offset = vec_idx * C;

    // Step 1: Compute mean using Welford's algorithm for numerical stability
    // Each thread handles multiple elements in a strided pattern
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        local_sum += input[base_offset + c];
    }

    // Reduce to get total sum
    float total_sum = block_reduce_sum(local_sum, shared_mem);

    // Thread 0 computes and broadcasts mean
    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)C;
    }
    __syncthreads();

    float mean = mean_shared;

    // Step 2: Compute variance
    float local_var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = input[base_offset + c];
        float diff = x - mean;
        local_var_sum += diff * diff;
    }

    // Reduce to get total variance sum
    float total_var_sum = block_reduce_sum(local_var_sum, shared_mem);

    // Thread 0 computes and broadcasts reciprocal std
    if (threadIdx.x == 0) {
        float variance = total_var_sum / (float)C;
        rstd_shared = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rstd = rstd_shared;

    // Step 3: Normalize and apply weight/bias
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = input[base_offset + c];
        float normalized = (x - mean) * rstd;
        output[base_offset + c] = normalized * weight[c] + bias[c];
    }
}

hipError_t launch_layer_norm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C,          // Channel dimension (normalize over this)
    int N,          // Number of vectors
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0) {
        return hipSuccess;
    }

    // Each block handles one vector
    int block_size = LAYERNORM_BLOCK_SIZE;
    int grid_size = N;

    hipLaunchKernelGGL(kernel_layer_norm_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, weight, bias, output, C, N, eps);

    return hipGetLastError();
}

// ============================================================================
// Group Normalization Kernel
// Normalizes input tensor per group of channels
// Input shape: [C, N, 1, 1] - C is channel dimension (fastest axis)
// With G groups, each group has C/G channels
// For each of the N vectors, normalize each group independently
// ============================================================================

// Group normalization kernel
// Each block handles one (vector, group) pair
// Grid: (num_groups, N)
__global__ void kernel_group_norm_f32(
    const float* __restrict__ input,      // [C, N, 1, 1]
    const float* __restrict__ weight,     // [C, 1, 1, 1]
    const float* __restrict__ bias,       // [C, 1, 1, 1]
    float* __restrict__ output,           // [C, N, 1, 1]
    int C,                                 // Total channels
    int N,                                 // Number of vectors
    int G,                                 // Number of groups
    float eps                              // Epsilon for numerical stability
) {
    int group_idx = blockIdx.x;
    int vec_idx = blockIdx.y;

    if (group_idx >= G || vec_idx >= N) return;

    int channels_per_group = C / G;

    // Shared memory for block reduction
    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float mean_shared;
    __shared__ float rstd_shared;

    // Base offset for this vector in memory
    int vec_offset = vec_idx * C;
    // Start channel for this group
    int group_start = group_idx * channels_per_group;

    // Step 1: Compute mean for this group
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        local_sum += input[vec_offset + c];
    }

    float total_sum = block_reduce_sum(local_sum, shared_mem);

    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)channels_per_group;
    }
    __syncthreads();

    float mean = mean_shared;

    // Step 2: Compute variance for this group
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        float x = input[vec_offset + c];
        float diff = x - mean;
        local_var_sum += diff * diff;
    }

    float total_var_sum = block_reduce_sum(local_var_sum, shared_mem);

    if (threadIdx.x == 0) {
        float variance = total_var_sum / (float)channels_per_group;
        rstd_shared = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rstd = rstd_shared;

    // Step 3: Normalize and apply per-channel weight/bias
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        float x = input[vec_offset + c];
        float normalized = (x - mean) * rstd;
        output[vec_offset + c] = normalized * weight[c] + bias[c];
    }
}

hipError_t launch_group_norm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C,          // Total channels
    int N,          // Number of vectors
    int G,          // Number of groups
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0 || G <= 0) {
        return hipSuccess;
    }

    // Each block handles one (group, vector) pair
    int block_size = LAYERNORM_BLOCK_SIZE;
    dim3 grid_size(G, N);

    hipLaunchKernelGGL(kernel_group_norm_f32, grid_size, dim3(block_size), 0, stream,
                       input, weight, bias, output, C, N, G, eps);

    return hipGetLastError();
}

// ============================================================================
// L2 Normalization Kernel (Key Normalization)
// Normalizes vectors to unit L2 norm, per head
// Used for key normalization in RWKV7
// ============================================================================

__global__ void kernel_l2_norm_f32(
    const float* __restrict__ input,      // [C, N, 1, 1] where C = H * head_size
    float* __restrict__ output,           // [C, N, 1, 1]
    int C,                                 // Total channels
    int N,                                 // Number of vectors
    int head_size,                         // Size of each head (normalize over this)
    float eps                              // Epsilon for numerical stability
) {
    int H = C / head_size;  // Number of heads
    int head_idx = blockIdx.x;
    int vec_idx = blockIdx.y;

    if (head_idx >= H || vec_idx >= N) return;

    // Shared memory for block reduction
    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float rnorm_shared;  // 1 / sqrt(sum_sq + eps)

    // Base offset for this vector in memory
    int vec_offset = vec_idx * C;
    // Start channel for this head
    int head_start = head_idx * head_size;

    // Step 1: Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        int c = head_start + i;
        float x = input[vec_offset + c];
        local_sum_sq += x * x;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_mem);

    if (threadIdx.x == 0) {
        rnorm_shared = rsqrtf(total_sum_sq + eps);
    }
    __syncthreads();

    float rnorm = rnorm_shared;

    // Step 2: Normalize
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        int c = head_start + i;
        output[vec_offset + c] = input[vec_offset + c] * rnorm;
    }
}

hipError_t launch_l2_norm_f32(
    const float* input,
    float* output,
    int C,          // Total channels (H * head_size)
    int N,          // Number of vectors
    int head_size,  // Size per head
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0 || head_size <= 0) {
        return hipSuccess;
    }

    int H = C / head_size;
    int block_size = LAYERNORM_BLOCK_SIZE;
    dim3 grid_size(H, N);

    hipLaunchKernelGGL(kernel_l2_norm_f32, grid_size, dim3(block_size), 0, stream,
                       input, output, C, N, head_size, eps);

    return hipGetLastError();
}

// ============================================================================
// FP16 kernel variants (FP16 I/O, FP32 accumulation)
// ============================================================================

__device__ __forceinline__ float f16_to_f32(__half v) {
    return __half2float(v);
}

__device__ __forceinline__ __half f32_to_f16(float v) {
    return __float2half(v);
}

__global__ void kernel_layer_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int C,
    int N,
    float eps
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float mean_shared;
    __shared__ float rstd_shared;

    int base_offset = vec_idx * C;

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        local_sum += f16_to_f32(input[base_offset + c]);
    }

    float total_sum = block_reduce_sum(local_sum, shared_mem);
    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)C;
    }
    __syncthreads();

    float mean = mean_shared;
    float local_var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = f16_to_f32(input[base_offset + c]);
        float diff = x - mean;
        local_var_sum += diff * diff;
    }

    float total_var_sum = block_reduce_sum(local_var_sum, shared_mem);
    if (threadIdx.x == 0) {
        float variance = total_var_sum / (float)C;
        rstd_shared = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rstd = rstd_shared;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = f16_to_f32(input[base_offset + c]);
        float normalized = (x - mean) * rstd;
        float w = f16_to_f32(weight[c]);
        float b = f16_to_f32(bias[c]);
        output[base_offset + c] = f32_to_f16(normalized * w + b);
    }
}

hipError_t launch_layer_norm_f16(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int C,
    int N,
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0) {
        return hipSuccess;
    }
    int block_size = LAYERNORM_BLOCK_SIZE;
    int grid_size = N;
    hipLaunchKernelGGL(kernel_layer_norm_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, weight, bias, output, C, N, eps);
    return hipGetLastError();
}

__global__ void kernel_group_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int C,
    int N,
    int G,
    float eps
) {
    int group_idx = blockIdx.x;
    int vec_idx = blockIdx.y;
    if (group_idx >= G || vec_idx >= N) return;

    int channels_per_group = C / G;
    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float mean_shared;
    __shared__ float rstd_shared;

    int vec_offset = vec_idx * C;
    int group_start = group_idx * channels_per_group;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        local_sum += f16_to_f32(input[vec_offset + c]);
    }

    float total_sum = block_reduce_sum(local_sum, shared_mem);
    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)channels_per_group;
    }
    __syncthreads();

    float mean = mean_shared;
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        float x = f16_to_f32(input[vec_offset + c]);
        float diff = x - mean;
        local_var_sum += diff * diff;
    }

    float total_var_sum = block_reduce_sum(local_var_sum, shared_mem);
    if (threadIdx.x == 0) {
        float variance = total_var_sum / (float)channels_per_group;
        rstd_shared = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rstd = rstd_shared;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int c = group_start + i;
        float x = f16_to_f32(input[vec_offset + c]);
        float normalized = (x - mean) * rstd;
        float w = f16_to_f32(weight[c]);
        float b = f16_to_f32(bias[c]);
        output[vec_offset + c] = f32_to_f16(normalized * w + b);
    }
}

hipError_t launch_group_norm_f16(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int C,
    int N,
    int G,
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0 || G <= 0) {
        return hipSuccess;
    }
    int block_size = LAYERNORM_BLOCK_SIZE;
    dim3 grid_size(G, N);
    hipLaunchKernelGGL(kernel_group_norm_f16, grid_size, dim3(block_size), 0, stream,
                       input, weight, bias, output, C, N, G, eps);
    return hipGetLastError();
}

__global__ void kernel_l2_norm_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int C,
    int N,
    int head_size,
    float eps
) {
    int H = C / head_size;
    int head_idx = blockIdx.x;
    int vec_idx = blockIdx.y;
    if (head_idx >= H || vec_idx >= N) return;

    __shared__ float shared_mem[LAYERNORM_BLOCK_SIZE / 32];
    __shared__ float rnorm_shared;

    int vec_offset = vec_idx * C;
    int head_start = head_idx * head_size;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        int c = head_start + i;
        float x = f16_to_f32(input[vec_offset + c]);
        local_sum_sq += x * x;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_mem);
    if (threadIdx.x == 0) {
        rnorm_shared = rsqrtf(total_sum_sq + eps);
    }
    __syncthreads();

    float rnorm = rnorm_shared;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        int c = head_start + i;
        float x = f16_to_f32(input[vec_offset + c]);
        output[vec_offset + c] = f32_to_f16(x * rnorm);
    }
}

hipError_t launch_l2_norm_f16(
    const __half* input,
    __half* output,
    int C,
    int N,
    int head_size,
    float eps,
    hipStream_t stream
) {
    if (C <= 0 || N <= 0 || head_size <= 0) {
        return hipSuccess;
    }
    int H = C / head_size;
    int block_size = LAYERNORM_BLOCK_SIZE;
    dim3 grid_size(H, N);
    hipLaunchKernelGGL(kernel_l2_norm_f16, grid_size, dim3(block_size), 0, stream,
                       input, output, C, N, head_size, eps);
    return hipGetLastError();
}

} // extern "C"
