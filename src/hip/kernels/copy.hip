// HIP kernels for RWKV inference backend
#include <hip/hip_runtime.h>

extern "C" {

__global__ void kernel_copy_f32(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Launch wrapper callable from Rust via FFI
hipError_t launch_copy_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_copy_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

// ============================================================================
// Decay Exponential Kernel: out = exp(-exp(x))
// Used for RWKV7 time decay computation
// ============================================================================

__global__ void kernel_decay_exp_f32(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Clamp input to avoid overflow in exp(x)
        // exp(88) ≈ 1.6e38, close to f32 max
        // For very positive x, exp(-exp(x)) → 0
        // For very negative x, exp(-exp(x)) → 1
        if (x > 80.0f) {
            output[idx] = 0.0f;
        } else if (x < -80.0f) {
            output[idx] = 1.0f;
        } else {
            output[idx] = __expf(-__expf(x));
        }
    }
}

hipError_t launch_decay_exp_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_decay_exp_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

// ============================================================================
// Lerp Kernel: out = a + t * (b - a) (linear interpolation)
// Used for RWKV7 mixing operations
// ============================================================================

__global__ void kernel_lerp_f32(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 const float* __restrict__ t,
                                 float* __restrict__ output,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = a[idx];
        float b_val = b[idx];
        float t_val = t[idx];
        // lerp: a + t * (b - a) = a * (1 - t) + b * t
        output[idx] = a_val + t_val * (b_val - a_val);
    }
}

hipError_t launch_lerp_f32(const float* a, const float* b, const float* t,
                           float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_lerp_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       a, b, t, output, n);

    return hipGetLastError();
}

// ============================================================================
// Sigmoid Kernel: out = 1 / (1 + exp(-x))
// Standard sigmoid activation function
// ============================================================================

__global__ void kernel_sigmoid_f32(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Numerically stable sigmoid:
        // For large positive x: 1 / (1 + exp(-x)) ≈ 1
        // For large negative x: 1 / (1 + exp(-x)) ≈ 0
        // Using __expf for fast math on GPU
        output[idx] = 1.0f / (1.0f + __expf(-x));
    }
}

hipError_t launch_sigmoid_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_sigmoid_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

// ============================================================================
// Squared ReLU Kernel: out = max(0, x)^2
// Used in RWKV7 channel mixing
// ============================================================================

__global__ void kernel_squared_relu_f32(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float relu = fmaxf(0.0f, x);
        output[idx] = relu * relu;
    }
}

hipError_t launch_squared_relu_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_squared_relu_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

// ============================================================================
// Softplus Decay Kernel: out = log(sigmoid(x)) - 0.5 = -softplus(-x) - 0.5
// Used for RWKV7 time decay computation
// ============================================================================

__global__ void kernel_softplus_decay_f32(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // log(sigmoid(x)) - 0.5 = -log(1 + exp(-x)) - 0.5
        // Numerically stable: for large positive x, exp(-x) -> 0, so result -> -0.5
        //                     for large negative x, result -> x - 0.5
        float log_sigmoid;
        if (x > 20.0f) {
            // For large positive x, log(sigmoid(x)) ≈ 0
            log_sigmoid = 0.0f;
        } else if (x < -20.0f) {
            // For large negative x, log(sigmoid(x)) ≈ x
            log_sigmoid = x;
        } else {
            log_sigmoid = -__logf(1.0f + __expf(-x));
        }
        output[idx] = log_sigmoid - 0.5f;
    }
}

hipError_t launch_softplus_decay_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_softplus_decay_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

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
// Tanh Activation Kernel
// Standard tanh activation: out = tanh(x)
// ============================================================================

__global__ void kernel_tanh_f32(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

hipError_t launch_tanh_f32(const float* input, float* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_tanh_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);

    return hipGetLastError();
}

// ============================================================================
// Token Shift Kernel
// Implements token shift/time mixing for RWKV
// Formula: output[t] = x[t] + mix * (prev[t] - x[t])
// Where prev[0] = state_in, prev[t>0] = x[t-1]
// This blends the current token with the previous token/state
// State is updated to x[last] for the next batch
// ============================================================================

__global__ void kernel_token_shift_f32(
    const float* __restrict__ x,          // [C, T, 1, 1] - input tensor
    const float* __restrict__ state_in,   // [C, 1, 1, 1] - previous state
    const float* __restrict__ mix,        // [C, 1, 1, 1] - mixing factor
    float* __restrict__ output,           // [C, T, 1, 1] - shifted output
    float* __restrict__ state_out,        // [C, 1, 1, 1] - new state (= x[:, T-1])
    int C,                                 // Channel dimension
    int T                                  // Number of tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float mix_val = mix[idx];
    float prev = state_in[idx];

    // Process each token in sequence
    for (int t = 0; t < T; t++) {
        int offset = t * C + idx;  // x is [C, T, 1, 1], so x[c, t] is at t*C + c
        float curr = x[offset];

        // Token shift: blend current with previous
        // output = curr + mix * (prev - curr) = lerp(curr, prev, mix)
        output[offset] = curr + mix_val * (prev - curr);

        // Current becomes previous for next iteration
        prev = curr;
    }

    // State out is the last token
    state_out[idx] = prev;
}

hipError_t launch_token_shift_f32(
    const float* x,
    const float* state_in,
    const float* mix,
    float* output,
    float* state_out,
    int C,
    int T,
    hipStream_t stream
) {
    if (C <= 0 || T <= 0) {
        return hipSuccess;
    }

    int block_size = 256;
    int grid_size = (C + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_token_shift_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       x, state_in, mix, output, state_out, C, T);

    return hipGetLastError();
}

// ============================================================================
// Channel-Mix State Kernel
// Same as token shift but with batch dimension
// Input: [C, T, B, 1], state: [C, B, 1, 1], x_k: [C, 1, 1, 1]
// Formula: output[t] = x[t] + x_k * (prev[t] - x[t])
// Where prev[0] = state_in, prev[t>0] = x[t-1]
// State is updated to x[T-1] for each batch
// ============================================================================

__global__ void kernel_channel_mix_state_f32(
    const float* __restrict__ x,          // [C, T, B, 1] - input tensor
    const float* __restrict__ state_in,   // [C, B, 1, 1] - previous state per batch
    const float* __restrict__ x_k,        // [C, 1, 1, 1] - mixing factor
    float* __restrict__ output,           // [C, T, B, 1] - output
    float* __restrict__ state_out,        // [C, B, 1, 1] - new state per batch
    int C,                                 // Channel dimension
    int T,                                 // Number of tokens
    int B                                  // Batch size
) {
    // Each thread handles one (channel, batch) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * B;
    if (idx >= total) return;

    int c = idx % C;        // Channel index
    int b = idx / C;        // Batch index

    float mix_val = x_k[c];
    // state_in is [C, B, 1, 1] so element (c, b) is at b*C + c
    float prev = state_in[b * C + c];

    // Process each token in sequence for this (channel, batch)
    for (int t = 0; t < T; t++) {
        // x is [C, T, B, 1] so element (c, t, b) is at b*T*C + t*C + c
        int offset = b * T * C + t * C + c;
        float curr = x[offset];

        // Token shift: blend current with previous
        output[offset] = curr + mix_val * (prev - curr);

        prev = curr;
    }

    // State out is the last token for this batch
    state_out[b * C + c] = prev;
}

hipError_t launch_channel_mix_state_f32(
    const float* x,
    const float* state_in,
    const float* x_k,
    float* output,
    float* state_out,
    int C,
    int T,
    int B,
    hipStream_t stream
) {
    if (C <= 0 || T <= 0 || B <= 0) {
        return hipSuccess;
    }

    int total = C * B;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_channel_mix_state_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       x, state_in, x_k, output, state_out, C, T, B);

    return hipGetLastError();
}

// Get device count for initialization check
hipError_t hip_get_device_count(int* count) {
    return hipGetDeviceCount(count);
}

// Get device properties (legacy - struct layout may not match Rust)
hipError_t hip_get_device_properties(hipDeviceProp_t* props, int device_id) {
    return hipGetDeviceProperties(props, device_id);
}

// Safe property accessors that extract individual fields to avoid ABI issues
hipError_t hip_get_device_name(int device_id, char* name, int max_len) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    int i;
    for (i = 0; i < max_len - 1 && props.name[i] != '\0'; i++) {
        name[i] = props.name[i];
    }
    name[i] = '\0';
    return hipSuccess;
}

hipError_t hip_get_device_gcn_arch_name(int device_id, char* name, int max_len) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    int i;
    for (i = 0; i < max_len - 1 && props.gcnArchName[i] != '\0'; i++) {
        name[i] = props.gcnArchName[i];
    }
    name[i] = '\0';
    return hipSuccess;
}

hipError_t hip_get_device_total_memory(int device_id, size_t* total_mem) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *total_mem = props.totalGlobalMem;
    return hipSuccess;
}

hipError_t hip_get_device_mp_count(int device_id, int* count) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *count = props.multiProcessorCount;
    return hipSuccess;
}

hipError_t hip_get_device_warp_size(int device_id, int* warp_size) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *warp_size = props.warpSize;
    return hipSuccess;
}

hipError_t hip_get_device_compute_capability(int device_id, int* major, int* minor) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *major = props.major;
    *minor = props.minor;
    return hipSuccess;
}

hipError_t hip_is_device_integrated(int device_id, int* integrated) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *integrated = props.integrated;
    return hipSuccess;
}

hipError_t hip_supports_cooperative_launch(int device_id, int* supported) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) return err;
    *supported = props.cooperativeLaunch;
    return hipSuccess;
}

// Set active device
hipError_t hip_set_device(int device_id) {
    return hipSetDevice(device_id);
}

// Get current device
hipError_t hip_get_device(int* device_id) {
    return hipGetDevice(device_id);
}

// Memory allocation
hipError_t hip_malloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

hipError_t hip_malloc_managed(void** ptr, size_t size) {
    return hipMallocManaged(ptr, size);
}

hipError_t hip_free(void* ptr) {
    return hipFree(ptr);
}

hipError_t hip_memset(void* ptr, int value, size_t size) {
    return hipMemset(ptr, value, size);
}

// Memory copy
hipError_t hip_memcpy_h2d(void* dst, const void* src, size_t size, hipStream_t stream) {
    return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
}

hipError_t hip_memcpy_d2h(void* dst, const void* src, size_t size, hipStream_t stream) {
    return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
}

// Stream management
hipError_t hip_stream_create(hipStream_t* stream) {
    return hipStreamCreate(stream);
}

hipError_t hip_stream_destroy(hipStream_t stream) {
    return hipStreamDestroy(stream);
}

hipError_t hip_stream_synchronize(hipStream_t stream) {
    return hipStreamSynchronize(stream);
}

hipError_t hip_device_synchronize() {
    return hipDeviceSynchronize();
}

// Error string
const char* hip_get_error_string(hipError_t error) {
    return hipGetErrorString(error);
}

} // extern "C"
