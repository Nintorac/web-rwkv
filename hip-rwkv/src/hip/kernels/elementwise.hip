// HIP kernels for RWKV inference backend
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>


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
            output[idx] = expf(-expf(x));
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
        // Using expf (not __expf) for full FP32 precision
        output[idx] = 1.0f / (1.0f + expf(-x));
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
            log_sigmoid = -logf(1.0f + expf(-x));
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
// Elementwise operations for GPU-native forward pass
// ============================================================================

// Elementwise add: output = a + b
__global__ void kernel_add_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

hipError_t launch_add_f32(const float* a, const float* b, float* output, int n, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_add_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       a, b, output, n);
    return hipGetLastError();
}

// Elementwise multiply: output = a * b
__global__ void kernel_mul_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

hipError_t launch_mul_f32(const float* a, const float* b, float* output, int n, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_mul_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       a, b, output, n);
    return hipGetLastError();
}

// Negate: output = -input
__global__ void kernel_negate_f32(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -input[idx];
    }
}

hipError_t launch_negate_f32(const float* input, float* output, int n, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_negate_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

// Exp: output = exp(input)
__global__ void kernel_exp_f32(const float* __restrict__ input,
                               float* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

hipError_t launch_exp_f32(const float* input, float* output, int n, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_exp_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

// Broadcast add: output[i] = input[i] + bias[i % bias_len]
// Used for adding per-channel biases to batched data
__global__ void kernel_broadcast_add_f32(const float* __restrict__ input,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int n, int bias_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + bias[idx % bias_len];
    }
}

hipError_t launch_broadcast_add_f32(const float* input, const float* bias, float* output,
                                    int n, int bias_len, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_broadcast_add_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, bias, output, n, bias_len);
    return hipGetLastError();
}

// Broadcast multiply: output[i] = input[i] * scale[i % scale_len]
// Used for per-channel scaling (e.g., k * k_k in RWKV7)
__global__ void kernel_broadcast_mul_f32(const float* __restrict__ input,
                                         const float* __restrict__ scale,
                                         float* __restrict__ output,
                                         int n, int scale_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale[idx % scale_len];
    }
}

hipError_t launch_broadcast_mul_f32(const float* input, const float* scale, float* output,
                                    int n, int scale_len, hipStream_t stream) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_broadcast_mul_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, scale, output, n, scale_len);
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

__global__ void kernel_copy_f16(const __half* __restrict__ input,
                                __half* __restrict__ output,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

hipError_t launch_copy_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_copy_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

__global__ void kernel_decay_exp_f16(const __half* __restrict__ input,
                                     __half* __restrict__ output,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(input[idx]);
        float out;
        if (x > 80.0f) {
            out = 0.0f;
        } else if (x < -80.0f) {
            out = 1.0f;
        } else {
            out = expf(-expf(x));
        }
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_decay_exp_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_decay_exp_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

__global__ void kernel_lerp_f16(const __half* __restrict__ a,
                                const __half* __restrict__ b,
                                const __half* __restrict__ t,
                                __half* __restrict__ output,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = f16_to_f32(a[idx]);
        float b_val = f16_to_f32(b[idx]);
        float t_val = f16_to_f32(t[idx]);
        float out = a_val + t_val * (b_val - a_val);
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_lerp_f16(const __half* a, const __half* b, const __half* t,
                           __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_lerp_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       a, b, t, output, n);
    return hipGetLastError();
}

__global__ void kernel_sigmoid_f16(const __half* __restrict__ input,
                                   __half* __restrict__ output,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(input[idx]);
        float out = 1.0f / (1.0f + expf(-x));
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_sigmoid_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_sigmoid_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

__global__ void kernel_squared_relu_f16(const __half* __restrict__ input,
                                        __half* __restrict__ output,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(input[idx]);
        float relu = fmaxf(0.0f, x);
        output[idx] = f32_to_f16(relu * relu);
    }
}

hipError_t launch_squared_relu_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_squared_relu_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

__global__ void kernel_softplus_decay_f16(const __half* __restrict__ input,
                                          __half* __restrict__ output,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(input[idx]);
        float log_sigmoid;
        if (x > 20.0f) {
            log_sigmoid = 0.0f;
        } else if (x < -20.0f) {
            log_sigmoid = x;
        } else {
            log_sigmoid = -logf(1.0f + expf(-x));
        }
        output[idx] = f32_to_f16(log_sigmoid - 0.5f);
    }
}

hipError_t launch_softplus_decay_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_softplus_decay_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

__global__ void kernel_tanh_f16(const __half* __restrict__ input,
                                __half* __restrict__ output,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(input[idx]);
        output[idx] = f32_to_f16(tanhf(x));
    }
}

hipError_t launch_tanh_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(kernel_tanh_f16, dim3(grid_size), dim3(block_size), 0, stream,
                       input, output, n);
    return hipGetLastError();
}

#define BLOCK_SIZE 256

__global__ void kernel_add_f16(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float out = f16_to_f32(a[idx]) + f16_to_f32(b[idx]);
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_add_f16(const __half* a, const __half* b, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_add_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, a, b, output, n);
    return hipGetLastError();
}

__global__ void kernel_mul_f16(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float out = f16_to_f32(a[idx]) * f16_to_f32(b[idx]);
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_mul_f16(const __half* a, const __half* b, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_mul_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, a, b, output, n);
    return hipGetLastError();
}

__global__ void kernel_negate_f16(const __half* __restrict__ input,
                                  __half* __restrict__ output,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = f32_to_f16(-f16_to_f32(input[idx]));
    }
}

hipError_t launch_negate_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_negate_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, input, output, n);
    return hipGetLastError();
}

__global__ void kernel_exp_f16(const __half* __restrict__ input,
                               __half* __restrict__ output,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = f32_to_f16(expf(f16_to_f32(input[idx])));
    }
}

hipError_t launch_exp_f16(const __half* input, __half* output, int n, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_exp_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, input, output, n);
    return hipGetLastError();
}

__global__ void kernel_broadcast_add_f16(const __half* __restrict__ input,
                                         const __half* __restrict__ bias,
                                         __half* __restrict__ output,
                                         int n, int bias_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int c_idx = idx % bias_len;
        float out = f16_to_f32(input[idx]) + f16_to_f32(bias[c_idx]);
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_broadcast_add_f16(const __half* input, const __half* bias, __half* output,
                                    int n, int bias_len, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_broadcast_add_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, bias, output, n, bias_len);
    return hipGetLastError();
}

__global__ void kernel_broadcast_mul_f16(const __half* __restrict__ input,
                                         const __half* __restrict__ scale,
                                         __half* __restrict__ output,
                                         int n, int scale_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int c_idx = idx % scale_len;
        float out = f16_to_f32(input[idx]) * f16_to_f32(scale[c_idx]);
        output[idx] = f32_to_f16(out);
    }
}

hipError_t launch_broadcast_mul_f16(const __half* input, const __half* scale, __half* output,
                                    int n, int scale_len, hipStream_t stream) {
    if (n <= 0) {
        return hipSuccess;
    }
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_broadcast_mul_f16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       input, scale, output, n, scale_len);
    return hipGetLastError();
}

} // extern "C"
