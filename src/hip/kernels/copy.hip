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
