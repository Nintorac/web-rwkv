//! HIP backend for AMD GPUs.
//!
//! This module provides a HIP-based backend for running RWKV inference on AMD GPUs.
//! It requires a ROCm/TheRock installation with support for your GPU architecture.

mod blas;
mod blaslt;
mod buffer;
mod device;
mod ffi;
mod kernels;
mod model;
mod pinned;
mod runtime;
mod scratch;
mod tensor;

#[cfg(feature = "hip-probes")]
pub mod probe;

// Re-export runtime types
pub use runtime::{softmax_one_cpu, HipRuntime};

// Re-export probe types when feature is enabled
#[cfg(feature = "hip-probes")]
pub use probe::{
    HipHook, HipProbeBuilder, HipProbeMap, HipProbeMapRef, ProbeContext, MAX_SHAPE_DIMS,
};

// Re-export FFI types and error handling
pub use ffi::{
    device_supports_cooperative_launch, error_string, get_device, get_device_compute_capability,
    get_device_count, get_device_mp_count, get_device_name, get_device_properties,
    get_device_total_memory, get_device_warp_size, get_gcn_arch_name, is_device_integrated,
    set_device, HipDeviceArch, HipDeviceProp, HipError, HipErrorKind, HipEvent, HipStream,
    HipblasLtHandle, HipblasStatus, Result, RocblasHandle, RocblasStatus,
    HIPBLAS_STATUS_ALLOC_FAILED, HIPBLAS_STATUS_NOT_SUPPORTED, HIPBLAS_STATUS_SUCCESS,
    HIP_ERROR_NOT_READY, HIP_HOST_MALLOC_DEFAULT, HIP_HOST_MALLOC_MAPPED, HIP_HOST_MALLOC_PORTABLE,
    HIP_SUCCESS, ROCBLAS_STATUS_SUCCESS,
};

// Re-export device types
pub use device::{device_synchronize, Event, HipContext, Stream};

// Re-export buffer types
pub use buffer::{DeviceBuffer, MemoryType};

// Re-export pinned memory types
pub use pinned::PinnedBuffer;

// Re-export tensor types
pub use tensor::{TensorHip, TensorShape, TensorView};

// Re-export scratch buffer types
pub use scratch::{DecodeConfig, DecodeScratch, HipRuntimeConfig, HipScratch, LoraDims, PrefillConfig, PrefillScratch};

// Re-export kernel functions
pub use kernels::{
    // Elementwise operations for GPU-native forward
    add_f32,
    broadcast_add_f32,
    broadcast_mul_f32,
    channel_mix_state_f32,
    control_k_f32,
    copy_f32,
    // GPU-to-GPU copy
    copy_tensor_f32,
    decay_exp_f32,
    exp_f32,
    group_norm_f32,
    hip_channel_mix_state,
    hip_control_k,
    hip_copy_kernel,
    hip_decay_exp,
    hip_group_norm,
    hip_l2_norm,
    hip_layer_norm,
    hip_lerp,
    hip_sigmoid,
    hip_softplus_decay,
    hip_squared_relu,
    hip_tanh,
    hip_token_shift,
    hip_wkv7_gemv,
    hip_wkv_bonus,
    l2_norm_f32,
    layer_norm_f32,
    lerp_f32,
    mul_f32,
    negate_f32,
    sigmoid_f32,
    softplus_decay_f32,
    squared_relu_f32,
    tanh_f32,
    token_shift_f32,
    // WKV7 rocBLAS GEMV implementation
    wkv7_gemv_f32,
    wkv7_wave_reduce,
    wkv7_fused_t1,
    wkv_bonus_f32,
};

// Re-export BLAS functions and context
pub use blas::{
    hgemm_f16, hip_hgemm, hip_sgemm, rocblas_create, rocblas_destroy, rocblas_set_stream,
    sgemm_f32, HipBlasContext,
};

// Re-export hipBLASLt functions and context
pub use blaslt::{
    hip_hipblaslt_hgemm, hip_hipblaslt_hgemm_f32_out, hipblaslt_create, hipblaslt_destroy,
    HipBlasLtContext,
};

// Re-export model types
pub use model::{
    AttentionHip, EmbedHip, FfnHip, HeadHip, HipDecode, HipPrefill, HipState, LayerHip,
    LayerNormHip, ModelLoadError, Rwkv7Hip, Rwkv7Model, Rwkv7ModelInfo, StateLayout,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_feature_compiles() {
        // This test exists - if it compiles with --features hip,
        // the build system works
    }

    #[test]
    fn test_hip_device_detection() {
        let count = get_device_count().expect("Failed to get device count");
        assert!(count > 0, "No HIP devices found");

        let name = get_device_name(0).expect("Failed to get device name");
        println!("Device 0: {}", name);

        let arch = get_gcn_arch_name(0).expect("Failed to get arch name");
        println!("Architecture: {}", arch);
    }

    #[test]
    fn test_simple_kernel_runs() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let output = hip_copy_kernel(&input).expect("Copy kernel failed");
        assert_eq!(input, output, "Copy kernel output mismatch");
    }
}
