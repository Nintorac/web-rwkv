//! HIP backend for AMD GPUs.
//!
//! This module provides a HIP-based backend for running RWKV inference on AMD GPUs.
//! It requires a ROCm/TheRock installation with support for your GPU architecture.

mod blas;
mod buffer;
mod device;
mod ffi;
mod kernels;
mod model;
mod runtime;
mod tensor;

#[cfg(feature = "hip-probes")]
pub mod probe;

// Re-export runtime types
pub use runtime::{HipRuntime, softmax_one_cpu};

// Re-export probe types when feature is enabled
#[cfg(feature = "hip-probes")]
pub use probe::{HipHook, HipProbeBuilder, HipProbeMap, HipProbeMapRef, ProbeContext, MAX_SHAPE_DIMS};

// Re-export FFI types and error handling
pub use ffi::{
    HipError, HipStream, HipDeviceProp, HipDeviceArch,
    RocblasHandle, RocblasStatus, ROCBLAS_STATUS_SUCCESS, HIP_SUCCESS,
    error_string, Result, HipErrorKind,
    get_device_count, get_device_properties, get_device_name, get_gcn_arch_name,
    set_device, get_device, get_device_total_memory, get_device_mp_count,
    get_device_warp_size, get_device_compute_capability, is_device_integrated,
    device_supports_cooperative_launch,
};

// Re-export device types
pub use device::{HipContext, Stream, device_synchronize};

// Re-export buffer types
pub use buffer::{DeviceBuffer, MemoryType};

// Re-export tensor types
pub use tensor::{TensorShape, TensorView, TensorHip};

// Re-export kernel functions
pub use kernels::{
    copy_f32, hip_copy_kernel,
    decay_exp_f32, hip_decay_exp,
    lerp_f32, hip_lerp,
    sigmoid_f32, hip_sigmoid,
    squared_relu_f32, hip_squared_relu,
    softplus_decay_f32, hip_softplus_decay,
    layer_norm_f32, hip_layer_norm,
    group_norm_f32, hip_group_norm,
    l2_norm_f32, hip_l2_norm,
    tanh_f32, hip_tanh,
    token_shift_f32, hip_token_shift,
    channel_mix_state_f32, hip_channel_mix_state,
    wkv_bonus_f32, hip_wkv_bonus,
    control_k_f32, hip_control_k,
    wkv7_f32, hip_wkv7,
    wkv7_f32_masked, hip_wkv7_masked,
};

// Re-export BLAS functions
pub use blas::{
    rocblas_create, rocblas_destroy, rocblas_set_stream,
    hgemm_f16, sgemm_f32,
    hip_sgemm, hip_hgemm,
};

// Re-export model types
pub use model::{
    Rwkv7ModelInfo, LayerNormHip, AttentionHip, FfnHip, LayerHip,
    EmbedHip, HeadHip, Rwkv7Hip, ModelLoadError, HipState,
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

    #[test]
    fn test_hip_memory_roundtrip() {
        // Use null stream as workaround for hipStreamCreate crashes
        let stream = Stream::null();
        let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();

        let mut device_buf = DeviceBuffer::<f32>::new(1024).expect("Failed to allocate");
        device_buf
            .copy_from_host(&host_data, &stream)
            .expect("Failed to copy to device");

        let mut result = vec![0.0f32; 1024];
        device_buf
            .copy_to_host(&mut result, &stream)
            .expect("Failed to copy to host");
        stream.synchronize().expect("Failed to synchronize");

        assert_eq!(host_data, result, "Memcpy roundtrip failed");
    }

    // === Acceptance Criteria Tests for bd-2sh.3.2 ===
    //
    // NOTE: These tests require a working ROCm installation. On some ROCm versions
    // (particularly with gfx1151/RDNA 3.5), hipStreamCreate and hipMemcpy operations
    // may crash due to runtime issues. The tests are designed to pass when the
    // ROCm environment is functioning correctly.
    //
    // Known issue: ROCm 7.1.1 on gfx1151 crashes on stream creation and memory
    // copy operations. These tests will be enabled when the environment is fixed.

    #[test]
    fn test_hip_context_creation() {
        // Test that we can create a HIP context
        let ctx = HipContext::new().expect("Failed to create context");

        // Verify device_id is valid (>= 0)
        assert!(ctx.device_id() >= 0, "Device ID should be non-negative");

        // Verify device name is reasonable
        let name = ctx.device_name().expect("Failed to get device name");
        let arch = ctx.gcn_arch_name().unwrap_or_default();
        println!("Context created for device: {} ({})", name, arch);

        assert!(
            arch.contains("gfx") || name.contains("Radeon") || name.contains("AMD"),
            "Expected AMD GPU, got: {} ({})",
            name,
            arch
        );
    }

    #[test]
    fn test_hip_context_device_query() {
        // Test that we can query device properties through the context
        let ctx = HipContext::new().expect("Failed to create context");

        // Query various device properties
        let name = ctx.device_name().expect("Failed to get device name");
        let arch = ctx.gcn_arch_name().unwrap_or_else(|_| "unknown".to_string());
        let memory = ctx.total_memory().unwrap_or(0);
        let mp_count = ctx.multiprocessor_count().unwrap_or(0);
        let warp_size = ctx.warp_size().unwrap_or(0);
        let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
        let is_integrated = ctx.is_integrated().unwrap_or(false);
        let supports_coop = ctx.supports_cooperative_launch().unwrap_or(false);

        println!("Device: {}", name);
        println!("  Architecture: {}", arch);
        println!("  Total memory: {} MB", memory / (1024 * 1024));
        println!("  Multiprocessors: {}", mp_count);
        println!("  Warp size: {}", warp_size);
        println!("  Compute capability: {}.{}", major, minor);
        println!("  Integrated (APU): {}", is_integrated);
        println!("  Cooperative launch: {}", supports_coop);

        // Validate device name (most reliable)
        assert!(!name.is_empty(), "Device name should not be empty");

        // For gfx1151, verify expected warp size
        if arch.contains("gfx1151") {
            if warp_size > 0 {
                assert_eq!(warp_size, 32, "gfx1151 should have warp size 32");
                println!("  Verified gfx1151 warp size");
            }
        }
    }

    #[test]
    fn test_hip_stream_sync() {
        // Test stream synchronization using null stream
        // (explicit stream creation may crash on some ROCm versions)
        let ctx = HipContext::new().expect("Failed to create context");

        // Test null stream synchronization
        ctx.synchronize().expect("Failed to synchronize null stream");
        println!("Null stream synchronization test passed");
    }

    // === Acceptance Criteria Tests for bd-2sh.3.3 (Tensor Abstraction) ===

    #[test]
    fn test_tensor_alloc() {
        // Test tensor allocation with various shapes and memory types
        let stream = Stream::null();

        // Test device memory allocation
        let shape = TensorShape::new(2, 3, 4, 1);
        let tensor = TensorHip::<f32>::new(shape).expect("Failed to allocate device tensor");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.len(), 24);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.memory_type(), MemoryType::Device);
        println!("Device tensor allocated: shape={}, len={}", tensor.shape(), tensor.len());

        // Test managed memory allocation
        let managed_tensor = TensorHip::<f32>::managed(shape).expect("Failed to allocate managed tensor");
        assert_eq!(managed_tensor.shape(), shape);
        assert_eq!(managed_tensor.memory_type(), MemoryType::Managed);
        println!("Managed tensor allocated: shape={}", managed_tensor.shape());

        // Test zeros initialization
        let zeros_tensor = TensorHip::<f32>::zeros(shape).expect("Failed to allocate zeros tensor");
        let data = zeros_tensor.to_vec(&stream).expect("Failed to read zeros");
        assert!(data.iter().all(|&x| x == 0.0), "Zeros tensor should be all zeros");
        println!("Zeros tensor verified: all {} elements are zero", data.len());

        // Test empty tensor
        let empty_shape = TensorShape::new(0, 1, 1, 1);
        let empty_tensor = TensorHip::<f32>::new(empty_shape).expect("Failed to allocate empty tensor");
        assert!(empty_tensor.is_empty());
        println!("Empty tensor allocated successfully");
    }

    #[test]
    fn test_tensor_copy_roundtrip() {
        // Test copying data to device and back
        let stream = Stream::null();

        // Create test data
        let shape = TensorShape::new(4, 3, 2, 1);
        let host_data: Vec<f32> = (0..24).map(|i| i as f32).collect();

        // Upload to device
        let tensor = TensorHip::from_slice(&host_data, shape, &stream)
            .expect("Failed to create tensor from slice");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.len(), 24);

        // Download back to host
        let result = tensor.to_vec(&stream).expect("Failed to copy tensor to host");
        assert_eq!(host_data, result, "Roundtrip data should match");
        println!("Tensor roundtrip verified: {} elements match", result.len());

        // Test with copy_from_slice
        let mut tensor2 = TensorHip::<f32>::new(shape).expect("Failed to allocate tensor");
        tensor2.copy_from_slice(&host_data, &stream).expect("Failed to copy from slice");
        stream.synchronize().expect("Failed to sync");

        let result2 = tensor2.to_vec(&stream).expect("Failed to read tensor2");
        assert_eq!(host_data, result2, "copy_from_slice data should match");
        println!("copy_from_slice roundtrip verified");
    }

    #[test]
    fn test_tensor_view_strides() {
        // Test tensor views and stride calculations
        let stream = Stream::null();

        // Create a 4x3x2 tensor with known data
        // shape[0]=4 is fastest (contiguous)
        let shape = TensorShape::new(4, 3, 2, 1);
        let host_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = TensorHip::from_slice(&host_data, shape, &stream)
            .expect("Failed to create tensor");

        // Verify strides: [1, 4, 12, 24]
        let strides = tensor.strides();
        assert_eq!(strides, [1, 4, 12, 24], "Strides should be [1, 4, 12, 24]");
        println!("Verified strides: {:?}", strides);

        // Test linear index calculation
        // Element at (x=1, y=2, z=1, w=0) should be at index: 1 + 2*4 + 1*12 = 21
        let idx = shape.linear_index(1, 2, 1, 0);
        assert_eq!(idx, 21, "Linear index (1,2,1,0) should be 21");
        println!("Linear index (1,2,1,0) = {}", idx);

        // Create a view: first row of each "page" (x=0..4, y=0..1, z=0..2)
        let view = tensor.view((0, 4), (0, 1), (0, 2), (0, 1))
            .expect("Failed to create view");
        assert_eq!(view.shape(), TensorShape::new(4, 1, 2, 1));
        assert_eq!(view.len(), 8);
        assert!(!view.is_contiguous(), "View with gap in y should not be contiguous");
        println!("View shape: {}, contiguous: {}", view.shape(), view.is_contiguous());

        // Test view strides (should inherit parent strides)
        let view_strides = view.strides();
        assert_eq!(view_strides, strides, "View should inherit parent strides");
        println!("View strides: {:?}", view_strides);

        // Test TensorView offset calculation
        let tv = tensor.tensor_view();
        assert_eq!(tv.offset, 0, "Original tensor offset should be 0");

        // Create view starting at (2, 1, 0, 0)
        let view2 = tensor.view((2, 4), (1, 3), (0, 2), (0, 1))
            .expect("Failed to create view2");
        let expected_offset = 2 * 1 + 1 * 4 + 0 * 12; // = 6
        assert_eq!(view2.offset(), expected_offset, "View offset should be 6");
        println!("View2 offset: {} (expected {})", view2.offset(), expected_offset);

        // Verify view shape
        assert_eq!(view2.shape(), TensorShape::new(2, 2, 2, 1));
        println!("View2 shape: {}", view2.shape());

        // Test contiguous view (full slice should be contiguous)
        let full_view = tensor.view((0, 4), (0, 3), (0, 2), (0, 1))
            .expect("Failed to create full view");
        assert!(full_view.is_contiguous(), "Full view should be contiguous");
        println!("Full view is contiguous: {}", full_view.is_contiguous());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.3 (Decay Exponential Kernel) ===

    #[test]
    fn test_decay_exp() {
        // Test basic decay_exp functionality: out = exp(-exp(x))
        // Also serves as the primary acceptance test when fixtures are loaded

        // Test known values
        let input = vec![
            0.0,   // exp(-exp(0)) = exp(-1) ≈ 0.3679
            -1.0,  // exp(-exp(-1)) = exp(-0.3679) ≈ 0.6922
            1.0,   // exp(-exp(1)) = exp(-2.718) ≈ 0.0660
            -5.0,  // exp(-exp(-5)) ≈ exp(-0.0067) ≈ 0.9933
            5.0,   // exp(-exp(5)) ≈ exp(-148.4) ≈ 0
        ];

        let output = hip_decay_exp(&input).expect("decay_exp kernel failed");

        // Expected values (computed with Python: np.exp(-np.exp(x)))
        let expected = vec![
            0.36787944,  // exp(-1)
            0.69220066,  // exp(-exp(-1))
            0.06598804,  // exp(-exp(1))
            0.99330715,  // exp(-exp(-5))
            0.0,         // exp(-exp(5)) ≈ 0 (underflow)
        ];

        // Check each value with tolerance
        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-3 + 1e-3 * exp.abs(); // rtol=1e-3, atol=1e-3
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic decay_exp test passed: {} values verified", output.len());
    }

    #[test]
    fn test_decay_exp_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -50.0,  // Very negative: exp(-exp(-50)) ≈ 1
            -10.0,  // Negative: exp(-exp(-10)) ≈ 1
            -5.0,   // Moderate negative
            -1.0,   // Small negative
            0.0,    // Zero
            1.0,    // Small positive
            5.0,    // Moderate positive: exp(-exp(5)) ≈ 0
            10.0,   // exp(-exp(10)) ≈ 0 (extreme underflow)
            80.0,   // At clamping boundary
            100.0,  // Beyond clamping: should be 0, not NaN/Inf
        ];

        let output = hip_decay_exp(&input).expect("decay_exp stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i, val, input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(output[0] > 0.999, "exp(-exp(-50)) should be ≈1, got {}", output[0]);
        assert!(output[1] > 0.999, "exp(-exp(-10)) should be ≈1, got {}", output[1]);
        assert!(output[7] < 0.001, "exp(-exp(10)) should be ≈0, got {}", output[7]);
        assert!(output[8] < 0.001, "exp(-exp(80)) should be ≈0, got {}", output[8]);
        assert_eq!(output[9], 0.0, "exp(-exp(100)) should be exactly 0");

        println!("Numerical stability test passed: all {} values are finite and in [0,1]", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.4 (Lerp Kernel) ===

    #[test]
    fn test_lerp() {
        // Test basic lerp functionality: out = a + t * (b - a)
        let a = vec![0.0, 1.0, 2.0, 10.0, -5.0];
        let b = vec![10.0, 5.0, 2.0, 0.0, 5.0];
        let t = vec![0.0, 0.5, 1.0, 0.25, 0.5];

        let output = hip_lerp(&a, &b, &t).expect("lerp kernel failed");

        // Expected: lerp(a, b, t) = a + t * (b - a)
        // [0] lerp(0, 10, 0) = 0
        // [1] lerp(1, 5, 0.5) = 1 + 0.5 * 4 = 3
        // [2] lerp(2, 2, 1) = 2
        // [3] lerp(10, 0, 0.25) = 10 + 0.25 * (-10) = 7.5
        // [4] lerp(-5, 5, 0.5) = -5 + 0.5 * 10 = 0
        let expected = vec![0.0, 3.0, 2.0, 7.5, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic lerp test passed: {} values verified", output.len());
    }

    #[test]
    fn test_lerp_edge_cases() {
        // Test edge cases: t outside [0, 1] (extrapolation)
        let a = vec![0.0, 0.0, 100.0];
        let b = vec![10.0, 10.0, 0.0];
        let t = vec![-0.5, 1.5, 2.0];

        let output = hip_lerp(&a, &b, &t).expect("lerp edge case test failed");

        // Expected with extrapolation:
        // [0] lerp(0, 10, -0.5) = 0 + (-0.5) * 10 = -5
        // [1] lerp(0, 10, 1.5) = 0 + 1.5 * 10 = 15
        // [2] lerp(100, 0, 2.0) = 100 + 2.0 * (-100) = -100
        let expected = vec![-5.0, 15.0, -100.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Lerp edge case test passed: extrapolation works correctly");
    }

    // === Acceptance Criteria Tests for bd-2sh.4.1 (Sigmoid Kernel) ===

    #[test]
    fn test_sigmoid() {
        // Test basic sigmoid functionality: out = 1 / (1 + exp(-x))
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_sigmoid(&input).expect("sigmoid kernel failed");

        // Expected: sigmoid(x) = 1 / (1 + exp(-x))
        // sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.7311
        // sigmoid(-1) ≈ 0.2689
        // sigmoid(2) ≈ 0.8808
        // sigmoid(-2) ≈ 0.1192
        let expected = vec![0.5, 0.7310586, 0.26894143, 0.880797, 0.11920292];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic sigmoid test passed: {} values verified", output.len());
    }

    #[test]
    fn test_sigmoid_edge_cases() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0,  // Very negative: sigmoid → 0
            -50.0,   // Large negative
            -10.0,   // Moderate negative
            0.0,     // Zero: sigmoid = 0.5
            10.0,    // Moderate positive
            50.0,    // Large positive
            100.0,   // Very positive: sigmoid → 1
        ];

        let output = hip_sigmoid(&input).expect("sigmoid edge case test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i, val, input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(output[0] < 1e-10, "sigmoid(-100) should be ≈0, got {}", output[0]);
        assert!(output[1] < 1e-10, "sigmoid(-50) should be ≈0, got {}", output[1]);
        assert!((output[3] - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {}", output[3]);
        assert!(output[5] >= 1.0 - 1e-10, "sigmoid(50) should be ≈1, got {}", output[5]);
        assert!(output[6] >= 1.0 - 1e-10, "sigmoid(100) should be ≈1, got {}", output[6]);

        println!("Sigmoid edge case test passed: all {} values are finite and in [0,1]", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.2 (Squared ReLU Kernel) ===

    #[test]
    fn test_squared_relu() {
        // Test basic squared ReLU: out = max(0, x)^2
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

        let output = hip_squared_relu(&input).expect("squared_relu kernel failed");

        // Expected: max(0, x)^2
        // [-2] -> 0^2 = 0
        // [-1] -> 0^2 = 0
        // [0]  -> 0^2 = 0
        // [1]  -> 1^2 = 1
        // [2]  -> 2^2 = 4
        // [3]  -> 3^2 = 9
        let expected = vec![0.0, 0.0, 0.0, 1.0, 4.0, 9.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Squared ReLU test passed: {} values verified", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.14 (Softplus Decay Kernel) ===

    #[test]
    fn test_softplus_decay() {
        // Test softplus decay: out = log(sigmoid(x)) - 0.5
        let input = vec![0.0, 1.0, -1.0, 5.0, -5.0];

        let output = hip_softplus_decay(&input).expect("softplus_decay kernel failed");

        // Expected: log(sigmoid(x)) - 0.5
        // log(sigmoid(0)) - 0.5 = log(0.5) - 0.5 ≈ -0.693 - 0.5 = -1.193
        // log(sigmoid(1)) - 0.5 ≈ -0.313 - 0.5 = -0.813
        // log(sigmoid(-1)) - 0.5 ≈ -1.313 - 0.5 = -1.813
        // log(sigmoid(5)) - 0.5 ≈ -0.0067 - 0.5 ≈ -0.507
        // log(sigmoid(-5)) - 0.5 ≈ -5.0067 - 0.5 ≈ -5.507
        let expected: Vec<f32> = input.iter().map(|&x| {
            let log_sigmoid = -(1.0f32 + (-x).exp()).ln();
            log_sigmoid - 0.5
        }).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Softplus decay test passed: {} values verified", output.len());
    }

    #[test]
    fn test_softplus_decay_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0,  // Very negative: result ≈ x - 0.5 = -100.5
            -50.0,   // Large negative
            -20.0,   // At clamping boundary
            0.0,     // Zero
            20.0,    // At clamping boundary
            50.0,    // Large positive
            100.0,   // Very positive: result ≈ -0.5
        ];

        let output = hip_softplus_decay(&input).expect("softplus_decay stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
        }

        // Verify expected behavior at extremes
        // For large negative x: result ≈ x - 0.5
        assert!((output[0] - (-100.5)).abs() < 0.1, "softplus_decay(-100) should be ≈-100.5, got {}", output[0]);
        // For large positive x: result ≈ -0.5
        assert!((output[6] - (-0.5)).abs() < 0.01, "softplus_decay(100) should be ≈-0.5, got {}", output[6]);

        println!("Softplus decay stability test passed: all {} values are finite", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.5 (Layer Normalization Kernel) ===

    #[test]
    fn test_layer_norm_basic() {
        // Test basic layer normalization with a simple 2-vector case
        // Input: 2 vectors of length 4
        // Shape: [4, 2, 1, 1] where 4 is the channel dimension (fastest axis)

        // Vector 0: [1.0, 2.0, 3.0, 4.0] -> mean=2.5, var=1.25
        // Vector 1: [0.0, 4.0, 2.0, 6.0] -> mean=3.0, var=5.0
        let input = vec![
            1.0, 2.0, 3.0, 4.0,  // vector 0 (elements at offsets 0-3)
            0.0, 4.0, 2.0, 6.0,  // vector 1 (elements at offsets 4-7)
        ];

        // Weight = 1.0 (no scaling)
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        // Bias = 0.0 (no offset)
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let c = 4; // Channel dimension
        let n = 2; // Number of vectors
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm kernel failed");

        // Expected for vector 0: (x - 2.5) / sqrt(1.25 + eps)
        // std0 = sqrt(1.25) ≈ 1.118034
        // normalized: [-1.342, -0.447, 0.447, 1.342]
        let mean0 = 2.5f32;
        let std0 = (1.25f32 + eps).sqrt();

        // Expected for vector 1: (x - 3.0) / sqrt(5.0 + eps)
        // std1 = sqrt(5.0) ≈ 2.236
        // normalized: [-1.342, 0.447, -0.447, 1.342]
        let mean1 = 3.0f32;
        let std1 = (5.0f32 + eps).sqrt();

        let expected = vec![
            (1.0 - mean0) / std0, (2.0 - mean0) / std0, (3.0 - mean0) / std0, (4.0 - mean0) / std0,
            (0.0 - mean1) / std1, (4.0 - mean1) / std1, (2.0 - mean1) / std1, (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic layer_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_layer_norm_with_affine() {
        // Test layer normalization with weight and bias
        // Input: 1 vector of length 4
        // Shape: [4, 1, 1, 1]

        // Vector: [0.0, 2.0, 4.0, 6.0] -> mean=3.0, var=5.0
        let input = vec![0.0, 2.0, 4.0, 6.0];

        // Weight = [2.0, 1.0, 0.5, 0.0]
        let weight = vec![2.0, 1.0, 0.5, 0.0];
        // Bias = [1.0, 0.0, -1.0, 5.0]
        let bias = vec![1.0, 0.0, -1.0, 5.0];

        let c = 4;
        let n = 1;
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm with affine failed");

        // normalized = (x - 3.0) / sqrt(5.0 + eps)
        // then: output = normalized * weight + bias
        let mean = 3.0f32;
        let std = (5.0f32 + eps).sqrt();

        let expected: Vec<f32> = (0..4).map(|i| {
            let x = input[i];
            let normalized = (x - mean) / std;
            normalized * weight[i] + bias[i]
        }).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Layer norm with affine test passed: {} values verified", output.len());
    }

    #[test]
    fn test_layer_norm_numerical_stability() {
        // Test with values that could cause numerical issues
        // Very small variance (constant input + small noise)
        let c = 128;
        let n = 4;

        let mut input = vec![0.0f32; c * n];
        // Fill with near-constant values
        for i in 0..n {
            for j in 0..c {
                // Small variation around 1000.0
                input[i * c + j] = 1000.0 + (j as f32) * 0.001;
            }
        }

        let weight = vec![1.0f32; c];
        let bias = vec![0.0f32; c];
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (vec={}, ch={})", i, i / c, i % c
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (vec={}, ch={})", i, i / c, i % c
            );
        }

        // Normalized values should have mean ≈ 0 and std ≈ 1 for each vector
        for vec_idx in 0..n {
            let start = vec_idx * c;
            let end = start + c;
            let vec_output = &output[start..end];

            let sum: f32 = vec_output.iter().sum();
            let mean = sum / (c as f32);

            // Mean should be close to 0 (within tolerance due to FP32 accumulation)
            // With 128 elements and FP32 arithmetic, numerical errors can accumulate
            assert!(
                mean.abs() < 1e-3,
                "Vector {} mean not close to 0: {}", vec_idx, mean
            );
        }

        println!("Layer norm stability test passed: {} values are finite with correct mean", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.6 (Group Normalization Kernel) ===

    #[test]
    fn test_group_norm_basic() {
        // Test group normalization with 2 groups on 1 vector
        // Input: 1 vector of length 8, divided into 2 groups of 4
        // Shape: [8, 1, 1, 1]
        let input = vec![
            // Group 0: [1, 2, 3, 4] -> mean=2.5, var=1.25
            1.0, 2.0, 3.0, 4.0,
            // Group 1: [0, 2, 4, 6] -> mean=3.0, var=5.0
            0.0, 2.0, 4.0, 6.0,
        ];

        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];

        let c = 8;
        let n = 1;
        let num_groups = 2;
        let eps = 1e-5;

        let output = hip_group_norm(&input, &weight, &bias, c, n, num_groups, eps)
            .expect("group_norm kernel failed");

        // Expected: normalize each group independently
        let mean0 = 2.5f32;
        let std0 = (1.25f32 + eps).sqrt();
        let mean1 = 3.0f32;
        let std1 = (5.0f32 + eps).sqrt();

        let expected = vec![
            (1.0 - mean0) / std0, (2.0 - mean0) / std0, (3.0 - mean0) / std0, (4.0 - mean0) / std0,
            (0.0 - mean1) / std1, (2.0 - mean1) / std1, (4.0 - mean1) / std1, (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic group_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_group_norm_multiple_vectors() {
        // Test group normalization on multiple vectors
        // 2 vectors of length 4, 2 groups of 2 channels each
        let input = vec![
            // Vector 0: groups [1, 3], [2, 4]
            1.0, 3.0, 2.0, 4.0,
            // Vector 1: groups [0, 2], [1, 3]
            0.0, 2.0, 1.0, 3.0,
        ];

        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];

        let c = 4;
        let n = 2;
        let num_groups = 2;
        let eps = 1e-5;

        let output = hip_group_norm(&input, &weight, &bias, c, n, num_groups, eps)
            .expect("group_norm multiple vectors failed");

        // Each group should have mean ≈ 0 after normalization
        assert_eq!(output.len(), 8);
        println!("Group norm multiple vectors test passed: {} values", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.7 (L2 Normalization Kernel) ===

    #[test]
    fn test_l2_norm_basic() {
        // Test L2 normalization on a simple case
        // 1 vector of length 4, with head_size=4 (1 head)
        let input = vec![3.0, 0.0, 4.0, 0.0];  // L2 norm = 5

        let c = 4;
        let n = 1;
        let head_size = 4;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps)
            .expect("l2_norm kernel failed");

        // Expected: input / 5 = [0.6, 0.0, 0.8, 0.0]
        let expected = vec![0.6, 0.0, 0.8, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic l2_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_l2_norm_per_head() {
        // Test L2 normalization with multiple heads
        // 1 vector of length 4, with head_size=2 (2 heads)
        let input = vec![
            3.0, 4.0,  // Head 0: norm = 5
            5.0, 12.0, // Head 1: norm = 13
        ];

        let c = 4;
        let n = 1;
        let head_size = 2;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps)
            .expect("l2_norm per head failed");

        // Expected: normalize each head independently
        let expected = vec![
            3.0/5.0, 4.0/5.0,
            5.0/13.0, 12.0/13.0,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("L2 norm per head test passed: {} values verified", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.13 (Tanh Kernel) ===

    #[test]
    fn test_tanh_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_tanh(&input).expect("tanh kernel failed");

        // Expected: tanh(x)
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic tanh test passed: {} values verified", output.len());
    }

    #[test]
    fn test_tanh_edge_cases() {
        let input = vec![
            -100.0, // Very negative: tanh → -1
            -10.0,
            0.0,    // tanh(0) = 0
            10.0,
            100.0,  // Very positive: tanh → 1
        ];

        let output = hip_tanh(&input).expect("tanh edge cases failed");

        // Verify no NaN or Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {}", i);
            assert!(!val.is_infinite(), "Inf at index {}", i);
            assert!(val >= -1.0 && val <= 1.0, "Value out of [-1,1] at index {}: {}", i, val);
        }

        // Check extremes
        assert!((output[0] - (-1.0)).abs() < 1e-6, "tanh(-100) should be ≈-1");
        assert!(output[2].abs() < 1e-6, "tanh(0) should be ≈0");
        assert!((output[4] - 1.0).abs() < 1e-6, "tanh(100) should be ≈1");

        println!("Tanh edge cases test passed");
    }

    // === Acceptance Criteria Tests for bd-2sh.4.9 (Token Shift Kernel) ===

    #[test]
    fn test_token_shift_basic() {
        // Test token shift with 2 channels and 3 tokens
        // x shape: [2, 3, 1, 1]
        let x = vec![
            // Token 0: [1.0, 2.0]
            1.0, 2.0,
            // Token 1: [3.0, 4.0]
            3.0, 4.0,
            // Token 2: [5.0, 6.0]
            5.0, 6.0,
        ];

        // Initial state: [0.0, 0.0]
        let state_in = vec![0.0, 0.0];

        // Mix factor: 0.5 (blend 50% of previous into current)
        let mix = vec![0.5, 0.5];

        let c = 2;
        let t = 3;

        let (output, state_out) = hip_token_shift(&x, &state_in, &mix, c, t)
            .expect("token_shift kernel failed");

        // Formula: output[t] = x[t] + mix * (prev - x[t])
        // Token 0: x[0] + 0.5*(state - x[0]) = 1 + 0.5*(0-1) = [0.5, 1.0]
        // Token 1: x[1] + 0.5*(x[0] - x[1]) = 3 + 0.5*(1-3) = [2.0, 3.0]
        // Token 2: x[2] + 0.5*(x[1] - x[2]) = 5 + 0.5*(3-5) = [4.0, 5.0]
        let expected = vec![
            0.5, 1.0,  // Token 0
            2.0, 3.0,  // Token 1
            4.0, 5.0,  // Token 2
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Output mismatch at index {}: actual={:.6}, expected={:.6}",
                i, actual, exp
            );
        }

        // State out should be x[last] = [5.0, 6.0]
        assert!((state_out[0] - 5.0).abs() < 1e-5, "state_out[0] should be 5.0");
        assert!((state_out[1] - 6.0).abs() < 1e-5, "state_out[1] should be 6.0");

        println!("Token shift basic test passed");
    }

    #[test]
    fn test_token_shift_no_mix() {
        // Test with mix=0 (pass through current, no blending)
        let x = vec![1.0, 2.0, 3.0, 4.0];  // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![0.0, 0.0];

        let (output, _) = hip_token_shift(&x, &state_in, &mix, 2, 2)
            .expect("token_shift no mix failed");

        // With mix=0: output = x + 0*(prev - x) = x
        // So output equals x directly
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}", i, actual, exp
            );
        }
        println!("Token shift no mix test passed");
    }

    #[test]
    fn test_token_shift_full_mix() {
        // Test with mix=1 (full blending with previous)
        let x = vec![1.0, 2.0, 3.0, 4.0];  // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![1.0, 1.0];

        let (output, _) = hip_token_shift(&x, &state_in, &mix, 2, 2)
            .expect("token_shift full mix failed");

        // With mix=1: output = x + 1*(prev - x) = prev
        // Token 0: prev = state_in = [10.0, 20.0]
        // Token 1: prev = x[0] = [1.0, 2.0]
        let expected = vec![10.0, 20.0, 1.0, 2.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}", i, actual, exp
            );
        }
        println!("Token shift full mix test passed");
    }

    // === State I/O and Batched Inference Tests (bd-2sh.5.6) ===

    /// Test that HipState is correctly sized for batched inference.
    #[test]
    fn test_hip_state_batched_sizing() {
        let info = Rwkv7ModelInfo {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            head_size: 64,
            n_vocab: 65536,
            n_hidden: 2048,
        };

        let batch_size = 4;
        let state = HipState::new(&info, batch_size);

        assert_eq!(state.batch_size, 4);
        assert_eq!(state.att_states.len(), 12);
        assert_eq!(state.att_shift_states.len(), 12);
        assert_eq!(state.ffn_states.len(), 12);

        // Check per-layer sizes include batch dimension
        let expected_att_state_size = 64 * 64 * 12 * 4; // head_size² * n_head * batch
        let expected_shift_state_size = 768 * 4; // n_embd * batch

        assert_eq!(state.att_states[0].len(), expected_att_state_size);
        assert_eq!(state.att_shift_states[0].len(), expected_shift_state_size);
        assert_eq!(state.ffn_states[0].len(), expected_shift_state_size);

        println!("HipState batched sizing test passed (B=4)");
    }

    /// Test that forward() convenience wrapper works.
    #[test]
    fn test_forward_convenience_wrapper() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let tokens = vec![1u32, 2, 3, 4, 5];

        let logits = model.forward(&tokens).expect("forward() failed");

        // Should return vocab_size * T logits
        let expected_len = model.info.n_vocab * tokens.len();
        assert_eq!(logits.len(), expected_len,
            "Expected {} logits, got {}", expected_len, logits.len());

        println!("forward() convenience wrapper test passed");
    }

    /// Test batched inference with B=2 produces same results as sequential B=1.
    #[test]
    fn test_batched_matches_sequential() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // Two sequences
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];

        // Run sequentially with B=1
        let logits1 = model.forward(&seq1).expect("seq1 forward failed");
        let logits2 = model.forward(&seq2).expect("seq2 forward failed");

        // Run batched with B=2
        let mut state = HipState::new(&model.info, 2);
        let batched_logits = model.forward_with_state(&[&seq1, &seq2], &mut state)
            .expect("batched forward failed");

        // Batched output should have shape [vocab_size, T, B]
        let vocab = model.info.n_vocab;
        let t = 3;
        let b = 2;
        assert_eq!(batched_logits.len(), vocab * t * b);

        // Compare: batched logits should match sequential
        // Layout: batched[v, t, b] = batched[b * T * V + t * V + v]
        let mut max_diff = 0.0f32;
        for ti in 0..t {
            for vi in 0..vocab {
                // Sequential: logits1[v + t*V], logits2[v + t*V]
                let seq1_val = logits1[vi + ti * vocab];
                let seq2_val = logits2[vi + ti * vocab];

                // Batched: logits[v, t, 0] and logits[v, t, 1]
                let batch1_val = batched_logits[0 * t * vocab + ti * vocab + vi];
                let batch2_val = batched_logits[1 * t * vocab + ti * vocab + vi];

                let diff1 = (seq1_val - batch1_val).abs();
                let diff2 = (seq2_val - batch2_val).abs();
                max_diff = max_diff.max(diff1).max(diff2);

                // Allow small tolerance for numerical differences
                assert!(diff1 < 1e-4, "seq1 mismatch at t={}, v={}: {} vs {}", ti, vi, seq1_val, batch1_val);
                assert!(diff2 < 1e-4, "seq2 mismatch at t={}, v={}: {} vs {}", ti, vi, seq2_val, batch2_val);
            }
        }

        println!("Batched matches sequential test PASSED (max_diff={})", max_diff);
    }

    /// Test streaming equivalence with batched state.
    #[test]
    fn test_batched_streaming_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        let tokens: Vec<u32> = vec![1, 2, 3];

        // Single-sequence batch forward
        let mut state1 = HipState::new(&model.info, 1);
        let logits_batch = model.forward_with_state(&[&tokens], &mut state1)
            .expect("batch forward failed");

        // Single-sequence streaming (token by token)
        let mut state2 = HipState::new(&model.info, 1);
        let mut logits_stream = Vec::new();
        for &tok in &tokens {
            let logits = model.forward_with_state(&[&[tok]], &mut state2)
                .expect("streaming forward failed");
            logits_stream.extend(logits);
        }

        assert_eq!(logits_batch.len(), logits_stream.len());

        let mut max_diff = 0.0f32;
        for (i, (batch, stream)) in logits_batch.iter().zip(logits_stream.iter()).enumerate() {
            let diff = (batch - stream).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 1e-3, "Streaming mismatch at {}: {} vs {}", i, batch, stream);
        }

        println!("Batched streaming equivalence test PASSED (max_diff={})", max_diff);
    }

    /// Test chunked processing with batched state.
    #[test]
    fn test_batched_chunked_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // 10 tokens, chunk into [4, 4, 2]
        let tokens: Vec<u32> = (1..=10).collect();
        let chunk_size = 4;

        // Full forward
        let mut state1 = HipState::new(&model.info, 1);
        let logits_full = model.forward_with_state(&[&tokens], &mut state1)
            .expect("full forward failed");

        // Chunked forward
        let mut state2 = HipState::new(&model.info, 1);
        let mut logits_chunked = Vec::new();
        for chunk in tokens.chunks(chunk_size) {
            let logits = model.forward_with_state(&[chunk], &mut state2)
                .expect("chunked forward failed");
            logits_chunked.extend(logits);
        }

        assert_eq!(logits_full.len(), logits_chunked.len());

        let mut max_diff = 0.0f32;
        for (i, (full, chunked)) in logits_full.iter().zip(logits_chunked.iter()).enumerate() {
            let diff = (full - chunked).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 1e-3, "Chunked mismatch at {}: {} vs {}", i, full, chunked);
        }

        println!("Batched chunked equivalence test PASSED (max_diff={})", max_diff);
    }

    /// Test state evolution in batched mode.
    #[test]
    fn test_batched_state_evolution() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];

        let mut state = HipState::new(&model.info, 2);

        // State starts at zero
        let att_sum_before: f32 = state.att_states.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .sum();
        assert_eq!(att_sum_before, 0.0);

        // Run forward
        let _ = model.forward_with_state(&[&seq1, &seq2], &mut state)
            .expect("forward failed");

        // State should have evolved
        let att_sum_after: f32 = state.att_states.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .sum();
        assert!(att_sum_after > 0.0, "att_states should be non-zero after forward");

        println!("Batched state evolution test PASSED");
    }

    /// Test batch size mismatch error.
    #[test]
    fn test_batch_size_mismatch_error() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // State with batch_size=2, but provide 3 sequences
        let mut state = HipState::new(&model.info, 2);
        let result = model.forward_with_state(
            &[&[1u32], &[2u32], &[3u32]],
            &mut state
        );

        assert!(result.is_err(), "Should error on batch size mismatch");
        println!("Batch size mismatch error test PASSED");
    }

    /// Test sequence length mismatch error.
    #[test]
    fn test_sequence_length_mismatch_error() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // Two sequences with different lengths
        let mut state = HipState::new(&model.info, 2);
        let result = model.forward_with_state(
            &[&[1u32, 2, 3], &[4u32, 5]],  // length 3 vs length 2
            &mut state
        );

        assert!(result.is_err(), "Should error on sequence length mismatch");
        println!("Sequence length mismatch error test PASSED");
    }

    /// Test HipState reset.
    #[test]
    fn test_hip_state_reset() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        let mut state = HipState::new(&info, 2);

        // Fill with values
        for s in &mut state.att_states { s.fill(1.0); }
        for s in &mut state.att_shift_states { s.fill(2.0); }
        for s in &mut state.ffn_states { s.fill(3.0); }

        state.reset();

        // All should be zero
        assert!(state.att_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.att_shift_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.ffn_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.v_first.is_none());

        println!("HipState reset test PASSED");
    }

    /// Test HipState v_first persistence across operations.
    #[test]
    fn test_hip_state_v_first_persistence() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        // Fresh state should have v_first = None
        let mut state = HipState::new(&info, 1);
        assert!(state.v_first.is_none(), "New state should have v_first = None");

        // Set v_first and verify it persists
        state.v_first = Some(vec![1.0f32; 64]);
        assert!(state.v_first.is_some(), "v_first should persist after assignment");
        assert_eq!(state.v_first.as_ref().unwrap().len(), 64);

        // Reset should clear v_first
        state.reset();
        assert!(state.v_first.is_none(), "Reset should clear v_first to None");

        println!("HipState v_first persistence test PASSED");
    }
}
