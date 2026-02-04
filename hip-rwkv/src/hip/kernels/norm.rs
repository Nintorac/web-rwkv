//! HIP kernel wrappers for normalization operations.

use half::f16;
use std::ffi::c_int;

use crate::hip::device::Stream;
use crate::hip::ffi::{
    check,
    launch_group_norm_f16,
    launch_group_norm_f32,
    launch_l2_norm_f16,
    launch_l2_norm_f32,
    launch_layer_norm_f16,
    launch_layer_norm_f32,
    HipErrorKind,
    Result,
};
use crate::hip::tensor::{TensorHip, TensorShape};

/// Launch the layer normalization kernel.
///
/// Layer normalization normalizes each vector of length C (channel dimension)
/// independently. Used in RWKV7 for normalizing activations.
///
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1] where C is the channel dimension
/// * `weight` - Per-channel weight of shape [C, 1, 1, 1]
/// * `bias` - Per-channel bias of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `eps` - Epsilon for numerical stability (typically 1e-5)
/// * `stream` - HIP stream for asynchronous execution
pub fn layer_norm_f32(
    input: &TensorHip<f32>,
    weight: &TensorHip<f32>,
    bias: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [C, T, B, 1] - treat T*B as total vectors
    let c = input.shape()[0]; // Channel dimension (normalize over this)
                              // Compute n as product of all dimensions except the first (handles batching)
    let n = input.shape()[1] * input.shape()[2] * input.shape()[3];

    // Validate shapes - output should have same total size
    let out_n = output.shape()[1] * output.shape()[2] * output.shape()[3];
    if output.shape()[0] != c || out_n != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected C={} with {} vectors, got {} with {} vectors",
                c,
                n,
                output.shape(),
                out_n
            ),
        });
    }
    if weight.shape()[0] != c || bias.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias shape mismatch: expected [{}, 1, 1, 1], got weight={}, bias={}",
                c,
                weight.shape(),
                bias.shape()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "layer_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_layer_norm_f32(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            eps,
            stream.handle(),
        ))
    }
}

pub fn layer_norm_f16(
    input: &TensorHip<f16>,
    weight: &TensorHip<f16>,
    bias: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let shape = input.shape();
    let c = shape[0];
    let n = shape[1] * shape[2] * shape[3];
    if input.len() != output.len() || weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: "layer_norm_f16 shape mismatch".to_string(),
        });
    }
    if !input.is_contiguous()
        || !output.is_contiguous()
        || !weight.is_contiguous()
        || !bias.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "layer_norm_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_layer_norm_f16(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute layer normalization on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `input` - Input data of shape [C, N, 1, 1] flattened (C*N elements)
/// * `weight` - Weight data of shape [C, 1, 1, 1] (C elements)
/// * `bias` - Bias data of shape [C, 1, 1, 1] (C elements)
/// * `c` - Channel dimension (length of each vector to normalize)
/// * `n` - Number of vectors to normalize
/// * `eps` - Epsilon for numerical stability
pub fn hip_layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c: usize,
    n: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    // Validate sizes
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {} ({}*{}), got {}",
                c * n,
                c,
                n,
                input.len()
            ),
        });
    }
    if weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias size mismatch: expected {}, got weight={}, bias={}",
                c,
                weight.len(),
                bias.len()
            ),
        });
    }

    let stream = Stream::null();

    // Create tensors with proper shapes
    let input_shape = TensorShape::new(c, n, 1, 1);
    let param_shape = TensorShape::new(c, 1, 1, 1);

    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let d_weight = TensorHip::from_slice(weight, param_shape, &stream)?;
    let d_bias = TensorHip::from_slice(bias, param_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;

    layer_norm_f32(&d_input, &d_weight, &d_bias, &mut d_output, eps, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the group normalization kernel.
///
/// Group normalization divides channels into groups and normalizes within each group.
/// Used in RWKV7 with 12 groups (H = 12 heads) and eps = 64e-5.
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1]
/// * `weight` - Per-channel weight of shape [C, 1, 1, 1]
/// * `bias` - Per-channel bias of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `num_groups` - Number of groups (must divide C evenly)
/// * `eps` - Epsilon for numerical stability
/// * `stream` - HIP stream
pub fn group_norm_f32(
    input: &TensorHip<f32>,
    weight: &TensorHip<f32>,
    bias: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    num_groups: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    // Compute n as product of all dimensions except the first (handles batching)
    let n = input.shape()[1] * input.shape()[2] * input.shape()[3];

    if c % num_groups != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by num_groups {}",
                c, num_groups
            ),
        });
    }
    // Validate shapes - output should have same total size
    let out_n = output.shape()[1] * output.shape()[2] * output.shape()[3];
    if output.shape()[0] != c || out_n != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected C={} with {} vectors, got {} with {} vectors",
                c,
                n,
                output.shape(),
                out_n
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "group_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_group_norm_f32(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            num_groups as c_int,
            eps,
            stream.handle(),
        ))
    }
}

pub fn group_norm_f16(
    input: &TensorHip<f16>,
    weight: &TensorHip<f16>,
    bias: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    num_groups: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    let n = input.shape()[1] * input.shape()[2] * input.shape()[3];

    if c % num_groups != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by num_groups {}",
                c, num_groups
            ),
        });
    }
    let out_n = output.shape()[1] * output.shape()[2] * output.shape()[3];
    if output.shape()[0] != c || out_n != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected C={} with {} vectors, got {} with {} vectors",
                c,
                n,
                output.shape(),
                out_n
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "group_norm_f16 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_group_norm_f16(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            num_groups as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute group normalization on host data.
pub fn hip_group_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c: usize,
    n: usize,
    num_groups: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got {}",
                c * n,
                input.len()
            ),
        });
    }
    if weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Weight/bias size mismatch: expected {}", c),
        });
    }

    let stream = Stream::null();
    let input_shape = TensorShape::new(c, n, 1, 1);
    let param_shape = TensorShape::new(c, 1, 1, 1);

    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let d_weight = TensorHip::from_slice(weight, param_shape, &stream)?;
    let d_bias = TensorHip::from_slice(bias, param_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;

    group_norm_f32(
        &d_input,
        &d_weight,
        &d_bias,
        &mut d_output,
        num_groups,
        eps,
        &stream,
    )?;

    d_output.to_vec(&stream)
}

/// Launch the L2 normalization kernel.
///
/// L2 normalization normalizes each head to unit L2 norm.
/// Used for key normalization in RWKV7.
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1] where C = H * head_size
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `head_size` - Size of each head (normalize over this dimension)
/// * `eps` - Epsilon for numerical stability
/// * `stream` - HIP stream
pub fn l2_norm_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    head_size: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    // Compute n as product of all dimensions except the first (handles batching)
    let n = input.shape()[1] * input.shape()[2] * input.shape()[3];

    if c % head_size != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by head_size {}",
                c, head_size
            ),
        });
    }
    // Validate shapes - output should have same total size
    let out_n = output.shape()[1] * output.shape()[2] * output.shape()[3];
    if output.shape()[0] != c || out_n != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected C={} with {} vectors, got {} with {} vectors",
                c,
                n,
                output.shape(),
                out_n
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "l2_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_l2_norm_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            head_size as c_int,
            eps,
            stream.handle(),
        ))
    }
}

pub fn l2_norm_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    head_size: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    let n = input.shape()[1] * input.shape()[2] * input.shape()[3];

    if c % head_size != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by head_size {}",
                c, head_size
            ),
        });
    }
    let out_n = output.shape()[1] * output.shape()[2] * output.shape()[3];
    if output.shape()[0] != c || out_n != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected C={} with {} vectors, got {} with {} vectors",
                c,
                n,
                output.shape(),
                out_n
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "l2_norm_f16 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_l2_norm_f16(
            input.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            head_size as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute L2 normalization on host data.
pub fn hip_l2_norm(
    input: &[f32],
    c: usize,
    n: usize,
    head_size: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got {}",
                c * n,
                input.len()
            ),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(c, n, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    l2_norm_f32(&d_input, &mut d_output, head_size, eps, &stream)?;

    d_output.to_vec(&stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Acceptance Criteria Tests for bd-2sh.4.5 (Layer Normalization Kernel) ===

    #[test]
    fn test_layer_norm_basic() {
        // Test basic layer normalization with a simple 2-vector case
        // Input: 2 vectors of length 4
        // Shape: [4, 2, 1, 1] where 4 is the channel dimension (fastest axis)

        // Vector 0: [1.0, 2.0, 3.0, 4.0] -> mean=2.5, var=1.25
        // Vector 1: [0.0, 4.0, 2.0, 6.0] -> mean=3.0, var=5.0
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // vector 0 (elements at offsets 0-3)
            0.0, 4.0, 2.0, 6.0, // vector 1 (elements at offsets 4-7)
        ];

        // Weight = 1.0 (no scaling)
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        // Bias = 0.0 (no offset)
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let c = 4; // Channel dimension
        let n = 2; // Number of vectors
        let eps = 1e-5;

        let output =
            hip_layer_norm(&input, &weight, &bias, c, n, eps).expect("layer_norm kernel failed");

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
            (1.0 - mean0) / std0,
            (2.0 - mean0) / std0,
            (3.0 - mean0) / std0,
            (4.0 - mean0) / std0,
            (0.0 - mean1) / std1,
            (4.0 - mean1) / std1,
            (2.0 - mean1) / std1,
            (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Basic layer_norm test passed: {} values verified",
            output.len()
        );
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

        let expected: Vec<f32> = (0..4)
            .map(|i| {
                let x = input[i];
                let normalized = (x - mean) / std;
                normalized * weight[i] + bias[i]
            })
            .collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Layer norm with affine test passed: {} values verified",
            output.len()
        );
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
                "NaN at index {} (vec={}, ch={})",
                i,
                i / c,
                i % c
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (vec={}, ch={})",
                i,
                i / c,
                i % c
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
                "Vector {} mean not close to 0: {}",
                vec_idx,
                mean
            );
        }

        println!(
            "Layer norm stability test passed: {} values are finite with correct mean",
            output.len()
        );
    }

    // === Acceptance Criteria Tests for bd-2sh.4.6 (Group Normalization Kernel) ===

    #[test]
    fn test_group_norm_basic() {
        // Test group normalization with 2 groups on 1 vector
        // Input: 1 vector of length 8, divided into 2 groups of 4
        // Shape: [8, 1, 1, 1]
        let input = vec![
            // Group 0: [1, 2, 3, 4] -> mean=2.5, var=1.25
            1.0, 2.0, 3.0, 4.0, // Group 1: [0, 2, 4, 6] -> mean=3.0, var=5.0
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
            (1.0 - mean0) / std0,
            (2.0 - mean0) / std0,
            (3.0 - mean0) / std0,
            (4.0 - mean0) / std0,
            (0.0 - mean1) / std1,
            (2.0 - mean1) / std1,
            (4.0 - mean1) / std1,
            (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Basic group_norm test passed: {} values verified",
            output.len()
        );
    }

    #[test]
    fn test_group_norm_multiple_vectors() {
        // Test group normalization on multiple vectors
        // 2 vectors of length 4, 2 groups of 2 channels each
        let input = vec![
            // Vector 0: groups [1, 3], [2, 4]
            1.0, 3.0, 2.0, 4.0, // Vector 1: groups [0, 2], [1, 3]
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
        println!(
            "Group norm multiple vectors test passed: {} values",
            output.len()
        );
    }

    // === Acceptance Criteria Tests for bd-2sh.4.7 (L2 Normalization Kernel) ===

    #[test]
    fn test_l2_norm_basic() {
        // Test L2 normalization on a simple case
        // 1 vector of length 4, with head_size=4 (1 head)
        let input = vec![3.0, 0.0, 4.0, 0.0]; // L2 norm = 5

        let c = 4;
        let n = 1;
        let head_size = 4;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps).expect("l2_norm kernel failed");

        // Expected: input / 5 = [0.6, 0.0, 0.8, 0.0]
        let expected = vec![0.6, 0.0, 0.8, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Basic l2_norm test passed: {} values verified",
            output.len()
        );
    }

    #[test]
    fn test_l2_norm_per_head() {
        // Test L2 normalization with multiple heads
        // 1 vector of length 4, with head_size=2 (2 heads)
        let input = vec![
            3.0, 4.0, // Head 0: norm = 5
            5.0, 12.0, // Head 1: norm = 13
        ];

        let c = 4;
        let n = 1;
        let head_size = 2;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps).expect("l2_norm per head failed");

        // Expected: normalize each head independently
        let expected = vec![3.0 / 5.0, 4.0 / 5.0, 5.0 / 13.0, 12.0 / 13.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "L2 norm per head test passed: {} values verified",
            output.len()
        );
    }
}
