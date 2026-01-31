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
