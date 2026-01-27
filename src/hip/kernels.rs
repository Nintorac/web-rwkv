//! HIP kernel wrapper functions.

use std::ffi::c_int;

use super::ffi::{
    HipErrorKind, Result, check,
    launch_copy_f32, launch_decay_exp_f32, launch_lerp_f32,
    launch_sigmoid_f32, launch_squared_relu_f32, launch_softplus_decay_f32,
    launch_layer_norm_f32, launch_group_norm_f32, launch_l2_norm_f32,
    launch_tanh_f32, launch_token_shift_f32, launch_channel_mix_state_f32,
    launch_wkv_bonus_f32, launch_control_k_f32, launch_wkv7_f32, launch_wkv7_f32_masked,
    // Elementwise operations
    launch_add_f32, launch_mul_f32, launch_negate_f32, launch_exp_f32,
    launch_broadcast_add_f32, launch_broadcast_mul_f32,
};
use super::device::Stream;
use super::buffer::DeviceBuffer;
use super::tensor::{TensorShape, TensorHip};

#[cfg(feature = "hip-probes")]
use crate::hip_probe;


/// Launch the copy kernel to copy f32 data from input to output
pub fn copy_f32(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check(launch_copy_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// GPU-to-GPU copy for TensorHip: output = input
///
/// Copies data from one GPU tensor to another without CPU round-trip.
/// Both tensors must have the same length and be contiguous.
pub fn copy_tensor_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "copy_tensor_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_copy_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Copy f32 data from host, through GPU copy kernel, back to host.
/// This is a convenience function for testing.
pub fn hip_copy_kernel(input: &[f32]) -> Result<Vec<f32>> {
    // Use the null stream as a workaround for hipStreamCreate crashes
    let stream = Stream::null();

    let mut d_input = DeviceBuffer::<f32>::new(input.len())?;
    let mut d_output = DeviceBuffer::<f32>::new(input.len())?;

    d_input.copy_from_host(input, &stream)?;
    copy_f32(&d_input, &mut d_output, &stream)?;

    let mut output = vec![0.0f32; input.len()];
    d_output.copy_to_host(&mut output, &stream)?;
    stream.synchronize()?;

    Ok(output)
}

/// Launch the decay exponential kernel: out = exp(-exp(x))
///
/// This is the time decay transformation used in RWKV7.
/// Numerically stable for all finite inputs.
pub fn decay_exp_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "decay_exp_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_decay_exp_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute decay exponential on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_decay_exp(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    decay_exp_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the lerp kernel: out = a + t * (b - a) (linear interpolation)
///
/// This is used for mixing operations in RWKV7.
/// All tensors must have the same length and be contiguous.
pub fn lerp_f32(
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    t: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let n = a.len();
    if b.len() != n || t.len() != n || output.len() != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, t={}, output={}",
                n, b.len(), t.len(), output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !t.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "lerp_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_lerp_f32(
            a.as_ptr(),
            b.as_ptr(),
            t.as_ptr(),
            output.as_mut_ptr(),
            n as c_int,
            stream.handle(),
        ))
    }
}

/// Compute linear interpolation on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_lerp(a: &[f32], b: &[f32], t: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() || a.len() != t.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Size mismatch: a={}, b={}, t={}", a.len(), b.len(), t.len()),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(a.len(), 1, 1, 1);

    let d_a = TensorHip::from_slice(a, shape, &stream)?;
    let d_b = TensorHip::from_slice(b, shape, &stream)?;
    let d_t = TensorHip::from_slice(t, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    lerp_f32(&d_a, &d_b, &d_t, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the sigmoid kernel: out = 1 / (1 + exp(-x))
///
/// Standard sigmoid activation function.
pub fn sigmoid_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "sigmoid_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_sigmoid_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute sigmoid on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_sigmoid(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    sigmoid_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the squared ReLU kernel: out = max(0, x)^2
///
/// Used in RWKV7 channel mixing.
pub fn squared_relu_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "squared_relu_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_squared_relu_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute squared ReLU on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_squared_relu(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    squared_relu_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the softplus decay kernel: out = log(sigmoid(x)) - 0.5
///
/// Used for RWKV7 time decay computation.
/// Numerically stable for all finite inputs.
pub fn softplus_decay_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "softplus_decay_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_softplus_decay_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute softplus decay on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_softplus_decay(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    softplus_decay_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

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
    // Input shape: [C, N, 1, 1]
    let c = input.shape()[0];  // Channel dimension (normalize over this)
    let n = input.shape()[1];  // Number of vectors

    // Validate shapes
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
            ),
        });
    }
    if weight.shape()[0] != c || bias.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias shape mismatch: expected [{}, 1, 1, 1], got weight={}, bias={}",
                c, weight.shape(), bias.shape()
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
                c * n, c, n, input.len()
            ),
        });
    }
    if weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias size mismatch: expected {}, got weight={}, bias={}",
                c, weight.len(), bias.len()
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
    let n = input.shape()[1];

    if c % num_groups != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by num_groups {}",
                c, num_groups
            ),
        });
    }
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
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
            message: format!("Input size mismatch: expected {}, got {}", c * n, input.len()),
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

    group_norm_f32(&d_input, &d_weight, &d_bias, &mut d_output, num_groups, eps, &stream)?;

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
    let n = input.shape()[1];

    if c % head_size != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by head_size {}",
                c, head_size
            ),
        });
    }
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
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
            message: format!("Input size mismatch: expected {}, got {}", c * n, input.len()),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(c, n, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    l2_norm_f32(&d_input, &mut d_output, head_size, eps, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the tanh kernel: out = tanh(x)
pub fn tanh_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "tanh_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_tanh_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute tanh on host data.
pub fn hip_tanh(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    tanh_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the token shift kernel.
///
/// Token shift implements RWKV's time-mixing operation:
/// - output[t] = x[t] + mix * (prev[t] - x[t])
/// - Where prev[0] = state_in, prev[t>0] = x[t-1]
/// - state_out = x[T-1] (last token becomes state for next batch)
///
/// # Arguments
/// * `x` - Input tensor of shape [C, T, 1, 1]
/// * `state_in` - Previous state of shape [C, 1, 1, 1]
/// * `mix` - Per-channel mixing factor of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, T, 1, 1]
/// * `state_out` - New state of shape [C, 1, 1, 1]
/// * `stream` - HIP stream
pub fn token_shift_f32(
    x: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    mix: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = x.shape()[0];
    let t = x.shape()[1];

    if output.shape()[0] != c || output.shape()[1] != t {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, t, output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_out.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, 1, 1, 1]",
                c
            ),
        });
    }
    if mix.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Mix shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, mix.shape()
            ),
        });
    }

    unsafe {
        check(launch_token_shift_f32(
            x.as_ptr(),
            state_in.as_ptr(),
            mix.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            c as c_int,
            t as c_int,
            stream.handle(),
        ))
    }
}

/// Compute token shift on host data, returning (output, state_out).
pub fn hip_token_shift(
    x: &[f32],
    state_in: &[f32],
    mix: &[f32],
    c: usize,
    t: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if x.len() != c * t {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x size mismatch: expected {}, got {}", c * t, x.len()),
        });
    }
    if state_in.len() != c || mix.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state/mix size mismatch: expected {}", c),
        });
    }

    let stream = Stream::null();

    let x_shape = TensorShape::new(c, t, 1, 1);
    let state_shape = TensorShape::new(c, 1, 1, 1);

    let d_x = TensorHip::from_slice(x, x_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_mix = TensorHip::from_slice(mix, state_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(x_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    token_shift_f32(&d_x, &d_state_in, &d_mix, &mut d_output, &mut d_state_out, &stream)?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// Launch the channel-mix state kernel.
///
/// Same as token shift but with batch dimension.
/// Input: [C, T, B, 1], state: [C, B, 1, 1], x_k: [C, 1, 1, 1]
///
/// # Arguments
/// * `x` - Input tensor of shape [C, T, B, 1]
/// * `state_in` - Previous state per batch of shape [C, B, 1, 1]
/// * `x_k` - Per-channel mixing factor of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, T, B, 1]
/// * `state_out` - New state per batch of shape [C, B, 1, 1]
/// * `stream` - HIP stream
pub fn channel_mix_state_f32(
    x: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    x_k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = x.shape()[0];
    let t = x.shape()[1];
    let b = x.shape()[2];

    if output.shape() != x.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                x.shape(), output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, b, state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, x_k.shape()
            ),
        });
    }

    unsafe {
        check(launch_channel_mix_state_f32(
            x.as_ptr(),
            state_in.as_ptr(),
            x_k.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            c as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute channel-mix state on host data, returning (output, state_out).
pub fn hip_channel_mix_state(
    x: &[f32],
    state_in: &[f32],
    x_k: &[f32],
    c: usize,
    t: usize,
    b: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if x.len() != c * t * b {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x size mismatch: expected {}, got {}", c * t * b, x.len()),
        });
    }
    if state_in.len() != c * b {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", c * b, state_in.len()),
        });
    }
    if x_k.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x_k size mismatch: expected {}, got {}", c, x_k.len()),
        });
    }

    let stream = Stream::null();

    let x_shape = TensorShape::new(c, t, b, 1);
    let state_shape = TensorShape::new(c, b, 1, 1);
    let xk_shape = TensorShape::new(c, 1, 1, 1);

    let d_x = TensorHip::from_slice(x, x_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_x_k = TensorHip::from_slice(x_k, xk_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(x_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    channel_mix_state_f32(&d_x, &d_state_in, &d_x_k, &mut d_output, &mut d_state_out, &stream)?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// Launch the WKV bonus kernel (time_first).
///
/// Computes: output = (r * k * r_k).sum(dim=head_size) * v
/// This is the "time_first" bonus attention on the current token.
///
/// # Arguments
/// * `r` - Receptance tensor of shape [N, H, T, B] where N=head_size, H=n_heads
/// * `k` - Key tensor of shape [N, H, T, B]
/// * `v` - Value tensor of shape [N, H, T, B]
/// * `r_k` - Per-head bonus weight of shape [N, H, 1, 1]
/// * `output` - Output tensor of shape [N, H, T, B]
/// * `stream` - HIP stream
pub fn wkv_bonus_f32(
    r: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    r_k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    // Shape: [N, H, T, B] where N=head_size
    let n = r.shape()[0];  // head_size
    let h = r.shape()[1];  // n_heads
    let t = r.shape()[2];  // tokens
    let b = r.shape()[3];  // batch

    // Validate shapes
    if k.shape() != r.shape() || v.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Shape mismatch: r={}, k={}, v={}",
                r.shape(), k.shape(), v.shape()
            ),
        });
    }
    if output.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                r.shape(), output.shape()
            ),
        });
    }
    if r_k.shape()[0] != n || r_k.shape()[1] != h {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "r_k shape mismatch: expected [{}, {}, 1, 1], got {}",
                n, h, r_k.shape()
            ),
        });
    }
    if !r.is_contiguous() || !k.is_contiguous() || !v.is_contiguous()
        || !r_k.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv_bonus_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv_bonus_f32(
            r.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            r_k.as_ptr(),
            output.as_mut_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV bonus on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `r` - Receptance data of shape [N, H, T, B] flattened
/// * `k` - Key data of shape [N, H, T, B] flattened
/// * `v` - Value data of shape [N, H, T, B] flattened
/// * `r_k` - Per-head bonus weight of shape [N, H, 1, 1] flattened
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens
/// * `b` - batch
pub fn hip_wkv_bonus(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    r_k: &[f32],
    n: usize,  // head_size
    h: usize,  // n_heads
    t: usize,  // tokens
    b: usize,  // batch
) -> Result<Vec<f32>> {
    let expected_len = n * h * t * b;
    let rk_len = n * h;

    if r.len() != expected_len || k.len() != expected_len || v.len() != expected_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got r={}, k={}, v={}",
                expected_len, r.len(), k.len(), v.len()
            ),
        });
    }
    if r_k.len() != rk_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("r_k size mismatch: expected {}, got {}", rk_len, r_k.len()),
        });
    }

    let stream = Stream::null();

    let data_shape = TensorShape::new(n, h, t, b);
    let rk_shape = TensorShape::new(n, h, 1, 1);

    let d_r = TensorHip::from_slice(r, data_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, data_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, data_shape, &stream)?;
    let d_r_k = TensorHip::from_slice(r_k, rk_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(data_shape)?;

    wkv_bonus_f32(&d_r, &d_k, &d_v, &d_r_k, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the control-K kernel (replacement key).
///
/// Computes: output = k * (1 + (a - 1) * k_a)
/// This creates the replacement key for RWKV7 time mixing.
///
/// # Arguments
/// * `k_a` - Per-channel control weight of shape [C, 1, 1, 1]
/// * `a` - Attention tensor of shape [C, T, B, 1]
/// * `k` - Key tensor of shape [C, T, B, 1]
/// * `output` - Output tensor of shape [C, T, B, 1]
/// * `stream` - HIP stream
pub fn control_k_f32(
    k_a: &TensorHip<f32>,
    a: &TensorHip<f32>,
    k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = k.shape()[0];
    let t = k.shape()[1];
    let b = k.shape()[2];

    if output.shape() != k.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                k.shape(), output.shape()
            ),
        });
    }
    if a.shape() != k.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "a shape mismatch: expected {}, got {}",
                k.shape(), a.shape()
            ),
        });
    }
    if k_a.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "k_a shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, k_a.shape()
            ),
        });
    }
    if !k_a.is_contiguous() || !a.is_contiguous() || !k.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "control_k_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_control_k_f32(
            k_a.as_ptr(),
            a.as_ptr(),
            k.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute control-K on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `k_a` - Per-channel control weight of size C
/// * `a` - Attention data of shape [C, T, B, 1] flattened
/// * `k` - Key data of shape [C, T, B, 1] flattened
/// * `c` - Channel dimension
/// * `t` - Token dimension
/// * `b` - Batch dimension
pub fn hip_control_k(
    k_a: &[f32],
    a: &[f32],
    k: &[f32],
    c: usize,
    t: usize,
    b: usize,
) -> Result<Vec<f32>> {
    let expected_len = c * t * b;

    if k.len() != expected_len || a.len() != expected_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got k={}, a={}",
                expected_len, k.len(), a.len()
            ),
        });
    }
    if k_a.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("k_a size mismatch: expected {}, got {}", c, k_a.len()),
        });
    }

    let stream = Stream::null();

    let data_shape = TensorShape::new(c, t, b, 1);
    let ka_shape = TensorShape::new(c, 1, 1, 1);

    let d_k_a = TensorHip::from_slice(k_a, ka_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, data_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, data_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(data_shape)?;

    control_k_f32(&d_k_a, &d_a, &d_k, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the WKV7 core kernel.
///
/// Implements RWKV7's time mixing attention mechanism:
///   1. sa = sum(a[j] * state[i,j])  - attention over state
///   2. state[i,j] = state[i,j] * w[j] + sa * b[j] + k[j] * v[i]  - update
///   3. y[i] = sum(state[i,j] * q[j])  - output
///
/// # Arguments
/// * `w_decay` - Pre-computed decay tensor [N, H, T, B] where decay = exp(-exp(w_raw))
/// * `q` - Query tensor [N, H, T, B]
/// * `k` - Key tensor [N, H, T, B]
/// * `v` - Value tensor [N, H, T, B]
/// * `a` - Attention component tensor [N, H, T, B]
/// * `b` - Attention component tensor [N, H, T, B]
/// * `state_in` - Input state tensor [N, N, H, B]
/// * `output` - Output tensor [N, H, T, B]
/// * `state_out` - Output state tensor [N, N, H, B]
/// * `stream` - HIP stream
pub fn wkv7_f32(
    w_decay: &TensorHip<f32>,
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [N, H, T, B]
    let n = w_decay.shape()[0];  // head_size
    let h = w_decay.shape()[1];  // n_heads
    let t = w_decay.shape()[2];  // tokens
    let b_size = w_decay.shape()[3];  // batch

    // Validate input shapes
    let input_shape = w_decay.shape();
    if q.shape() != input_shape || k.shape() != input_shape || v.shape() != input_shape
        || a.shape() != input_shape || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input shape mismatch: w_decay={}, q={}, k={}, v={}, a={}, b={}",
                w_decay.shape(), q.shape(), k.shape(), v.shape(), a.shape(), b.shape()
            ),
        });
    }

    // Validate output shape
    if output.shape() != input_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                input_shape, output.shape()
            ),
        });
    }

    // Validate state shapes: [N, N, H, B]
    let state_shape = TensorShape::new(n, n, h, b_size);
    if state_in.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_in shape mismatch: expected {}, got {}",
                state_shape, state_in.shape()
            ),
        });
    }
    if state_out.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_out shape mismatch: expected {}, got {}",
                state_shape, state_out.shape()
            ),
        });
    }

    // Check contiguity
    if !w_decay.is_contiguous() || !q.is_contiguous() || !k.is_contiguous()
        || !v.is_contiguous() || !a.is_contiguous() || !b.is_contiguous()
        || !state_in.is_contiguous() || !output.is_contiguous() || !state_out.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_f32(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state_in.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV7 on host data, returning (output, state_out).
/// This is a convenience function for testing.
///
/// # Arguments
/// * `w_decay` - Pre-computed decay data [N, H, T, B] flattened
/// * `q`, `k`, `v`, `a`, `b` - Input data [N, H, T, B] flattened
/// * `state_in` - Input state [N, N, H, B] flattened
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens
/// * `b` - batch
pub fn hip_wkv7(
    w_decay: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    state_in: &[f32],
    n: usize,
    h: usize,
    t: usize,
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let input_len = n * h * t * batch;
    let state_len = n * n * h * batch;

    if w_decay.len() != input_len || q.len() != input_len || k.len() != input_len
        || v.len() != input_len || a.len() != input_len || b.len() != input_len
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got w={}, q={}, k={}, v={}, a={}, b={}",
                input_len, w_decay.len(), q.len(), k.len(), v.len(), a.len(), b.len()
            ),
        });
    }
    if state_in.len() != state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", state_len, state_in.len()),
        });
    }

    let stream = Stream::null();

    let input_shape = TensorShape::new(n, h, t, batch);
    let state_shape = TensorShape::new(n, n, h, batch);

    let d_w_decay = TensorHip::from_slice(w_decay, input_shape, &stream)?;
    let d_q = TensorHip::from_slice(q, input_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, input_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, input_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, input_shape, &stream)?;
    let d_b = TensorHip::from_slice(b, input_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    wkv7_f32(
        &d_w_decay, &d_q, &d_k, &d_v, &d_a, &d_b,
        &d_state_in, &mut d_output, &mut d_state_out, &stream
    )?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// WKV7 kernel with length masking for variable-length batched sequences.
/// Skips state updates for padding positions, preserving: state(seq + padding) == state(seq).
///
/// # Arguments
/// * `w_decay` - Pre-computed decay tensor [N, H, T, B]
/// * `q`, `k`, `v`, `a`, `b` - Input tensors [N, H, T, B]
/// * `state_in` - Input state tensor [N, N, H, B]
/// * `output` - Output tensor [N, H, T, B]
/// * `state_out` - Output state tensor [N, N, H, B]
/// * `lengths` - Real sequence length per batch [B] (as i32)
/// * `stream` - HIP stream
pub fn wkv7_f32_masked(
    w_decay: &TensorHip<f32>,
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    lengths: &TensorHip<i32>,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [N, H, T, B]
    let n = w_decay.shape()[0];  // head_size
    let h = w_decay.shape()[1];  // n_heads
    let t = w_decay.shape()[2];  // tokens (padded)
    let b_size = w_decay.shape()[3];  // batch

    // Validate input shapes
    let input_shape = w_decay.shape();
    if q.shape() != input_shape || k.shape() != input_shape || v.shape() != input_shape
        || a.shape() != input_shape || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input shape mismatch: w_decay={}, q={}, k={}, v={}, a={}, b={}",
                w_decay.shape(), q.shape(), k.shape(), v.shape(), a.shape(), b.shape()
            ),
        });
    }

    // Validate output shape
    if output.shape() != input_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                input_shape, output.shape()
            ),
        });
    }

    // Validate state shapes: [N, N, H, B]
    let state_shape = TensorShape::new(n, n, h, b_size);
    if state_in.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_in shape mismatch: expected {}, got {}",
                state_shape, state_in.shape()
            ),
        });
    }
    if state_out.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_out shape mismatch: expected {}, got {}",
                state_shape, state_out.shape()
            ),
        });
    }

    // Validate lengths shape: should have b_size elements
    let lengths_len = lengths.shape().len();
    if lengths_len != b_size {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "lengths size mismatch: expected {}, got {}",
                b_size, lengths_len
            ),
        });
    }

    // Check contiguity
    if !w_decay.is_contiguous() || !q.is_contiguous() || !k.is_contiguous()
        || !v.is_contiguous() || !a.is_contiguous() || !b.is_contiguous()
        || !state_in.is_contiguous() || !output.is_contiguous() || !state_out.is_contiguous()
        || !lengths.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_f32_masked requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_f32_masked(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state_in.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            lengths.as_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Extract shift states at the correct positions based on per-batch lengths.
///
/// For each batch element b, extracts x[:, length[b]-1, b] instead of x[:, T-1, b].
/// This is used to correctly handle padded sequences where we want the state
/// at the last valid position, not the last padded position.
///
/// # Arguments
/// * `x` - Input tensor [C, T, B] flattened
/// * `lengths` - Real sequence length per batch [B]
/// * `c` - Embedding dimension
/// * `t` - Padded sequence length
/// * `b` - Batch size
///
/// # Returns
/// State tensor [C, B] containing x[:, length[i]-1, i] for each batch i
pub(crate) fn extract_shift_state_at_lengths(
    x: &[f32],
    lengths: &[usize],
    c: usize,
    t: usize,
    b: usize,
) -> Vec<f32> {
    let mut state_out = vec![0.0f32; c * b];
    for batch_idx in 0..b {
        let time_idx = lengths[batch_idx] - 1; // Last valid position
        for channel in 0..c {
            // x layout: [C, T, B] = x[batch_idx * T * C + time_idx * C + channel]
            let x_idx = batch_idx * t * c + time_idx * c + channel;
            let state_idx = batch_idx * c + channel;
            state_out[state_idx] = x[x_idx];
        }
    }
    state_out
}

/// Compute WKV7 with length masking on host data, returning (output, state_out).
/// This is a convenience function for testing.
///
/// # Arguments
/// * `w_decay` - Pre-computed decay data [N, H, T, B] flattened
/// * `q`, `k`, `v`, `a`, `b` - Input data [N, H, T, B] flattened
/// * `state_in` - Input state [N, N, H, B] flattened
/// * `lengths` - Real sequence length per batch [B]
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens (padded)
/// * `batch` - batch size
pub fn hip_wkv7_masked(
    w_decay: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    state_in: &[f32],
    lengths: &[i32],
    n: usize,
    h: usize,
    t: usize,
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let input_len = n * h * t * batch;
    let state_len = n * n * h * batch;

    if w_decay.len() != input_len || q.len() != input_len || k.len() != input_len
        || v.len() != input_len || a.len() != input_len || b.len() != input_len
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got w={}, q={}, k={}, v={}, a={}, b={}",
                input_len, w_decay.len(), q.len(), k.len(), v.len(), a.len(), b.len()
            ),
        });
    }
    if state_in.len() != state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", state_len, state_in.len()),
        });
    }
    if lengths.len() != batch {
        return Err(HipErrorKind {
            code: -1,
            message: format!("lengths size mismatch: expected {}, got {}", batch, lengths.len()),
        });
    }

    let stream = Stream::null();

    let input_shape = TensorShape::new(n, h, t, batch);
    let state_shape = TensorShape::new(n, n, h, batch);
    let lengths_shape = TensorShape::new(batch, 1, 1, 1);

    let d_w_decay = TensorHip::from_slice(w_decay, input_shape, &stream)?;
    let d_q = TensorHip::from_slice(q, input_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, input_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, input_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, input_shape, &stream)?;
    let d_b = TensorHip::from_slice(b, input_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_lengths = TensorHip::from_slice(lengths, lengths_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    wkv7_f32_masked(
        &d_w_decay, &d_q, &d_k, &d_v, &d_a, &d_b,
        &d_state_in, &mut d_output, &mut d_state_out,
        &d_lengths, &stream
    )?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

// ============================================================================
// Elementwise operations for GPU-native forward pass
// ============================================================================

/// Elementwise add: output = a + b
///
/// Both inputs must have the same shape and be contiguous.
pub fn add_f32(
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, output={}",
                a.len(), b.len(), output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "add_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_add_f32(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            a.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Elementwise multiply: output = a * b
///
/// Both inputs must have the same shape and be contiguous.
pub fn mul_f32(
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, output={}",
                a.len(), b.len(), output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "mul_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_mul_f32(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            a.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Negate: output = -input
pub fn negate_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(), output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "negate_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_negate_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Exponential: output = exp(input)
pub fn exp_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(), output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "exp_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_exp_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Broadcast add: output[i] = input[i] + bias[i % bias_len]
///
/// Used for adding per-channel biases to batched data.
/// Input shape: [C, T, B], bias shape: [C], output shape: [C, T, B]
pub fn broadcast_add_f32(
    input: &TensorHip<f32>,
    bias: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(), output.len()
            ),
        });
    }
    if input.len() % bias.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by bias len {}",
                input.len(), bias.len()
            ),
        });
    }
    if !input.is_contiguous() || !bias.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "broadcast_add_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_broadcast_add_f32(
            input.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            bias.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Broadcast multiply: output[i] = input[i] * scale[i % scale_len]
///
/// Used for per-channel scaling (e.g., k * k_k in RWKV7).
/// Input shape: [C, T, B], scale shape: [C], output shape: [C, T, B]
pub fn broadcast_mul_f32(
    input: &TensorHip<f32>,
    scale: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(), output.len()
            ),
        });
    }
    if input.len() % scale.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by scale len {}",
                input.len(), scale.len()
            ),
        });
    }
    if !input.is_contiguous() || !scale.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "broadcast_mul_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_broadcast_mul_f32(
            input.as_ptr(),
            scale.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            scale.len() as c_int,
            stream.handle(),
        ))
    }
}

