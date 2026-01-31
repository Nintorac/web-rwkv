//! HIP kernel wrappers for elementwise operations.

use half::f16;
use std::ffi::c_int;

use crate::hip::buffer::DeviceBuffer;
use crate::hip::device::Stream;
use crate::hip::ffi::{
    check,
    launch_add_f16,
    launch_add_f32,
    launch_broadcast_add_f16,
    launch_broadcast_add_f32,
    launch_broadcast_mul_f16,
    launch_broadcast_mul_f32,
    launch_copy_f16,
    launch_copy_f32,
    launch_decay_exp_f16,
    launch_decay_exp_f32,
    launch_exp_f16,
    launch_exp_f32,
    launch_lerp_f16,
    launch_lerp_f32,
    launch_mul_f16,
    launch_mul_f32,
    launch_negate_f16,
    launch_negate_f32,
    launch_sigmoid_f16,
    launch_sigmoid_f32,
    launch_softplus_decay_f16,
    launch_softplus_decay_f32,
    launch_squared_relu_f16,
    launch_squared_relu_f32,
    launch_tanh_f16,
    launch_tanh_f32,
    HipErrorKind,
    Result,
};
use crate::hip::tensor::{TensorHip, TensorShape};

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

/// GPU-to-GPU copy for TensorHip<f16>: output = input
pub fn copy_tensor_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "copy_tensor_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_copy_f16(
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

pub fn decay_exp_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "decay_exp_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_decay_exp_f16(
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
                n,
                b.len(),
                t.len(),
                output.len()
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

pub fn lerp_f16(
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    t: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    let n = a.len();
    if b.len() != n || t.len() != n || output.len() != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, t={}, output={}",
                n,
                b.len(),
                t.len(),
                output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !t.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "lerp_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_lerp_f16(
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

pub fn sigmoid_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "sigmoid_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_sigmoid_f16(
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

pub fn squared_relu_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "squared_relu_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_squared_relu_f16(
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

pub fn softplus_decay_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "softplus_decay_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_softplus_decay_f16(
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

pub fn tanh_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
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
            message: "tanh_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_tanh_f16(
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
                a.len(),
                b.len(),
                output.len()
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
                a.len(),
                b.len(),
                output.len()
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
                input.len(),
                output.len()
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
pub fn exp_f32(input: &TensorHip<f32>, output: &mut TensorHip<f32>, stream: &Stream) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
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
                input.len(),
                output.len()
            ),
        });
    }
    if input.len() % bias.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by bias len {}",
                input.len(),
                bias.len()
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
                input.len(),
                output.len()
            ),
        });
    }
    if input.len() % scale.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by scale len {}",
                input.len(),
                scale.len()
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

pub fn add_f16(
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, output={}",
                a.len(),
                b.len(),
                output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "add_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_add_f16(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            a.len() as c_int,
            stream.handle(),
        ))
    }
}

pub fn mul_f16(
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, output={}",
                a.len(),
                b.len(),
                output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "mul_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_mul_f16(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            a.len() as c_int,
            stream.handle(),
        ))
    }
}

pub fn negate_f16(
    input: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "negate_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_negate_f16(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

pub fn exp_f16(input: &TensorHip<f16>, output: &mut TensorHip<f16>, stream: &Stream) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "exp_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_exp_f16(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

pub fn broadcast_add_f16(
    input: &TensorHip<f16>,
    bias: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
            ),
        });
    }
    if input.len() % bias.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by bias len {}",
                input.len(),
                bias.len()
            ),
        });
    }
    if !input.is_contiguous() || !bias.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "broadcast_add_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_broadcast_add_f16(
            input.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            bias.len() as c_int,
            stream.handle(),
        ))
    }
}

pub fn broadcast_mul_f16(
    input: &TensorHip<f16>,
    scale: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
            ),
        });
    }
    if input.len() % scale.len() != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by scale len {}",
                input.len(),
                scale.len()
            ),
        });
    }
    if !input.is_contiguous() || !scale.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "broadcast_mul_f16 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_broadcast_mul_f16(
            input.as_ptr(),
            scale.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            scale.len() as c_int,
            stream.handle(),
        ))
    }
}
