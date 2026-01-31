//! HIP kernel wrappers for RWKV-specific operations.

use half::f16;
use std::ffi::c_int;

use crate::hip::device::Stream;
use crate::hip::ffi::{
    check,
    launch_channel_mix_state_f16,
    launch_channel_mix_state_f16_masked,
    launch_channel_mix_state_f32,
    launch_channel_mix_state_f32_masked,
    launch_control_k_f16,
    launch_control_k_f32,
    launch_copy_f16_to_f32,
    launch_token_shift_f32,
    // WKV7 GEMV operations
    launch_wkv7_gemv,
    HipErrorKind,
    Result,
    RocblasHandle,
};
use crate::hip::tensor::{TensorHip, TensorShape};

/// Convert f16 tensor to f32 tensor on GPU.
///
/// This avoids CPU conversion overhead by performing the f16->f32 cast on the GPU.
/// Useful for logits download where we want f32 output but computation is in f16.
pub fn copy_f16_to_f32(
    input: &TensorHip<f16>,
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
            message: "copy_f16_to_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_copy_f16_to_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
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
                c,
                t,
                output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_out.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("State shape mismatch: expected [{}, 1, 1, 1]", c),
        });
    }
    if mix.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Mix shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                mix.shape()
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

    token_shift_f32(
        &d_x,
        &d_state_in,
        &d_mix,
        &mut d_output,
        &mut d_state_out,
        &stream,
    )?;

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
                x.shape(),
                output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c,
                b,
                state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                x_k.shape()
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

/// Channel-mix state computation with length masking for variable-length sequences.
///
/// Same as `channel_mix_state_f32` but respects per-batch sequence lengths.
/// Only processes tokens [0, lengths[b]) for each batch, so state_out
/// contains x[lengths[b]-1] instead of x[T-1].
///
/// # Arguments
/// * `x` - Input tensor of shape [C, T, B, 1]
/// * `state_in` - Previous state per batch of shape [C, B, 1, 1]
/// * `x_k` - Per-channel mixing factor of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, T, B, 1]
/// * `state_out` - New state per batch of shape [C, B, 1, 1]
/// * `lengths` - GPU tensor of real sequence lengths per batch [B]
/// * `stream` - HIP stream
pub fn channel_mix_state_f32_masked(
    x: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    x_k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    lengths: &TensorHip<i32>,
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
                x.shape(),
                output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c,
                b,
                state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                x_k.shape()
            ),
        });
    }
    if lengths.shape()[0] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Lengths shape mismatch: expected [{}, 1, 1, 1], got {}",
                b,
                lengths.shape()
            ),
        });
    }

    unsafe {
        check(launch_channel_mix_state_f32_masked(
            x.as_ptr(),
            state_in.as_ptr(),
            x_k.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            lengths.as_ptr() as *const c_int,
            c as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

pub fn channel_mix_state_f16(
    x: &TensorHip<f16>,
    state_in: &TensorHip<f16>,
    x_k: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    state_out: &mut TensorHip<f16>,
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
                x.shape(),
                output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c,
                b,
                state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                x_k.shape()
            ),
        });
    }

    unsafe {
        check(launch_channel_mix_state_f16(
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

pub fn channel_mix_state_f16_masked(
    x: &TensorHip<f16>,
    state_in: &TensorHip<f16>,
    x_k: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    state_out: &mut TensorHip<f16>,
    lengths: &TensorHip<i32>,
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
                x.shape(),
                output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c,
                b,
                state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                x_k.shape()
            ),
        });
    }
    if lengths.shape()[0] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Lengths shape mismatch: expected [{}, 1, 1, 1], got {}",
                b,
                lengths.shape()
            ),
        });
    }

    unsafe {
        check(launch_channel_mix_state_f16_masked(
            x.as_ptr(),
            state_in.as_ptr(),
            x_k.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            lengths.as_ptr() as *const c_int,
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
            message: format!(
                "state_in size mismatch: expected {}, got {}",
                c * b,
                state_in.len()
            ),
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

    channel_mix_state_f32(
        &d_x,
        &d_state_in,
        &d_x_k,
        &mut d_output,
        &mut d_state_out,
        &stream,
    )?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
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
                k.shape(),
                output.shape()
            ),
        });
    }
    if a.shape() != k.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "a shape mismatch: expected {}, got {}",
                k.shape(),
                a.shape()
            ),
        });
    }
    if k_a.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "k_a shape mismatch: expected [{}, 1, 1, 1], got {}",
                c,
                k_a.shape()
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

pub fn control_k_f16(
    k_a: &TensorHip<f16>,
    a: &TensorHip<f16>,
    k: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    let c = a.shape()[0];
    let t = a.shape()[1];
    let b = a.shape()[2];

    if k_a.shape()[0] != c || k.shape() != a.shape() || output.shape() != a.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "control_k_f16 shape mismatch: k_a={}, a={}, k={}, output={}",
                k_a.shape(),
                a.shape(),
                k.shape(),
                output.shape()
            ),
        });
    }
    if !k_a.is_contiguous() || !a.is_contiguous() || !k.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "control_k_f16 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_control_k_f16(
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
                expected_len,
                k.len(),
                a.len()
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
        // Skip empty sequences - keep state as zeros
        if lengths[batch_idx] == 0 {
            continue;
        }
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

// ============================================================================
// WKV7 rocBLAS GEMV Implementation
// ============================================================================

/// Compute WKV7 using rocBLAS batched GEMV operations.
///
/// This is an alternative implementation of WKV7 that uses rocBLAS
/// `sgemv_strided_batched` for the matrix-vector operations instead of
/// a custom fused kernel. The algorithm:
///
/// ```text
/// For each timestep t:
///   1. sa = state @ a  (batched GEMV)
///   2. state = state * w + outer(sa, b) + outer(v, k)  (element-wise)
///   3. y = state @ q  (batched GEMV)
/// ```
///
/// # Arguments
/// * `handle` - rocBLAS handle (must be set to the correct stream)
/// * `w_decay` - Pre-computed decay tensor [N, H, T, B]
/// * `q`, `k`, `v`, `a`, `b` - Input tensors [N, H, T, B]
/// * `state_in` - Input state tensor [N, N, H, B]
/// * `output` - Output tensor [N, H, T, B]
/// * `state_out` - Output state tensor [N, N, H, B]
/// * `sa_tmp` - Scratch buffer for intermediate sa computation [N, H, B]
/// * `stream` - HIP stream
pub fn wkv7_gemv_f32(
    handle: RocblasHandle,
    w_decay: &TensorHip<f32>,
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    sa_tmp: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [N, H, T, B]
    let n = w_decay.shape()[0]; // head_size
    let h = w_decay.shape()[1]; // n_heads
    let t = w_decay.shape()[2]; // tokens
    let b_size = w_decay.shape()[3]; // batch

    // Validate input shapes
    let input_shape = w_decay.shape();
    if q.shape() != input_shape
        || k.shape() != input_shape
        || v.shape() != input_shape
        || a.shape() != input_shape
        || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input shape mismatch: w_decay={}, q={}, k={}, v={}, a={}, b={}",
                w_decay.shape(),
                q.shape(),
                k.shape(),
                v.shape(),
                a.shape(),
                b.shape()
            ),
        });
    }

    // Validate output shape
    if output.shape() != input_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                input_shape,
                output.shape()
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
                state_shape,
                state_in.shape()
            ),
        });
    }
    if state_out.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_out shape mismatch: expected {}, got {}",
                state_shape,
                state_out.shape()
            ),
        });
    }

    // Validate scratch buffer shape: [N, H, B]
    let sa_shape = TensorShape::new(n, h, b_size, 1);
    if sa_tmp.shape() != sa_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "sa_tmp shape mismatch: expected {}, got {}",
                sa_shape,
                sa_tmp.shape()
            ),
        });
    }

    // Check contiguity
    if !w_decay.is_contiguous()
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !a.is_contiguous()
        || !b.is_contiguous()
        || !state_in.is_contiguous()
        || !output.is_contiguous()
        || !state_out.is_contiguous()
        || !sa_tmp.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_gemv_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_gemv(
            handle,
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state_in.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            sa_tmp.as_mut_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV7 using rocBLAS GEMV on host data, returning (output, state_out).
/// This is a convenience function for testing.
///
/// # Arguments
/// * `w_decay` - Pre-computed decay data [N, H, T, B] flattened
/// * `q`, `k`, `v`, `a`, `b` - Input data [N, H, T, B] flattened
/// * `state_in` - Input state [N, N, H, B] flattened
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens
/// * `batch` - batch size
pub fn hip_wkv7_gemv(
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
    use crate::hip::blas::{rocblas_create, rocblas_destroy, rocblas_set_stream};

    let input_len = n * h * t * batch;
    let state_len = n * n * h * batch;
    let sa_len = n * h * batch;

    if w_decay.len() != input_len
        || q.len() != input_len
        || k.len() != input_len
        || v.len() != input_len
        || a.len() != input_len
        || b.len() != input_len
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got w={}, q={}, k={}, v={}, a={}, b={}",
                input_len,
                w_decay.len(),
                q.len(),
                k.len(),
                v.len(),
                a.len(),
                b.len()
            ),
        });
    }
    if state_in.len() != state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_in size mismatch: expected {}, got {}",
                state_len,
                state_in.len()
            ),
        });
    }

    let stream = Stream::null();

    // Create rocBLAS handle
    let handle = rocblas_create()?;
    rocblas_set_stream(handle, &stream)?;

    let input_shape = TensorShape::new(n, h, t, batch);
    let state_shape = TensorShape::new(n, n, h, batch);
    let sa_shape = TensorShape::new(n, h, batch, 1);

    let d_w_decay = TensorHip::from_slice(w_decay, input_shape, &stream)?;
    let d_q = TensorHip::from_slice(q, input_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, input_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, input_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, input_shape, &stream)?;
    let d_b = TensorHip::from_slice(b, input_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;
    let mut d_sa_tmp = TensorHip::<f32>::new(sa_shape)?;

    wkv7_gemv_f32(
        handle,
        &d_w_decay,
        &d_q,
        &d_k,
        &d_v,
        &d_a,
        &d_b,
        &d_state_in,
        &mut d_output,
        &mut d_state_out,
        &mut d_sa_tmp,
        &stream,
    )?;

    // Clean up rocBLAS handle
    rocblas_destroy(handle)?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Acceptance Criteria Tests for bd-2sh.4.9 (Token Shift Kernel) ===

    #[test]
    fn test_token_shift_basic() {
        // Test token shift with 2 channels and 3 tokens
        // x shape: [2, 3, 1, 1]
        let x = vec![
            // Token 0: [1.0, 2.0]
            1.0, 2.0, // Token 1: [3.0, 4.0]
            3.0, 4.0, // Token 2: [5.0, 6.0]
            5.0, 6.0,
        ];

        // Initial state: [0.0, 0.0]
        let state_in = vec![0.0, 0.0];

        // Mix factor: 0.5 (blend 50% of previous into current)
        let mix = vec![0.5, 0.5];

        let c = 2;
        let t = 3;

        let (output, state_out) =
            hip_token_shift(&x, &state_in, &mix, c, t).expect("token_shift kernel failed");

        // Formula: output[t] = x[t] + mix * (prev - x[t])
        // Token 0: x[0] + 0.5*(state - x[0]) = 1 + 0.5*(0-1) = [0.5, 1.0]
        // Token 1: x[1] + 0.5*(x[0] - x[1]) = 3 + 0.5*(1-3) = [2.0, 3.0]
        // Token 2: x[2] + 0.5*(x[1] - x[2]) = 5 + 0.5*(3-5) = [4.0, 5.0]
        let expected = vec![
            0.5, 1.0, // Token 0
            2.0, 3.0, // Token 1
            4.0, 5.0, // Token 2
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Output mismatch at index {}: actual={:.6}, expected={:.6}",
                i,
                actual,
                exp
            );
        }

        // State out should be x[last] = [5.0, 6.0]
        assert!(
            (state_out[0] - 5.0).abs() < 1e-5,
            "state_out[0] should be 5.0"
        );
        assert!(
            (state_out[1] - 6.0).abs() < 1e-5,
            "state_out[1] should be 6.0"
        );

        println!("Token shift basic test passed");
    }

    #[test]
    fn test_token_shift_no_mix() {
        // Test with mix=0 (pass through current, no blending)
        let x = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![0.0, 0.0];

        let (output, _) =
            hip_token_shift(&x, &state_in, &mix, 2, 2).expect("token_shift no mix failed");

        // With mix=0: output = x + 0*(prev - x) = x
        // So output equals x directly
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                actual,
                exp
            );
        }
        println!("Token shift no mix test passed");
    }

    #[test]
    fn test_token_shift_full_mix() {
        // Test with mix=1 (full blending with previous)
        let x = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![1.0, 1.0];

        let (output, _) =
            hip_token_shift(&x, &state_in, &mix, 2, 2).expect("token_shift full mix failed");

        // With mix=1: output = x + 1*(prev - x) = prev
        // Token 0: prev = state_in = [10.0, 20.0]
        // Token 1: prev = x[0] = [1.0, 2.0]
        let expected = vec![10.0, 20.0, 1.0, 2.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                actual,
                exp
            );
        }
        println!("Token shift full mix test passed");
    }
}
