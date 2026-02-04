//! HIP kernel wrappers for WKV7 operations.

use half::f16;
use std::ffi::c_int;

use crate::hip::device::Stream;
use crate::hip::ffi::{
    check,
    launch_wkv_bonus_f16,
    launch_wkv_bonus_f32,
    HipErrorKind,
    Result,
};
use crate::hip::tensor::{TensorHip, TensorShape};

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
    let n = r.shape()[0]; // head_size
    let h = r.shape()[1]; // n_heads
    let t = r.shape()[2]; // tokens
    let b = r.shape()[3]; // batch

    // Validate shapes
    if k.shape() != r.shape() || v.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Shape mismatch: r={}, k={}, v={}",
                r.shape(),
                k.shape(),
                v.shape()
            ),
        });
    }
    if output.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                r.shape(),
                output.shape()
            ),
        });
    }
    if r_k.shape()[0] != n || r_k.shape()[1] != h {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "r_k shape mismatch: expected [{}, {}, 1, 1], got {}",
                n,
                h,
                r_k.shape()
            ),
        });
    }
    if !r.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !r_k.is_contiguous()
        || !output.is_contiguous()
    {
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

pub fn wkv_bonus_f16(
    r: &TensorHip<f16>,
    k: &TensorHip<f16>,
    v: &TensorHip<f16>,
    r_k: &TensorHip<f16>,
    output: &mut TensorHip<f16>,
    stream: &Stream,
) -> Result<()> {
    let n = r.shape()[0];
    let h = r.shape()[1];
    let t = r.shape()[2];
    let b = r.shape()[3];

    if !r.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !r_k.is_contiguous()
        || !output.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv_bonus_f16 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv_bonus_f16(
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
    n: usize, // head_size
    h: usize, // n_heads
    t: usize, // tokens
    b: usize, // batch
) -> Result<Vec<f32>> {
    let expected_len = n * h * t * b;
    let rk_len = n * h;

    if r.len() != expected_len || k.len() != expected_len || v.len() != expected_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got r={}, k={}, v={}",
                expected_len,
                r.len(),
                k.len(),
                v.len()
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

/// High-occupancy WKV7 variant using wave-cooperative reductions.
///
/// This kernel stores state in LDS (shared memory) and uses explicit
/// shuffle-based reductions instead of atomics. It should have better
/// occupancy than the original register-based kernel.
///
/// State is updated in-place. The HIP kernel loads state into LDS before
/// writing, so passing the same buffer as both state_in and state_out
/// to the FFI function is safe.
pub fn wkv7_wave_reduce(
    w_decay: &TensorHip<f16>,
    q: &TensorHip<f16>,
    k: &TensorHip<f16>,
    v: &TensorHip<f16>,
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    state: &mut TensorHip<f32>,
    output: &mut TensorHip<f16>,
    lengths: &TensorHip<i32>,
    stream: &Stream,
) -> Result<()> {
    use crate::hip::ffi::launch_wkv7_wave_reduce;

    let n = w_decay.shape()[0];
    let h = w_decay.shape()[1];
    let t = w_decay.shape()[2];
    let b_size = w_decay.shape()[3];

    // Validate shapes
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
                "Shape mismatch: w_decay={:?}, q={:?}, k={:?}, v={:?}, a={:?}, b={:?}",
                w_decay.shape(),
                q.shape(),
                k.shape(),
                v.shape(),
                a.shape(),
                b.shape()
            ),
        });
    }

    let state_shape = TensorShape::new(n, n, h, b_size);
    if state.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected {}, got {}",
                state_shape,
                state.shape()
            ),
        });
    }

    let output_shape = TensorShape::new(n, h, t, b_size);
    if output.shape() != output_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                output_shape,
                output.shape()
            ),
        });
    }

    // Pass state.as_ptr() as state_in and state.as_mut_ptr() as state_out.
    // The kernel loads state into LDS before writing, so aliasing is safe.
    unsafe {
        check(launch_wkv7_wave_reduce(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state.as_ptr(),
            output.as_mut_ptr(),
            state.as_mut_ptr(),
            lengths.as_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Fused WKV7 kernel for T=1 decode.
///
/// Same 256-thread wave-reduce architecture as wave_reduce_t1 but with:
/// - Fused sa + update + y in a single per-row pass (state read once, cached in VGPRs)
/// - Zero barriers between sa/update/y phases
/// - In-place state (single mutable tensor, no separate in/out)
/// - Params as f32 in LDS (no per-use half->float conversion)
pub fn wkv7_fused_t1(
    w_decay: &TensorHip<f16>,
    q: &TensorHip<f16>,
    k: &TensorHip<f16>,
    v: &TensorHip<f16>,
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    state: &mut TensorHip<f32>,  // in-place
    output: &mut TensorHip<f16>,
    lengths: &TensorHip<i32>,
    stream: &Stream,
) -> Result<()> {
    use crate::hip::ffi::launch_wkv7_fused_t1;

    let n = w_decay.shape()[0];
    let h = w_decay.shape()[1];
    let t = w_decay.shape()[2];
    let b_size = w_decay.shape()[3];

    if t != 1 {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_fused_t1 requires T=1".to_string(),
        });
    }

    let input_shape = w_decay.shape();
    if q.shape() != input_shape
        || k.shape() != input_shape
        || v.shape() != input_shape
        || a.shape() != input_shape
        || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: "Input shape mismatch".to_string(),
        });
    }

    let state_shape = TensorShape::new(n, n, h, b_size);
    if state.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected {}, got {}",
                state_shape,
                state.shape()
            ),
        });
    }

    let output_shape = TensorShape::new(n, h, t, b_size);
    if output.shape() != output_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                output_shape,
                output.shape()
            ),
        });
    }

    if lengths.shape().len() != b_size {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "lengths size mismatch: expected {}, got {}",
                b_size,
                lengths.shape().len()
            ),
        });
    }

    if !w_decay.is_contiguous()
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !a.is_contiguous()
        || !b.is_contiguous()
        || !state.is_contiguous()
        || !output.is_contiguous()
        || !lengths.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_fused_t1 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_fused_t1(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state.as_mut_ptr(),
            output.as_mut_ptr(),
            lengths.as_ptr(),
            n as c_int,
            h as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}
