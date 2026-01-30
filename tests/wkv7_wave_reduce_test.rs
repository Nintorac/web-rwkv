//! Validate the experimental wave-reduce WKV kernel against the legacy register kernel.
//!
//! Run with:
//!   ROCM_PATH=/opt/rocm cargo test --release --features hip --test wkv7_wave_reduce_test -- --nocapture

#![cfg(feature = "hip")]

use half::f16;
use web_rwkv::hip::{wkv7_f16_masked, wkv7_wave_reduce, Stream, TensorHip, TensorShape};

fn random_f16(len: usize) -> Vec<f16> {
    (0..len)
        .map(|_| f16::from_f32(fastrand::f32() * 2.0 - 1.0))
        .collect()
}

fn random_f32(len: usize) -> Vec<f32> {
    (0..len).map(|_| fastrand::f32() * 2.0 - 1.0).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_rel_error(a: &[f32], b: &[f32]) -> f32 {
    let max_abs_a = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_abs_b = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_abs = max_abs_a.max(max_abs_b);
    if max_abs < 1e-10 {
        return 0.0;
    }
    max_abs_diff(a, b) / max_abs
}

#[test]
fn wave_reduce_matches_register_kernel() {
    let n = 64; // head_size
    let h = 2; // heads
    let t = 3; // tokens
    let b = 2; // batch

    let input_len = n * h * t * b;
    let state_len = n * n * h * b;

    let w_decay = random_f16(input_len);
    let q = random_f16(input_len);
    let k = random_f16(input_len);
    let v = random_f16(input_len);
    let a = random_f16(input_len);
    let b_vec = random_f16(input_len);
    let state_in = random_f32(state_len);
    let lengths = vec![2_i32, 3_i32]; // differing lengths to exercise masking

    let stream = Stream::null();
    let data_shape = TensorShape::new(n, h, t, b);
    let state_shape = TensorShape::new(n, n, h, b);
    let lengths_shape = TensorShape::new(b, 1, 1, 1);

    // Legacy register kernel
    let d_w_decay = TensorHip::from_slice(&w_decay, data_shape, &stream).unwrap();
    let d_q = TensorHip::from_slice(&q, data_shape, &stream).unwrap();
    let d_k = TensorHip::from_slice(&k, data_shape, &stream).unwrap();
    let d_v = TensorHip::from_slice(&v, data_shape, &stream).unwrap();
    let d_a = TensorHip::from_slice(&a, data_shape, &stream).unwrap();
    let d_b = TensorHip::from_slice(&b_vec, data_shape, &stream).unwrap();
    let d_state_in = TensorHip::from_slice(&state_in, state_shape, &stream).unwrap();
    let d_lengths = TensorHip::from_slice(&lengths, lengths_shape, &stream).unwrap();
    let mut d_output = TensorHip::<f16>::new(data_shape).unwrap();
    let mut d_state_out = TensorHip::<f32>::new(state_shape).unwrap();

    wkv7_f16_masked(
        &d_w_decay,
        &d_q,
        &d_k,
        &d_v,
        &d_a,
        &d_b,
        &d_state_in,
        &mut d_output,
        &mut d_state_out,
        &d_lengths,
        &stream,
    )
    .unwrap();

    let output_register: Vec<f16> = d_output.to_vec(&stream).unwrap();
    let state_register: Vec<f32> = d_state_out.to_vec(&stream).unwrap();

    // Wave-reduce kernel
    let d_w_decay2 = TensorHip::from_slice(&w_decay, data_shape, &stream).unwrap();
    let d_q2 = TensorHip::from_slice(&q, data_shape, &stream).unwrap();
    let d_k2 = TensorHip::from_slice(&k, data_shape, &stream).unwrap();
    let d_v2 = TensorHip::from_slice(&v, data_shape, &stream).unwrap();
    let d_a2 = TensorHip::from_slice(&a, data_shape, &stream).unwrap();
    let d_b2 = TensorHip::from_slice(&b_vec, data_shape, &stream).unwrap();
    let d_state_in2 = TensorHip::from_slice(&state_in, state_shape, &stream).unwrap();
    let d_lengths2 = TensorHip::from_slice(&lengths, lengths_shape, &stream).unwrap();
    let mut d_output2 = TensorHip::<f16>::new(data_shape).unwrap();
    let mut d_state_out2 = TensorHip::<f32>::new(state_shape).unwrap();

    wkv7_wave_reduce(
        &d_w_decay2,
        &d_q2,
        &d_k2,
        &d_v2,
        &d_a2,
        &d_b2,
        &d_state_in2,
        &mut d_output2,
        &mut d_state_out2,
        &d_lengths2,
        &stream,
    )
    .unwrap();

    let output_wave: Vec<f16> = d_output2.to_vec(&stream).unwrap();
    let state_wave: Vec<f32> = d_state_out2.to_vec(&stream).unwrap();

    // Compare
    let output_register_f32: Vec<f32> = output_register.iter().map(|x| x.to_f32()).collect();
    let output_wave_f32: Vec<f32> = output_wave.iter().map(|x| x.to_f32()).collect();

    let output_rel_err = max_rel_error(&output_register_f32, &output_wave_f32);
    let state_rel_err = max_rel_error(&state_register, &state_wave);

    let tol = 5e-3;
    assert!(
        output_rel_err < tol,
        "Output mismatch: rel_err={} tol={}",
        output_rel_err,
        tol
    );
    assert!(
        state_rel_err < tol,
        "State mismatch: rel_err={} tol={}",
        state_rel_err,
        tol
    );
}
