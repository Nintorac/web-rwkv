//! Simple WKV profiling binary for rocprof
#![cfg(feature = "hip")]

use web_rwkv::hip::hip_wkv7;

fn random_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 1234567) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

fn main() {
    let n = 64;
    let h = 12;
    let t = 1;
    let b = 256;

    let input_len = n * h * t * b;
    let state_len = n * n * h * b;

    let w_decay = random_f32(input_len);
    let q = random_f32(input_len);
    let k = random_f32(input_len);
    let v = random_f32(input_len);
    let a = random_f32(input_len);
    let b_vec = random_f32(input_len);
    let state_in = random_f32(state_len);

    // Warmup
    let _ = hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);

    // Profile iterations
    for _ in 0..10 {
        let _ = hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
    }

    println!("Done");
}
