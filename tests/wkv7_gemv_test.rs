//! Test WKV7 rocBLAS GEMV implementation against the custom kernel.
//!
//! Run with:
//!   ROCM_PATH=/opt/rocm cargo test --release --features hip --test wkv7_gemv_test -- --nocapture

#![cfg(feature = "hip")]

use std::time::Instant;

/// Generate pseudo-random f32 data for testing using fastrand.
fn random_f32(n: usize) -> Vec<f32> {
    // Using thread-local fastrand doesn't need mutable borrow
    (0..n).map(|_| fastrand::f32() * 2.0 - 1.0).collect()
}

/// Compute max absolute difference between two slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, diff| acc.max(diff))
}

/// Compute relative error: max(|a-b|) / max(|a|, |b|)
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
fn test_wkv7_gemv_correctness() {
    use web_rwkv::hip::{hip_wkv7, hip_wkv7_gemv};

    // Small test case: N=64 (head_size), H=2 (heads), T=4 (tokens), B=2 (batch)
    let n = 64; // head_size (must be 64 for WKV7)
    let h = 2; // number of heads
    let t = 4; // timesteps
    let b = 2; // batch size

    let input_len = n * h * t * b;
    let state_len = n * n * h * b;

    // Generate random inputs
    let w_decay = random_f32(input_len); // Should be in (0, 1) range, but random for testing
    let q = random_f32(input_len);
    let k = random_f32(input_len);
    let v = random_f32(input_len);
    let a = random_f32(input_len);
    let b_vec = random_f32(input_len);
    let state_in = random_f32(state_len);

    // Run original WKV7 kernel
    let (output_orig, state_out_orig) =
        hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b)
            .expect("Original WKV7 failed");

    // Run rocBLAS GEMV WKV7
    let (output_gemv, state_out_gemv) =
        hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b)
            .expect("GEMV WKV7 failed");

    // Compare outputs
    let output_diff = max_abs_diff(&output_orig, &output_gemv);
    let output_rel_err = max_rel_error(&output_orig, &output_gemv);
    let state_diff = max_abs_diff(&state_out_orig, &state_out_gemv);
    let state_rel_err = max_rel_error(&state_out_orig, &state_out_gemv);

    println!("WKV7 Correctness Test (N={}, H={}, T={}, B={})", n, h, t, b);
    println!("  Output max abs diff: {:.6e}", output_diff);
    println!("  Output max rel err:  {:.6e}", output_rel_err);
    println!("  State max abs diff:  {:.6e}", state_diff);
    println!("  State max rel err:   {:.6e}", state_rel_err);

    // Tolerances - rocBLAS GEMV may have different numerical behavior
    let tolerance = 1e-4;
    assert!(
        output_rel_err < tolerance,
        "Output relative error {} exceeds tolerance {}",
        output_rel_err,
        tolerance
    );
    assert!(
        state_rel_err < tolerance,
        "State relative error {} exceeds tolerance {}",
        state_rel_err,
        tolerance
    );

    println!("  PASSED!");
}

#[test]
#[ignore] // Run with --ignored for benchmarking
fn benchmark_wkv7_gemv() {
    use web_rwkv::hip::{hip_wkv7, hip_wkv7_gemv};

    // Production-like test case
    let n = 64; // head_size
    let h = 12; // typical number of heads
    let t = 1; // decode mode (one token)
    let b = 256; // large batch

    let input_len = n * h * t * b;
    let state_len = n * n * h * b;

    // Generate random inputs
    let w_decay = random_f32(input_len);
    let q = random_f32(input_len);
    let k = random_f32(input_len);
    let v = random_f32(input_len);
    let a = random_f32(input_len);
    let b_vec = random_f32(input_len);
    let state_in = random_f32(state_len);

    println!("\nWKV7 Benchmark (N={}, H={}, T={}, B={})", n, h, t, b);
    println!("  State matrices: {} ({}x{} each)", h * b, n, n);
    println!(
        "  State size: {:.2} MB",
        (state_len * 4) as f64 / 1024.0 / 1024.0
    );

    // Warmup
    let _ = hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
    let _ = hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);

    // Benchmark original WKV7
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
    }
    let orig_time = start.elapsed() / iterations;

    // Benchmark GEMV WKV7
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
    }
    let gemv_time = start.elapsed() / iterations;

    println!("  Original WKV7: {:?}", orig_time);
    println!("  rocBLAS GEMV:  {:?}", gemv_time);
    println!(
        "  Speedup: {:.2}x",
        orig_time.as_secs_f64() / gemv_time.as_secs_f64()
    );

    // Verify correctness
    let (output_orig, _) =
        hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b).unwrap();
    let (output_gemv, _) =
        hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b).unwrap();
    let output_rel_err = max_rel_error(&output_orig, &output_gemv);
    println!("  Output relative error: {:.6e}", output_rel_err);
}

#[test]
#[ignore] // Run with --ignored for benchmarking
fn benchmark_wkv7_batch_sweep() {
    use web_rwkv::hip::{hip_wkv7, hip_wkv7_coalesced, hip_wkv7_gemv};

    let n = 64;
    let h = 12;
    let t = 1;

    println!("\nWKV7 Batch Sweep (N={}, H={}, T={})", n, h, t);
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "Batch", "Original", "Coalesced", "GEMV", "Coal/Orig"
    );
    println!("{:-<8} {:-<12} {:-<12} {:-<12} {:-<10}", "", "", "", "", "");

    for b in [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
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
        let _ = hip_wkv7_coalesced(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
        let _ = hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);

        // Benchmark
        let iterations = 50;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hip_wkv7(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
        }
        let orig_time = start.elapsed() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hip_wkv7_coalesced(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
        }
        let coal_time = start.elapsed() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hip_wkv7_gemv(&w_decay, &q, &k, &v, &a, &b_vec, &state_in, n, h, t, b);
        }
        let gemv_time = start.elapsed() / iterations;

        let speedup = coal_time.as_secs_f64() / orig_time.as_secs_f64();

        println!(
            "{:>8} {:>9.2} ms {:>9.2} ms {:>9.2} ms {:>8.2}x",
            b,
            orig_time.as_secs_f64() * 1000.0,
            coal_time.as_secs_f64() * 1000.0,
            gemv_time.as_secs_f64() * 1000.0,
            speedup
        );
    }
}
