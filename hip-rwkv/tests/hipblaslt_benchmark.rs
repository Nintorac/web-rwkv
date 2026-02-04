//! hipBLASLt vs rocBLAS benchmark
//!
//! Run with:
//! ```
//! cargo test --release --test hipblaslt_benchmark -- --nocapture --ignored
//! ```

use std::time::Instant;

use half::f16;

use hip_rwkv::hip::{
    device_synchronize, HipBlasContext, HipBlasLtContext, Stream, TensorHip, TensorShape,
};

/// Test matrix multiply dimensions (typical head layer)
/// vocab_size=65536, n_embd=768, tokens=256
const M: usize = 65536; // vocab_size (output features)
const K: usize = 768; // n_embd (input features)
const N: usize = 256; // tokens (batch * seq_len)

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

#[test]
#[ignore = "requires AMD GPU"]
fn benchmark_head_gemm_rocblas_vs_hipblaslt() {
    // Initialize data with random-ish values
    let weight: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32((i % 1000) as f32 / 1000.0 - 0.5))
        .collect();
    let input: Vec<f16> = (0..K * N)
        .map(|i| f16::from_f32((i % 777) as f32 / 777.0 - 0.5))
        .collect();

    println!("\n=== Head GEMM Benchmark: M={}, N={}, K={} ===\n", M, N, K);
    println!("Matrix dimensions:");
    println!("  Weight: [{}, {}] (vocab_size x n_embd)", M, K);
    println!("  Input:  [{}, {}] (n_embd x tokens)", K, N);
    println!("  Output: [{}, {}] (vocab_size x tokens, F32)", M, N);
    println!();

    // rocBLAS benchmark
    let rocblas_time = benchmark_rocblas(&weight, &input);
    println!(
        "rocBLAS (hgemm_f32_out): {:.3} ms average",
        rocblas_time * 1000.0
    );

    // hipBLASLt benchmark
    match benchmark_hipblaslt(&weight, &input) {
        Ok(hipblaslt_time) => {
            println!(
                "hipBLASLt (hgemm_f32_out): {:.3} ms average",
                hipblaslt_time * 1000.0
            );
            println!();
            let speedup = rocblas_time / hipblaslt_time;
            if speedup > 1.0 {
                println!("hipBLASLt is {:.2}x FASTER than rocBLAS", speedup);
            } else {
                println!("rocBLAS is {:.2}x faster than hipBLASLt", 1.0 / speedup);
            }
        }
        Err(e) => {
            println!("hipBLASLt not available: {}", e);
        }
    }
}

fn benchmark_rocblas(weight: &[f16], input: &[f16]) -> f64 {
    let ctx = HipBlasContext::new().expect("Failed to create rocBLAS context");
    let stream = ctx.stream();

    let weight_shape = TensorShape::new(M, K, 1, 1);
    let input_shape = TensorShape::new(K, N, 1, 1);
    let output_shape = TensorShape::new(M, N, 1, 1);

    let d_weight =
        TensorHip::from_slice(weight, weight_shape, stream).expect("Failed to upload weight");
    let d_input =
        TensorHip::from_slice(input, input_shape, stream).expect("Failed to upload input");
    let mut d_output = TensorHip::<f32>::new(output_shape).expect("Failed to allocate output");

    // Warmup
    for _ in 0..WARMUP_ITERS {
        ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output)
            .expect("rocBLAS GEMM failed");
    }
    device_synchronize().expect("Sync failed");

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output)
            .expect("rocBLAS GEMM failed");
    }
    device_synchronize().expect("Sync failed");
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / BENCH_ITERS as f64
}

fn benchmark_hipblaslt(weight: &[f16], input: &[f16]) -> Result<f64, String> {
    let ctx = HipBlasLtContext::new().map_err(|e| format!("{}", e))?;
    let stream = ctx.stream();

    let weight_shape = TensorShape::new(M, K, 1, 1);
    let input_shape = TensorShape::new(K, N, 1, 1);
    let output_shape = TensorShape::new(M, N, 1, 1);

    let d_weight =
        TensorHip::from_slice(weight, weight_shape, stream).map_err(|e| format!("{}", e))?;
    let d_input =
        TensorHip::from_slice(input, input_shape, stream).map_err(|e| format!("{}", e))?;
    let mut d_output = TensorHip::<f32>::new(output_shape).map_err(|e| format!("{}", e))?;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output)
            .map_err(|e| format!("{}", e))?;
    }
    device_synchronize().map_err(|e| format!("{}", e))?;

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output)
            .map_err(|e| format!("{}", e))?;
    }
    device_synchronize().map_err(|e| format!("{}", e))?;
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / BENCH_ITERS as f64)
}

#[test]
#[ignore = "requires AMD GPU"]
fn verify_hipblaslt_correctness() {
    // Small test to verify results match
    let m = 32;
    let k = 16;
    let n = 8;

    // Weight: identity-like matrix
    let mut weight: Vec<f16> = vec![f16::from_f32(0.0); m * k];
    for i in 0..k.min(m) {
        weight[i * m + i] = f16::from_f32(1.0); // column-major: (i, i) = i * m + i
    }

    // Input: simple sequence
    let input: Vec<f16> = (0..k * n).map(|i| f16::from_f32(i as f32)).collect();

    println!(
        "\n=== Correctness Verification: M={}, N={}, K={} ===\n",
        m, n, k
    );

    // rocBLAS result
    let rocblas_ctx = HipBlasContext::new().expect("Failed to create rocBLAS context");
    let stream = rocblas_ctx.stream();

    let weight_shape = TensorShape::new(m, k, 1, 1);
    let input_shape = TensorShape::new(k, n, 1, 1);
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight =
        TensorHip::from_slice(&weight, weight_shape, stream).expect("Weight upload failed");
    let d_input = TensorHip::from_slice(&input, input_shape, stream).expect("Input upload failed");
    let mut d_output_rocblas = TensorHip::<f32>::new(output_shape).expect("Output alloc failed");

    rocblas_ctx
        .hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output_rocblas)
        .expect("rocBLAS failed");
    let rocblas_result = d_output_rocblas.to_vec(stream).expect("Download failed");

    // hipBLASLt result
    match HipBlasLtContext::new() {
        Ok(hipblaslt_ctx) => {
            let stream = hipblaslt_ctx.stream();
            let d_weight =
                TensorHip::from_slice(&weight, weight_shape, stream).expect("Weight upload failed");
            let d_input =
                TensorHip::from_slice(&input, input_shape, stream).expect("Input upload failed");
            let mut d_output_hipblaslt =
                TensorHip::<f32>::new(output_shape).expect("Output alloc failed");

            match hipblaslt_ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output_hipblaslt)
            {
                Ok(_) => {
                    let hipblaslt_result =
                        d_output_hipblaslt.to_vec(stream).expect("Download failed");

                    // Compare
                    let mut max_diff = 0.0f32;
                    let mut max_diff_idx = 0;
                    for (i, (&r, &h)) in rocblas_result
                        .iter()
                        .zip(hipblaslt_result.iter())
                        .enumerate()
                    {
                        let diff = (r - h).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            max_diff_idx = i;
                        }
                    }

                    println!("Max difference: {} at index {}", max_diff, max_diff_idx);
                    if max_diff < 0.01 {
                        println!("PASS: Results match within tolerance");
                    } else {
                        println!("FAIL: Results differ significantly");
                    }
                }
                Err(e) => {
                    println!("hipBLASLt GEMM not supported: {}", e);
                }
            }
        }
        Err(e) => {
            println!("hipBLASLt not available: {}", e);
        }
    }
}
