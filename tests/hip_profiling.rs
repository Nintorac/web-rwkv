//! HIP profiling test - run with:
//! ```
//! WEB_RWKV_HIP_PROF=1 cargo test --release --features hip,hip-prof --test hip_profiling -- --nocapture --ignored
//! ```

#![cfg(all(feature = "hip", feature = "tokio"))]

use std::path::Path;

use anyhow::Result;

use web_rwkv::hip::{HipRuntime, HipRuntimeConfig, Rwkv7Hip};

const MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

fn model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
}

/// Run decode-only profiling at batch size 256.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_batch_256() -> Result<()> {
    profile_decode(256, 32).await
}

/// Run decode-only profiling at batch size 64 for comparison.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_batch_64() -> Result<()> {
    profile_decode(64, 32).await
}

/// Run decode-only profiling at batch size 16.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_batch_16() -> Result<()> {
    profile_decode(16, 32).await
}

/// Run decode-only profiling at batch size 1.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_batch_1() -> Result<()> {
    profile_decode(1, 128).await
}

async fn profile_decode(batch_size: usize, decode_steps: usize) -> Result<()> {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return Ok(());
    }

    eprintln!(
        "\n=== HIP Decode Profiling: batch_size={}, decode_steps={} ===\n",
        batch_size, decode_steps
    );

    // Load model
    let model = Rwkv7Hip::load(MODEL_PATH)?;
    let info = &model.info;

    eprintln!("Model: {} v7", MODEL_PATH);
    eprintln!(
        "  Vocab: {}, Layers: {}, Embed: {}",
        info.n_vocab, info.n_layer, info.n_embd
    );

    // Create HIP runtime with chunk_size=1 for decode-only workload
    // This enables the fast path that skips GPU→CPU token staging roundtrip
    let config = HipRuntimeConfig::new(1, batch_size);
    let runtime = HipRuntime::with_config(model, config)?;

    // Warmup runs
    eprintln!("\nWarming up (3 iterations)...");
    for _ in 0..3 {
        // Create batch of single tokens
        let sequences: Vec<Vec<u32>> = (0..batch_size).map(|_| vec![1u32]).collect();
        let seq_refs: Vec<&[u32]> = sequences.iter().map(|s| s.as_slice()).collect();
        let _ = runtime.infer(&seq_refs)?;
    }

    // Reset state for timed run
    runtime.reset_state();

    // Timed decode steps
    eprintln!("Running {} decode steps...\n", decode_steps);

    let mut step_times = Vec::with_capacity(decode_steps);

    for step in 0..decode_steps {
        // Create batch of single tokens (decode step)
        let sequences: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| {
                // Use different tokens to avoid any caching effects
                let token = ((step * batch_size + i) % 1000 + 1) as u32;
                vec![token]
            })
            .collect();
        let seq_refs: Vec<&[u32]> = sequences.iter().map(|s| s.as_slice()).collect();

        let step_start = std::time::Instant::now();
        let _ = runtime.infer(&seq_refs)?;
        let step_time = step_start.elapsed();
        step_times.push(step_time);
    }

    // Calculate statistics
    let total_time: std::time::Duration = step_times.iter().sum();
    let total_tokens = batch_size * decode_steps;
    let tokens_per_sec = total_tokens as f64 / total_time.as_secs_f64();

    let step_times_ms: Vec<f64> = step_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    let mean_step_ms = step_times_ms.iter().sum::<f64>() / step_times_ms.len() as f64;
    let min_step_ms = step_times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_step_ms = step_times_ms
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = step_times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_ms = sorted[sorted.len() / 2];
    let p95_idx = (sorted.len() as f64 * 0.95) as usize;
    let p95_ms = sorted[p95_idx.min(sorted.len() - 1)];

    eprintln!("=== Results ===");
    eprintln!("Batch size:      {}", batch_size);
    eprintln!("Decode steps:    {}", decode_steps);
    eprintln!("Total tokens:    {}", total_tokens);
    eprintln!(
        "Total time:      {:.3} ms",
        total_time.as_secs_f64() * 1000.0
    );
    eprintln!("Throughput:      {:.1} tokens/sec", tokens_per_sec);
    eprintln!("");
    eprintln!("Step latency:");
    eprintln!("  Mean:          {:.3} ms", mean_step_ms);
    eprintln!("  Min:           {:.3} ms", min_step_ms);
    eprintln!("  Max:           {:.3} ms", max_step_ms);
    eprintln!("  P50:           {:.3} ms", p50_ms);
    eprintln!("  P95:           {:.3} ms", p95_ms);

    Ok(())
}

/// Run a sweep across multiple batch sizes.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_sweep() -> Result<()> {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return Ok(());
    }

    eprintln!("\n=== HIP Decode Batch Size Sweep ===\n");

    let batch_sizes = [32, 64, 128, 256];
    let decode_steps = 8; // Match benchmark config

    eprintln!("Model: {}", MODEL_PATH);

    eprintln!(
        "\n{:>10} {:>12} {:>12} {:>12}",
        "batch_size", "tok/s", "step_ms", "total_ms"
    );
    eprintln!("{}", "-".repeat(50));

    for &batch_size in &batch_sizes {
        // Build fresh model for each batch size with chunk_size=1 for decode
        let model = Rwkv7Hip::load(MODEL_PATH)?;
        let config = HipRuntimeConfig::new(1, batch_size);
        let runtime = HipRuntime::with_config(model, config)?;

        // Warmup
        for _ in 0..3 {
            let sequences: Vec<Vec<u32>> = (0..batch_size).map(|_| vec![1u32]).collect();
            let seq_refs: Vec<&[u32]> = sequences.iter().map(|s| s.as_slice()).collect();
            let _ = runtime.infer(&seq_refs)?;
        }

        // Reset state
        runtime.reset_state();

        // Timed run
        let start = std::time::Instant::now();
        for step in 0..decode_steps {
            let sequences: Vec<Vec<u32>> = (0..batch_size)
                .map(|i| {
                    let token = ((step * batch_size + i) % 1000 + 1) as u32;
                    vec![token]
                })
                .collect();
            let seq_refs: Vec<&[u32]> = sequences.iter().map(|s| s.as_slice()).collect();
            let _ = runtime.infer(&seq_refs)?;
        }
        let elapsed = start.elapsed();

        let total_tokens = batch_size * decode_steps;
        let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
        let step_ms = elapsed.as_secs_f64() * 1000.0 / decode_steps as f64;
        let total_ms = elapsed.as_secs_f64() * 1000.0;

        eprintln!(
            "{:>10} {:>12.1} {:>12.3} {:>12.1}",
            batch_size, tokens_per_sec, step_ms, total_ms
        );
    }

    eprintln!("\nDone.");
    Ok(())
}
