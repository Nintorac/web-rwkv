//! WGPU profiling test - run with:
//! ```
//! cargo test --release --features wgpu-prof --test wgpu_profiling -- --nocapture --ignored
//! ```
//!
//! For per-operation GPU timing, ensure the `wgpu-prof` feature is enabled.
//! The test will use GPU timestamp queries to measure time spent in each operation
//! (embed, layer, head).

use std::fs::File;

use anyhow::Result;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use web_rwkv::{
    context::ContextBuilder,
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder},
        v7, TokioRuntime,
    },
};
#[cfg(feature = "wgpu-prof")]
use web_rwkv::context::Context;

const MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

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
    eprintln!("\n=== WGPU Decode Profiling: batch_size={}, decode_steps={} ===\n", batch_size, decode_steps);

    // Load model
    let file = File::open(MODEL_PATH)?;
    let data = unsafe { Mmap::map(&file)? };
    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;

    eprintln!("Model: {} v{:?}", MODEL_PATH, info.version);
    eprintln!("  Vocab: {}, Layers: {}, Embed: {}", info.num_vocab, info.num_layer, info.num_emb);

    // Create WGPU context
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("failed to find adapter");

    let adapter_info = adapter.get_info();
    eprintln!("Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);

    #[cfg(feature = "wgpu-prof")]
    {
        let features = adapter.features();
        eprintln!("Timestamp support: {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await?;

    // Build model
    let builder = ModelBuilder::new(&context, model);
    let model = builder.build_v7().await?;

    // Create runtime with specified batch size
    let bundle = v7::Bundle::<f16>::new(model, batch_size);
    let runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;

    // Warmup runs
    eprintln!("\nWarming up (3 iterations)...");
    for _ in 0..3 {
        let batches: Vec<_> = (0..batch_size)
            .map(|_| RnnInputBatch::new(vec![1u32], RnnOption::Last))
            .collect();
        let input = RnnInput::new(batches, 1);  // rounds to 32 (MIN_TOKEN_CHUNK_SIZE)
        let _ = runtime.infer(input).await?;
    }

    // Timed decode steps
    eprintln!("Running {} decode steps...\n", decode_steps);

    let mut step_times = Vec::with_capacity(decode_steps);

    for step in 0..decode_steps {
        // Create batch of single tokens (decode step)
        let batches: Vec<_> = (0..batch_size)
            .map(|i| {
                // Use different tokens to avoid any caching effects
                let token = ((step * batch_size + i) % 1000 + 1) as u32;
                RnnInputBatch::new(vec![token], RnnOption::Last)
            })
            .collect();
        let input = RnnInput::new(batches, 1);  // rounds to 32 (MIN_TOKEN_CHUNK_SIZE)

        let step_start = std::time::Instant::now();
        let _ = runtime.infer(input).await?;
        let step_time = step_start.elapsed();
        step_times.push(step_time);
    }

    // Calculate statistics
    let total_time: std::time::Duration = step_times.iter().sum();
    let total_tokens = batch_size * decode_steps;
    let tokens_per_sec = total_tokens as f64 / total_time.as_secs_f64();

    let step_times_ms: Vec<f64> = step_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let mean_step_ms = step_times_ms.iter().sum::<f64>() / step_times_ms.len() as f64;
    let min_step_ms = step_times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_step_ms = step_times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = step_times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_ms = sorted[sorted.len() / 2];
    let p95_idx = (sorted.len() as f64 * 0.95) as usize;
    let p95_ms = sorted[p95_idx.min(sorted.len() - 1)];

    eprintln!("=== Results ===");
    eprintln!("Batch size:      {}", batch_size);
    eprintln!("Decode steps:    {}", decode_steps);
    eprintln!("Total tokens:    {}", total_tokens);
    eprintln!("Total time:      {:.3} ms", total_time.as_secs_f64() * 1000.0);
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

/// Detailed profiling: measure forward and readback time separately.
/// When the `wgpu-prof` feature is enabled, also shows per-operation GPU timing
/// for embed, layer, and head operations.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_detailed() -> Result<()> {
    use web_rwkv::runtime::{Dispatcher, Job, JobInput};

    let batch_size = 256usize;
    let decode_steps = 32usize;

    eprintln!("\n=== WGPU Detailed Profiling: batch_size={}, decode_steps={} ===\n", batch_size, decode_steps);

    // Load model
    let file = File::open(MODEL_PATH)?;
    let data = unsafe { Mmap::map(&file)? };
    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;

    eprintln!("Model: {} v{:?}", MODEL_PATH, info.version);
    eprintln!("  Vocab: {}, Layers: {}, Embed: {}", info.num_vocab, info.num_layer, info.num_emb);

    // Create WGPU context
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("failed to find adapter");

    let adapter_info = adapter.get_info();
    eprintln!("Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);

    #[cfg(feature = "wgpu-prof")]
    {
        let features = adapter.features();
        eprintln!("Timestamp support: {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await?;

    // Build model
    let builder = ModelBuilder::new(&context, model);
    let model = builder.build_v7().await?;

    // Create bundle with specified batch size
    let bundle = v7::Bundle::<f16>::new(model, batch_size);

    // Warmup
    eprintln!("\nWarming up (3 iterations)...");
    for _ in 0..3 {
        let batches: Vec<_> = (0..batch_size)
            .map(|_| RnnInputBatch::new(vec![1u32], RnnOption::Last))
            .collect();
        let input = RnnInput::new(batches, 1);
        let rnn_info = (&input).into_iter().next().unwrap();
        let chunk = input.chunk();
        let mut job = bundle.dispatch(rnn_info)?;
        job.load(&chunk)?;
        job.submit();

        // Resolve timestamps after submit (during warmup to initialize profiler)
        #[cfg(feature = "wgpu-prof")]
        resolve_timestamps(&bundle, &context);

        let _ = job.back().await?;

        // Accumulate timestamps after GPU work completes (during warmup)
        #[cfg(feature = "wgpu-prof")]
        accumulate_timestamps(&bundle, &context);
    }

    // Clear profiler totals after warmup
    #[cfg(feature = "wgpu-prof")]
    {
        let profiler = bundle.profiler();
        if let Ok(mut prof) = profiler.lock() {
            prof.clear();
        };
    }

    // Timed decode steps with detailed breakdown
    eprintln!("Running {} decode steps with detailed timing...\n", decode_steps);

    let mut dispatch_times = Vec::with_capacity(decode_steps);
    let mut load_times = Vec::with_capacity(decode_steps);
    let mut submit_times = Vec::with_capacity(decode_steps);
    let mut back_times = Vec::with_capacity(decode_steps);
    let mut total_times = Vec::with_capacity(decode_steps);

    for step in 0..decode_steps {
        let batches: Vec<_> = (0..batch_size)
            .map(|i| {
                let token = ((step * batch_size + i) % 1000 + 1) as u32;
                RnnInputBatch::new(vec![token], RnnOption::Last)
            })
            .collect();
        let input = RnnInput::new(batches, 1);
        let rnn_info = (&input).into_iter().next().unwrap();
        let chunk = input.chunk();

        let step_start = std::time::Instant::now();

        // Dispatch (build GPU commands)
        let dispatch_start = std::time::Instant::now();
        let mut job = bundle.dispatch(rnn_info)?;
        let dispatch_time = dispatch_start.elapsed();

        // Load (CPU work: prepare input)
        let load_start = std::time::Instant::now();
        job.load(&chunk)?;
        let load_time = load_start.elapsed();

        // Submit (queue GPU commands)
        let submit_start = std::time::Instant::now();
        job.submit();
        let submit_time = submit_start.elapsed();

        // Resolve timestamps after submit
        #[cfg(feature = "wgpu-prof")]
        resolve_timestamps(&bundle, &context);

        // Back (wait for GPU + readback)
        let back_start = std::time::Instant::now();
        let _ = job.back().await?;
        let back_time = back_start.elapsed();

        // Accumulate timestamps after GPU work completes
        #[cfg(feature = "wgpu-prof")]
        accumulate_timestamps(&bundle, &context);

        let total_time = step_start.elapsed();

        dispatch_times.push(dispatch_time);
        load_times.push(load_time);
        submit_times.push(submit_time);
        back_times.push(back_time);
        total_times.push(total_time);
    }

    // Calculate statistics
    let calc_stats = |times: &[std::time::Duration]| -> (f64, f64, f64) {
        let times_ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, min, max)
    };

    let (dispatch_mean, dispatch_min, dispatch_max) = calc_stats(&dispatch_times);
    let (load_mean, load_min, load_max) = calc_stats(&load_times);
    let (submit_mean, submit_min, submit_max) = calc_stats(&submit_times);
    let (back_mean, back_min, back_max) = calc_stats(&back_times);
    let (total_mean, total_min, total_max) = calc_stats(&total_times);

    let total_tokens = batch_size * decode_steps;
    let total_time: std::time::Duration = total_times.iter().sum();
    let tokens_per_sec = total_tokens as f64 / total_time.as_secs_f64();

    // Calculate output size for bandwidth
    // Output is f32, shape [vocab_size, batch_size, 1, 1]
    let output_size_bytes = info.num_vocab * batch_size * 4;
    let bandwidth_gb_s = (output_size_bytes as f64 / 1e9) / (back_mean / 1000.0);

    eprintln!("=== Results ===");
    eprintln!("Batch size:      {}", batch_size);
    eprintln!("Decode steps:    {}", decode_steps);
    eprintln!("Throughput:      {:.1} tokens/sec", tokens_per_sec);
    eprintln!("");
    eprintln!("Output size:     {:.2} MB (f32)", output_size_bytes as f64 / 1e6);
    eprintln!("");
    eprintln!("Timing breakdown (mean/min/max ms):");
    eprintln!("  dispatch:      {:.3} / {:.3} / {:.3}", dispatch_mean, dispatch_min, dispatch_max);
    eprintln!("  load:          {:.3} / {:.3} / {:.3}", load_mean, load_min, load_max);
    eprintln!("  submit:        {:.3} / {:.3} / {:.3}", submit_mean, submit_min, submit_max);
    eprintln!("  back:          {:.3} / {:.3} / {:.3}", back_mean, back_min, back_max);
    eprintln!("  total:         {:.3} / {:.3} / {:.3}", total_mean, total_min, total_max);
    eprintln!("");
    eprintln!("Readback bandwidth: {:.2} GB/s", bandwidth_gb_s);

    // Print per-operation GPU timing (wgpu-prof feature)
    #[cfg(feature = "wgpu-prof")]
    {
        eprintln!("");
        let label = format!("b={} t=1 steps={}", batch_size, decode_steps);
        print_profile(&bundle, &label);
    }

    Ok(())
}

/// Resolve timestamps after GPU work submission.
/// Creates a command encoder, resolves the query set, and submits.
#[cfg(feature = "wgpu-prof")]
fn resolve_timestamps(bundle: &v7::Bundle<f16>, context: &Context) {
    let profiler = bundle.profiler();
    let mut encoder = context.device.create_command_encoder(&Default::default());
    if let Ok(prof) = profiler.lock() {
        prof.resolve(&mut encoder);
    }
    context.queue.submit(Some(encoder.finish()));
}

/// Accumulate timestamp results after GPU work completes.
#[cfg(feature = "wgpu-prof")]
fn accumulate_timestamps(bundle: &v7::Bundle<f16>, context: &Context) {
    let profiler = bundle.profiler();
    if let Ok(mut prof) = profiler.lock() {
        prof.accumulate(&context.device);
    };
}

/// Print collected profiling results.
#[cfg(feature = "wgpu-prof")]
fn print_profile(bundle: &v7::Bundle<f16>, context_str: &str) {
    let profiler = bundle.profiler();
    if let Ok(prof) = profiler.lock() {
        prof.print(context_str);
    };
}

/// Run a sweep across multiple batch sizes.
#[tokio::test]
#[ignore = "requires model file and GPU"]
async fn profile_decode_sweep() -> Result<()> {
    eprintln!("\n=== WGPU Decode Batch Size Sweep ===\n");

    let batch_sizes = [1, 4, 16, 64, 128, 256];
    let decode_steps = 32;

    // Load model once
    let file = File::open(MODEL_PATH)?;
    let data = unsafe { Mmap::map(&file)? };
    let model_data = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model_data)?;

    eprintln!("Model: {} v{:?}", MODEL_PATH, info.version);

    // Create WGPU context
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("failed to find adapter");

    let adapter_info = adapter.get_info();
    eprintln!("Adapter: {} ({:?})\n", adapter_info.name, adapter_info.backend);

    eprintln!("{:>10} {:>12} {:>12} {:>12}", "batch_size", "tok/s", "step_ms", "total_ms");
    eprintln!("{}", "-".repeat(50));

    for &batch_size in &batch_sizes {
        // Build fresh model for each batch size
        let context = ContextBuilder::new(adapter.clone())
            .auto_limits(&info)
            .build()
            .await?;

        let model_data = SafeTensors::deserialize(&data)?;
        let builder = ModelBuilder::new(&context, model_data);
        let model = builder.build_v7().await?;
        let bundle = v7::Bundle::<f16>::new(model, batch_size);
        let runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;

        // Warmup
        for _ in 0..3 {
            let batches: Vec<_> = (0..batch_size)
                .map(|_| RnnInputBatch::new(vec![1u32], RnnOption::Last))
                .collect();
            let input = RnnInput::new(batches, 1);  // rounds to 32 (MIN_TOKEN_CHUNK_SIZE)
            let _ = runtime.infer(input).await?;
        }

        // Timed run
        let start = std::time::Instant::now();
        for step in 0..decode_steps {
            let batches: Vec<_> = (0..batch_size)
                .map(|i| {
                    let token = ((step * batch_size + i) % 1000 + 1) as u32;
                    RnnInputBatch::new(vec![token], RnnOption::Last)
                })
                .collect();
            let input = RnnInput::new(batches, 1);  // rounds to 32 (MIN_TOKEN_CHUNK_SIZE)
            let _ = runtime.infer(input).await?;
        }
        let elapsed = start.elapsed();

        let total_tokens = batch_size * decode_steps;
        let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
        let step_ms = elapsed.as_secs_f64() * 1000.0 / decode_steps as f64;
        let total_ms = elapsed.as_secs_f64() * 1000.0;

        eprintln!("{:>10} {:>12.1} {:>12.3} {:>12.1}", batch_size, tokens_per_sec, step_ms, total_ms);
    }

    eprintln!("\nDone.");
    Ok(())
}
