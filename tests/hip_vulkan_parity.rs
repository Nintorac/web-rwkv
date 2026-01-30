//! HIP vs Vulkan/WGPU Backend Parity Tests
//!
//! These tests diagnose where and why the HIP and Vulkan backends diverge.
//! Key finding: Token 0 matches perfectly but subsequent tokens diverge,
//! suggesting state accumulation issues.
//!
//! Run with: cargo test --features hip,tokio hip_vulkan_parity -- --nocapture

#![cfg(all(feature = "hip", feature = "tokio"))]

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;

use web_rwkv::{
    context::{ContextBuilder, InstanceExt},
    hip::{HipRuntime, HipState, Rwkv7Hip},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelVersion},
        v7, Runtime as RuntimeTrait, TokioRuntime,
    },
    tensor::{TensorCpu, TensorInit, TensorShape as _},
};

const MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

fn model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
}

/// Statistics for comparing two logit vectors.
#[derive(Debug, Clone)]
struct LogitStats {
    cosine_sim: f64,
    max_diff: f32,
    mean_diff: f64,
    top1_match: bool,
    top1_hip: usize,
    top1_wgpu: usize,
    top10_overlap: usize,
    top100_overlap: usize,
    kl_divergence: f64,
}

impl LogitStats {
    fn compute(hip: &[f32], wgpu: &[f32]) -> Self {
        assert_eq!(hip.len(), wgpu.len());
        let n = hip.len();

        // Raw statistics
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut dot_product = 0.0f64;
        let mut hip_norm_sq = 0.0f64;
        let mut wgpu_norm_sq = 0.0f64;

        for (h, w) in hip.iter().zip(wgpu.iter()) {
            let diff = (h - w).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff as f64;
            dot_product += (*h as f64) * (*w as f64);
            hip_norm_sq += (*h as f64).powi(2);
            wgpu_norm_sq += (*w as f64).powi(2);
        }

        let cosine_sim = dot_product / (hip_norm_sq.sqrt() * wgpu_norm_sq.sqrt());
        let mean_diff = sum_diff / n as f64;

        // Top-k indices
        let mut hip_indexed: Vec<(usize, f32)> = hip.iter().cloned().enumerate().collect();
        let mut wgpu_indexed: Vec<(usize, f32)> = wgpu.iter().cloned().enumerate().collect();
        hip_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        wgpu_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top1_hip = hip_indexed[0].0;
        let top1_wgpu = wgpu_indexed[0].0;
        let top1_match = top1_hip == top1_wgpu;

        let hip_top10: HashSet<usize> = hip_indexed.iter().take(10).map(|(i, _)| *i).collect();
        let wgpu_top10: HashSet<usize> = wgpu_indexed.iter().take(10).map(|(i, _)| *i).collect();
        let top10_overlap = hip_top10.intersection(&wgpu_top10).count();

        let hip_top100: HashSet<usize> = hip_indexed.iter().take(100).map(|(i, _)| *i).collect();
        let wgpu_top100: HashSet<usize> = wgpu_indexed.iter().take(100).map(|(i, _)| *i).collect();
        let top100_overlap = hip_top100.intersection(&wgpu_top100).count();

        // KL divergence
        let hip_probs = stable_softmax(hip);
        let wgpu_probs = stable_softmax(wgpu);
        let mut kl_div = 0.0f64;
        for (p, q) in hip_probs.iter().zip(wgpu_probs.iter()) {
            if *p > 1e-10 && *q > 1e-10 {
                kl_div += (*p as f64) * ((*p as f64).ln() - (*q as f64).ln());
            }
        }

        Self {
            cosine_sim,
            max_diff,
            mean_diff,
            top1_match,
            top1_hip,
            top1_wgpu,
            top10_overlap,
            top100_overlap,
            kl_divergence: kl_div,
        }
    }
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|x| x / sum).collect()
}

/// Test 1: Single token forward - should match perfectly.
///
/// This tests that the basic forward computation is correct without
/// any state accumulation effects.
#[tokio::test]
async fn test_single_token_parity() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: Single Token Forward Parity ===\n");

    // === HIP Backend ===
    let hip_model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model.info.n_vocab;
    let hip_runtime = HipRuntime::new(hip_model, 1);

    // Process single token
    let tokens = vec![42u32]; // Arbitrary token
    let hip_logits = hip_runtime
        .infer_one(&tokens)
        .expect("HIP inference failed");
    let hip_data = hip_logits.data();

    // === WGPU Backend ===
    let file = File::open(MODEL_PATH).expect("Failed to open model file");
    let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
    let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
    let info = Loader::info(&model).expect("Failed to get model info");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await
        .expect("Failed to get WGPU adapter");

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await
        .expect("Failed to create WGPU context");

    let wgpu_model = ModelBuilder::new(&context, model)
        .build_v7()
        .await
        .expect("Failed to build WGPU model");

    let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
    let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    let batch = RnnInputBatch::new(tokens.clone(), RnnOption::Full);
    let input = RnnInput::new(vec![batch], 128);
    let (_input, output) = runtime.infer(input).await.expect("WGPU inference failed");
    let wgpu_data = output[0].0.data();

    // Compare
    let stats = LogitStats::compute(hip_data, wgpu_data);

    println!(
        "Single token ({}): cosine={:.6}, top1={} (HIP={}, WGPU={}), max_diff={:.6e}",
        tokens[0],
        stats.cosine_sim,
        if stats.top1_match { "✓" } else { "✗" },
        stats.top1_hip,
        stats.top1_wgpu,
        stats.max_diff
    );

    // Single token should have very high parity
    assert!(
        stats.cosine_sim > 0.999,
        "Single token cosine should be > 0.999, got {:.6}",
        stats.cosine_sim
    );
    assert!(
        stats.top1_match,
        "Single token should have matching top-1 prediction"
    );
    assert!(
        stats.top10_overlap >= 9,
        "Single token should have >= 9/10 top-10 overlap, got {}",
        stats.top10_overlap
    );

    println!("\n✓ Single token parity test PASSED");
}

/// Test 2: Token-by-token with state reset between calls.
///
/// This isolates whether the issue is in forward computation or state management.
/// By resetting state between each token, we test if each individual forward
/// pass is correct.
#[tokio::test]
async fn test_token_by_token_with_reset() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: Token-by-Token with State Reset ===\n");

    // === HIP Backend ===
    let hip_model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model.info.n_vocab;
    let hip_runtime = HipRuntime::new(hip_model, 1);

    // === WGPU Backend ===
    let file = File::open(MODEL_PATH).expect("Failed to open model file");
    let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
    let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
    let info = Loader::info(&model).expect("Failed to get model info");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await
        .expect("Failed to get WGPU adapter");

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await
        .expect("Failed to create WGPU context");

    let wgpu_model = ModelBuilder::new(&context, model)
        .build_v7()
        .await
        .expect("Failed to build WGPU model");

    let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
    let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    // Test multiple tokens, resetting state each time
    let tokens_to_test = [1u32, 2, 3, 4, 42, 100, 1000];

    for &token in &tokens_to_test {
        // Reset HIP state
        hip_runtime.reset_state();

        // HIP forward
        let hip_logits = hip_runtime
            .infer_one(&[token])
            .expect("HIP inference failed");
        let hip_data = hip_logits.data();

        // WGPU forward (fresh state per call? need to check)
        // Note: TokioRuntime maintains internal state, we need a fresh runtime
        // For this test, we'll create a new runtime each time (expensive but accurate)
        let file2 = File::open(MODEL_PATH).expect("Failed to open model file");
        let data2 = unsafe { Mmap::map(&file2).expect("Failed to mmap model") };
        let model2 = SafeTensors::deserialize(&data2).expect("Failed to deserialize model");
        let wgpu_model2 = ModelBuilder::new(&context, model2)
            .build_v7()
            .await
            .expect("Failed to build WGPU model");
        let bundle2 = v7::Bundle::<f16>::new(wgpu_model2, 1);
        let runtime2: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle2).await);

        let batch = RnnInputBatch::new(vec![token], RnnOption::Full);
        let input = RnnInput::new(vec![batch], 128);
        let (_input, output) = runtime2.infer(input).await.expect("WGPU inference failed");
        let wgpu_data = output[0].0.data();

        let stats = LogitStats::compute(hip_data, wgpu_data);

        let status = if stats.top1_match && stats.cosine_sim > 0.999 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "Token {:5}: {} cosine={:.6}, top1={}/{}, top10={}/10",
            token, status, stats.cosine_sim, stats.top1_hip, stats.top1_wgpu, stats.top10_overlap
        );
    }

    println!("\nToken-by-token with reset test complete.");
}

/// Test 3: Multi-token prefill - process all tokens at once.
///
/// This tests whether multi-token processing differs between backends.
#[tokio::test]
async fn test_multi_token_prefill() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: Multi-Token Prefill ===\n");

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    // === HIP Backend ===
    let hip_model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model.info.n_vocab;
    let hip_runtime = HipRuntime::new(hip_model, 1);

    let hip_logits = hip_runtime
        .infer_one(&tokens)
        .expect("HIP inference failed");
    let hip_data = hip_logits.data();

    // === WGPU Backend ===
    let file = File::open(MODEL_PATH).expect("Failed to open model file");
    let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
    let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
    let info = Loader::info(&model).expect("Failed to get model info");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await
        .expect("Failed to get WGPU adapter");

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await
        .expect("Failed to create WGPU context");

    let wgpu_model = ModelBuilder::new(&context, model)
        .build_v7()
        .await
        .expect("Failed to build WGPU model");

    let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
    let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    let batch = RnnInputBatch::new(tokens.clone(), RnnOption::Full);
    let input = RnnInput::new(vec![batch], 128);
    let (_input, output) = runtime.infer(input).await.expect("WGPU inference failed");
    let wgpu_data = output[0].0.data();

    // Compare per-token
    println!("Tokens: {:?}", tokens);
    println!("Vocab size: {}, Num tokens: {}\n", vocab_size, tokens.len());

    let mut all_top1_match = true;
    for t in 0..tokens.len() {
        let start = t * vocab_size;
        let end = start + vocab_size;
        let hip_slice = &hip_data[start..end];
        let wgpu_slice = &wgpu_data[start..end];

        let stats = LogitStats::compute(hip_slice, wgpu_slice);

        if !stats.top1_match {
            all_top1_match = false;
        }

        let status = if stats.top1_match && stats.cosine_sim > 0.99 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "Token {} ({}): {} cos={:.6}, top1={}/{}, top10={}/10, KL={:.2e}",
            t,
            tokens[t],
            status,
            stats.cosine_sim,
            stats.top1_hip,
            stats.top1_wgpu,
            stats.top10_overlap,
            stats.kl_divergence
        );
    }

    // Summary
    println!("\n=== Summary ===");
    if all_top1_match {
        println!("✓ All tokens have matching top-1 predictions");
    } else {
        println!("✗ Some tokens have divergent top-1 predictions");
    }
}

/// Test 4: Sequential generation - one token at a time with state carried forward.
///
/// This is the critical test for understanding state accumulation issues.
#[tokio::test]
async fn test_sequential_generation_parity() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: Sequential Generation (Token by Token with State) ===\n");

    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4];

    // === HIP Backend ===
    let hip_model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model.info.n_vocab;
    let hip_runtime = HipRuntime::new(hip_model, 1);
    hip_runtime.reset_state();

    // === WGPU Backend ===
    let file = File::open(MODEL_PATH).expect("Failed to open model file");
    let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
    let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
    let info = Loader::info(&model).expect("Failed to get model info");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await
        .expect("Failed to get WGPU adapter");

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await
        .expect("Failed to create WGPU context");

    let wgpu_model = ModelBuilder::new(&context, model)
        .build_v7()
        .await
        .expect("Failed to build WGPU model");

    let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
    let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    println!("Processing tokens one at a time with state carried forward:\n");

    // Track divergence progression
    let mut divergence_started_at: Option<usize> = None;

    for (i, &token) in prompt_tokens.iter().enumerate() {
        // HIP: Process single token (state maintained internally)
        let hip_logits = hip_runtime
            .infer_one(&[token])
            .expect("HIP inference failed");
        let hip_data = hip_logits.data();

        // WGPU: Process single token (state maintained internally)
        let batch = RnnInputBatch::new(vec![token], RnnOption::Full);
        let input = RnnInput::new(vec![batch], 128);
        let (_remaining, output) = runtime.infer(input).await.expect("WGPU inference failed");
        let wgpu_data = output[0].0.data();

        let stats = LogitStats::compute(hip_data, wgpu_data);

        let status = if stats.top1_match && stats.cosine_sim > 0.99 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "Step {:2} (token {:5}): {} cos={:.6}, top1={}/{}, top10={}/10, KL={:.2e}",
            i,
            token,
            status,
            stats.cosine_sim,
            stats.top1_hip,
            stats.top1_wgpu,
            stats.top10_overlap,
            stats.kl_divergence
        );

        // Track when divergence starts
        if !stats.top1_match && divergence_started_at.is_none() {
            divergence_started_at = Some(i);
        }
    }

    println!("\n=== Analysis ===");
    if let Some(step) = divergence_started_at {
        println!(
            "Divergence started at step {} (token {})",
            step, prompt_tokens[step]
        );
        println!(
            "This suggests state accumulation issues starting after {} tokens",
            step
        );
    } else {
        println!("✓ No divergence detected in sequential processing");
    }
}

/// Test 5: Compare prefill vs sequential processing on same backend.
///
/// This tests whether the HIP backend gives the same results when processing
/// tokens all at once vs one at a time.
#[tokio::test]
async fn test_hip_prefill_vs_sequential() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: HIP Prefill vs Sequential Processing ===\n");

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    // === Prefill mode ===
    let hip_model_prefill = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model_prefill.info.n_vocab;
    let runtime_prefill = HipRuntime::new(hip_model_prefill, 1);

    let prefill_logits = runtime_prefill
        .infer_one(&tokens)
        .expect("Prefill inference failed");
    let prefill_data = prefill_logits.data();

    // === Sequential mode ===
    let hip_model_seq = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let runtime_seq = HipRuntime::new(hip_model_seq, 1);

    let mut seq_logits_all: Vec<f32> = Vec::new();
    for &token in &tokens {
        let logits = runtime_seq
            .infer_one(&[token])
            .expect("Sequential inference failed");
        seq_logits_all.extend_from_slice(logits.data());
    }

    // Compare
    println!("Tokens: {:?}\n", tokens);

    let mut all_match = true;
    for t in 0..tokens.len() {
        let start = t * vocab_size;
        let end = start + vocab_size;
        let prefill_slice = &prefill_data[start..end];
        let seq_slice = &seq_logits_all[start..end];

        let stats = LogitStats::compute(prefill_slice, seq_slice);

        let status = if stats.cosine_sim > 0.9999 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "Token {} ({}): {} cos={:.6}, max_diff={:.2e}",
            t, tokens[t], status, stats.cosine_sim, stats.max_diff
        );

        if stats.cosine_sim < 0.9999 {
            all_match = false;
        }
    }

    println!("\n=== Result ===");
    if all_match {
        println!("✓ Prefill and sequential processing produce identical results on HIP");
    } else {
        println!("✗ Prefill and sequential processing differ on HIP");
        println!("  This indicates an issue with the HIP state management");
    }
}

/// Test 6: Deep dive into what happens at the divergence point.
///
/// Process tokens and dump detailed statistics at each step.
#[tokio::test]
async fn test_divergence_deep_dive() {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    println!("\n=== Test: Divergence Deep Dive ===\n");

    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    // HIP prefill
    let hip_model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load HIP model");
    let vocab_size = hip_model.info.n_vocab;
    let hip_runtime = HipRuntime::new(hip_model, 1);

    let hip_logits = hip_runtime
        .infer_one(&tokens)
        .expect("HIP inference failed");
    let hip_data = hip_logits.data();

    // WGPU prefill
    let file = File::open(MODEL_PATH).expect("Failed to open model file");
    let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
    let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
    let info = Loader::info(&model).expect("Failed to get model info");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await
        .expect("Failed to get WGPU adapter");

    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await
        .expect("Failed to create WGPU context");

    let wgpu_model = ModelBuilder::new(&context, model)
        .build_v7()
        .await
        .expect("Failed to build WGPU model");

    let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
    let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    let batch = RnnInputBatch::new(tokens.clone(), RnnOption::Full);
    let input = RnnInput::new(vec![batch], 128);
    let (_input, output) = runtime.infer(input).await.expect("WGPU inference failed");
    let wgpu_data = output[0].0.data();

    // Deep dive into each token
    for t in 0..tokens.len() {
        let start = t * vocab_size;
        let end = start + vocab_size;
        let hip_slice = &hip_data[start..end];
        let wgpu_slice = &wgpu_data[start..end];

        println!("\n========== Token {} (input: {}) ==========", t, tokens[t]);

        // Logit ranges
        let hip_min = hip_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hip_max = hip_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let hip_mean: f32 = hip_slice.iter().sum::<f32>() / hip_slice.len() as f32;

        let wgpu_min = wgpu_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let wgpu_max = wgpu_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let wgpu_mean: f32 = wgpu_slice.iter().sum::<f32>() / wgpu_slice.len() as f32;

        println!("Logit ranges:");
        println!(
            "  HIP:  min={:.4}, max={:.4}, mean={:.4}",
            hip_min, hip_max, hip_mean
        );
        println!(
            "  WGPU: min={:.4}, max={:.4}, mean={:.4}",
            wgpu_min, wgpu_max, wgpu_mean
        );

        // Top-5 predictions
        let mut hip_indexed: Vec<(usize, f32)> = hip_slice.iter().cloned().enumerate().collect();
        let mut wgpu_indexed: Vec<(usize, f32)> = wgpu_slice.iter().cloned().enumerate().collect();
        hip_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        wgpu_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nTop-5 predictions:");
        println!(
            "  HIP:  {:?}",
            hip_indexed
                .iter()
                .take(5)
                .map(|(i, v)| (*i, format!("{:.3}", v)))
                .collect::<Vec<_>>()
        );
        println!(
            "  WGPU: {:?}",
            wgpu_indexed
                .iter()
                .take(5)
                .map(|(i, v)| (*i, format!("{:.3}", v)))
                .collect::<Vec<_>>()
        );

        // Overall stats
        let stats = LogitStats::compute(hip_slice, wgpu_slice);
        println!("\nComparison:");
        println!("  Cosine similarity: {:.6}", stats.cosine_sim);
        println!("  Max difference: {:.6e}", stats.max_diff);
        println!("  Mean difference: {:.6e}", stats.mean_diff);
        println!(
            "  Top-1 match: {} (HIP={}, WGPU={})",
            stats.top1_match, stats.top1_hip, stats.top1_wgpu
        );
        println!("  Top-10 overlap: {}/10", stats.top10_overlap);
        println!("  KL divergence: {:.6e}", stats.kl_divergence);
    }
}
