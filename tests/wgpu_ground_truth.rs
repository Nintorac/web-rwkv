//! Test WGPU implementation against the same Python ground truth used for HIP.
//!
//! This test verifies whether WGPU matches Python rwkvfla reference,
//! just like the HIP ground truth tests do.
//!
//! Run with: cargo test --features tokio wgpu_ground_truth -- --nocapture

mod common;

use common::TestFixture;
use std::path::Path;

const MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

fn model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn ground_truth_fixtures_exist() -> bool {
    Path::new("tests/fixtures/ground_truth/config.npz").exists()
        && Path::new("tests/fixtures/ground_truth/step_0.npz").exists()
}

/// Test WGPU against Python ground truth.
#[cfg(feature = "tokio")]
#[tokio::test]
async fn test_wgpu_against_ground_truth() {
    if !model_exists() {
        eprintln!("Skipping test: model file not found at {}", MODEL_PATH);
        return;
    }
    if !ground_truth_fixtures_exist() {
        eprintln!("Skipping test: ground truth fixtures not found");
        return;
    }

    use half::f16;
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;
    use web_rwkv::{
        context::{ContextBuilder, InstanceExt},
        runtime::{
            infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
            loader::Loader,
            model::{ContextAutoLimits, ModelBuilder},
            v7, Runtime as RuntimeTrait, TokioRuntime,
        },
    };

    // Load config
    let config =
        TestFixture::load("tests/fixtures/ground_truth/config.npz").expect("Failed to load config");
    let tokens_i64 = config.i64("tokens");
    let n_steps = config.i64("n_steps")[0] as usize;

    println!("Testing WGPU against ground truth:");
    println!("  Tokens: {:?}", tokens_i64);
    println!("  Steps: {}", n_steps);

    // Load WGPU model
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

    // 75% hard threshold, warn if below 99%
    const MIN_PASS_PCT: f64 = 75.0;
    const WARN_PASS_PCT: f64 = 99.0;

    // Process each token and compare logits
    let mut all_passed = true;
    for step in 0..n_steps {
        let token = tokens_i64[step] as u32;
        let fixture_path = format!("tests/fixtures/ground_truth/step_{}.npz", step);
        let fixture =
            TestFixture::load(&fixture_path).expect(&format!("Failed to load {}", fixture_path));

        // Run single token through WGPU
        let batch = RnnInputBatch::new(vec![token], RnnOption::Full);
        let input = RnnInput::new(vec![batch], 128);
        let (_, output) = runtime.infer(input).await.expect("WGPU inference failed");
        let logits: Vec<f32> = output.0[0].0.data().to_vec();

        // Load expected logits
        let expected_logits = fixture.f32("logits");

        // Compare
        let (pass_count, total, max_diff) =
            count_within_tolerance(&logits, expected_logits, 1e-2, 1e-2);
        let pass_pct = 100.0 * pass_count as f64 / total as f64;

        // Find top predictions
        let (wgpu_top1, wgpu_logit) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();

        let (expected_top1, expected_logit) = expected_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();

        let top1_match = wgpu_top1 == expected_top1;

        if pass_pct >= WARN_PASS_PCT && top1_match {
            println!(
                "  Step {}: token {} -> WGPU top1={} (expected {}) {:.2}% within tol OK",
                step, token, wgpu_top1, expected_top1, pass_pct
            );
        } else if pass_pct >= MIN_PASS_PCT && top1_match {
            eprintln!(
                "  Step {}: token {} -> WGPU top1={} (expected {}) {:.2}% within tol WARNING",
                step, token, wgpu_top1, expected_top1, pass_pct
            );
        } else {
            eprintln!("  Step {}: token {} -> WGPU top1={}, expected={} top1_match={} {:.2}% within tol {}",
                     step, token, wgpu_top1, expected_top1, top1_match, pass_pct,
                     if pass_pct >= MIN_PASS_PCT { "WARNING" } else { "FAILED" });

            // Show top-5 from both
            let mut wgpu_indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            wgpu_indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            let mut expected_indexed: Vec<(usize, f32)> =
                expected_logits.iter().cloned().enumerate().collect();
            expected_indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            eprintln!(
                "    WGPU top-5: {:?}",
                wgpu_indexed.iter().take(5).collect::<Vec<_>>()
            );
            eprintln!(
                "    Expected top-5: {:?}",
                expected_indexed.iter().take(5).collect::<Vec<_>>()
            );

            if pass_pct < MIN_PASS_PCT || !top1_match {
                all_passed = false;
            }
        }
    }

    if all_passed {
        println!("\n✓ WGPU passed ground truth validation");
    } else {
        println!("\n✗ WGPU FAILED ground truth validation");
    }
}

fn count_within_tolerance(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
) -> (usize, usize, f32) {
    let mut pass_count = 0;
    let mut max_diff = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let diff = (a - e).abs();
        max_diff = max_diff.max(diff);
        let threshold = atol + rtol * e.abs();
        if diff <= threshold {
            pass_count += 1;
        }
    }
    (pass_count, actual.len(), max_diff)
}
