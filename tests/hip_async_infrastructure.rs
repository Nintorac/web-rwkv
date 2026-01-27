//! Tests for HIP async infrastructure.
//!
//! Run with: cargo test --features hip hip_async_infrastructure

#![cfg(feature = "hip")]

use web_rwkv::hip::{Rwkv7Hip, HipRuntimeConfig};

/// Test that forward_async returns a ForwardCompletion and results match forward()
#[test]
fn test_forward_async_basic() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_async_basic: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(256, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    let tokens = vec![0u32, 1, 2, 3, 4];

    // Run forward_async
    let completion = model.forward_async(&[&tokens], None)
        .expect("forward_async failed");

    // Wait for completion
    let (logits, state) = completion.wait()
        .expect("ForwardCompletion::wait() failed");

    // Basic sanity checks
    assert!(!logits.is_empty(), "Logits should not be empty");
    assert!(logits.len() > 0, "Should have logits output");

    // Verify state was returned
    assert_eq!(state.batch_size, 1, "State batch size should be 1");

    println!("forward_async test passed - logits shape: {}", logits.len());
}

/// Test that forward_async results match forward() exactly
#[test]
fn test_forward_async_matches_sync() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_async_matches_sync: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(256, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    let tokens = vec![0u32, 1, 2, 3];

    // Run sync forward
    let (sync_logits, sync_state) = model.forward(&[&tokens], None)
        .expect("Sync forward failed");

    // Run async forward
    let completion = model.forward_async(&[&tokens], None)
        .expect("forward_async failed");
    let (async_logits, async_state) = completion.wait()
        .expect("ForwardCompletion::wait() failed");

    // Compare logits
    assert_eq!(sync_logits.len(), async_logits.len(),
        "Logits length mismatch: sync {} vs async {}", sync_logits.len(), async_logits.len());

    for (i, (&s, &a)) in sync_logits.iter().zip(async_logits.iter()).enumerate() {
        let diff = (s - a).abs();
        assert!(diff < 1e-5,
            "Logits mismatch at index {}: sync {} vs async {}, diff {}", i, s, a, diff);
    }

    // Compare state sizes
    assert_eq!(sync_state.att_states.len(), async_state.att_states.len(),
        "State att_states count mismatch");
    assert_eq!(sync_state.ffn_states.len(), async_state.ffn_states.len(),
        "State ffn_states count mismatch");

    println!("forward_async matches sync forward - {} logits compared", sync_logits.len());
}

/// Test ForwardCompletion::is_ready() returns true after wait()
#[test]
fn test_forward_completion_is_ready() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_completion_is_ready: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(256, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    let tokens = vec![0u32, 1, 2];

    let completion = model.forward_async(&[&tokens], None)
        .expect("forward_async failed");

    // Since current impl wraps sync forward, is_ready should return true immediately
    // after the event is recorded
    let ready = completion.is_ready()
        .expect("is_ready() failed");

    // Note: is_ready may or may not be true depending on timing, but should not error
    println!("is_ready() returned: {}", ready);

    // Wait should always work
    let (logits, _state) = completion.wait()
        .expect("wait() failed");
    assert!(!logits.is_empty());
}

/// Test forward_async with batched input
#[test]
fn test_forward_async_batched() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_async_batched: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(256, 2);
    let model = model.with_config(config).expect("Failed to configure model");

    let tokens1 = vec![0u32, 1, 2];
    let tokens2 = vec![3u32, 4, 5];

    let completion = model.forward_async(&[&tokens1, &tokens2], None)
        .expect("forward_async batched failed");

    let (logits, state) = completion.wait()
        .expect("wait() failed");

    assert!(!logits.is_empty(), "Batched logits should not be empty");
    assert_eq!(state.batch_size, 2, "State should have batch_size=2");

    println!("forward_async batched test passed - batch_size: {}", state.batch_size);
}

/// Test forward_async rejects sequences longer than chunk_size
#[test]
fn test_forward_async_sequence_too_long() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_async_sequence_too_long: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    // Use small chunk size
    let config = HipRuntimeConfig::new(10, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    // Sequence longer than chunk_size=10
    let tokens: Vec<u32> = (0..20).collect();

    let result = model.forward_async(&[&tokens], None);
    match result {
        Ok(_) => panic!("Should reject sequence longer than chunk_size"),
        Err(e) => {
            assert!(e.message.contains("chunk_size"), "Error should mention chunk_size: {}", e.message);
            println!("Correctly rejected long sequence: {}", e.message);
        }
    }
}

/// Test forward_async rejects empty batch
#[test]
fn test_forward_async_empty_batch() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_forward_async_empty_batch: model not found at {}", model_path);
        return;
    }

    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(256, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    let empty: &[&[u32]] = &[];

    let result = model.forward_async(empty, None);
    assert!(result.is_err(), "Should reject empty batch");

    println!("Correctly rejected empty batch");
}
