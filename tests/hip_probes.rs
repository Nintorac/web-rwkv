//! Tests for HIP probe system.
//!
//! Run with: cargo test --features hip-probes hip_probes

#![cfg(feature = "hip-probes")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use web_rwkv::hip::{HipHook, HipProbeBuilder, Rwkv7Hip};

/// Test that probes capture intermediate values during forward pass.
#[test]
fn test_probe_captures_intermediates() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_probe_captures_intermediates: model not found at {}", model_path);
        return;
    }

    // Storage for captured values: (hook, layer) -> first 100 values
    let captured: Arc<Mutex<HashMap<(HipHook, Option<usize>), Vec<f32>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // Register probes for several hook points
    let captured_att_ln = captured.clone();
    let captured_wkv = captured.clone();
    let captured_ffn = captured.clone();
    let captured_head = captured.clone();

    let probes = HipProbeBuilder::new()
        .on(HipHook::PostAttLayerNorm, move |data, ctx| {
            let key = (HipHook::PostAttLayerNorm, ctx.layer);
            let n = data.len().min(100);
            captured_att_ln.lock().unwrap()
                .insert(key, data[..n].to_vec());
        })
        .on(HipHook::PostWkv, move |data, ctx| {
            let key = (HipHook::PostWkv, ctx.layer);
            let n = data.len().min(100);
            captured_wkv.lock().unwrap()
                .insert(key, data[..n].to_vec());
        })
        .on(HipHook::PostFfn, move |data, ctx| {
            let key = (HipHook::PostFfn, ctx.layer);
            let n = data.len().min(100);
            captured_ffn.lock().unwrap()
                .insert(key, data[..n].to_vec());
        })
        .on(HipHook::PostHead, move |data, ctx| {
            let key = (HipHook::PostHead, ctx.layer);
            let n = data.len().min(100);
            captured_head.lock().unwrap()
                .insert(key, data[..n].to_vec());
        })
        .build();

    use web_rwkv::hip::HipRuntimeConfig;
    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model")
        .with_probes(probes);
    let config = HipRuntimeConfig::new(256, 1);
    let model = model.with_config(config).expect("Failed to configure model");

    let n_layer = model.info.n_layer;

    // Run forward pass
    let (_logits, _state) = model.forward(&[&[0, 1, 2]], None)
        .expect("Forward failed");

    // Check captured values
    let captured = captured.lock().unwrap();

    // Should have captured PostAttLayerNorm for all layers
    for layer in 0..n_layer {
        assert!(
            captured.contains_key(&(HipHook::PostAttLayerNorm, Some(layer))),
            "Missing PostAttLayerNorm for layer {}", layer
        );
    }

    // Should have captured PostWkv for all layers
    for layer in 0..n_layer {
        assert!(
            captured.contains_key(&(HipHook::PostWkv, Some(layer))),
            "Missing PostWkv for layer {}", layer
        );
    }

    // Should have captured PostFfn for all layers
    for layer in 0..n_layer {
        assert!(
            captured.contains_key(&(HipHook::PostFfn, Some(layer))),
            "Missing PostFfn for layer {}", layer
        );
    }

    // Should have captured PostHead (no layer)
    assert!(
        captured.contains_key(&(HipHook::PostHead, None)),
        "Missing PostHead probe"
    );

    // Verify captured data is non-trivial (not all zeros)
    let post_head = captured.get(&(HipHook::PostHead, None)).unwrap();
    let has_nonzero = post_head.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "PostHead data is all zeros - something is wrong");

    println!("Captured {} probe points", captured.len());
    println!("PostHead first 10 values: {:?}", &post_head[..10.min(post_head.len())]);
}

/// Test probe context contains correct metadata.
#[test]
fn test_probe_context() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_probe_context: model not found at {}", model_path);
        return;
    }

    // Storage for captured context info
    let contexts: Arc<Mutex<Vec<(HipHook, Option<usize>, usize, usize, Vec<usize>)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let ctx_clone = contexts.clone();
    let probes = HipProbeBuilder::new()
        .on(HipHook::PostAttLayerNorm, move |_data, ctx| {
            ctx_clone.lock().unwrap().push((
                HipHook::PostAttLayerNorm,
                ctx.layer,
                ctx.batch_size,
                ctx.seq_len,
                ctx.shape().to_vec(),
            ));
        })
        .build();

    use web_rwkv::hip::HipRuntimeConfig;
    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model")
        .with_probes(probes);
    let config = HipRuntimeConfig::new(256, 2);
    let model = model.with_config(config).expect("Failed to configure model");

    // Test with batch_size=2, seq_len=4
    let tokens = vec![0u32, 1, 2, 3];
    let (_logits, _state) = model.forward(&[&tokens, &tokens], None)
        .expect("Forward failed");

    let contexts = contexts.lock().unwrap();

    // Should have one context per layer
    assert_eq!(contexts.len(), model.info.n_layer);

    // Check first layer context
    let (hook, layer, batch_size, seq_len, shape) = &contexts[0];
    assert_eq!(*hook, HipHook::PostAttLayerNorm);
    assert_eq!(*layer, Some(0));
    assert_eq!(*batch_size, 2);
    assert_eq!(*seq_len, 4);
    assert_eq!(shape, &[model.info.n_embd, 4, 2]); // [C, T, B]

    println!("Context test passed - captured {} contexts", contexts.len());
}

/// Test that probes work with variable-length forward pass.
#[test]
fn test_probe_with_masked_forward() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_probe_with_masked_forward: model not found at {}", model_path);
        return;
    }

    let captured: Arc<Mutex<HashMap<HipHook, usize>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let cap = captured.clone();
    let probes = HipProbeBuilder::new()
        .on(HipHook::PostWkv, move |_data, _ctx| {
            let mut map = cap.lock().unwrap();
            *map.entry(HipHook::PostWkv).or_insert(0) += 1;
        })
        .build();

    use web_rwkv::hip::HipRuntimeConfig;
    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model")
        .with_probes(probes);
    let config = HipRuntimeConfig::new(256, 2);
    let model = model.with_config(config).expect("Failed to configure model");

    let n_layer = model.info.n_layer;

    // Use forward with variable lengths (new API handles this automatically)
    let seq1 = vec![0u32, 1, 2];       // length 3
    let seq2 = vec![0u32, 1, 2, 3, 4]; // length 5

    let (_logits, _state) = model.forward(&[&seq1, &seq2], None)
        .expect("Forward failed");

    let captured = captured.lock().unwrap();
    let wkv_count = captured.get(&HipHook::PostWkv).copied().unwrap_or(0);

    // Should have one PostWkv call per layer
    assert_eq!(wkv_count, n_layer, "Expected {} PostWkv calls, got {}", n_layer, wkv_count);

    println!("Masked forward test passed - {} PostWkv calls", wkv_count);
}
