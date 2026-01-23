//! Tests to verify fixture loading infrastructure works correctly.
//!
//! These tests validate that:
//! 1. Generated NPZ fixtures can be loaded from Rust
//! 2. Shape arrays are correctly parsed
//! 3. Data arrays have expected properties

mod common;

use common::{assert_tensors_close, TestFixture};
use std::path::Path;

/// Check if fixtures have been generated
fn fixtures_exist() -> bool {
    Path::new("tests/fixtures/kernels/sigmoid/basic.npz").exists()
}

#[test]
fn test_fixture_load_sigmoid() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated. Run: python scripts/generate_test_fixtures.py");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/basic.npz")
        .expect("Failed to load sigmoid fixture");

    // Verify expected keys exist
    assert!(fixture.contains("input"), "Missing 'input' array");
    assert!(fixture.contains("input_shape"), "Missing 'input_shape' array");
    assert!(fixture.contains("expected"), "Missing 'expected' array");
    assert!(fixture.contains("expected_shape"), "Missing 'expected_shape' array");

    // Verify shape
    let shape = fixture.shape4("input");
    assert_eq!(shape[0], 768, "Expected C=768 as first dimension");
    assert_eq!(shape[2], 1, "Expected third dimension to be 1");
    assert_eq!(shape[3], 1, "Expected fourth dimension to be 1");

    // Verify data sizes match shapes
    let input = fixture.f32("input");
    let expected_size: usize = shape.iter().product();
    assert_eq!(
        input.len(),
        expected_size,
        "Input array size doesn't match shape"
    );

    println!(
        "Sigmoid fixture loaded: {} elements, shape {:?}",
        input.len(),
        shape
    );
}

#[test]
fn test_fixture_load_wkv7() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/single_token.npz")
        .expect("Failed to load WKV7 fixture");

    // WKV7 should have these arrays
    let expected_keys = ["q", "k", "v", "a", "b", "w_decay", "state_in", "expected_output", "expected_state"];
    for key in &expected_keys {
        assert!(
            fixture.contains(key),
            "Missing expected key '{}' in WKV7 fixture",
            key
        );
    }

    // Verify state shape is [N, N, H, B] = [64, 64, 12, 1]
    let state_shape = fixture.shape4("state_in");
    assert_eq!(state_shape, [64, 64, 12, 1], "Unexpected state shape");

    // Verify state has correct number of elements
    let state = fixture.f32("state_in");
    assert_eq!(state.len(), 64 * 64 * 12 * 1, "State array size mismatch");

    println!(
        "WKV7 fixture loaded: state shape {:?}, {} elements",
        state_shape,
        state.len()
    );
}

#[test]
fn test_fixture_load_layer_norm() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/layer_norm/basic.npz")
        .expect("Failed to load layer norm fixture");

    assert!(fixture.contains("input"));
    assert!(fixture.contains("weight"));
    assert!(fixture.contains("bias"));
    assert!(fixture.contains("expected"));

    let input_shape = fixture.shape4("input");
    let weight_shape = fixture.shape4("weight");

    // Weight should be 1D (C, 1, 1, 1)
    assert_eq!(weight_shape[1], 1);
    assert_eq!(weight_shape[2], 1);
    assert_eq!(weight_shape[3], 1);

    println!(
        "LayerNorm fixture loaded: input shape {:?}, weight shape {:?}",
        input_shape, weight_shape
    );
}

#[test]
fn test_tensor_comparison() {
    // Test the comparison utility itself
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0001, 2.0001, 3.0001, 4.0001];

    // Should pass with reasonable tolerance
    assert!(
        assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok(),
        "Should pass with small differences"
    );

    // Should fail with large differences
    let c = vec![1.1, 2.0, 3.0, 4.0];
    assert!(
        assert_tensors_close(&a, &c, 1e-3, 1e-4).is_err(),
        "Should fail with 10% difference"
    );

    // Length mismatch should fail
    let d = vec![1.0, 2.0, 3.0];
    assert!(
        assert_tensors_close(&a, &d, 1e-3, 1e-4).is_err(),
        "Should fail with length mismatch"
    );
}

/// Test decay_exp kernel against Python fixtures.
/// This is an acceptance criteria test for bd-2sh.4.3.
#[test]
#[cfg(feature = "hip")]
fn test_decay_exp_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/decay_exp/basic.npz")
        .expect("Failed to load decay_exp fixture");

    let input = fixture.f32("input");
    let expected = fixture.f32("expected");
    let shape = fixture.shape4("input");

    println!(
        "Testing decay_exp: {} elements, shape {:?}",
        input.len(),
        shape
    );

    // Run kernel
    let output = web_rwkv::hip::hip_decay_exp(input)
        .expect("decay_exp kernel failed");

    // Compare with fixture
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("decay_exp output doesn't match fixture");

    println!("decay_exp fixture test passed: {} elements match", output.len());
}

/// Test decay_exp numerical stability against edge case fixtures.
/// This is an acceptance criteria test for bd-2sh.4.3.
#[test]
#[cfg(feature = "hip")]
fn test_decay_exp_stability_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/decay_exp/stability.npz")
        .expect("Failed to load decay_exp stability fixture");

    let input = fixture.f32("input");
    let expected = fixture.f32("expected");

    println!(
        "Testing decay_exp stability: {} edge case values",
        input.len()
    );

    // Run kernel
    let output = web_rwkv::hip::hip_decay_exp(input)
        .expect("decay_exp stability kernel failed");

    // Verify no NaN/Inf
    for (i, &val) in output.iter().enumerate() {
        assert!(!val.is_nan(), "NaN at index {} (input={})", i, input[i]);
        assert!(!val.is_infinite(), "Inf at index {} (input={})", i, input[i]);
    }

    // Compare with fixture
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("decay_exp stability output doesn't match fixture");

    println!("decay_exp stability fixture test passed");
}

/// Test lerp kernel against Python fixtures.
/// This is an acceptance criteria test for bd-2sh.4.4.
#[test]
#[cfg(feature = "hip")]
fn test_lerp_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/lerp/basic.npz")
        .expect("Failed to load lerp fixture");

    let a = fixture.f32("a");
    let b = fixture.f32("b");
    let t = fixture.f32("t");
    let expected = fixture.f32("expected");
    let shape = fixture.shape4("a");

    println!(
        "Testing lerp: {} elements, shape {:?}",
        a.len(),
        shape
    );

    // Run kernel
    let output = web_rwkv::hip::hip_lerp(a, b, t)
        .expect("lerp kernel failed");

    // Compare with fixture (< 1e-3 relative error per acceptance criteria)
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("lerp output doesn't match fixture");

    println!("lerp fixture test passed: {} elements match", output.len());
}

/// Test sigmoid kernel against Python fixtures.
/// This is an acceptance criteria test for bd-2sh.4.1.
#[test]
#[cfg(feature = "hip")]
fn test_sigmoid_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/basic.npz")
        .expect("Failed to load sigmoid fixture");

    let input = fixture.f32("input");
    let expected = fixture.f32("expected");
    let shape = fixture.shape4("input");

    println!(
        "Testing sigmoid: {} elements, shape {:?}",
        input.len(),
        shape
    );

    // Run kernel
    let output = web_rwkv::hip::hip_sigmoid(input)
        .expect("sigmoid kernel failed");

    // Compare with fixture (< 1e-3 relative error per acceptance criteria)
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("sigmoid output doesn't match fixture");

    println!("sigmoid fixture test passed: {} elements match", output.len());
}

/// Test sigmoid edge cases against Python fixtures.
/// This is an acceptance criteria test for bd-2sh.4.1.
#[test]
#[cfg(feature = "hip")]
fn test_sigmoid_edge_cases_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/edge_cases.npz")
        .expect("Failed to load sigmoid edge_cases fixture");

    let input = fixture.f32("input");
    let expected = fixture.f32("expected");

    println!(
        "Testing sigmoid edge cases: {} elements",
        input.len()
    );

    // Run kernel
    let output = web_rwkv::hip::hip_sigmoid(input)
        .expect("sigmoid edge cases kernel failed");

    // Verify no NaN/Inf
    for (i, &val) in output.iter().enumerate() {
        assert!(!val.is_nan(), "NaN at index {} (input={})", i, input[i]);
        assert!(!val.is_infinite(), "Inf at index {} (input={})", i, input[i]);
        assert!(val >= 0.0 && val <= 1.0, "Value out of [0,1] at index {}: {}", i, val);
    }

    // Compare with fixture
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("sigmoid edge cases output doesn't match fixture");

    println!("sigmoid edge cases fixture test passed");
}

/// Test squared ReLU kernel against Python fixtures.
/// This is an acceptance criteria test for bd-2sh.4.2.
#[test]
#[cfg(feature = "hip")]
fn test_squared_relu_fixture() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let fixture = TestFixture::load("tests/fixtures/kernels/squared_relu/basic.npz")
        .expect("Failed to load squared_relu fixture");

    let input = fixture.f32("input");
    let expected = fixture.f32("expected");
    let shape = fixture.shape4("input");

    println!(
        "Testing squared_relu: {} elements, shape {:?}",
        input.len(),
        shape
    );

    // Run kernel
    let output = web_rwkv::hip::hip_squared_relu(input)
        .expect("squared_relu kernel failed");

    // Compare with fixture (< 1e-3 relative error per acceptance criteria)
    assert_tensors_close(&output, expected, 1e-3, 1e-4)
        .expect("squared_relu output doesn't match fixture");

    println!("squared_relu fixture test passed: {} elements match", output.len());
}

#[test]
fn test_all_kernel_fixtures_loadable() {
    if !fixtures_exist() {
        eprintln!("Skipping test: fixtures not generated");
        return;
    }

    let kernel_fixtures = [
        "kernels/sigmoid/basic.npz",
        "kernels/sigmoid/edge_cases.npz",
        "kernels/squared_relu/basic.npz",
        "kernels/decay_exp/basic.npz",
        "kernels/tanh/basic.npz",
        "kernels/lerp/basic.npz",
        "kernels/layer_norm/basic.npz",
        "kernels/group_norm/basic.npz",
        "kernels/l2_norm/basic.npz",
        "kernels/matmul/gemv.npz",
        "kernels/matmul/gemm_batched.npz",
        "kernels/token_shift/single.npz",
        "kernels/wkv7/single_token.npz",
        "kernels/wkv7/batched.npz",
        "kernels/wkv_bonus/basic.npz",
        "kernels/control_k/basic.npz",
        "kernels/channel_mix_state/basic.npz",
    ];

    let mut loaded = 0;
    for fixture_path in &kernel_fixtures {
        let full_path = format!("tests/fixtures/{}", fixture_path);
        if Path::new(&full_path).exists() {
            match TestFixture::load(&full_path) {
                Ok(f) => {
                    loaded += 1;
                    println!("  OK: {} ({} arrays)", fixture_path, f.data.len());
                }
                Err(e) => {
                    panic!("Failed to load {}: {}", fixture_path, e);
                }
            }
        }
    }

    println!("\nLoaded {} kernel fixtures successfully", loaded);
    assert!(loaded > 0, "No fixtures were loaded");
}
