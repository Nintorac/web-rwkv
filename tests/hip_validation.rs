//! Limit validation and concurrent access tests for HipRuntime.
//!
//! Section A: Verifies that invalid inputs produce the correct error messages.
//! Section B: Verifies concurrent access to a shared HipRuntime does not panic.

use std::path::Path;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn model_exists() -> bool {
    Path::new("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").exists()
}

/// Load the model and create a HipRuntime with the given config.
#[cfg(feature = "hip")]
fn make_runtime(max_prefill_chunk: usize, batch_size: usize) -> web_rwkv::hip::HipRuntime {
    use web_rwkv::hip::{HipRuntime, HipRuntimeConfig, Rwkv7Hip};

    let model = Rwkv7Hip::load("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st")
        .expect("Failed to load model");
    let config = HipRuntimeConfig::new(max_prefill_chunk, batch_size);
    HipRuntime::with_config(model, config).expect("Failed to create runtime")
}

// ══════════════════════════════════════════════════════════════════════════════
// Section A: Limit Validation Tests
// ══════════════════════════════════════════════════════════════════════════════

/// infer(&[]) must return "Empty batch".
#[test]
#[cfg(feature = "hip")]
fn test_infer_empty_batch_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(256, 1);
    let result = rt.infer(&[]);
    assert!(result.is_err(), "Expected error for empty batch");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Empty batch"),
        "Expected 'Empty batch' in error, got: {}",
        msg
    );
}

/// infer(&[&[1,2], &[3,4]]) with B=1 must fail with batch size exceeded.
/// Goes through prefill path (T>1), which checks batch size.
#[test]
#[cfg(feature = "hip")]
fn test_infer_batch_exceeds_max_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(256, 1);
    // Use T>1 sequences to hit prefill path where batch size is validated
    let result = rt.infer(&[&[1, 2], &[3, 4]]);
    assert!(result.is_err(), "Expected error for batch exceeding max");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Batch size 2 exceeds configured max 1"),
        "Expected 'Batch size 2 exceeds configured max 1' in error, got: {}",
        msg
    );
}

/// infer(&[&[]]) must return "All sequences are empty".
#[test]
#[cfg(feature = "hip")]
fn test_infer_all_empty_sequences_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(256, 1);
    let result = rt.infer(&[&[]]);
    assert!(result.is_err(), "Expected error for all-empty sequences");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("All sequences are empty"),
        "Expected 'All sequences are empty' in error, got: {}",
        msg
    );
}

/// infer_one(&[1,2,3,4,5]) with chunk=4 must fail with effective token count exceeded.
#[test]
#[cfg(feature = "hip")]
fn test_infer_token_count_exceeds_chunk_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(4, 1);
    let result = rt.infer_one(&[1, 2, 3, 4, 5]);
    assert!(result.is_err(), "Expected error for token count exceeding chunk");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("effective token count"),
        "Expected 'effective token count' in error, got: {}",
        msg
    );
}

/// infer(&[&[1,2,3], &[4,5]]) with chunk=4, B=2 must fail on effective token count.
/// Multi-batch prefill: effective = B * max_len = 2 * 3 = 6 > 4.
#[test]
#[cfg(feature = "hip")]
fn test_infer_multi_batch_exceeds_chunk_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(4, 2);
    let result = rt.infer(&[&[1, 2, 3], &[4, 5]]);
    assert!(result.is_err(), "Expected error for multi-batch token count exceeding chunk");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("effective token count"),
        "Expected 'effective token count' in error, got: {}",
        msg
    );
}

/// step(&[&[1], &[2]], None) with B=1 must fail via decode batch size check.
/// Both sequences are T=1, so step() dispatches to decode path.
#[test]
#[cfg(feature = "hip")]
fn test_step_decode_batch_exceeds_max_error() {
    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = make_runtime(256, 1);
    let result = rt.step(&[&[1], &[2]], None);
    assert!(result.is_err(), "Expected error for decode batch exceeding max");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("batch size") && msg.contains("exceeds"),
        "Expected 'batch size' and 'exceeds' in error, got: {}",
        msg
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Section B: Concurrent Access Tests
// ══════════════════════════════════════════════════════════════════════════════

/// 4 threads x 10 iterations of infer_one() on shared Arc<HipRuntime>.
/// Verifies no panic, all logits finite, all calls return Ok.
#[test]
#[cfg(feature = "hip")]
fn test_concurrent_infer_no_panic() {
    use std::sync::Arc;

    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = Arc::new(make_runtime(256, 1));
    let n_threads = 4;
    let n_iters = 10;

    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let rt = Arc::clone(&rt);
            std::thread::spawn(move || {
                for i in 0..n_iters {
                    let result = rt.infer_one(&[1]);
                    match result {
                        Ok(tensor) => {
                            let data = tensor.data();
                            assert!(
                                data.iter().all(|v| v.is_finite()),
                                "Thread {} iter {}: non-finite logit found",
                                tid,
                                i
                            );
                        }
                        Err(e) => {
                            panic!("Thread {} iter {}: infer_one failed: {}", tid, i, e);
                        }
                    }
                }
            })
        })
        .collect();

    for (tid, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|e| {
            panic!("Thread {} panicked: {:?}", tid, e);
        });
    }
}

/// Thread 1: prefill calls (T=4), Thread 2: decode calls (T=1), concurrently.
/// Verifies no panic, no deadlock, both threads complete.
#[test]
#[cfg(feature = "hip")]
fn test_concurrent_mixed_prefill_decode() {
    use std::sync::Arc;

    if !model_exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let rt = Arc::new(make_runtime(256, 1));
    let n_iters = 10;

    // Thread 1: prefill (T=4)
    let rt1 = Arc::clone(&rt);
    let prefill_handle = std::thread::spawn(move || {
        for i in 0..n_iters {
            let result = rt1.infer_one(&[1, 2, 3, 4]);
            match result {
                Ok(tensor) => {
                    let data = tensor.data();
                    assert!(
                        data.iter().all(|v| v.is_finite()),
                        "Prefill thread iter {}: non-finite logit found",
                        i
                    );
                }
                Err(e) => {
                    panic!("Prefill thread iter {}: infer_one failed: {}", i, e);
                }
            }
        }
    });

    // Thread 2: decode (T=1)
    let rt2 = Arc::clone(&rt);
    let decode_handle = std::thread::spawn(move || {
        for i in 0..n_iters {
            let result = rt2.infer_one(&[1]);
            match result {
                Ok(tensor) => {
                    let data = tensor.data();
                    assert!(
                        data.iter().all(|v| v.is_finite()),
                        "Decode thread iter {}: non-finite logit found",
                        i
                    );
                }
                Err(e) => {
                    panic!("Decode thread iter {}: infer_one failed: {}", i, e);
                }
            }
        }
    });

    prefill_handle
        .join()
        .unwrap_or_else(|e| panic!("Prefill thread panicked: {:?}", e));
    decode_handle
        .join()
        .unwrap_or_else(|e| panic!("Decode thread panicked: {:?}", e));
}
