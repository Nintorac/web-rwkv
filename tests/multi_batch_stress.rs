//! Multi-batch, long-sequence, and stress tests for the HIP backend.
//!
//! Exercises:
//! - B=2 determinism (same input both slots, exact equality)
//! - B=2 quality against ground truth (step API, parameterized chunk)
//! - B=2 mixed-length inference (infer vs infer_one)
//! - Long-sequence stability (no panics, all finite)
//! - Long-sequence determinism (run twice, exact match)
//! - State continuity across chunked prefill (all-at-once vs split)

mod common;

use common::TestFixture;
use std::path::Path;
use test_case::test_case;

// ── Helpers ──────────────────────────────────────────────────────────────

fn model_exists() -> bool {
    Path::new("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").exists()
}

fn fixtures_exist() -> bool {
    Path::new("tests/fixtures/ground_truth/config.npz").exists()
        && Path::new("tests/fixtures/ground_truth/step_0.npz").exists()
}

/// Get top-k indices from logits, sorted by descending logit value.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

/// Calculate overlap between two top-k sets as a fraction.
fn top_k_overlap(actual: &[usize], expected: &[usize]) -> f64 {
    let expected_set: std::collections::HashSet<_> = expected.iter().collect();
    let matches = actual.iter().filter(|i| expected_set.contains(i)).count();
    matches as f64 / expected.len() as f64
}

/// Compute ranks for a slice of values (higher value = lower rank).
/// Handles ties by assigning average rank.
fn compute_ranks(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<_> = values.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute Spearman's rank correlation coefficient between two slices.
fn spearman_correlation(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");
    let n = a.len() as f64;

    let ranks_a = compute_ranks(a);
    let ranks_b = compute_ranks(b);

    let mean_a: f64 = ranks_a.iter().sum::<f64>() / n;
    let mean_b: f64 = ranks_b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..a.len() {
        let da = ranks_a[i] - mean_a;
        let db = ranks_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-10 || var_b < 1e-10 {
        return 1.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Load model and create a HipRuntime with given chunk size and batch size.
#[cfg(feature = "hip")]
fn make_runtime(
    chunk: usize,
    batch: usize,
) -> web_rwkv::hip::HipRuntime {
    use web_rwkv::hip::{HipRuntime, HipRuntimeConfig, Rwkv7Hip};

    let model =
        Rwkv7Hip::load("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st")
            .expect("Failed to load model");
    let config = HipRuntimeConfig::new(chunk, batch);
    HipRuntime::with_config(model, config).expect("Failed to configure runtime")
}

/// Load fixture tokens and n_steps from config.npz.
fn load_fixture_tokens() -> (Vec<u32>, usize) {
    let config =
        TestFixture::load("tests/fixtures/ground_truth/config.npz").expect("Failed to load config");
    let tokens_i64 = config.i64("tokens");
    let n_steps = config.i64("n_steps")[0] as usize;
    let tokens: Vec<u32> = tokens_i64.iter().map(|&t| t as u32).collect();
    (tokens, n_steps)
}

// ── Tests ────────────────────────────────────────────────────────────────

/// B=2 determinism: same 4-token sequence in both batch slots.
/// Both slots must produce exactly equal logits.
#[test]
#[cfg(feature = "hip")]
fn test_multi_batch_determinism() {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, n_steps) = load_fixture_tokens();
    // Take up to 4 tokens from fixtures
    let len = n_steps.min(4);
    let toks = &tokens[..len];

    let rt = make_runtime(256, 2);

    // Feed same tokens to both batch slots via step()
    let (logits, _state) = rt
        .step(&[toks, toks], None)
        .expect("step failed for B=2 determinism");

    let n_vocab = rt.info().n_vocab;
    // Logits layout is batch-sequential: [batch0_t0..batch0_tN, batch1_t0..batch1_tN]
    let total = logits.len();
    assert_eq!(
        total,
        n_vocab * len * 2,
        "Expected {} logits, got {}",
        n_vocab * len * 2,
        total
    );

    // Compare slot 0 vs slot 1 for each token position
    for t in 0..len {
        let offset_0 = t * n_vocab;               // batch 0 starts at 0
        let offset_1 = (len + t) * n_vocab;       // batch 1 starts at len * n_vocab
        let slot0 = &logits[offset_0..offset_0 + n_vocab];
        let slot1 = &logits[offset_1..offset_1 + n_vocab];

        assert_eq!(
            slot0, slot1,
            "Batch slot 0 and 1 logits differ at token position {}",
            t
        );
    }

    println!("test_multi_batch_determinism: PASS (B=2, {} tokens, exact equality)", len);
}

/// B=2 quality against ground truth via step() API, parameterized by chunk size.
/// Feed same token to both batch slots each step; compare each slot against fixtures.
#[test_case(1  ; "chunk1_decode")]
#[test_case(14 ; "chunk14_fla")]
#[cfg(feature = "hip")]
fn test_multi_batch_quality(chunk_size: usize) {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, n_steps) = load_fixture_tokens();
    let n_vocab: usize = 65536;

    // max_prefill_chunk must accommodate batch_size * chunk_size effective tokens
    let rt = make_runtime(chunk_size * 2, 2);

    let mut state: Option<web_rwkv::hip::HipState> = None;
    let mut top1_matches = 0usize;
    let mut total_rho = 0.0f64;

    let mut pos = 0;
    while pos < n_steps {
        let end = (pos + chunk_size).min(n_steps);
        let chunk_tokens: Vec<u32> = (pos..end).map(|i| tokens[i]).collect();

        // Feed same chunk to both batch slots
        let (logits, new_state) = rt
            .step(&[&chunk_tokens, &chunk_tokens], state)
            .unwrap_or_else(|e| panic!("step failed at pos {}: {}", pos, e));
        state = Some(new_state);

        let chunk_len = end - pos;
        for t in 0..chunk_len {
            let step_idx = pos + t;
            let fixture = TestFixture::load(&format!(
                "tests/fixtures/ground_truth/step_{}.npz",
                step_idx
            ))
            .expect("load fixture");
            let expected = fixture.f32("logits");

            // Logits layout is batch-sequential: [batch0_t0..batch0_tN, batch1_t0..batch1_tN]
            let offset_0 = t * n_vocab;
            let offset_1 = (chunk_len + t) * n_vocab;
            let slot0 = &logits[offset_0..offset_0 + n_vocab];
            let slot1 = &logits[offset_1..offset_1 + n_vocab];

            // Both slots must match ground truth (catches asymmetric bugs)
            let expected_top1 = top_k_indices(expected, 1)[0];
            if top_k_indices(slot0, 1)[0] == expected_top1 {
                top1_matches += 1;
            }
            assert_eq!(
                top_k_indices(slot1, 1)[0], expected_top1,
                "Slot1 top-1 mismatch at step {} (chunk_size={})",
                step_idx, chunk_size
            );

            // Spearman against ground truth for both slots
            let rho = spearman_correlation(slot0, expected);
            let rho1 = spearman_correlation(slot1, expected);
            total_rho += rho;
            assert!(
                rho1 >= 0.999,
                "Slot1 Spearman {:.6} < 0.999 at step {} (chunk_size={})",
                rho1, step_idx, chunk_size
            );
        }

        pos = end;
    }

    let avg_rho = total_rho / n_steps as f64;
    println!(
        "multi_batch_quality (chunk={}): top-1={}/{} ({:.1}%), avg_rho={:.6}",
        chunk_size,
        top1_matches,
        n_steps,
        100.0 * top1_matches as f64 / n_steps as f64,
        avg_rho
    );
    assert_eq!(
        top1_matches, n_steps,
        "Top-1 mismatch: {}/{} (chunk_size={})",
        top1_matches, n_steps, chunk_size
    );
    assert!(
        avg_rho >= 0.999,
        "Spearman avg {:.6} < 0.999 (chunk_size={})",
        avg_rho,
        chunk_size
    );
}

/// Mixed-length batched inference: infer(&[&tokens[0..5], &tokens[0..3]])
/// vs two separate infer_one() runs. Compare last-token logits.
#[test]
#[cfg(feature = "hip")]
fn test_multi_batch_mixed_lengths() {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, n_steps) = load_fixture_tokens();
    let long_len = n_steps.min(5);
    let short_len = n_steps.min(3);
    let long_seq = &tokens[..long_len];
    let short_seq = &tokens[..short_len];

    let n_vocab: usize = 65536;

    // Run batched inference with mixed lengths, then drop runtime to free GPU memory
    let (batched_long_last, batched_short_last) = {
        let rt_batched = make_runtime(256, 2);
        let batched_logits = rt_batched
            .infer(&[long_seq, short_seq])
            .expect("batched infer failed");
        let batched_data: &[f32] = &batched_logits;

        let total_tokens = long_len + short_len;
        assert_eq!(
            batched_data.len(),
            n_vocab * total_tokens,
            "Unexpected batched logits length"
        );

        let batched_long_last_offset = (long_len - 1) * n_vocab;
        let batched_short_last_offset = (long_len + short_len - 1) * n_vocab;
        (
            batched_data[batched_long_last_offset..batched_long_last_offset + n_vocab].to_vec(),
            batched_data[batched_short_last_offset..batched_short_last_offset + n_vocab].to_vec(),
        )
        // rt_batched dropped here
    };

    // Run separate single-sequence inferences, dropping each runtime before the next
    let single_long_last = {
        let rt_single = make_runtime(256, 1);
        let single_long_logits = rt_single
            .infer_one(long_seq)
            .expect("single infer_one (long) failed");
        let single_long_data: &[f32] = &single_long_logits;
        let offset = (long_len - 1) * n_vocab;
        single_long_data[offset..offset + n_vocab].to_vec()
        // rt_single dropped here
    };

    let single_short_last = {
        let rt_single2 = make_runtime(256, 1);
        let single_short_logits = rt_single2
            .infer_one(short_seq)
            .expect("single infer_one (short) failed");
        let single_short_data: &[f32] = &single_short_logits;
        let offset = (short_len - 1) * n_vocab;
        single_short_data[offset..offset + n_vocab].to_vec()
        // rt_single2 dropped here
    };

    // Compare long sequence last-token logits
    let long_top5_overlap = top_k_overlap(
        &top_k_indices(&batched_long_last, 5),
        &top_k_indices(&single_long_last, 5),
    );
    let long_rho = spearman_correlation(&batched_long_last, &single_long_last);

    println!(
        "mixed_lengths long (T={}): top-5 overlap={:.2}, rho={:.6}",
        long_len, long_top5_overlap, long_rho
    );
    assert!(
        long_top5_overlap >= 0.8,
        "Long seq top-5 overlap {:.2} < 0.8",
        long_top5_overlap
    );
    assert!(
        long_rho >= 0.99,
        "Long seq Spearman {:.6} < 0.99",
        long_rho
    );

    // Compare short sequence last-token logits
    let short_top5_overlap = top_k_overlap(
        &top_k_indices(&batched_short_last, 5),
        &top_k_indices(&single_short_last, 5),
    );
    let short_rho = spearman_correlation(&batched_short_last, &single_short_last);

    println!(
        "mixed_lengths short (T={}): top-5 overlap={:.2}, rho={:.6}",
        short_len, short_top5_overlap, short_rho
    );
    assert!(
        short_top5_overlap >= 0.8,
        "Short seq top-5 overlap {:.2} < 0.8",
        short_top5_overlap
    );
    assert!(
        short_rho >= 0.99,
        "Short seq Spearman {:.6} < 0.99",
        short_rho
    );

    println!("test_multi_batch_mixed_lengths: PASS");
}

/// Long-sequence stability: 256 tokens (cycling fixture tokens), verify all
/// logits are finite and non-zero. Marked #[ignore] because it is slow.
#[test]
#[ignore]
#[cfg(feature = "hip")]
fn test_long_sequence_no_panic() {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, _) = load_fixture_tokens();
    let target_len: usize = 256;

    // Cycle fixture tokens to reach target length
    let long_tokens: Vec<u32> = tokens.iter().cycle().take(target_len).copied().collect();

    {
        let rt = make_runtime(256, 1);
        let logits = rt.infer_one(&long_tokens).expect("infer_one failed for 256 tokens");
        let logits_data: &[f32] = &logits;

        for (i, &v) in logits_data.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite logit at index {}: {}", i, v);
        }
        let any_nonzero = logits_data.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "All logits are zero for 256-token sequence");
        // rt dropped here
    }

    // Also test chunked-64 calls via step()
    let rt2 = make_runtime(64, 1);
    let mut state: Option<web_rwkv::hip::HipState> = None;
    let mut pos = 0;
    while pos < target_len {
        let end = (pos + 64).min(target_len);
        let chunk = &long_tokens[pos..end];
        let (chunk_logits, new_state) = rt2
            .step(&[chunk], state)
            .unwrap_or_else(|e| panic!("step failed at pos {}: {}", pos, e));
        state = Some(new_state);

        for (i, &v) in chunk_logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Non-finite logit at chunk pos {}, index {}: {}",
                pos, i, v
            );
        }
        let any_nonzero = chunk_logits.iter().any(|&v| v != 0.0);
        assert!(
            any_nonzero,
            "All logits zero at chunk starting pos {}",
            pos
        );
        pos = end;
    }

    println!(
        "test_long_sequence_no_panic: PASS ({} tokens, single + chunked-64)",
        target_len
    );
}

/// Long-sequence determinism: 128 tokens, run twice from reset state,
/// verify exact f32 equality. Marked #[ignore] because it is slow.
#[test]
#[ignore]
#[cfg(feature = "hip")]
fn test_long_sequence_determinism() {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, _) = load_fixture_tokens();
    let target_len: usize = 128;

    let long_tokens: Vec<u32> = tokens.iter().cycle().take(target_len).copied().collect();

    // Run 1
    let logits1 = {
        let rt1 = make_runtime(256, 1);
        let (logits, _state) = rt1
            .step(&[&long_tokens], None)
            .expect("step run 1 failed");
        logits
        // rt1 dropped here
    };

    // Run 2 (fresh runtime, same model)
    let logits2 = {
        let rt2 = make_runtime(256, 1);
        let (logits, _state) = rt2
            .step(&[&long_tokens], None)
            .expect("step run 2 failed");
        logits
        // rt2 dropped here
    };

    assert_eq!(
        logits1.len(),
        logits2.len(),
        "Logit lengths differ: {} vs {}",
        logits1.len(),
        logits2.len()
    );
    assert_eq!(
        logits1, logits2,
        "Long-sequence logits are not deterministic (128 tokens)"
    );

    println!("test_long_sequence_determinism: PASS (128 tokens, exact equality)");
}

/// State continuity across chunks: 14 tokens all-at-once vs split 7+7
/// via step() with state carry. Compare last-token logits.
#[test]
#[cfg(feature = "hip")]
fn test_state_continuity_across_chunks() {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (tokens, n_steps) = load_fixture_tokens();
    let total_len = n_steps.min(14);
    let split = total_len / 2; // 7 if total_len == 14
    let all_tokens = &tokens[..total_len];

    let n_vocab: usize = 65536;

    // All-at-once (drop runtime before creating the next one)
    let all_last = {
        let rt_all = make_runtime(256, 1);
        let (logits_all, _state_all) = rt_all
            .step(&[all_tokens], None)
            .expect("step (all-at-once) failed");
        let offset = (total_len - 1) * n_vocab;
        logits_all[offset..offset + n_vocab].to_vec()
        // rt_all dropped here
    };

    // Split: first half, then second half with state carry
    let split_last = {
        let rt_split = make_runtime(256, 1);
        let (_, state_mid) = rt_split
            .step(&[&all_tokens[..split]], None)
            .expect("step (first half) failed");

        let (logits_second, _state_final) = rt_split
            .step(&[&all_tokens[split..]], Some(state_mid))
            .expect("step (second half) failed");

        let second_len = total_len - split;
        let offset = (second_len - 1) * n_vocab;
        logits_second[offset..offset + n_vocab].to_vec()
        // rt_split dropped here
    };

    // Compare
    let overlap = top_k_overlap(
        &top_k_indices(&all_last, 5),
        &top_k_indices(&split_last, 5),
    );
    let rho = spearman_correlation(&all_last, &split_last);

    println!(
        "state_continuity (T={}, split={}+{}): top-5 overlap={:.2}, rho={:.6}",
        total_len, split, total_len - split, overlap, rho
    );
    assert!(
        overlap >= 0.8,
        "State continuity top-5 overlap {:.2} < 0.8",
        overlap
    );
    assert!(
        rho >= 0.999,
        "State continuity Spearman {:.6} < 0.999",
        rho
    );

    println!("test_state_continuity_across_chunks: PASS");
}
