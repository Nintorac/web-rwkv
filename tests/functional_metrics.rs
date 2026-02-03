//! Functional quality metrics for HIP RWKV7: top-k overlap, Spearman rank correlation.
//!
//! Parameterized by chunk_size to exercise different kernel paths:
//!   chunk_size=1  → FusedT1Wkv (T=1 decode)
//!   chunk_size=2  → FLA (minimal chunked prefill)
//!   chunk_size=7  → FLA (odd size, remainder handling)
//!   chunk_size=14 → FLA (full sequence in one call)

mod common;

use common::TestFixture;
use std::path::Path;
use test_case::test_case;

// ── Quality thresholds ─────────────────────────────────────────────────────

/// Minimum average Spearman rank correlation (full 65k-token distribution).
const SPEARMAN_MIN_AVG: f64 = 0.999;

/// Minimum average top-5 token overlap fraction.
const TOP5_MIN_AVG: f64 = 0.80;

/// Minimum average top-10 token overlap fraction.
const TOP10_MIN_AVG: f64 = 0.70;

// ── Helpers ────────────────────────────────────────────────────────────────

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

/// Load model and run all fixture tokens in chunks of `chunk_size`.
/// Returns per-step logits for comparison against fixtures.
#[cfg(feature = "hip")]
fn run_chunked(chunk_size: usize) -> (Vec<(usize, Vec<f32>)>, usize) {
    let config =
        TestFixture::load("tests/fixtures/ground_truth/config.npz").expect("Failed to load config");
    let tokens_i64 = config.i64("tokens");
    let n_steps = config.i64("n_steps")[0] as usize;
    let n_vocab = 65536;

    use web_rwkv::hip::{HipRuntime, HipRuntimeConfig};
    let model =
        web_rwkv::hip::Rwkv7Hip::load("/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st")
            .expect("Failed to load model");
    let rt_config = HipRuntimeConfig::new(256, 1);
    let model = HipRuntime::with_config(model, rt_config)
        .expect("Failed to configure runtime");

    let mut state: Option<web_rwkv::hip::HipState> = None;
    let mut results = Vec::with_capacity(n_steps);

    let mut pos = 0;
    while pos < n_steps {
        let end = (pos + chunk_size).min(n_steps);
        let tokens: Vec<u32> = (pos..end).map(|i| tokens_i64[i] as u32).collect();

        let (logits, new_state) = model
            .step(&[&tokens], state)
            .unwrap_or_else(|e| panic!("step failed at pos {}: {}", pos, e));
        state = Some(new_state);

        let chunk_len = end - pos;
        for t in 0..chunk_len {
            let start = t * n_vocab;
            let tok_logits = logits[start..start + n_vocab].to_vec();
            results.push((pos + t, tok_logits));
        }
        pos = end;
    }

    (results, n_steps)
}

// ── Parameterized tests ────────────────────────────────────────────────────

#[test_case(1  ; "t1_fused")]
#[test_case(2  ; "t2_fla")]
#[test_case(7  ; "t7_fla")]
#[test_case(14 ; "t14_fla")]
#[cfg(feature = "hip")]
fn test_top1_chunked(chunk_size: usize) {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (results, n_steps) = run_chunked(chunk_size);
    let mut matches = 0;

    for &(step, ref logits) in &results {
        let fixture = TestFixture::load(&format!("tests/fixtures/ground_truth/step_{}.npz", step))
            .expect("load fixture");
        let expected = fixture.f32("logits");

        if top_k_indices(logits, 1)[0] == top_k_indices(expected, 1)[0] {
            matches += 1;
        } else {
            println!(
                "  Step {:2}: chunk_size={} HIP={} expected={} mismatch",
                step, chunk_size, top_k_indices(logits, 1)[0], top_k_indices(expected, 1)[0]
            );
        }
    }

    println!(
        "Top-1 (chunk_size={}): {}/{} ({:.1}%)",
        chunk_size, matches, n_steps,
        100.0 * matches as f64 / n_steps as f64
    );
    assert_eq!(matches, n_steps, "Top-1 mismatch with chunk_size={}", chunk_size);
}

#[test_case(1  ; "t1_fused")]
#[test_case(2  ; "t2_fla")]
#[test_case(7  ; "t7_fla")]
#[test_case(14 ; "t14_fla")]
#[cfg(feature = "hip")]
fn test_top5_chunked(chunk_size: usize) {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (results, n_steps) = run_chunked(chunk_size);
    let mut total_overlap = 0.0;

    for &(step, ref logits) in &results {
        let fixture = TestFixture::load(&format!("tests/fixtures/ground_truth/step_{}.npz", step))
            .expect("load fixture");
        let expected = fixture.f32("logits");
        let overlap = top_k_overlap(&top_k_indices(logits, 5), &top_k_indices(expected, 5));
        total_overlap += overlap;
    }

    let avg = total_overlap / n_steps as f64;
    println!("Top-5 (chunk_size={}): avg {:.1}%", chunk_size, avg * 100.0);
    assert!(avg >= TOP5_MIN_AVG, "Top-5 avg {:.1}% < {:.0}% (chunk_size={})",
        avg * 100.0, TOP5_MIN_AVG * 100.0, chunk_size);
}

#[test_case(1  ; "t1_fused")]
#[test_case(2  ; "t2_fla")]
#[test_case(7  ; "t7_fla")]
#[test_case(14 ; "t14_fla")]
#[cfg(feature = "hip")]
fn test_top10_chunked(chunk_size: usize) {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (results, n_steps) = run_chunked(chunk_size);
    let mut total_overlap = 0.0;

    for &(step, ref logits) in &results {
        let fixture = TestFixture::load(&format!("tests/fixtures/ground_truth/step_{}.npz", step))
            .expect("load fixture");
        let expected = fixture.f32("logits");
        let overlap = top_k_overlap(&top_k_indices(logits, 10), &top_k_indices(expected, 10));
        total_overlap += overlap;
    }

    let avg = total_overlap / n_steps as f64;
    println!("Top-10 (chunk_size={}): avg {:.1}%", chunk_size, avg * 100.0);
    assert!(avg >= TOP10_MIN_AVG, "Top-10 avg {:.1}% < {:.0}% (chunk_size={})",
        avg * 100.0, TOP10_MIN_AVG * 100.0, chunk_size);
}

#[test_case(1  ; "t1_fused")]
#[test_case(2  ; "t2_fla")]
#[test_case(7  ; "t7_fla")]
#[test_case(14 ; "t14_fla")]
#[cfg(feature = "hip")]
fn test_spearman_chunked(chunk_size: usize) {
    if !model_exists() || !fixtures_exist() {
        eprintln!("Skipping: model or fixtures not found");
        return;
    }

    let (results, n_steps) = run_chunked(chunk_size);
    let mut total_rho = 0.0;
    let mut worst_rho = 1.0;

    for &(step, ref logits) in &results {
        let fixture = TestFixture::load(&format!("tests/fixtures/ground_truth/step_{}.npz", step))
            .expect("load fixture");
        let expected = fixture.f32("logits");

        let rho = spearman_correlation(logits, expected);
        total_rho += rho;
        if rho < worst_rho {
            worst_rho = rho;
        }

        let status = if rho >= SPEARMAN_MIN_AVG { "ok" } else { "FAIL" };
        println!("  Step {:2}: chunk_size={} rho={:.6} {}", step, chunk_size, rho, status);
    }

    let avg = total_rho / n_steps as f64;
    println!("Spearman (chunk_size={}): avg={:.6} min={:.6}", chunk_size, avg, worst_rho);
    assert!(avg >= SPEARMAN_MIN_AVG, "Spearman avg {:.6} < {:.3} (chunk_size={})",
        avg, SPEARMAN_MIN_AVG, chunk_size);
}
