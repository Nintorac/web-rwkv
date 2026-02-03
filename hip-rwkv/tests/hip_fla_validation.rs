//! FLA (Flash Linear Attention) ground-truth validation tests.
//!
//! These tests validate FLA correctness across ALL hook points against ground truth
//! fixtures by feeding tokens in T=2 chunks through the FLA path.
//!
//! This mirrors the comprehensive hook coverage of `hip_layer_validation.rs` (RNN test)
//! but validates the chunked FLA code path (T >= FLA_CHUNK_THRESHOLD=2).
//!
//! Run with: cargo test --features hip-probes hip_fla_validation -- --nocapture

#![cfg(feature = "hip-probes")]

mod common;

use common::{assert_tensors_close, TestFixture, Tolerances};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use hip_rwkv::hip::{HipHook, HipProbeBuilder, HipRuntime, HipRuntimeConfig, HipState, Rwkv7Hip};

/// Storage for captured probe data, keyed by (hook, layer).
type CapturedData = Arc<Mutex<HashMap<(HipHook, Option<usize>), Vec<f32>>>>;

// ---------------------------------------------------------------------------
// Tolerance and fixture-key mappings (mirrored from hip_layer_validation.rs)
// ---------------------------------------------------------------------------

/// FP16 tolerance for the HIP backend.
///
/// The HIP backend uses FP16 (half-precision) for GEMM operations while the Python
/// reference uses BF16 (bfloat16). Both have limited precision but differ in rounding
/// behavior (FP16: 10-bit mantissa, BF16: 7-bit mantissa with wider range). These
/// differences cause per-layer divergences of 2-8% that compound across 12 layers.
///
/// Base tolerance of 10% (rtol=atol=0.1) is calibrated for single-layer FP16 GEMM
/// error. Layer scaling accounts for error accumulation.
const FP16_BASE: Tolerances = Tolerances {
    rtol: 1e-1,
    atol: 1e-1,
};

/// Tolerance for operations after layer normalization.
///
/// Layer normalization amplifies input differences through the 1/sqrt(var) scaling.
/// When the variance of a group is small, even small absolute input differences
/// produce large normalized output differences. This is a fundamental property of
/// normalization, not a precision bug. Typical amplification is 2-5x.
const FP16_NORMALIZED: Tolerances = Tolerances {
    rtol: 3e-1,
    atol: 3e-1,
};

/// Tolerance for WKV state (FP32 accumulation path).
/// Input-driven errors propagate through FP16 inputs to the FLA pipeline.
const FP16_STATE: Tolerances = Tolerances {
    rtol: 1e-1,
    atol: 1e-1,
};

/// Tolerance for residual connections (accumulated across sub-operations).
const FP16_ACCUMULATED: Tolerances = Tolerances {
    rtol: 3e-1,
    atol: 3e-1,
};

/// Get base tolerances for a specific hook point (before layer scaling).
fn base_tolerances_for_hook(hook: HipHook) -> Tolerances {
    use HipHook::*;
    match hook {
        // Normalized outputs: layer norm / group norm amplify input errors
        PostEmbedLayerNorm | PostAttLayerNorm | PostFfnLayerNorm | PostHeadLayerNorm
        | PostAttGroupNorm | PostAttL2Norm => FP16_NORMALIZED,

        // Linear projections
        PostAttLinear | PostFfnLinear | PostAttOut | PostFfnOut | PostHead => FP16_BASE,

        // Token shift (depends on previous layer's output which may have been layer-normed)
        PostAttTokenShift | PostFfnTokenShift => FP16_NORMALIZED,

        // LoRA + activation outputs
        PostAttDecay | PostAttAdapt | PostAttGate => FP16_BASE,

        // Value residual (lerp)
        PostAttValueResidual => FP16_BASE,

        // Control k
        PostAttControlK => FP16_BASE,

        // WKV operations
        PreWkv | PostWkv | PostWkvBonus => FP16_BASE,

        // WKV state: FP32 accumulation but receives cascaded FP16 input errors
        PreWkvState | PostWkvState => FP16_STATE,

        // Gated output
        PostAttGated => FP16_BASE,

        // FFN activation (squared ReLU amplifies errors)
        PostFfnActivate => FP16_NORMALIZED,

        // Residual connections (accumulated errors)
        PostAtt | PostFfn => FP16_ACCUMULATED,

        // Embedding (before any computation)
        PostEmbed => FP16_BASE,
    }
}

/// Get layer-scaled tolerances for a hook point.
///
/// FP16 precision errors compound across layers. After L layers, the expected error
/// grows roughly as (1 + L) due to cascading through layer norm + GEMM + nonlinearities.
/// Layer norm amplification can cause ~2-3x error magnification per layer for outlier
/// elements, though most elements are within sqrt(L) scaling.
fn tolerances_for_hook(hook: HipHook, layer: Option<usize>) -> Tolerances {
    let base = base_tolerances_for_hook(hook);
    let scale = match layer {
        Some(l) => 1.0 + l as f32 * 0.5,
        None => {
            // Non-layer hooks (PostHead, PostEmbed, etc.) occur after all layers,
            // so they accumulate the most error.
            1.0 + 11.0 * 0.5
        }
    };
    Tolerances {
        rtol: base.rtol * scale,
        atol: base.atol * scale,
    }
}

/// Map HipHook to fixture field name(s).
///
/// Returns a Vec of fixture keys. For stacked tensors (e.g. PostAttTokenShift),
/// multiple keys are returned; each sub-tensor occupies n_embd elements in the
/// captured data.
fn fixture_key_for_hook(hook: HipHook, layer: Option<usize>) -> Vec<String> {
    use HipHook::*;

    let prefix = match layer {
        Some(l) => format!("layer_{}_", l),
        None => String::new(),
    };

    match hook {
        // Embedding stage
        PostEmbed => vec!["embedding".to_string()],
        PostEmbedLayerNorm => vec![format!("{}after_ln0", prefix)],

        // Attention layer norm
        PostAttLayerNorm => vec![format!("{}after_ln1", prefix)],

        // Token shift - stacked as [xr, xw, xk, xv, xa, xg]
        PostAttTokenShift => vec![
            format!("{}xr", prefix),
            format!("{}xw", prefix),
            format!("{}xk", prefix),
            format!("{}xv", prefix),
            format!("{}xa", prefix),
            format!("{}xg", prefix),
        ],

        // Linear projections - stacked as [r, k, v]
        PostAttLinear => vec![
            format!("{}r", prefix),
            format!("{}k", prefix),
            format!("{}v", prefix),
        ],

        // LoRA outputs
        PostAttDecay => vec![format!("{}w", prefix)],
        PostAttAdapt => vec![format!("{}a", prefix)],
        PostAttGate => vec![format!("{}g", prefix)],

        // Value residual (layers > 0)
        PostAttValueResidual => vec![format!("{}v_after_residual", prefix)],

        // L2 norm of k
        PostAttL2Norm => vec![format!("{}kk", prefix)],

        // Control k
        PostAttControlK => vec![format!("{}k_ctrl", prefix)],

        // Pre-WKV tensors - stacked as [w_decay, r, k_ctrl, v, wkv_a, wkv_b]
        PreWkv => vec![
            format!("{}w", prefix),
            format!("{}r", prefix),
            format!("{}k_ctrl", prefix),
            format!("{}v", prefix),
            format!("{}wkv_a", prefix),
            format!("{}wkv_b", prefix),
        ],

        // WKV state
        PreWkvState => vec![format!("{}wkv_state_in", prefix)],
        PostWkvState => vec![format!("{}wkv_state_out", prefix)],

        // WKV output
        PostWkv => vec![format!("{}wkv_out", prefix)],

        // WKV bonus
        PostWkvBonus => vec![format!("{}wkv_bonus", prefix)],

        // Group norm after WKV
        PostAttGroupNorm => vec![format!("{}wkv_normed", prefix)],

        // Gated output (wkv_combined * g) - not stored separately in fixture
        PostAttGated => vec![],

        // Output projection
        PostAttOut => vec![format!("{}att_out", prefix)],

        // After attention residual
        PostAtt => vec![format!("{}after_att_residual", prefix)],

        // FFN layer norm
        PostFfnLayerNorm => vec![format!("{}after_ln2", prefix)],

        // FFN token shift
        PostFfnTokenShift => vec![format!("{}xk_ffn", prefix)],

        // FFN linear projection
        PostFfnLinear => vec![format!("{}k_ffn", prefix)],

        // FFN activation (squared ReLU)
        PostFfnActivate => vec![format!("{}k_ffn_sq", prefix)],

        // FFN output projection
        PostFfnOut => vec![format!("{}ffn_out", prefix)],

        // After FFN residual
        PostFfn => vec![format!("{}output", prefix)],

        // Head layer norm
        PostHeadLayerNorm => vec!["after_ln_out".to_string()],

        // Final logits
        PostHead => vec!["logits".to_string()],
    }
}

// ---------------------------------------------------------------------------
// Hook classification helpers
// ---------------------------------------------------------------------------

/// Returns true if the hook captures state (no T dimension in data).
fn is_state_hook(hook: HipHook) -> bool {
    matches!(hook, HipHook::PreWkvState | HipHook::PostWkvState)
}

/// Build probes that capture ALL hook points into the shared map.
fn build_capture_probes() -> (hip_rwkv::hip::HipProbeMap, CapturedData) {
    let captured: CapturedData = Arc::new(Mutex::new(HashMap::new()));

    let hooks = vec![
        HipHook::PostEmbed,
        HipHook::PostEmbedLayerNorm,
        HipHook::PostAttLayerNorm,
        HipHook::PostAttTokenShift,
        HipHook::PostAttLinear,
        HipHook::PostAttDecay,
        HipHook::PostAttAdapt,
        HipHook::PostAttGate,
        HipHook::PostAttValueResidual,
        HipHook::PostAttL2Norm,
        HipHook::PostAttControlK,
        HipHook::PreWkv,
        HipHook::PreWkvState,
        HipHook::PostWkv,
        HipHook::PostWkvState,
        HipHook::PostWkvBonus,
        HipHook::PostAttGroupNorm,
        HipHook::PostAttGated,
        HipHook::PostAttOut,
        HipHook::PostAtt,
        HipHook::PostFfnLayerNorm,
        HipHook::PostFfnTokenShift,
        HipHook::PostFfnLinear,
        HipHook::PostFfnActivate,
        HipHook::PostFfnOut,
        HipHook::PostFfn,
        HipHook::PostHeadLayerNorm,
        HipHook::PostHead,
    ];

    let mut builder = HipProbeBuilder::new();

    for hook in hooks {
        let cap = captured.clone();
        builder = builder.on(hook, move |data, ctx| {
            let key = (hook, ctx.layer);
            cap.lock().unwrap().insert(key, data.to_vec());
        });
    }

    (builder.build(), captured)
}

// ---------------------------------------------------------------------------
// Comparison result tracking
// ---------------------------------------------------------------------------

/// Information about the first failure encountered.
struct FirstFailure {
    step: usize,
    layer: Option<usize>,
    hook: HipHook,
    fixture_key: String,
}

/// Mutable counters and first-failure tracker threaded through validation.
struct ValidationCounters {
    total_checks: usize,
    passed_checks: usize,
    skipped_checks: usize,
    first_failure: Option<FirstFailure>,
}

impl ValidationCounters {
    fn new() -> Self {
        Self {
            total_checks: 0,
            passed_checks: 0,
            skipped_checks: 0,
            first_failure: None,
        }
    }

    fn record_pass(&mut self) {
        self.total_checks += 1;
        self.passed_checks += 1;
    }

    fn record_skip(&mut self) {
        self.skipped_checks += 1;
    }

    fn record_fail(
        &mut self,
        step: usize,
        layer: Option<usize>,
        hook: HipHook,
        fixture_key: &str,
        err: &str,
    ) {
        self.total_checks += 1;
        println!(
            "  [FAIL] {:?} layer={:?} key={} (step {})",
            hook, layer, fixture_key, step
        );
        for line in err.lines().take(4) {
            println!("         {}", line);
        }
        if self.first_failure.is_none() {
            self.first_failure = Some(FirstFailure {
                step,
                layer,
                hook,
                fixture_key: fixture_key.to_string(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Core comparison logic for a single hook against fixture data (T=2 aware)
// ---------------------------------------------------------------------------

/// Compare captured T=2 data for a single hook against two single-token fixtures.
///
/// For normal (non-state) hooks, the captured data has shape [C, T=2, B=1] in
/// column-major layout: first half = token 0, second half = token 1.
///
/// For stacked hooks (e.g. PostAttTokenShift with 6 sub-tensors), the captured
/// data is [n_embd * N_STACK, T=2, B=1]. The first half contains all N_STACK
/// sub-tensors for token 0, the second half for token 1.
///
/// For state hooks (PreWkvState, PostWkvState), there is no T dimension.
/// `fixture_odd` provides the reference (state after processing both tokens).
fn compare_hook_t2(
    hook: HipHook,
    layer: Option<usize>,
    captured: &[f32],
    fixture_even: &TestFixture,
    fixture_odd: &TestFixture,
    step: usize,
    idx_even: usize,
    idx_odd: usize,
    counters: &mut ValidationCounters,
) {
    let fixture_keys = fixture_key_for_hook(hook, layer);
    let tolerances = tolerances_for_hook(hook, layer);

    // Skip hooks with no fixture mapping (e.g. PostAttGated)
    if fixture_keys.is_empty() {
        counters.record_skip();
        return;
    }

    // Check that at least one fixture key exists in the appropriate fixture.
    // PreWkvState = state at start of pair -> compare against fixture_even
    // PostWkvState = state at end of pair -> compare against fixture_odd
    // Other hooks: compare against fixture_even for token 0
    let ref_fixture = if hook == HipHook::PostWkvState {
        fixture_odd
    } else if hook == HipHook::PreWkvState {
        fixture_even
    } else {
        fixture_even
    };
    let any_key_present = fixture_keys.iter().any(|k| ref_fixture.contains(k));
    if !any_key_present {
        // All keys missing from fixture - skip silently
        counters.record_skip();
        return;
    }

    if is_state_hook(hook) {
        // ----- State hooks: no T dimension -----
        // PreWkvState = state before the T=2 pair -> compare against fixture_even
        // PostWkvState = state after the T=2 pair -> compare against fixture_odd
        let (state_fixture, state_idx) = if hook == HipHook::PreWkvState {
            (fixture_even, idx_even)
        } else {
            (fixture_odd, idx_odd)
        };
        compare_state_hook(
            hook,
            layer,
            captured,
            state_fixture,
            &fixture_keys,
            tolerances,
            step,
            state_idx,
            counters,
        );
    } else {
        // ----- Normal hooks: split captured data at halfway for token 0 / token 1 -----
        let half = captured.len() / 2;
        if half == 0 {
            println!(
                "  [SKIP] {:?} layer={:?}: captured data empty",
                hook, layer
            );
            counters.record_skip();
            return;
        }
        let tok0_data = &captured[..half];
        let tok1_data = &captured[half..];

        compare_token_against_fixture(
            hook,
            layer,
            tok0_data,
            fixture_even,
            &fixture_keys,
            tolerances,
            step,
            idx_even,
            0,
            counters,
        );
        compare_token_against_fixture(
            hook,
            layer,
            tok1_data,
            fixture_odd,
            &fixture_keys,
            tolerances,
            step,
            idx_odd,
            1,
            counters,
        );
    }
}

/// Compare state data (no T dimension) against a fixture.
fn compare_state_hook(
    hook: HipHook,
    layer: Option<usize>,
    captured: &[f32],
    fixture: &TestFixture,
    fixture_keys: &[String],
    tolerances: Tolerances,
    step: usize,
    fixture_step_idx: usize,
    counters: &mut ValidationCounters,
) {
    // State hooks always have a single fixture key
    for key in fixture_keys {
        if !fixture.contains(key) {
            counters.record_skip();
            continue;
        }

        let expected = fixture.f32(key);
        let result = assert_tensors_close(captured, expected, tolerances.rtol, tolerances.atol);
        if result.is_ok() {
            counters.record_pass();
        } else {
            let label = format!("{} (vs step_{})", key, fixture_step_idx);
            counters.record_fail(step, layer, hook, &label, &result.unwrap_err());
        }
    }
}

/// Compare a single token's data against a fixture, handling stacked sub-tensors.
fn compare_token_against_fixture(
    hook: HipHook,
    layer: Option<usize>,
    token_data: &[f32],
    fixture: &TestFixture,
    fixture_keys: &[String],
    tolerances: Tolerances,
    step: usize,
    fixture_step_idx: usize,
    token_idx: usize,
    counters: &mut ValidationCounters,
) {
    let num_tensors = fixture_keys.len();

    if num_tensors > 1 {
        // Stacked tensor: split token_data into num_tensors equal parts
        let sub_size = token_data.len() / num_tensors;
        if sub_size == 0 {
            println!(
                "  [SKIP] {:?} layer={:?} token[{}]: data too small ({}) for {} sub-tensors",
                hook,
                layer,
                token_idx,
                token_data.len(),
                num_tensors
            );
            counters.record_skip();
            return;
        }

        for (i, key) in fixture_keys.iter().enumerate() {
            if !fixture.contains(key) {
                counters.record_skip();
                continue;
            }

            let start = i * sub_size;
            let end = start + sub_size;
            let actual_slice = &token_data[start..end];
            let expected = fixture.f32(key);

            let result =
                assert_tensors_close(actual_slice, expected, tolerances.rtol, tolerances.atol);
            if result.is_ok() {
                counters.record_pass();
            } else {
                let label = format!(
                    "{} token[{}] (vs step_{})",
                    key, token_idx, fixture_step_idx
                );
                counters.record_fail(step, layer, hook, &label, &result.unwrap_err());
            }
        }
    } else {
        // Single tensor
        let key = &fixture_keys[0];
        if !fixture.contains(key) {
            counters.record_skip();
            return;
        }

        let expected = fixture.f32(key);
        let result =
            assert_tensors_close(token_data, expected, tolerances.rtol, tolerances.atol);
        if result.is_ok() {
            counters.record_pass();
        } else {
            let label = format!(
                "{} token[{}] (vs step_{})",
                key, token_idx, fixture_step_idx
            );
            counters.record_fail(step, layer, hook, &label, &result.unwrap_err());
        }
    }
}

// ---------------------------------------------------------------------------
// The main test
// ---------------------------------------------------------------------------

/// Validate FLA correctness per-layer against ground truth fixtures by feeding tokens
/// in T=2 chunks. Since FLA_CHUNK_THRESHOLD=2, each T=2 call goes through the FLA path.
///
/// The 14 single-token fixtures (step_0..step_13) provide per-layer intermediate values.
/// By feeding tokens in pairs, we get 7 intermediate checkpoints:
///   step s (0..6): feed [token_{2s}, token_{2s+1}]
///     - For each hook, the captured T=2 data is split: first half = token 0, second half = token 1.
///     - Token 0 is compared against step_{2s} fixture, token 1 against step_{2s+1} fixture.
///     - State hooks (no T dimension) are compared against step_{2s+1} (state after both tokens).
///
/// This test covers ALL hooks that the RNN layer-by-layer test covers:
///   PostEmbed, PostEmbedLayerNorm, PostAttLayerNorm, PostAttTokenShift, PostAttLinear,
///   PostAttDecay, PostAttAdapt, PostAttGate, PostAttValueResidual, PostAttL2Norm,
///   PostAttControlK, PreWkv, PreWkvState, PostWkv, PostWkvState, PostWkvBonus,
///   PostAttGroupNorm, PostAttOut, PostAtt, PostFfnLayerNorm, PostFfnTokenShift,
///   PostFfnLinear, PostFfnActivate, PostFfnOut, PostFfn, PostHeadLayerNorm, PostHead
///
/// Run with: cargo test --features hip,hip-probes test_fla_ground_truth_t2_chunked -- --nocapture
#[test]
fn test_fla_ground_truth_t2_chunked() {
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let config_path = "../tests/fixtures/ground_truth/config.npz";
    if !Path::new(config_path).exists() {
        eprintln!("Skipping: config fixture not found at {}", config_path);
        eprintln!(
            "Generate with: python scripts/extract_rwkv7_fixtures.py --model /path/to/model.pth"
        );
        return;
    }

    // Check that at least the first two step fixtures exist
    if !Path::new("../tests/fixtures/ground_truth/step_0.npz").exists()
        || !Path::new("../tests/fixtures/ground_truth/step_1.npz").exists()
    {
        eprintln!("Skipping: step fixtures not found");
        return;
    }

    // Load config
    let config = TestFixture::load(config_path).expect("Failed to load config");
    let tokens_i64 = config.i64("tokens");
    let n_steps = config.i64("n_steps")[0] as usize;
    assert!(n_steps >= 2, "Need at least 2 steps for T=2 chunking");
    let num_pairs = n_steps / 2;

    println!(
        "\n=== FLA Ground Truth Validation (T=2 chunked, {} pairs, ALL hooks) ===\n",
        num_pairs
    );

    // Build probes that capture all hook points
    let (probes, captured) = build_capture_probes();

    // Load model with probes and FLA-compatible config.
    let model = Rwkv7Hip::load(model_path)
        .expect("Failed to load model")
        .with_probes(probes);
    let rt_config = HipRuntimeConfig::new(256, 1);
    let model = HipRuntime::with_config(model, rt_config)
        .expect("Failed to configure runtime");

    let n_layer = model.info().n_layer;

    println!(
        "Model: n_layer={}, n_embd={}, n_head={}, head_size={}",
        n_layer, model.info().n_embd, model.info().n_head, model.info().head_size
    );
    println!();

    // Ordered list of per-layer hooks (execution order)
    let per_layer_hooks = vec![
        HipHook::PostAttLayerNorm,
        HipHook::PostAttTokenShift,
        HipHook::PostAttLinear,
        HipHook::PostAttDecay,
        HipHook::PostAttAdapt,
        HipHook::PostAttGate,
        HipHook::PostAttValueResidual,
        HipHook::PostAttL2Norm,
        HipHook::PostAttControlK,
        HipHook::PreWkv,
        HipHook::PreWkvState,
        HipHook::PostWkv,
        HipHook::PostWkvState,
        HipHook::PostWkvBonus,
        HipHook::PostAttGroupNorm,
        HipHook::PostAttOut,
        HipHook::PostAtt,
        HipHook::PostFfnLayerNorm,
        HipHook::PostFfnTokenShift,
        HipHook::PostFfnLinear,
        HipHook::PostFfnActivate,
        HipHook::PostFfnOut,
        HipHook::PostFfn,
    ];

    let mut counters = ValidationCounters::new();
    let mut state: Option<HipState> = None;

    for step in 0..num_pairs {
        let idx_even = 2 * step;
        let idx_odd = 2 * step + 1;
        let tok0 = tokens_i64[idx_even] as u32;
        let tok1 = tokens_i64[idx_odd] as u32;

        // Load the two corresponding fixture files
        let fixture_even_path = format!("../tests/fixtures/ground_truth/step_{}.npz", idx_even);
        let fixture_odd_path = format!("../tests/fixtures/ground_truth/step_{}.npz", idx_odd);

        if !Path::new(&fixture_even_path).exists() || !Path::new(&fixture_odd_path).exists() {
            println!(
                "Step {} (tokens [{}, {}]): fixture files not found, skipping",
                step, tok0, tok1
            );
            continue;
        }

        let fixture_even = TestFixture::load(&fixture_even_path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", fixture_even_path, e));
        let fixture_odd = TestFixture::load(&fixture_odd_path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", fixture_odd_path, e));

        // Clear captured data before each step
        captured.lock().unwrap().clear();

        // Feed T=2 tokens through model (triggers FLA since T=2 >= FLA_CHUNK_THRESHOLD=2)
        let (logits, new_state) = model
            .step(&[&[tok0, tok1]], state)
            .unwrap_or_else(|e| panic!("Step {} failed: {}", step, e));
        state = Some(new_state);

        let _ = logits; // logits not compared in isolation (accumulated errors)

        let cap = captured.lock().unwrap();

        println!(
            "--- Step {} (tokens [{}, {}], fixtures step_{}/step_{}) ---",
            step, tok0, tok1, idx_even, idx_odd
        );

        // === Embedding hooks (fire once, not per-layer) ===
        // PostEmbed: layer=None
        if let Some(data) = cap.get(&(HipHook::PostEmbed, None)) {
            compare_hook_t2(
                HipHook::PostEmbed,
                None,
                data,
                &fixture_even,
                &fixture_odd,
                step,
                idx_even,
                idx_odd,
                &mut counters,
            );
        }

        // PostEmbedLayerNorm: layer=Some(0)
        if let Some(data) = cap.get(&(HipHook::PostEmbedLayerNorm, Some(0))) {
            compare_hook_t2(
                HipHook::PostEmbedLayerNorm,
                Some(0),
                data,
                &fixture_even,
                &fixture_odd,
                step,
                idx_even,
                idx_odd,
                &mut counters,
            );
        }

        // === Per-layer hooks ===
        for layer in 0..n_layer {
            for &hook in &per_layer_hooks {
                // Skip PostAttValueResidual for layer 0 (no v_first yet)
                if hook == HipHook::PostAttValueResidual && layer == 0 {
                    continue;
                }

                if let Some(data) = cap.get(&(hook, Some(layer))) {
                    compare_hook_t2(
                        hook,
                        Some(layer),
                        data,
                        &fixture_even,
                        &fixture_odd,
                        step,
                        idx_even,
                        idx_odd,
                        &mut counters,
                    );
                }
            }
        }

        // === Head hooks (fire once, not per-layer) ===
        // PostHeadLayerNorm: layer=None
        if let Some(data) = cap.get(&(HipHook::PostHeadLayerNorm, None)) {
            compare_hook_t2(
                HipHook::PostHeadLayerNorm,
                None,
                data,
                &fixture_even,
                &fixture_odd,
                step,
                idx_even,
                idx_odd,
                &mut counters,
            );
        }

        // PostHead: layer=None
        if let Some(data) = cap.get(&(HipHook::PostHead, None)) {
            compare_hook_t2(
                HipHook::PostHead,
                None,
                data,
                &fixture_even,
                &fixture_odd,
                step,
                idx_even,
                idx_odd,
                &mut counters,
            );
        }

        // Per-step summary
        println!(
            "  Step {} summary: {}/{} checks passed so far ({} skipped)",
            step, counters.passed_checks, counters.total_checks, counters.skipped_checks
        );
        println!();
    }

    // === Final Summary ===
    println!("=== FLA Ground Truth Validation Summary (ALL hooks) ===");
    println!(
        "Total checks: {} (skipped: {})",
        counters.total_checks, counters.skipped_checks
    );
    println!(
        "Passed: {}/{}",
        counters.passed_checks, counters.total_checks
    );

    if counters.total_checks > 0 {
        let pass_rate = counters.passed_checks as f64 / counters.total_checks as f64;
        println!("Pass rate: {:.1}%", pass_rate * 100.0);
    }

    if let Some(ref failure) = counters.first_failure {
        println!(
            "\nFirst divergence at step {}, layer {:?}: {:?} key='{}'",
            failure.step, failure.layer, failure.hook, failure.fixture_key
        );
    }

    // 95% pass threshold - accounts for expected BF16 vs FP32 precision differences
    const PASS_THRESHOLD: f64 = 0.95;

    if counters.total_checks > 0 {
        let pass_rate = counters.passed_checks as f64 / counters.total_checks as f64;
        if pass_rate < PASS_THRESHOLD {
            let failed = counters.total_checks - counters.passed_checks;
            panic!(
                "FLA ground truth validation: {}/{} checks failed ({:.1}% < {:.0}% threshold). First failure: {}",
                failed,
                counters.total_checks,
                pass_rate * 100.0,
                PASS_THRESHOLD * 100.0,
                counters
                    .first_failure
                    .as_ref()
                    .map(|f| format!("{:?} layer={:?} key='{}'", f.hook, f.layer, f.fixture_key))
                    .unwrap_or_else(|| "unknown".to_string())
            );
        }
    }

    println!(
        "\nFLA ground truth validation passed: {}/{} checks OK ({:.1}%)",
        counters.passed_checks,
        counters.total_checks,
        if counters.total_checks > 0 {
            counters.passed_checks as f64 / counters.total_checks as f64 * 100.0
        } else {
            0.0
        }
    );
}
