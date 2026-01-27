//! Tests for forward_with_scratch GPU-native forward pass.

#[cfg(feature = "hip")]
mod tests {
    use std::path::Path;
    use web_rwkv::hip::{HipState, Rwkv7Hip};
    use web_rwkv::hip::{HipScratch, HipRuntimeConfig};

    const MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

    fn model_available() -> bool {
        Path::new(MODEL_PATH).exists()
    }

    #[test]
    fn test_forward_with_scratch_basic() {
        if !model_available() {
            eprintln!("Skipping test: model not found at {}", MODEL_PATH);
            return;
        }

        let model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load model");
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

        // Create scratch buffers
        let config = HipRuntimeConfig::new(16, 1);
        let lora_dims = model.lora_dims();
        let mut scratch = HipScratch::new(&model.info, lora_dims, config)
            .expect("Failed to create scratch");

        let mut state = HipState::new(&model.info, 1);

        // Run forward_with_scratch
        let logits = model
            .forward_with_scratch(&[&tokens], &mut state, &mut scratch)
            .expect("forward_with_scratch failed");

        // Verify shape
        let expected_len = model.info.n_vocab * tokens.len();
        assert_eq!(logits.len(), expected_len, "Logits length mismatch");

        // Verify no NaN or Inf
        for (i, &val) in logits.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {}", i);
            assert!(!val.is_infinite(), "Inf at index {}", i);
        }

        println!("test_forward_with_scratch_basic PASSED");
    }

    #[test]
    fn test_forward_with_scratch_matches_forward_with_state() {
        if !model_available() {
            eprintln!("Skipping test: model not found at {}", MODEL_PATH);
            return;
        }

        let model = Rwkv7Hip::load(MODEL_PATH).expect("Failed to load model");
        let tokens: Vec<u32> = vec![1, 2, 3];

        // Run with forward_with_state (old API)
        let mut state_old = HipState::new(&model.info, 1);
        let logits_old = model
            .forward_with_state(&[&tokens], &mut state_old)
            .expect("forward_with_state failed");

        // Run with forward_with_scratch (new API)
        let config = HipRuntimeConfig::new(16, 1);
        let lora_dims = model.lora_dims();
        let mut scratch = HipScratch::new(&model.info, lora_dims, config)
            .expect("Failed to create scratch");
        let mut state_new = HipState::new(&model.info, 1);
        let logits_new = model
            .forward_with_scratch(&[&tokens], &mut state_new, &mut scratch)
            .expect("forward_with_scratch failed");

        // Compare results
        assert_eq!(logits_old.len(), logits_new.len());

        let mut max_diff = 0.0f32;
        for (old, new) in logits_old.iter().zip(logits_new.iter()) {
            max_diff = max_diff.max((old - new).abs());
        }

        println!("Max diff between old and new API: {}", max_diff);

        // Allow some tolerance due to floating point differences
        assert!(
            max_diff < 1e-3,
            "forward_with_scratch should match forward_with_state (max_diff={})",
            max_diff
        );

        println!("test_forward_with_scratch_matches_forward_with_state PASSED");
    }
}
