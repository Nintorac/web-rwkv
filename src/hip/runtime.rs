//! HIP runtime implementation for RWKV7.
//!
//! This module provides `HipRuntime`, which wraps `Rwkv7Hip` and manages
//! state with thread-safe access for integration with the web-rwkv runtime interface.

use std::sync::Mutex;

use super::{HipState, Rwkv7Hip, Rwkv7ModelInfo};

/// HIP-based runtime for RWKV7 inference.
///
/// Wraps a loaded `Rwkv7Hip` model and manages state with thread-safe access.
/// This struct will implement `Runtime<Rnn>` to integrate with the web-rwkv
/// runtime interface.
pub struct HipRuntime {
    model: Rwkv7Hip,
    state: Mutex<HipState>,
    num_batch: usize,
}

impl HipRuntime {
    /// Create a new HipRuntime.
    ///
    /// # Arguments
    /// * `model` - Loaded RWKV7 HIP model
    /// * `num_batch` - Maximum batch size for inference
    pub fn new(model: Rwkv7Hip, num_batch: usize) -> Self {
        let state = HipState::new(&model.info, num_batch);
        Self {
            model,
            state: Mutex::new(state),
            num_batch,
        }
    }

    /// Get model info.
    pub fn info(&self) -> &Rwkv7ModelInfo {
        &self.model.info
    }

    /// Get configured batch size.
    pub fn num_batch(&self) -> usize {
        self.num_batch
    }

    /// Reset all state to initial values.
    pub fn reset_state(&self) {
        let mut state = self.state.lock().unwrap();
        state.reset();
    }

    /// Get a snapshot of current state (for testing/debugging).
    pub fn get_state_snapshot(&self) -> HipState {
        let state = self.state.lock().unwrap();
        state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_hip_runtime_construction() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let vocab_size = model.info.n_vocab;
        let n_embd = model.info.n_embd;

        let runtime = HipRuntime::new(model, 4);

        assert_eq!(runtime.info().n_vocab, vocab_size);
        assert_eq!(runtime.info().n_embd, n_embd);
        assert_eq!(runtime.num_batch(), 4);

        println!("HipRuntime construction test PASSED");
    }

    #[test]
    fn test_hip_runtime_state_reset() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime = HipRuntime::new(model, 1);

        // Should not panic
        runtime.reset_state();

        // Verify state is reset
        let state = runtime.get_state_snapshot();
        assert!(state.v_first.is_none());

        println!("HipRuntime state reset test PASSED");
    }
}
