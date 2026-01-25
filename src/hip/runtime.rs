//! HIP runtime implementation for RWKV7.
//!
//! This module provides `HipRuntime`, which wraps `Rwkv7Hip` and manages
//! state with thread-safe access for integration with the web-rwkv runtime interface.

use std::sync::Mutex;

use super::{HipState, Rwkv7Hip, Rwkv7ModelInfo};
use crate::tensor::{TensorCpu, TensorError, TensorInit, TensorShape as TensorShapeTrait};

/// CPU-only softmax for HIP backend.
///
/// Applies softmax over the first dimension (vocab) for each token.
/// Uses numerically stable algorithm: subtract max before exp.
///
/// # Arguments
/// * `input` - Input tensor with shape [vocab_size, num_tokens, 1, 1]
///
/// # Returns
/// Tensor with same shape, where each column sums to 1.0
pub fn softmax_one_cpu(input: TensorCpu<f32>) -> Result<TensorCpu<f32>, TensorError> {
    let shape = input.shape();
    if shape.len() == 0 {
        return Ok(input);
    }

    let data = input.data();
    let vocab_size = shape[0];
    let num_tokens = data.len() / vocab_size;

    let mut output = Vec::with_capacity(data.len());

    for t in 0..num_tokens {
        let start = t * vocab_size;
        let end = start + vocab_size;
        let slice = &data[start..end];

        // Numerically stable: subtract max before exp
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = slice.iter().map(|&x| (x - max_val).exp()).sum();

        for &x in slice {
            output.push((x - max_val).exp() / exp_sum);
        }
    }

    TensorInit::from_data(shape, output)
}

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
    use crate::tensor::shape::Shape;
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

    #[test]
    fn test_softmax_cpu_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input: TensorCpu<f32> = TensorInit::from_data(Shape::new(4, 1, 1, 1), data).unwrap();
        let output = softmax_one_cpu(input).unwrap();

        // Sum should be 1.0
        let sum: f32 = output.data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax should sum to 1.0, got {}",
            sum
        );

        // Values should be in (0, 1) and monotonically increasing
        let data = output.data();
        assert!(data[0] < data[1] && data[1] < data[2] && data[2] < data[3]);

        println!("Softmax CPU basic test PASSED");
    }

    #[test]
    fn test_softmax_cpu_numerical_stability() {
        // Large values that would overflow naive exp()
        let data = vec![1000.0f32, 1001.0, 1002.0, 1003.0];
        let input: TensorCpu<f32> = TensorInit::from_data(Shape::new(4, 1, 1, 1), data).unwrap();
        let output = softmax_one_cpu(input).unwrap();

        // Should not produce NaN or Inf
        assert!(
            output.data().iter().all(|&x| x.is_finite()),
            "Softmax should handle large values without overflow"
        );

        // Sum should still be 1.0
        let sum: f32 = output.data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax should sum to 1.0, got {}",
            sum
        );

        println!("Softmax CPU numerical stability test PASSED");
    }

    #[test]
    fn test_softmax_cpu_multiple_tokens() {
        // 2 tokens, vocab size 3 - each token's probs should sum to 1.0
        // Shape [3, 2, 1, 1] means vocab_size=3, num_tokens=2
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: TensorCpu<f32> = TensorInit::from_data(Shape::new(3, 2, 1, 1), data).unwrap();
        let output = softmax_one_cpu(input).unwrap();

        let data = output.data();

        // First token (indices 0, 1, 2)
        let sum1: f32 = data[0..3].iter().sum();
        assert!(
            (sum1 - 1.0).abs() < 1e-6,
            "First token softmax should sum to 1.0, got {}",
            sum1
        );

        // Second token (indices 3, 4, 5)
        let sum2: f32 = data[3..6].iter().sum();
        assert!(
            (sum2 - 1.0).abs() < 1e-6,
            "Second token softmax should sum to 1.0, got {}",
            sum2
        );

        println!("Softmax CPU multiple tokens test PASSED");
    }

    #[test]
    fn test_softmax_cpu_empty() {
        // Empty tensor should be handled gracefully
        let data: Vec<f32> = vec![];
        let input: TensorCpu<f32> = TensorInit::from_data(Shape::new(0, 0, 1, 1), data).unwrap();
        let output = softmax_one_cpu(input).unwrap();

        assert_eq!(output.data().len(), 0);

        println!("Softmax CPU empty test PASSED");
    }
}
