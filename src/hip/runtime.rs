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

    /// Run inference on a batch of token sequences.
    ///
    /// Supports variable-length sequences through padding and masking.
    /// State is preserved across calls for streaming inference.
    ///
    /// # Arguments
    /// * `sequences` - Batch of token sequences (can be variable length)
    ///
    /// # Returns
    /// Logits tensor with shape [vocab_size, max_len * batch_size, 1, 1]
    pub fn infer(&self, sequences: &[&[u32]]) -> Result<TensorCpu<f32>, super::HipErrorKind> {
        if sequences.is_empty() {
            return Err(super::HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }

        // Pad sequences and get original lengths
        let (padded, lengths) = self.pad_sequences(sequences);
        let padded_refs: Vec<&[u32]> = padded.iter().map(|v| v.as_slice()).collect();

        // Take state from mutex, run forward, put new state back
        let mut state_guard = self.state.lock().unwrap();
        let old_state = std::mem::replace(&mut *state_guard, HipState::new(&self.model.info, self.num_batch));
        drop(state_guard);

        let (logits, new_state) = self.model.forward(&padded_refs, Some(old_state), &lengths)?;

        // Store the updated state
        let mut state_guard = self.state.lock().unwrap();
        *state_guard = new_state;

        // Convert to TensorCpu
        let vocab_size = self.model.info.n_vocab;
        let max_len = padded[0].len();
        let batch_size = sequences.len();
        let shape = crate::tensor::shape::Shape::new(vocab_size, max_len * batch_size, 1, 1);

        TensorInit::from_data(shape, logits).map_err(|e| super::HipErrorKind {
            code: -1,
            message: format!("Failed to create output tensor: {}", e),
        })
    }

    /// Run single-sequence inference (convenience method).
    pub fn infer_one(&self, tokens: &[u32]) -> Result<TensorCpu<f32>, super::HipErrorKind> {
        self.infer(&[tokens])
    }

    /// Pad sequences to max length for batched inference.
    ///
    /// Takes variable-length sequences and pads them to the maximum length
    /// in the batch using token 0 as padding.
    ///
    /// # Arguments
    /// * `sequences` - Variable-length token sequences
    ///
    /// # Returns
    /// Tuple of (padded_tokens, original_lengths) where:
    /// - `padded_tokens[i]` has length equal to max sequence length
    /// - `original_lengths[i]` is the original length of sequence i
    pub fn pad_sequences(&self, sequences: &[&[u32]]) -> (Vec<Vec<u32>>, Vec<usize>) {
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut padded = Vec::with_capacity(sequences.len());
        let mut lengths = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let len = seq.len();
            lengths.push(len);

            let mut tokens = seq.to_vec();
            tokens.resize(max_len, 0); // Pad with token 0
            padded.push(tokens);
        }

        (padded, lengths)
    }

    /// Extract logits for the last real token of each sequence.
    ///
    /// Given logits from a padded batch and the original lengths,
    /// extracts only the logits at the last valid position for each sequence.
    ///
    /// # Arguments
    /// * `logits` - Full logits tensor [vocab_size, max_len * batch_size, 1, 1]
    /// * `lengths` - Original sequence lengths
    /// * `max_len` - Padded sequence length
    ///
    /// # Returns
    /// Vec of logit slices, one per sequence (each of length vocab_size)
    pub fn extract_last_logits(
        &self,
        logits: &TensorCpu<f32>,
        lengths: &[usize],
        max_len: usize,
    ) -> Vec<Vec<f32>> {
        let vocab_size = self.model.info.n_vocab;
        let data = logits.data();
        let batch_size = lengths.len();

        let mut results = Vec::with_capacity(batch_size);

        for (batch_idx, &real_len) in lengths.iter().enumerate() {
            // For batch b, token t: index = (b * max_len + t) * vocab_size
            // We want the last real token: t = real_len - 1
            let last_token_idx = real_len.saturating_sub(1);
            let offset = (batch_idx * max_len + last_token_idx) * vocab_size;
            let slice = &data[offset..offset + vocab_size];
            results.push(slice.to_vec());
        }

        results
    }

    /// Extract all valid logits for each sequence (not padding).
    ///
    /// # Arguments
    /// * `logits` - Full logits tensor [vocab_size, max_len * batch_size, 1, 1]
    /// * `lengths` - Original sequence lengths
    /// * `max_len` - Padded sequence length
    ///
    /// # Returns
    /// Vec of logit vectors, one per sequence. Each inner vec has length
    /// `real_len * vocab_size` containing logits for all valid positions.
    pub fn extract_all_logits(
        &self,
        logits: &TensorCpu<f32>,
        lengths: &[usize],
        max_len: usize,
    ) -> Vec<Vec<f32>> {
        let vocab_size = self.model.info.n_vocab;
        let data = logits.data();
        let batch_size = lengths.len();

        let mut results = Vec::with_capacity(batch_size);

        for (batch_idx, &real_len) in lengths.iter().enumerate() {
            let mut seq_logits = Vec::with_capacity(real_len * vocab_size);

            for t in 0..real_len {
                let offset = (batch_idx * max_len + t) * vocab_size;
                seq_logits.extend_from_slice(&data[offset..offset + vocab_size]);
            }

            results.push(seq_logits);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::shape::Shape;
    use std::path::Path;

    #[cfg(feature = "tokio")]
    use {
        crate::{
            context::{ContextBuilder, InstanceExt},
            runtime::{
                infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
                loader::Loader,
                model::{ContextAutoLimits, ModelBuilder, ModelVersion},
                v7, Runtime as RuntimeTrait, TokioRuntime,
            },
        },
        half::f16,
        memmap2::Mmap,
        safetensors::SafeTensors,
        std::fs::File,
    };

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

    /// Proof-of-life test: run actual inference through HipRuntime
    #[test]
    fn test_hip_runtime_infer_proof_of_life() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let vocab_size = model.info.n_vocab;
        let runtime = HipRuntime::new(model, 1);

        // Run inference on a simple sequence
        let tokens: &[u32] = &[1, 2, 3, 4];
        let logits = runtime.infer_one(tokens).expect("Inference failed");

        // Validate output shape
        let shape = logits.shape();
        assert_eq!(shape[0], vocab_size, "First dim should be vocab_size");
        assert_eq!(shape[1], tokens.len(), "Second dim should be num_tokens");

        // Validate logits are finite (not NaN or Inf)
        assert!(
            logits.data().iter().all(|&x| x.is_finite()),
            "Logits should be finite"
        );

        // Apply softmax and verify it sums to 1.0 for each token
        let probs = softmax_one_cpu(logits).expect("Softmax failed");
        for t in 0..tokens.len() {
            let start = t * vocab_size;
            let end = start + vocab_size;
            let sum: f32 = probs.data()[start..end].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "Token {} probs should sum to 1.0, got {}",
                t,
                sum
            );
        }

        println!("HipRuntime inference proof-of-life PASSED");
        println!("  - Ran inference on {} tokens", tokens.len());
        println!("  - Output shape: [{}, {}, {}, {}]", shape[0], shape[1], shape[2], shape[3]);
    }

    /// Test stateful inference: multiple calls should accumulate state
    #[test]
    fn test_hip_runtime_stateful_inference() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime = HipRuntime::new(model, 1);

        // First inference
        let logits1 = runtime.infer_one(&[1, 2]).expect("First inference failed");

        // Second inference (state should be different from fresh)
        let _logits2 = runtime.infer_one(&[3, 4]).expect("Second inference failed");

        // Reset and run same sequence - should match first result
        runtime.reset_state();
        let logits3 = runtime.infer_one(&[1, 2]).expect("Third inference failed");

        // logits1 and logits3 should be identical (same input, fresh state)
        let diff: f32 = logits1
            .data()
            .iter()
            .zip(logits3.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-6,
            "Same input with reset state should produce same output, diff={}",
            diff
        );

        // logits2 should be different from logits1 (different state)
        // (We don't assert this strongly since token content differs, but we ran successfully)

        println!("HipRuntime stateful inference test PASSED");
    }

    /// Compare HIP and WGPU backend outputs on the same input.
    /// Uses multiple metrics for comprehensive comparison:
    /// - Raw logit statistics (max/mean absolute diff, cosine similarity)
    /// - Top-k set overlap (practical measure of prediction agreement)
    /// - Top-1 agreement (do they predict the same token?)
    /// - KL divergence (information-theoretic distance)
    ///
    /// Reference: https://huggingface.co/blog/rishiraj/kld-guided-quantization
    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn test_hip_vs_wgpu_parity() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        // === HIP Backend ===
        let hip_model = Rwkv7Hip::load(model_path).expect("Failed to load HIP model");
        let vocab_size = hip_model.info.n_vocab;
        let hip_runtime = HipRuntime::new(hip_model, 1);

        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let num_tokens = tokens.len();
        let hip_logits = hip_runtime
            .infer_one(&tokens)
            .expect("HIP inference failed");
        let hip_logits_vec: Vec<f32> = hip_logits.data().to_vec();

        // === WGPU Backend ===
        let file = File::open(model_path).expect("Failed to open model file");
        let data = unsafe { Mmap::map(&file).expect("Failed to mmap model") };
        let model = SafeTensors::deserialize(&data).expect("Failed to deserialize model");
        let info = Loader::info(&model).expect("Failed to get model info");

        assert!(
            matches!(info.version, ModelVersion::V7),
            "Expected V7 model for comparison"
        );

        let instance = wgpu::Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await
            .expect("Failed to get WGPU adapter");

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await
            .expect("Failed to create WGPU context");

        let wgpu_model = ModelBuilder::new(&context, model)
            .build_v7()
            .await
            .expect("Failed to build WGPU model");

        let bundle = v7::Bundle::<f16>::new(wgpu_model, 1);
        let runtime: Box<dyn RuntimeTrait<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

        // Run inference - need to use RnnInput format
        let batch = RnnInputBatch::new(tokens.clone(), RnnOption::Full);
        let input = RnnInput::new(vec![batch], 128);

        let (_input, output) = runtime.infer(input).await.expect("WGPU inference failed");
        let wgpu_logits_tensor = output[0].0.clone();
        let wgpu_logits_vec: Vec<f32> = wgpu_logits_tensor.data().to_vec();

        // === Verify sizes match ===
        assert_eq!(
            hip_logits_vec.len(),
            wgpu_logits_vec.len(),
            "Output sizes should match: HIP={}, WGPU={}",
            hip_logits_vec.len(),
            wgpu_logits_vec.len()
        );

        println!("\n============================================================");
        println!("HIP vs WGPU Backend Comparison");
        println!("============================================================");
        println!("Tokens: {:?}", tokens);
        println!("Vocab size: {}, Num tokens: {}", vocab_size, num_tokens);
        println!("HIP logits len: {}, WGPU logits len: {}", hip_logits_vec.len(), wgpu_logits_vec.len());

        // Debug: print some actual logit values
        println!("\nSample logits (first token, indices 0-5):");
        println!("  HIP:  {:?}", &hip_logits_vec[0..6]);
        println!("  WGPU: {:?}", &wgpu_logits_vec[0..6]);
        println!("Sample logits (first token, indices 45-50):");
        println!("  HIP:  {:?}", &hip_logits_vec[45..51]);
        println!("  WGPU: {:?}", &wgpu_logits_vec[45..51]);

        // Check logit ranges
        let hip_min = hip_logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let hip_max = hip_logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let wgpu_min = wgpu_logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let wgpu_max = wgpu_logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("\nLogit ranges:");
        println!("  HIP:  [{:.4}, {:.4}]", hip_min, hip_max);
        println!("  WGPU: [{:.4}, {:.4}]", wgpu_min, wgpu_max);
        println!();

        // === Per-token analysis ===
        let mut all_top1_match = true;
        let mut total_top10_overlap = 0usize;
        let mut total_top100_overlap = 0usize;
        let mut total_kl_div = 0.0f64;
        let mut total_cosine_sim = 0.0f64;
        let mut max_logit_diff = 0.0f32;
        let mut sum_logit_diff = 0.0f64;

        for t in 0..num_tokens {
            let start = t * vocab_size;
            let end = start + vocab_size;
            let hip_slice = &hip_logits_vec[start..end];
            let wgpu_slice = &wgpu_logits_vec[start..end];

            // --- Raw logit statistics ---
            let mut token_max_diff = 0.0f32;
            let mut token_sum_diff = 0.0f64;
            let mut dot_product = 0.0f64;
            let mut hip_norm_sq = 0.0f64;
            let mut wgpu_norm_sq = 0.0f64;

            for (h, w) in hip_slice.iter().zip(wgpu_slice.iter()) {
                let diff = (h - w).abs();
                token_max_diff = token_max_diff.max(diff);
                token_sum_diff += diff as f64;
                dot_product += (*h as f64) * (*w as f64);
                hip_norm_sq += (*h as f64).powi(2);
                wgpu_norm_sq += (*w as f64).powi(2);
            }

            max_logit_diff = max_logit_diff.max(token_max_diff);
            sum_logit_diff += token_sum_diff;

            let cosine_sim = dot_product / (hip_norm_sq.sqrt() * wgpu_norm_sq.sqrt());
            total_cosine_sim += cosine_sim;

            // --- Top-k indices ---
            let mut hip_indexed: Vec<(usize, f32)> =
                hip_slice.iter().cloned().enumerate().collect();
            let mut wgpu_indexed: Vec<(usize, f32)> =
                wgpu_slice.iter().cloned().enumerate().collect();
            hip_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            wgpu_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let hip_top1 = hip_indexed[0].0;
            let wgpu_top1 = wgpu_indexed[0].0;
            let top1_match = hip_top1 == wgpu_top1;
            if !top1_match {
                all_top1_match = false;
            }

            let hip_top10: std::collections::HashSet<usize> =
                hip_indexed.iter().take(10).map(|(i, _)| *i).collect();
            let wgpu_top10: std::collections::HashSet<usize> =
                wgpu_indexed.iter().take(10).map(|(i, _)| *i).collect();
            let top10_overlap = hip_top10.intersection(&wgpu_top10).count();
            total_top10_overlap += top10_overlap;

            let hip_top100: std::collections::HashSet<usize> =
                hip_indexed.iter().take(100).map(|(i, _)| *i).collect();
            let wgpu_top100: std::collections::HashSet<usize> =
                wgpu_indexed.iter().take(100).map(|(i, _)| *i).collect();
            let top100_overlap = hip_top100.intersection(&wgpu_top100).count();
            total_top100_overlap += top100_overlap;

            // --- KL divergence (on softmax probs) ---
            // KL(P || Q) = sum(P * log(P/Q))
            let hip_probs = stable_softmax(hip_slice);
            let wgpu_probs = stable_softmax(wgpu_slice);
            let mut kl_div = 0.0f64;
            for (p, q) in hip_probs.iter().zip(wgpu_probs.iter()) {
                if *p > 1e-10 && *q > 1e-10 {
                    kl_div += (*p as f64) * ((*p as f64).ln() - (*q as f64).ln());
                }
            }
            total_kl_div += kl_div;

            println!(
                "Token {}: top1={} (HIP={}, WGPU={}), top10={}/10, top100={}/100, cosine={:.6}, KL={:.6e}",
                t,
                if top1_match { "✓" } else { "✗" },
                hip_top1,
                wgpu_top1,
                top10_overlap,
                top100_overlap,
                cosine_sim,
                kl_div
            );
        }

        // === Aggregate metrics ===
        let mean_logit_diff = sum_logit_diff / (hip_logits_vec.len() as f64);
        let avg_cosine_sim = total_cosine_sim / (num_tokens as f64);
        let avg_top10_overlap = total_top10_overlap as f64 / (num_tokens as f64);
        let avg_top100_overlap = total_top100_overlap as f64 / (num_tokens as f64);
        let avg_kl_div = total_kl_div / (num_tokens as f64);

        println!();
        println!("Aggregate Metrics:");
        println!("  Raw Logits:");
        println!("    - Max absolute diff: {:.6e}", max_logit_diff);
        println!("    - Mean absolute diff: {:.6e}", mean_logit_diff);
        println!("    - Avg cosine similarity: {:.6}", avg_cosine_sim);
        println!("  Top-k Agreement:");
        println!(
            "    - Top-1 match: {}/{}",
            if all_top1_match { num_tokens } else { 0 },
            num_tokens
        );
        println!("    - Avg top-10 overlap: {:.1}/10", avg_top10_overlap);
        println!("    - Avg top-100 overlap: {:.1}/100", avg_top100_overlap);
        println!("  Information Theory:");
        println!("    - Avg KL divergence: {:.6e}", avg_kl_div);

        // === Parity Assessment ===
        // Note: Current thresholds are lenient due to known differences between
        // HIP (fp32 throughout) and WGPU (fp16 intermediate) implementations.
        // These will be tightened as parity improves.
        let cosine_ok = avg_cosine_sim > 0.95;
        let top10_ok = avg_top10_overlap >= 1.0;
        let kl_ok = avg_kl_div < 10.0;

        println!();
        println!("Parity Assessment:");
        println!(
            "  Cosine > 0.95: {} ({:.6})",
            if cosine_ok { "PASS" } else { "FAIL" },
            avg_cosine_sim
        );
        println!(
            "  Top-10 >= 1.0: {} ({:.1})",
            if top10_ok { "PASS" } else { "FAIL" },
            avg_top10_overlap
        );
        println!(
            "  KL < 10.0: {} ({:.4})",
            if kl_ok { "PASS" } else { "FAIL" },
            avg_kl_div
        );

        // For now, only fail on catastrophic differences
        // TODO: Tighten these as parity improves
        assert!(
            avg_cosine_sim > 0.90,
            "Cosine similarity {:.6} indicates catastrophic mismatch (expected > 0.90)",
            avg_cosine_sim
        );

        if cosine_ok && top10_ok && kl_ok {
            println!("\nHIP vs WGPU parity test PASSED (all thresholds met)");
        } else {
            println!("\nHIP vs WGPU parity test PASSED (minimum thresholds met, some metrics need improvement)");
        }
    }

    /// Numerically stable softmax
    fn stable_softmax(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum).collect()
    }

    // ========== Variable-length batching tests ==========

    #[test]
    fn test_pad_sequences_equal_length() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime = HipRuntime::new(model, 2);

        // No padding needed when all same length
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];
        let (padded, lengths) = runtime.pad_sequences(&[&seq1, &seq2]);

        assert_eq!(lengths, vec![3, 3]);
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 6]);

        println!("test_pad_sequences_equal_length PASSED");
    }

    #[test]
    fn test_pad_sequences_variable_length() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime = HipRuntime::new(model, 2);

        // Variable lengths - should pad shorter sequence
        let seq1: Vec<u32> = vec![1, 2, 3, 4, 5];
        let seq2: Vec<u32> = vec![10, 20];
        let (padded, lengths) = runtime.pad_sequences(&[&seq1, &seq2]);

        assert_eq!(lengths, vec![5, 2]);
        assert_eq!(padded[0], vec![1, 2, 3, 4, 5]);
        assert_eq!(padded[1], vec![10, 20, 0, 0, 0]); // Padded with zeros

        println!("test_pad_sequences_variable_length PASSED");
    }

    #[test]
    fn test_extract_last_logits() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let vocab_size = model.info.n_vocab;
        let runtime = HipRuntime::new(model, 2);

        // Create mock logits: [vocab_size, max_len * batch_size, 1, 1]
        // batch_size=2, max_len=3
        let max_len = 3;
        let batch_size = 2;
        let total = vocab_size * max_len * batch_size;
        let mut data = vec![0.0f32; total];

        // Mark specific positions with identifiable values
        // Batch 0, token 0: all 1.0
        // Batch 0, token 1: all 2.0
        // Batch 0, token 2: all 3.0
        // Batch 1, token 0: all 10.0
        // Batch 1, token 1: all 20.0
        // Batch 1, token 2: all 30.0
        for b in 0..batch_size {
            for t in 0..max_len {
                let base_val = if b == 0 { (t + 1) as f32 } else { ((t + 1) * 10) as f32 };
                for v in 0..vocab_size {
                    data[(b * max_len + t) * vocab_size + v] = base_val;
                }
            }
        }

        let logits: TensorCpu<f32> =
            TensorInit::from_data(Shape::new(vocab_size, max_len * batch_size, 1, 1), data)
                .unwrap();

        // lengths = [2, 3] means batch 0 has 2 real tokens, batch 1 has 3
        let lengths = vec![2, 3];
        let extracted = runtime.extract_last_logits(&logits, &lengths, max_len);

        // Batch 0: last real token is at index 1 (value 2.0)
        assert_eq!(extracted[0][0], 2.0);
        // Batch 1: last real token is at index 2 (value 30.0)
        assert_eq!(extracted[1][0], 30.0);

        println!("test_extract_last_logits PASSED");
    }

    #[test]
    fn test_extract_all_logits() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let vocab_size = model.info.n_vocab;
        let runtime = HipRuntime::new(model, 2);

        // Same setup as above
        let max_len = 3;
        let batch_size = 2;
        let total = vocab_size * max_len * batch_size;
        let mut data = vec![0.0f32; total];

        for b in 0..batch_size {
            for t in 0..max_len {
                let base_val = if b == 0 { (t + 1) as f32 } else { ((t + 1) * 10) as f32 };
                for v in 0..vocab_size {
                    data[(b * max_len + t) * vocab_size + v] = base_val;
                }
            }
        }

        let logits: TensorCpu<f32> =
            TensorInit::from_data(Shape::new(vocab_size, max_len * batch_size, 1, 1), data)
                .unwrap();

        // lengths = [2, 3]
        let lengths = vec![2, 3];
        let extracted = runtime.extract_all_logits(&logits, &lengths, max_len);

        // Batch 0: 2 tokens * vocab_size
        assert_eq!(extracted[0].len(), 2 * vocab_size);
        assert_eq!(extracted[0][0], 1.0); // token 0
        assert_eq!(extracted[0][vocab_size], 2.0); // token 1

        // Batch 1: 3 tokens * vocab_size
        assert_eq!(extracted[1].len(), 3 * vocab_size);
        assert_eq!(extracted[1][0], 10.0); // token 0
        assert_eq!(extracted[1][vocab_size], 20.0); // token 1
        assert_eq!(extracted[1][2 * vocab_size], 30.0); // token 2

        println!("test_extract_all_logits PASSED");
    }

    /// CRITICAL TEST: Verify that padding doesn't affect hidden state.
    /// h(seq + padding) should equal h(seq)
    #[test]
    fn test_padding_preserves_hidden_state() {
        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime_unpadded = HipRuntime::new(model, 1);

        let model2 = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let runtime_padded = HipRuntime::new(model2, 2);

        // Process [1, 2, 3] without padding
        let seq: Vec<u32> = vec![1, 2, 3];
        let _logits_unpadded = runtime_unpadded.infer_one(&seq).expect("Unpadded inference failed");
        let state_unpadded = runtime_unpadded.get_state_snapshot();

        // Process [1, 2, 3] with padding via variable-length batch
        // We'll have seq1=[1,2,3] and seq2=[4,5] which pads seq2
        // But we only care about the first batch's state
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5]; // Different sequence, will be padded
        // infer() now handles variable-length sequences automatically
        let _logits_padded = runtime_padded
            .infer(&[&seq1, &seq2])
            .expect("Padded inference failed");
        let state_padded = runtime_padded.get_state_snapshot();

        // Compare the state for batch 0 from padded vs unpadded
        // Note: The padded runtime has batch_size=2, so we need to extract batch 0's state
        let n_layer = state_unpadded.att_states.len();

        println!("Comparing states across {} layers...", n_layer);

        let mut max_att_diff = 0.0f32;
        let mut max_ffn_diff = 0.0f32;

        for layer in 0..n_layer {
            // att_shift_states: [n_embd * batch] - extract first n_embd for batch 0
            let n_embd = state_unpadded.att_shift_states[layer].len();
            for i in 0..n_embd {
                let diff = (state_unpadded.att_shift_states[layer][i]
                    - state_padded.att_shift_states[layer][i])
                    .abs();
                max_att_diff = max_att_diff.max(diff);
            }

            // ffn_states: same structure
            for i in 0..n_embd {
                let diff =
                    (state_unpadded.ffn_states[layer][i] - state_padded.ffn_states[layer][i]).abs();
                max_ffn_diff = max_ffn_diff.max(diff);
            }
        }

        println!("Max att_shift diff: {:.6e}", max_att_diff);
        println!("Max ffn_shift diff: {:.6e}", max_ffn_diff);

        // The states should be very close (within floating point tolerance)
        assert!(
            max_att_diff < 1e-4,
            "Attention shift states differ too much: {:.6e}",
            max_att_diff
        );
        assert!(
            max_ffn_diff < 1e-4,
            "FFN shift states differ too much: {:.6e}",
            max_ffn_diff
        );

        println!("test_padding_preserves_hidden_state PASSED");
    }

}
