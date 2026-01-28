//! Prefill-uniform scenario for benchmark execution.
//!
//! This module provides utilities for the prefill-uniform benchmark scenario, which measures
//! prompt processing performance as prompt length crosses chunk boundaries.
//!
//! # Overview
//!
//! In prefill-uniform, each batch has the same prompt length `L`. The benchmark measures:
//! - Time until each batch returns its first logits (local TTFT)
//! - Time to completion (all batches finished prompt consumption)
//! - Number of inference calls required when L > C
//!
//! # Length Sets
//!
//! Two modes are supported for determining sequence lengths:
//!
//! ## 1. Canonical Length List (per chunk size C)
//!
//! For each token_chunk_size `C`, generates lengths emphasizing:
//! - `<C` region
//! - exactly `C`
//! - `>C` boundaries
//! - multiples of `C`
//!
//! The canonical list is: `[1, C/4, C/2, C-1, C, C+1, 2C, 4C, 8C]`
//!
//! ## 2. Total-Token Target Mode
//!
//! For multi-batch comparability, lengths are computed from total token targets:
//! - Choose targets `T = [C/2, C, 2C, 4C, 8C]`
//! - For each `(B, C, T)`, set `L = ceil(T / B)` so `B*L ≈ T`
//!
//! # Deterministic Token Generation
//!
//! Tokens are generated deterministically from a seed to ensure reproducibility.
//! Uses a simple linear congruential generator (LCG) for fast, deterministic sequences.
//!
//! # Metrics
//!
//! - `prefill_total_ms`: Time until all batches reached first logits
//! - `ttft_ms_local[]`: Per-batch time to first token
//! - `ttft_min_ms`, `ttft_p50_ms`, `ttft_max_ms`: Derived TTFT statistics
//! - `num_infer_calls`: Number of inference calls (important when L > C)
//! - `total_prompt_tokens`: `B * L`
//! - `prefill_tok_per_s`: `total_prompt_tokens / (prefill_total_ms / 1000)`

use std::time::{Duration, Instant};

/// Generates the canonical length list for a given token_chunk_size.
///
/// The canonical list is: `[1, C/4, C/2, C-1, C, C+1, 2C, 4C, 8C]`
/// - All values are rounded to integers and ensured >= 1
/// - Deduplication is applied to remove any duplicates from rounding
///
/// # Arguments
///
/// * `token_chunk_size` - The chunk size C to generate lengths for
///
/// # Returns
///
/// A sorted, deduplicated vector of sequence lengths
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::canonical_length_list;
///
/// let lengths = canonical_length_list(128);
/// // Returns [1, 32, 64, 127, 128, 129, 256, 512, 1024]
/// assert!(lengths.contains(&1));
/// assert!(lengths.contains(&128));
/// assert!(lengths.contains(&1024));
/// ```
pub fn canonical_length_list(token_chunk_size: u32) -> Vec<u32> {
    let c = token_chunk_size;

    // Generate canonical list: [1, C/4, C/2, C-1, C, C+1, 2C, 4C, 8C]
    let mut lengths = vec![
        1,
        (c / 4).max(1),       // C/4, minimum 1
        (c / 2).max(1),       // C/2, minimum 1
        c.saturating_sub(1).max(1), // C-1, minimum 1
        c,                     // C
        c + 1,                 // C+1
        c * 2,                 // 2C
        c * 4,                 // 4C
        c * 8,                 // 8C
    ];

    // Sort and deduplicate
    lengths.sort_unstable();
    lengths.dedup();

    lengths
}

/// Total token targets for target mode.
///
/// Returns targets `T = [C/2, C, 2C, 4C, 8C]` for the given chunk size.
///
/// # Arguments
///
/// * `token_chunk_size` - The chunk size C to generate targets for
///
/// # Returns
///
/// A vector of total token targets
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::total_token_targets;
///
/// let targets = total_token_targets(256);
/// assert_eq!(targets, vec![128, 256, 512, 1024, 2048]);
/// ```
pub fn total_token_targets(token_chunk_size: u32) -> Vec<u32> {
    let c = token_chunk_size;
    vec![
        (c / 2).max(1),  // C/2, minimum 1
        c,               // C
        c * 2,           // 2C
        c * 4,           // 4C
        c * 8,           // 8C
    ]
}

/// Computes sequence length from total token target and batch size.
///
/// For a given total token target `T` and batch size `B`, computes `L = ceil(T / B)`.
/// This ensures that `B * L >= T` (total tokens processed is at least T).
///
/// # Arguments
///
/// * `total_tokens` - The total token target T
/// * `batch_size` - The batch size B
///
/// # Returns
///
/// The sequence length L such that `B * L >= T`
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::seq_len_from_total_tokens;
///
/// // For T=1024, B=4: L = ceil(1024/4) = 256
/// assert_eq!(seq_len_from_total_tokens(1024, 4), 256);
///
/// // For T=100, B=8: L = ceil(100/8) = 13
/// assert_eq!(seq_len_from_total_tokens(100, 8), 13);
/// ```
pub fn seq_len_from_total_tokens(total_tokens: u32, batch_size: u32) -> u32 {
    if batch_size == 0 {
        return 0;
    }
    // ceil(T / B) = (T + B - 1) / B
    (total_tokens + batch_size - 1) / batch_size
}

/// Generates sequence lengths for target mode.
///
/// For each total token target in `[C/2, C, 2C, 4C, 8C]`, computes `L = ceil(T / B)`.
///
/// # Arguments
///
/// * `token_chunk_size` - The chunk size C
/// * `batch_size` - The batch size B
///
/// # Returns
///
/// A sorted, deduplicated vector of sequence lengths for target mode
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::target_mode_lengths;
///
/// let lengths = target_mode_lengths(256, 4);
/// // For B=4, C=256:
/// // T=128 -> L=32, T=256 -> L=64, T=512 -> L=128, T=1024 -> L=256, T=2048 -> L=512
/// assert_eq!(lengths, vec![32, 64, 128, 256, 512]);
/// ```
pub fn target_mode_lengths(token_chunk_size: u32, batch_size: u32) -> Vec<u32> {
    let targets = total_token_targets(token_chunk_size);
    let mut lengths: Vec<u32> = targets
        .iter()
        .map(|&t| seq_len_from_total_tokens(t, batch_size))
        .collect();

    // Sort and deduplicate
    lengths.sort_unstable();
    lengths.dedup();

    lengths
}

/// Generates all lengths for prefill-uniform benchmarking.
///
/// Combines both canonical length list and target mode lengths into a single
/// sorted, deduplicated list for comprehensive coverage.
///
/// # Arguments
///
/// * `token_chunk_size` - The chunk size C
/// * `batch_size` - The batch size B
///
/// # Returns
///
/// A sorted, deduplicated vector containing all unique lengths from both modes
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::all_prefill_lengths;
///
/// let lengths = all_prefill_lengths(128, 4);
/// // Contains both canonical lengths and target-mode lengths
/// assert!(lengths.contains(&1));     // From canonical
/// assert!(lengths.contains(&128));   // From canonical (C)
/// assert!(lengths.contains(&1024));  // From canonical (8C)
/// ```
pub fn all_prefill_lengths(token_chunk_size: u32, batch_size: u32) -> Vec<u32> {
    let mut lengths = canonical_length_list(token_chunk_size);
    let target_lengths = target_mode_lengths(token_chunk_size, batch_size);

    lengths.extend(target_lengths);
    lengths.sort_unstable();
    lengths.dedup();

    lengths
}

/// Deterministic token generator using a linear congruential generator (LCG).
///
/// Produces reproducible token sequences from a seed value.
/// Uses the same parameters as Java's `java.util.Random` for well-tested properties.
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_uniform::TokenGenerator;
///
/// let mut gen = TokenGenerator::new(12345, 50257);
/// let tokens = gen.generate(10);
/// assert_eq!(tokens.len(), 10);
///
/// // Same seed produces same sequence
/// let mut gen2 = TokenGenerator::new(12345, 50257);
/// let tokens2 = gen2.generate(10);
/// assert_eq!(tokens, tokens2);
/// ```
pub struct TokenGenerator {
    state: u64,
    vocab_size: u32,
}

impl TokenGenerator {
    /// LCG multiplier (from Java)
    const MULTIPLIER: u64 = 0x5DEECE66D;
    /// LCG increment (from Java)
    const INCREMENT: u64 = 0xB;
    /// LCG mask for 48-bit state
    const MASK: u64 = (1u64 << 48) - 1;

    /// Creates a new token generator with the given seed and vocabulary size.
    ///
    /// # Arguments
    ///
    /// * `seed` - Initial seed value for reproducibility
    /// * `vocab_size` - Maximum token value (exclusive)
    pub fn new(seed: u64, vocab_size: u32) -> Self {
        Self {
            state: seed ^ Self::MULTIPLIER & Self::MASK,
            vocab_size,
        }
    }

    /// Advances the internal state and returns the next raw value.
    fn next_raw(&mut self) -> u64 {
        self.state = (self.state.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::INCREMENT)) & Self::MASK;
        self.state
    }

    /// Generates the next token in the sequence.
    pub fn next_token(&mut self) -> u16 {
        let raw = self.next_raw();
        // Use upper bits for better distribution
        ((raw >> 17) % (self.vocab_size as u64)) as u16
    }

    /// Generates a vector of `count` tokens.
    pub fn generate(&mut self, count: usize) -> Vec<u16> {
        (0..count).map(|_| self.next_token()).collect()
    }

    /// Generates tokens for a batch of sequences with the same length.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of sequences in the batch
    /// * `seq_len` - Length of each sequence
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector contains tokens for one batch element
    pub fn generate_batch(&mut self, batch_size: u32, seq_len: u32) -> Vec<Vec<u16>> {
        (0..batch_size)
            .map(|_| self.generate(seq_len as usize))
            .collect()
    }
}

/// Result of a prefill benchmark execution.
///
/// Contains all metrics required for JSONL output and analysis.
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Total time for prefill in milliseconds
    pub prefill_total_ms: f64,
    /// Total tokens processed (batch_size * seq_len)
    pub total_prompt_tokens: u32,
    /// Throughput in tokens per second
    pub prefill_tok_per_s: f64,
    /// Number of inference calls made
    pub num_infer_calls: u32,
    /// Per-batch time to first token in milliseconds
    pub ttft_ms_local: Vec<f64>,
    /// Minimum TTFT across batches
    pub ttft_min_ms: f64,
    /// Median TTFT across batches (50th percentile)
    pub ttft_p50_ms: f64,
    /// Maximum TTFT across batches
    pub ttft_max_ms: f64,
}

impl PrefillResult {
    /// Creates a PrefillResult from raw timing data.
    ///
    /// # Arguments
    ///
    /// * `ttft_ms_local` - Per-batch TTFT values in milliseconds
    /// * `total_prompt_tokens` - Total tokens processed
    /// * `num_infer_calls` - Number of inference calls made
    pub fn from_ttft(ttft_ms_local: Vec<f64>, total_prompt_tokens: u32, num_infer_calls: u32) -> Self {
        let prefill_total_ms = ttft_ms_local
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let prefill_tok_per_s = if prefill_total_ms > 0.0 {
            (total_prompt_tokens as f64) / (prefill_total_ms / 1000.0)
        } else {
            0.0
        };

        let (ttft_min_ms, ttft_p50_ms, ttft_max_ms) = compute_ttft_stats(&ttft_ms_local);

        Self {
            prefill_total_ms,
            total_prompt_tokens,
            prefill_tok_per_s,
            num_infer_calls,
            ttft_ms_local,
            ttft_min_ms,
            ttft_p50_ms,
            ttft_max_ms,
        }
    }
}

/// Computes TTFT statistics (min, median, max) from a slice of TTFT values.
///
/// # Arguments
///
/// * `ttft_values` - Slice of TTFT values in milliseconds
///
/// # Returns
///
/// Tuple of (min, median, max) TTFT values
fn compute_ttft_stats(ttft_values: &[f64]) -> (f64, f64, f64) {
    if ttft_values.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut sorted: Vec<f64> = ttft_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    // Compute median (p50)
    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    (min, median, max)
}

/// Tracks per-batch TTFT during prefill execution.
///
/// This struct records the time when each batch produces its first output,
/// enabling per-batch TTFT tracking even when inference is chunked.
#[derive(Debug)]
pub struct TtftTracker {
    /// Start time of the prefill operation
    start: Instant,
    /// Per-batch TTFT (None if batch hasn't produced output yet)
    ttft: Vec<Option<Duration>>,
    /// Number of inference calls made
    num_infer_calls: u32,
}

impl TtftTracker {
    /// Creates a new TTFT tracker for the given batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of batches to track
    pub fn new(batch_size: u32) -> Self {
        Self {
            start: Instant::now(),
            ttft: vec![None; batch_size as usize],
            num_infer_calls: 0,
        }
    }

    /// Starts the timer. Call this immediately before the first inference call.
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Records an inference call and checks which batches produced output.
    ///
    /// # Arguments
    ///
    /// * `batch_has_output` - A function that returns true if batch `i` has produced output
    ///
    /// # Returns
    ///
    /// `true` if all batches have produced output, `false` otherwise
    pub fn record_infer<F>(&mut self, batch_has_output: F) -> bool
    where
        F: Fn(usize) -> bool,
    {
        self.num_infer_calls += 1;
        let now = self.start.elapsed();

        let mut all_done = true;
        for (i, ttft) in self.ttft.iter_mut().enumerate() {
            if ttft.is_none() {
                if batch_has_output(i) {
                    *ttft = Some(now);
                } else {
                    all_done = false;
                }
            }
        }

        all_done
    }

    /// Checks if all batches have recorded their TTFT.
    pub fn all_done(&self) -> bool {
        self.ttft.iter().all(|t| t.is_some())
    }

    /// Returns the number of inference calls made.
    pub fn num_infer_calls(&self) -> u32 {
        self.num_infer_calls
    }

    /// Converts the tracker to a vector of TTFT values in milliseconds.
    ///
    /// # Panics
    ///
    /// Panics if any batch hasn't recorded its TTFT yet.
    pub fn to_ttft_ms(&self) -> Vec<f64> {
        self.ttft
            .iter()
            .map(|t| {
                t.expect("All batches should have recorded TTFT")
                    .as_secs_f64()
                    * 1000.0
            })
            .collect()
    }

    /// Computes the total prefill time (max TTFT across all batches).
    pub fn total_time(&self) -> Duration {
        self.ttft
            .iter()
            .filter_map(|t| *t)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Finalizes the tracker and returns a PrefillResult.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of batches
    /// * `seq_len` - Sequence length per batch
    ///
    /// # Panics
    ///
    /// Panics if any batch hasn't recorded its TTFT yet.
    pub fn finalize(&self, batch_size: u32, seq_len: u32) -> PrefillResult {
        let ttft_ms = self.to_ttft_ms();
        let total_prompt_tokens = batch_size * seq_len;
        PrefillResult::from_ttft(ttft_ms, total_prompt_tokens, self.num_infer_calls)
    }
}

/// Configuration for a prefill-uniform benchmark case.
#[derive(Debug, Clone)]
pub struct PrefillUniformConfig {
    /// Batch size
    pub batch_size: u32,
    /// Sequence length (uniform across batch)
    pub seq_len: u32,
    /// Token chunk size
    pub token_chunk_size: u32,
    /// Random seed for deterministic token generation
    pub seed: u64,
    /// Vocabulary size for token generation
    pub vocab_size: u32,
}

impl Default for PrefillUniformConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 128,
            token_chunk_size: 128,
            seed: 42,
            vocab_size: 65536, // Common RWKV vocab size
        }
    }
}

impl PrefillUniformConfig {
    /// Creates a new configuration with the given parameters.
    pub fn new(batch_size: u32, seq_len: u32, token_chunk_size: u32) -> Self {
        Self {
            batch_size,
            seq_len,
            token_chunk_size,
            ..Default::default()
        }
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the vocabulary size.
    pub fn with_vocab_size(mut self, vocab_size: u32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Generates input tokens for this configuration.
    pub fn generate_tokens(&self) -> Vec<Vec<u16>> {
        let mut gen = TokenGenerator::new(self.seed, self.vocab_size);
        gen.generate_batch(self.batch_size, self.seq_len)
    }

    /// Computes the total number of prompt tokens.
    pub fn total_prompt_tokens(&self) -> u32 {
        self.batch_size * self.seq_len
    }

    /// Estimates the number of inference calls needed.
    ///
    /// When seq_len > token_chunk_size, multiple infer() calls are needed.
    pub fn estimated_infer_calls(&self) -> u32 {
        if self.token_chunk_size == 0 {
            return 0;
        }
        (self.seq_len + self.token_chunk_size - 1) / self.token_chunk_size
    }
}

/// Describes the length mode used for prefill benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthMode {
    /// Canonical length list based on chunk size
    Canonical,
    /// Total-token target mode
    TotalTokenTarget,
}

/// Metadata about how a sequence length was derived.
#[derive(Debug, Clone)]
pub struct LengthMetadata {
    /// The sequence length
    pub seq_len: u32,
    /// Mode used to derive this length
    pub mode: LengthMode,
    /// For target mode: the original total token target
    pub total_token_target: Option<u32>,
    /// Human-readable description of how this length was derived
    pub description: String,
}

impl LengthMetadata {
    /// Creates metadata for a canonical length.
    pub fn canonical(seq_len: u32, token_chunk_size: u32) -> Self {
        let desc = describe_canonical_length(seq_len, token_chunk_size);
        Self {
            seq_len,
            mode: LengthMode::Canonical,
            total_token_target: None,
            description: desc,
        }
    }

    /// Creates metadata for a target-mode length.
    pub fn target_mode(seq_len: u32, total_tokens: u32, batch_size: u32) -> Self {
        Self {
            seq_len,
            mode: LengthMode::TotalTokenTarget,
            total_token_target: Some(total_tokens),
            description: format!("T={} / B={} = L={}", total_tokens, batch_size, seq_len),
        }
    }
}

/// Describes how a canonical length relates to the chunk size.
fn describe_canonical_length(seq_len: u32, token_chunk_size: u32) -> String {
    let c = token_chunk_size;
    if seq_len == 1 {
        "L=1".to_string()
    } else if seq_len == c / 4 {
        format!("L=C/4={}", seq_len)
    } else if seq_len == c / 2 {
        format!("L=C/2={}", seq_len)
    } else if seq_len == c.saturating_sub(1) {
        format!("L=C-1={}", seq_len)
    } else if seq_len == c {
        format!("L=C={}", seq_len)
    } else if seq_len == c + 1 {
        format!("L=C+1={}", seq_len)
    } else if seq_len == c * 2 {
        format!("L=2C={}", seq_len)
    } else if seq_len == c * 4 {
        format!("L=4C={}", seq_len)
    } else if seq_len == c * 8 {
        format!("L=8C={}", seq_len)
    } else {
        format!("L={}", seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_length_list_128() {
        let lengths = canonical_length_list(128);
        // Should contain: 1, 32, 64, 127, 128, 129, 256, 512, 1024
        assert!(lengths.contains(&1));
        assert!(lengths.contains(&32));   // C/4
        assert!(lengths.contains(&64));   // C/2
        assert!(lengths.contains(&127));  // C-1
        assert!(lengths.contains(&128));  // C
        assert!(lengths.contains(&129));  // C+1
        assert!(lengths.contains(&256));  // 2C
        assert!(lengths.contains(&512));  // 4C
        assert!(lengths.contains(&1024)); // 8C

        // Should be sorted
        for i in 1..lengths.len() {
            assert!(lengths[i] > lengths[i - 1], "Should be sorted");
        }

        // No duplicates
        let unique_len = lengths.len();
        let mut dedup = lengths.clone();
        dedup.dedup();
        assert_eq!(unique_len, dedup.len(), "Should have no duplicates");
    }

    #[test]
    fn test_canonical_length_list_256() {
        let lengths = canonical_length_list(256);
        // Should contain: 1, 64, 128, 255, 256, 257, 512, 1024, 2048
        assert!(lengths.contains(&1));
        assert!(lengths.contains(&64));    // C/4
        assert!(lengths.contains(&128));   // C/2
        assert!(lengths.contains(&255));   // C-1
        assert!(lengths.contains(&256));   // C
        assert!(lengths.contains(&257));   // C+1
        assert!(lengths.contains(&512));   // 2C
        assert!(lengths.contains(&1024));  // 4C
        assert!(lengths.contains(&2048));  // 8C
    }

    #[test]
    fn test_canonical_length_list_small_chunk() {
        // For very small chunk sizes, some values might collapse
        let lengths = canonical_length_list(4);
        // C=4: 1, 1, 2, 3, 4, 5, 8, 16, 32 -> after dedup: 1, 2, 3, 4, 5, 8, 16, 32
        assert!(lengths.contains(&1));
        assert!(lengths.contains(&2));
        assert!(lengths.contains(&3));
        assert!(lengths.contains(&4));
        assert!(lengths.contains(&5));
        assert!(lengths.contains(&8));
        assert!(lengths.contains(&16));
        assert!(lengths.contains(&32));
    }

    #[test]
    fn test_total_token_targets() {
        let targets = total_token_targets(256);
        assert_eq!(targets, vec![128, 256, 512, 1024, 2048]);

        let targets = total_token_targets(128);
        assert_eq!(targets, vec![64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_seq_len_from_total_tokens() {
        // Exact division
        assert_eq!(seq_len_from_total_tokens(1024, 4), 256);
        assert_eq!(seq_len_from_total_tokens(100, 10), 10);

        // Ceiling division
        assert_eq!(seq_len_from_total_tokens(100, 8), 13); // ceil(100/8) = 13
        assert_eq!(seq_len_from_total_tokens(101, 10), 11); // ceil(101/10) = 11
        assert_eq!(seq_len_from_total_tokens(1, 4), 1); // ceil(1/4) = 1

        // Edge case: batch_size = 0
        assert_eq!(seq_len_from_total_tokens(100, 0), 0);
    }

    #[test]
    fn test_target_mode_lengths() {
        let lengths = target_mode_lengths(256, 4);
        // For B=4, C=256:
        // T=128 -> L=32, T=256 -> L=64, T=512 -> L=128, T=1024 -> L=256, T=2048 -> L=512
        assert_eq!(lengths, vec![32, 64, 128, 256, 512]);
    }

    #[test]
    fn test_target_mode_lengths_batch_1() {
        let lengths = target_mode_lengths(128, 1);
        // For B=1, L = T
        // Targets: 64, 128, 256, 512, 1024
        assert_eq!(lengths, vec![64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_all_prefill_lengths() {
        let lengths = all_prefill_lengths(128, 4);

        // Should contain canonical lengths
        assert!(lengths.contains(&1));
        assert!(lengths.contains(&128)); // C
        assert!(lengths.contains(&1024)); // 8C

        // Should contain target-mode lengths
        // For B=4, C=128: T=[64,128,256,512,1024] -> L=[16,32,64,128,256]
        assert!(lengths.contains(&16));
        assert!(lengths.contains(&256));

        // Should be sorted and deduplicated
        for i in 1..lengths.len() {
            assert!(lengths[i] > lengths[i - 1], "Should be sorted with no duplicates");
        }
    }

    #[test]
    fn test_token_generator_deterministic() {
        let mut gen1 = TokenGenerator::new(12345, 50257);
        let tokens1 = gen1.generate(100);

        let mut gen2 = TokenGenerator::new(12345, 50257);
        let tokens2 = gen2.generate(100);

        assert_eq!(tokens1, tokens2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_token_generator_different_seeds() {
        let mut gen1 = TokenGenerator::new(12345, 50257);
        let tokens1 = gen1.generate(100);

        let mut gen2 = TokenGenerator::new(54321, 50257);
        let tokens2 = gen2.generate(100);

        assert_ne!(tokens1, tokens2, "Different seeds should produce different sequences");
    }

    #[test]
    fn test_token_generator_range() {
        let vocab_size = 1000u32;
        let mut gen = TokenGenerator::new(42, vocab_size);

        for _ in 0..1000 {
            let token = gen.next_token();
            assert!(
                (token as u32) < vocab_size,
                "Token {} should be < vocab_size {}",
                token,
                vocab_size
            );
        }
    }

    #[test]
    fn test_token_generator_batch() {
        let mut gen = TokenGenerator::new(42, 65536);
        let batch = gen.generate_batch(4, 128);

        assert_eq!(batch.len(), 4, "Should have 4 batch elements");
        for (i, tokens) in batch.iter().enumerate() {
            assert_eq!(
                tokens.len(),
                128,
                "Batch element {} should have 128 tokens",
                i
            );
        }
    }

    #[test]
    fn test_compute_ttft_stats() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (min, p50, max) = compute_ttft_stats(&values);

        assert_eq!(min, 10.0);
        assert_eq!(max, 50.0);
        assert_eq!(p50, 30.0); // Odd count: middle element

        let values = vec![10.0, 20.0, 30.0, 40.0];
        let (min, p50, max) = compute_ttft_stats(&values);

        assert_eq!(min, 10.0);
        assert_eq!(max, 40.0);
        assert_eq!(p50, 25.0); // Even count: average of middle two
    }

    #[test]
    fn test_compute_ttft_stats_empty() {
        let (min, p50, max) = compute_ttft_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(p50, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_compute_ttft_stats_single() {
        let (min, p50, max) = compute_ttft_stats(&[42.0]);
        assert_eq!(min, 42.0);
        assert_eq!(p50, 42.0);
        assert_eq!(max, 42.0);
    }

    #[test]
    fn test_prefill_result_from_ttft() {
        let ttft_ms = vec![10.0, 20.0, 30.0, 40.0];
        let result = PrefillResult::from_ttft(ttft_ms.clone(), 512, 2);

        assert_eq!(result.prefill_total_ms, 40.0); // Max TTFT
        assert_eq!(result.total_prompt_tokens, 512);
        assert_eq!(result.num_infer_calls, 2);
        assert_eq!(result.ttft_ms_local, ttft_ms);
        assert_eq!(result.ttft_min_ms, 10.0);
        assert_eq!(result.ttft_max_ms, 40.0);
        // p50 = (20 + 30) / 2 = 25
        assert_eq!(result.ttft_p50_ms, 25.0);

        // prefill_tok_per_s = 512 / (40ms / 1000) = 512 / 0.04 = 12800
        assert!((result.prefill_tok_per_s - 12800.0).abs() < 0.1);
    }

    #[test]
    fn test_prefill_uniform_config_default() {
        let config = PrefillUniformConfig::default();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.seq_len, 128);
        assert_eq!(config.token_chunk_size, 128);
        assert_eq!(config.seed, 42);
        assert_eq!(config.vocab_size, 65536);
    }

    #[test]
    fn test_prefill_uniform_config_builder() {
        let config = PrefillUniformConfig::new(4, 256, 128)
            .with_seed(12345)
            .with_vocab_size(50257);

        assert_eq!(config.batch_size, 4);
        assert_eq!(config.seq_len, 256);
        assert_eq!(config.token_chunk_size, 128);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.vocab_size, 50257);
    }

    #[test]
    fn test_prefill_uniform_config_total_tokens() {
        let config = PrefillUniformConfig::new(4, 256, 128);
        assert_eq!(config.total_prompt_tokens(), 1024);
    }

    #[test]
    fn test_prefill_uniform_config_estimated_infer_calls() {
        let config = PrefillUniformConfig::new(4, 256, 128);
        assert_eq!(config.estimated_infer_calls(), 2); // ceil(256/128) = 2

        let config = PrefillUniformConfig::new(4, 128, 128);
        assert_eq!(config.estimated_infer_calls(), 1); // ceil(128/128) = 1

        let config = PrefillUniformConfig::new(4, 129, 128);
        assert_eq!(config.estimated_infer_calls(), 2); // ceil(129/128) = 2

        let config = PrefillUniformConfig::new(4, 1024, 128);
        assert_eq!(config.estimated_infer_calls(), 8); // ceil(1024/128) = 8
    }

    #[test]
    fn test_prefill_uniform_config_generate_tokens() {
        let config = PrefillUniformConfig::new(4, 128, 128).with_seed(42);
        let tokens = config.generate_tokens();

        assert_eq!(tokens.len(), 4);
        for batch in &tokens {
            assert_eq!(batch.len(), 128);
        }

        // Should be deterministic
        let tokens2 = config.generate_tokens();
        assert_eq!(tokens, tokens2);
    }

    #[test]
    fn test_ttft_tracker() {
        let mut tracker = TtftTracker::new(4);
        tracker.start();

        // First infer: batches 0 and 1 produce output
        let done = tracker.record_infer(|i| i < 2);
        assert!(!done);
        assert_eq!(tracker.num_infer_calls(), 1);

        // Second infer: all batches produce output
        let done = tracker.record_infer(|_| true);
        assert!(done);
        assert_eq!(tracker.num_infer_calls(), 2);

        // All should be done
        assert!(tracker.all_done());

        // Should be able to get TTFT values
        let ttft_ms = tracker.to_ttft_ms();
        assert_eq!(ttft_ms.len(), 4);
    }

    #[test]
    fn test_length_metadata_canonical() {
        let meta = LengthMetadata::canonical(128, 128);
        assert_eq!(meta.seq_len, 128);
        assert_eq!(meta.mode, LengthMode::Canonical);
        assert!(meta.total_token_target.is_none());
        assert!(meta.description.contains("C="));
    }

    #[test]
    fn test_length_metadata_target_mode() {
        let meta = LengthMetadata::target_mode(256, 1024, 4);
        assert_eq!(meta.seq_len, 256);
        assert_eq!(meta.mode, LengthMode::TotalTokenTarget);
        assert_eq!(meta.total_token_target, Some(1024));
        assert!(meta.description.contains("T=1024"));
        assert!(meta.description.contains("B=4"));
        assert!(meta.description.contains("L=256"));
    }

    #[test]
    fn test_describe_canonical_length() {
        assert_eq!(describe_canonical_length(1, 128), "L=1");
        assert_eq!(describe_canonical_length(32, 128), "L=C/4=32");
        assert_eq!(describe_canonical_length(64, 128), "L=C/2=64");
        assert_eq!(describe_canonical_length(127, 128), "L=C-1=127");
        assert_eq!(describe_canonical_length(128, 128), "L=C=128");
        assert_eq!(describe_canonical_length(129, 128), "L=C+1=129");
        assert_eq!(describe_canonical_length(256, 128), "L=2C=256");
        assert_eq!(describe_canonical_length(512, 128), "L=4C=512");
        assert_eq!(describe_canonical_length(1024, 128), "L=8C=1024");
        assert_eq!(describe_canonical_length(100, 128), "L=100");
    }
}
