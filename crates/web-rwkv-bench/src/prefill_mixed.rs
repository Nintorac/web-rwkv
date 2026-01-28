//! Prefill-mixed benchmark scenario with named case patterns.
//!
//! This module provides deterministic length vector generation for mixed-batch
//! prefill benchmarks. Each named case pattern produces a reproducible sequence
//! of lengths from only `(mixed_case_id, batch_size, chunk_size)`.
//!
//! # Named Cases
//!
//! - `staircase_8`: Spread from very short to very long lengths
//! - `bimodal_half`: Two buckets - short and long
//! - `one_long_rest_short`: One slot at maximum, rest at minimum
//! - `realistic_chat_scaled`: Hand-curated pattern simulating chat workloads
//!
//! # Determinism
//!
//! The length vector for a mixed case is deterministic solely from
//! `(mixed_case_id, batch_size, chunk_size)`. No PRNG is involved in length
//! generation - only the token *contents* use PRNG seeds.
//!
//! # Scaling Rules
//!
//! For adapting base patterns to arbitrary batch sizes:
//! 1. Compute base integer lengths using integer arithmetic
//! 2. If `B <= base_len`: take the first `B` elements
//! 3. If `B > base_len`: repeat the base pattern until length >= B, then truncate
//! 4. Ensure all lengths are >= 1

use std::fmt;

/// Error type for mixed case pattern operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MixedCaseError {
    /// Unknown mixed case identifier
    UnknownCaseId(String),
    /// Invalid parameter (batch_size or chunk_size is 0)
    InvalidParameter(String),
}

impl fmt::Display for MixedCaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MixedCaseError::UnknownCaseId(id) => write!(f, "Unknown mixed case ID: {}", id),
            MixedCaseError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::error::Error for MixedCaseError {}

/// Result type for mixed case operations.
pub type MixedCaseResult<T> = Result<T, MixedCaseError>;

/// Named mixed case identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixedCaseId {
    /// Staircase pattern: spread from very short to very long
    Staircase8,
    /// Bimodal pattern: half short, half long
    BimodalHalf,
    /// One long sequence, rest short
    OneLongRestShort,
    /// Realistic chat workload pattern
    RealisticChatScaled,
}

impl MixedCaseId {
    /// All available mixed case IDs.
    pub const ALL: &'static [MixedCaseId] = &[
        MixedCaseId::Staircase8,
        MixedCaseId::BimodalHalf,
        MixedCaseId::OneLongRestShort,
        MixedCaseId::RealisticChatScaled,
    ];

    /// Convert from string identifier.
    pub fn from_str(s: &str) -> Option<MixedCaseId> {
        match s {
            "staircase_8" => Some(MixedCaseId::Staircase8),
            "bimodal_half" => Some(MixedCaseId::BimodalHalf),
            "one_long_rest_short" => Some(MixedCaseId::OneLongRestShort),
            "realistic_chat_scaled" => Some(MixedCaseId::RealisticChatScaled),
            _ => None,
        }
    }

    /// Convert to string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            MixedCaseId::Staircase8 => "staircase_8",
            MixedCaseId::BimodalHalf => "bimodal_half",
            MixedCaseId::OneLongRestShort => "one_long_rest_short",
            MixedCaseId::RealisticChatScaled => "realistic_chat_scaled",
        }
    }
}

impl fmt::Display for MixedCaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Generate a length vector for a mixed case pattern.
///
/// This function is the main entry point for generating deterministic length
/// vectors from `(mixed_case_id, batch_size, chunk_size)`.
///
/// # Arguments
///
/// * `mixed_case_id` - The named case pattern to use
/// * `batch_size` - Number of batch elements (B)
/// * `chunk_size` - Token chunk size (C)
///
/// # Returns
///
/// A vector of length `batch_size` where each element is a sequence length >= 1.
///
/// # Errors
///
/// Returns an error if `batch_size` or `chunk_size` is 0.
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_mixed::{generate_lengths, MixedCaseId};
///
/// let lengths = generate_lengths(MixedCaseId::Staircase8, 8, 256).unwrap();
/// assert_eq!(lengths.len(), 8);
/// assert!(lengths.iter().all(|&l| l >= 1));
/// ```
pub fn generate_lengths(
    mixed_case_id: MixedCaseId,
    batch_size: u32,
    chunk_size: u32,
) -> MixedCaseResult<Vec<u32>> {
    if batch_size == 0 {
        return Err(MixedCaseError::InvalidParameter(
            "batch_size must be > 0".to_string(),
        ));
    }
    if chunk_size == 0 {
        return Err(MixedCaseError::InvalidParameter(
            "chunk_size must be > 0".to_string(),
        ));
    }

    let lengths = match mixed_case_id {
        MixedCaseId::Staircase8 => generate_staircase_8(batch_size, chunk_size),
        MixedCaseId::BimodalHalf => generate_bimodal_half(batch_size, chunk_size),
        MixedCaseId::OneLongRestShort => generate_one_long_rest_short(batch_size, chunk_size),
        MixedCaseId::RealisticChatScaled => generate_realistic_chat_scaled(batch_size, chunk_size),
    };

    Ok(lengths)
}

/// Generate lengths from a string mixed_case_id.
///
/// This is a convenience wrapper around [`generate_lengths`] that parses the
/// mixed case ID from a string.
///
/// # Example
///
/// ```
/// use web_rwkv_bench::prefill_mixed::generate_lengths_from_str;
///
/// let lengths = generate_lengths_from_str("staircase_8", 8, 256).unwrap();
/// assert_eq!(lengths.len(), 8);
/// ```
pub fn generate_lengths_from_str(
    mixed_case_id: &str,
    batch_size: u32,
    chunk_size: u32,
) -> MixedCaseResult<Vec<u32>> {
    let case_id = MixedCaseId::from_str(mixed_case_id)
        .ok_or_else(|| MixedCaseError::UnknownCaseId(mixed_case_id.to_string()))?;
    generate_lengths(case_id, batch_size, chunk_size)
}

/// Generate staircase_8 pattern.
///
/// For B=8: `[C/16, C/8, C/4, C/2, 3C/4, C, 2C, 4C]`
///
/// For other B: truncate or repeat the pattern.
fn generate_staircase_8(batch_size: u32, chunk_size: u32) -> Vec<u32> {
    // Base pattern for B=8 (using integer division)
    let c = chunk_size;
    let base_pattern: Vec<u32> = vec![
        c / 16,    // Very short
        c / 8,     // Short
        c / 4,     // Quarter
        c / 2,     // Half
        3 * c / 4, // Three quarters
        c,         // Full chunk
        2 * c,     // Double chunk
        4 * c,     // Quadruple chunk
    ];

    scale_pattern_to_batch(&base_pattern, batch_size)
}

/// Generate bimodal_half pattern.
///
/// Two buckets: short `C/8`, long `4C`.
/// If B is odd, the "long" side gets the extra element.
/// Deterministic ordering: `[long x ceil(B/2)] + [short x floor(B/2)]`
fn generate_bimodal_half(batch_size: u32, chunk_size: u32) -> Vec<u32> {
    let short_len = ensure_min_one(chunk_size / 8);
    let long_len = ensure_min_one(4 * chunk_size);

    let num_long = (batch_size + 1) / 2; // ceil(B/2)
    let num_short = batch_size / 2; // floor(B/2)

    let mut lengths = Vec::with_capacity(batch_size as usize);

    // Long elements first
    for _ in 0..num_long {
        lengths.push(long_len);
    }

    // Short elements second
    for _ in 0..num_short {
        lengths.push(short_len);
    }

    lengths
}

/// Generate one_long_rest_short pattern.
///
/// `[8C] + [C/8] * (B-1)`
fn generate_one_long_rest_short(batch_size: u32, chunk_size: u32) -> Vec<u32> {
    let long_len = ensure_min_one(8 * chunk_size);
    let short_len = ensure_min_one(chunk_size / 8);

    let mut lengths = Vec::with_capacity(batch_size as usize);

    // One long element
    lengths.push(long_len);

    // Rest are short
    for _ in 1..batch_size {
        lengths.push(short_len);
    }

    lengths
}

/// Generate realistic_chat_scaled pattern.
///
/// A hand-curated increasing vector for C=256:
/// `[32, 64, 96, 128, 192, 256, 384, 512]`
///
/// Scale deterministically to other C using:
/// `len_i = max(1, base_i * C / 256)`
fn generate_realistic_chat_scaled(batch_size: u32, chunk_size: u32) -> Vec<u32> {
    // Base pattern for C=256
    const BASE_C: u32 = 256;
    let base_pattern: Vec<u32> = vec![32, 64, 96, 128, 192, 256, 384, 512];

    // Scale to current chunk_size
    let scaled_pattern: Vec<u32> = base_pattern
        .iter()
        .map(|&base_len| ensure_min_one(base_len * chunk_size / BASE_C))
        .collect();

    scale_pattern_to_batch(&scaled_pattern, batch_size)
}

/// Scale a base pattern to an arbitrary batch size.
///
/// Rules:
/// 1. If `B <= base_len`: take the first `B` elements
/// 2. If `B > base_len`: repeat the base pattern until length >= B, then truncate
/// 3. Ensure all lengths are >= 1
fn scale_pattern_to_batch(base_pattern: &[u32], batch_size: u32) -> Vec<u32> {
    let base_len = base_pattern.len();
    let b = batch_size as usize;

    let mut result = Vec::with_capacity(b);

    if b <= base_len {
        // Truncate: take first B elements
        result.extend(base_pattern[..b].iter().map(|&l| ensure_min_one(l)));
    } else {
        // Repeat: cycle through the pattern
        for i in 0..b {
            let idx = i % base_len;
            result.push(ensure_min_one(base_pattern[idx]));
        }
    }

    result
}

/// Ensure a length is at least 1.
#[inline]
fn ensure_min_one(len: u32) -> u32 {
    len.max(1)
}

/// Get all available mixed case IDs as strings.
pub fn all_mixed_case_ids() -> Vec<&'static str> {
    MixedCaseId::ALL.iter().map(|id| id.as_str()).collect()
}

/// Compute total tokens for a length vector.
pub fn total_tokens(lengths: &[u32]) -> u64 {
    lengths.iter().map(|&l| l as u64).sum()
}

// =============================================================================
// TTFT TRACKING FOR MIXED PREFILL
// =============================================================================

use std::time::{Duration, Instant};

use crate::prefill_uniform::compute_ttft_stats;

/// Result of a prefill-mixed benchmark execution.
///
/// Contains all metrics required for JSONL output and analysis.
/// Unlike `PrefillResult` in prefill_uniform, this struct handles
/// sequences of different lengths within a batch.
#[derive(Debug, Clone)]
pub struct PrefillMixedResult {
    /// Total time for prefill in milliseconds (max TTFT across all batches)
    pub prefill_total_ms: f64,
    /// Total tokens processed across all sequences (sum of all lengths)
    pub total_prompt_tokens: u32,
    /// Throughput in tokens per second (total_prompt_tokens / prefill_total_s)
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

impl PrefillMixedResult {
    /// Creates a PrefillMixedResult from raw timing data.
    ///
    /// This is the primary constructor for mixed-batch prefill results.
    ///
    /// # Arguments
    ///
    /// * `ttft_ms_local` - Per-batch TTFT values in milliseconds
    /// * `lengths` - Per-batch sequence lengths (to compute total tokens)
    /// * `num_infer_calls` - Number of inference calls made
    ///
    /// # Example
    ///
    /// ```
    /// use web_rwkv_bench::prefill_mixed::PrefillMixedResult;
    ///
    /// let ttft_ms = vec![10.0, 15.0, 25.0, 40.0];
    /// let lengths = vec![100, 200, 300, 400];
    /// let result = PrefillMixedResult::from_ttft(ttft_ms, &lengths, 4);
    ///
    /// assert_eq!(result.total_prompt_tokens, 1000); // 100+200+300+400
    /// assert_eq!(result.prefill_total_ms, 40.0);   // max TTFT
    /// assert_eq!(result.ttft_min_ms, 10.0);
    /// assert_eq!(result.ttft_max_ms, 40.0);
    /// ```
    pub fn from_ttft(ttft_ms_local: Vec<f64>, lengths: &[u32], num_infer_calls: u32) -> Self {
        let prefill_total_ms = ttft_ms_local
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let total_prompt_tokens = total_tokens(lengths) as u32;

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

/// Tracks per-batch TTFT during mixed-prefill execution.
///
/// This struct records the time when each batch produces its first output,
/// enabling per-batch TTFT tracking for mixed-length scenarios where
/// different batches have different sequence lengths.
///
/// In a mixed scenario, shorter sequences will typically complete first,
/// so their TTFT will be recorded before longer sequences.
#[derive(Debug)]
pub struct MixedTtftTracker {
    /// Start time of the prefill operation
    start: Instant,
    /// Per-batch sequence lengths
    lengths: Vec<u32>,
    /// Per-batch TTFT (None if batch hasn't produced output yet)
    ttft: Vec<Option<Duration>>,
    /// Number of inference calls made
    num_infer_calls: u32,
}

impl MixedTtftTracker {
    /// Creates a new mixed TTFT tracker with the given sequence lengths.
    ///
    /// # Arguments
    ///
    /// * `lengths` - Per-batch sequence lengths
    ///
    /// # Example
    ///
    /// ```
    /// use web_rwkv_bench::prefill_mixed::MixedTtftTracker;
    ///
    /// let lengths = vec![100, 200, 300, 400];
    /// let tracker = MixedTtftTracker::new(lengths);
    /// ```
    pub fn new(lengths: Vec<u32>) -> Self {
        let batch_size = lengths.len();
        Self {
            start: Instant::now(),
            lengths,
            ttft: vec![None; batch_size],
            num_infer_calls: 0,
        }
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.lengths.len()
    }

    /// Returns the sequence lengths.
    pub fn lengths(&self) -> &[u32] {
        &self.lengths
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

    /// Finalizes the tracker and returns a PrefillMixedResult.
    ///
    /// # Panics
    ///
    /// Panics if any batch hasn't recorded its TTFT yet.
    pub fn finalize(&self) -> PrefillMixedResult {
        let ttft_ms = self.to_ttft_ms();
        PrefillMixedResult::from_ttft(ttft_ms, &self.lengths, self.num_infer_calls)
    }
}

/// Configuration for a prefill-mixed benchmark case.
#[derive(Debug, Clone)]
pub struct PrefillMixedConfig {
    /// Mixed case identifier
    pub mixed_case_id: MixedCaseId,
    /// Batch size
    pub batch_size: u32,
    /// Token chunk size
    pub token_chunk_size: u32,
    /// Random seed for deterministic token generation
    pub seed: u64,
    /// Vocabulary size for token generation
    pub vocab_size: u32,
}

impl Default for PrefillMixedConfig {
    fn default() -> Self {
        Self {
            mixed_case_id: MixedCaseId::Staircase8,
            batch_size: 8,
            token_chunk_size: 256,
            seed: 42,
            vocab_size: 65536, // Common RWKV vocab size
        }
    }
}

impl PrefillMixedConfig {
    /// Creates a new configuration with the given parameters.
    pub fn new(mixed_case_id: MixedCaseId, batch_size: u32, token_chunk_size: u32) -> Self {
        Self {
            mixed_case_id,
            batch_size,
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

    /// Generates the length vector for this configuration.
    pub fn generate_lengths(&self) -> MixedCaseResult<Vec<u32>> {
        generate_lengths(self.mixed_case_id, self.batch_size, self.token_chunk_size)
    }

    /// Generates input tokens for this configuration.
    ///
    /// Returns a vector of token vectors, where each inner vector
    /// corresponds to one batch element with its specific length.
    pub fn generate_tokens(&self) -> MixedCaseResult<Vec<Vec<u16>>> {
        use crate::prefill_uniform::TokenGenerator;

        let lengths = self.generate_lengths()?;
        let mut gen = TokenGenerator::new(self.seed, self.vocab_size);

        let tokens = lengths
            .iter()
            .map(|&len| gen.generate(len as usize))
            .collect();

        Ok(tokens)
    }

    /// Computes the total number of prompt tokens.
    pub fn total_prompt_tokens(&self) -> MixedCaseResult<u64> {
        let lengths = self.generate_lengths()?;
        Ok(total_tokens(&lengths))
    }

    /// Creates a MixedTtftTracker for this configuration.
    pub fn create_tracker(&self) -> MixedCaseResult<MixedTtftTracker> {
        let lengths = self.generate_lengths()?;
        Ok(MixedTtftTracker::new(lengths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_case_id_roundtrip() {
        for &id in MixedCaseId::ALL {
            let s = id.as_str();
            let parsed = MixedCaseId::from_str(s);
            assert_eq!(parsed, Some(id), "Failed roundtrip for {}", s);
        }
    }

    #[test]
    fn test_unknown_case_id() {
        assert!(MixedCaseId::from_str("unknown_pattern").is_none());
    }

    #[test]
    fn test_generate_lengths_from_str_unknown() {
        let result = generate_lengths_from_str("unknown_pattern", 8, 256);
        assert!(matches!(result, Err(MixedCaseError::UnknownCaseId(_))));
    }

    #[test]
    fn test_staircase_8_base_case() {
        let lengths = generate_lengths(MixedCaseId::Staircase8, 8, 256).unwrap();
        assert_eq!(lengths.len(), 8);

        // Expected for C=256: [16, 32, 64, 128, 192, 256, 512, 1024]
        let expected = vec![16, 32, 64, 128, 192, 256, 512, 1024];
        assert_eq!(lengths, expected);
    }

    #[test]
    fn test_staircase_8_small_chunk() {
        // With C=32, some divisions would give 0, should be clamped to 1
        let lengths = generate_lengths(MixedCaseId::Staircase8, 8, 32).unwrap();
        assert_eq!(lengths.len(), 8);
        assert!(lengths.iter().all(|&l| l >= 1), "All lengths must be >= 1");
    }

    #[test]
    fn test_staircase_8_truncate() {
        // B=4 should take first 4 elements
        let lengths = generate_lengths(MixedCaseId::Staircase8, 4, 256).unwrap();
        assert_eq!(lengths.len(), 4);
        assert_eq!(lengths, vec![16, 32, 64, 128]);
    }

    #[test]
    fn test_staircase_8_repeat() {
        // B=16 should repeat the 8-element pattern twice
        let lengths = generate_lengths(MixedCaseId::Staircase8, 16, 256).unwrap();
        assert_eq!(lengths.len(), 16);

        // First 8 should equal second 8
        assert_eq!(lengths[0..8], lengths[8..16]);
    }

    #[test]
    fn test_bimodal_half_even() {
        // B=8: 4 long, 4 short
        let lengths = generate_lengths(MixedCaseId::BimodalHalf, 8, 256).unwrap();
        assert_eq!(lengths.len(), 8);

        // Expected: [1024, 1024, 1024, 1024, 32, 32, 32, 32]
        let long_len = 4 * 256; // 1024
        let short_len = 256 / 8; // 32
        assert_eq!(&lengths[0..4], &[long_len; 4]);
        assert_eq!(&lengths[4..8], &[short_len; 4]);
    }

    #[test]
    fn test_bimodal_half_odd() {
        // B=5: ceil(5/2)=3 long, floor(5/2)=2 short
        let lengths = generate_lengths(MixedCaseId::BimodalHalf, 5, 256).unwrap();
        assert_eq!(lengths.len(), 5);

        let long_len = 4 * 256; // 1024
        let short_len = 256 / 8; // 32
        assert_eq!(&lengths[0..3], &[long_len; 3]);
        assert_eq!(&lengths[3..5], &[short_len; 2]);
    }

    #[test]
    fn test_bimodal_half_b1() {
        // B=1: ceil(1/2)=1 long, floor(1/2)=0 short
        let lengths = generate_lengths(MixedCaseId::BimodalHalf, 1, 256).unwrap();
        assert_eq!(lengths.len(), 1);
        assert_eq!(lengths[0], 4 * 256);
    }

    #[test]
    fn test_one_long_rest_short() {
        // B=8: [8C] + [C/8] * 7
        let lengths = generate_lengths(MixedCaseId::OneLongRestShort, 8, 256).unwrap();
        assert_eq!(lengths.len(), 8);

        assert_eq!(lengths[0], 8 * 256); // 2048
        for i in 1..8 {
            assert_eq!(lengths[i], 256 / 8); // 32
        }
    }

    #[test]
    fn test_one_long_rest_short_b1() {
        // B=1: just one long element
        let lengths = generate_lengths(MixedCaseId::OneLongRestShort, 1, 256).unwrap();
        assert_eq!(lengths.len(), 1);
        assert_eq!(lengths[0], 8 * 256);
    }

    #[test]
    fn test_realistic_chat_scaled_base() {
        // C=256: base pattern
        let lengths = generate_lengths(MixedCaseId::RealisticChatScaled, 8, 256).unwrap();
        assert_eq!(lengths.len(), 8);
        assert_eq!(lengths, vec![32, 64, 96, 128, 192, 256, 384, 512]);
    }

    #[test]
    fn test_realistic_chat_scaled_double() {
        // C=512: double the base pattern
        let lengths = generate_lengths(MixedCaseId::RealisticChatScaled, 8, 512).unwrap();
        assert_eq!(lengths.len(), 8);
        assert_eq!(lengths, vec![64, 128, 192, 256, 384, 512, 768, 1024]);
    }

    #[test]
    fn test_realistic_chat_scaled_half() {
        // C=128: half the base pattern
        let lengths = generate_lengths(MixedCaseId::RealisticChatScaled, 8, 128).unwrap();
        assert_eq!(lengths.len(), 8);
        assert_eq!(lengths, vec![16, 32, 48, 64, 96, 128, 192, 256]);
    }

    #[test]
    fn test_realistic_chat_scaled_small_chunk() {
        // C=32: small pattern, some values should be clamped to 1
        let lengths = generate_lengths(MixedCaseId::RealisticChatScaled, 8, 32).unwrap();
        assert_eq!(lengths.len(), 8);
        assert!(lengths.iter().all(|&l| l >= 1));
        // Expected: [4, 8, 12, 16, 24, 32, 48, 64]
        assert_eq!(lengths, vec![4, 8, 12, 16, 24, 32, 48, 64]);
    }

    #[test]
    fn test_all_patterns_min_length() {
        // Test that all patterns produce lengths >= 1 for various inputs
        for &case_id in MixedCaseId::ALL {
            for batch_size in [1, 2, 4, 8, 16, 32] {
                for chunk_size in [32, 64, 128, 256, 512] {
                    let lengths = generate_lengths(case_id, batch_size, chunk_size).unwrap();
                    assert_eq!(
                        lengths.len(),
                        batch_size as usize,
                        "Wrong length for {:?} B={} C={}",
                        case_id,
                        batch_size,
                        chunk_size
                    );
                    assert!(
                        lengths.iter().all(|&l| l >= 1),
                        "Length < 1 for {:?} B={} C={}",
                        case_id,
                        batch_size,
                        chunk_size
                    );
                }
            }
        }
    }

    #[test]
    fn test_determinism() {
        // Same inputs should always produce same outputs
        for &case_id in MixedCaseId::ALL {
            let lengths1 = generate_lengths(case_id, 8, 256).unwrap();
            let lengths2 = generate_lengths(case_id, 8, 256).unwrap();
            assert_eq!(lengths1, lengths2, "Non-deterministic for {:?}", case_id);
        }
    }

    #[test]
    fn test_invalid_batch_size() {
        let result = generate_lengths(MixedCaseId::Staircase8, 0, 256);
        assert!(matches!(result, Err(MixedCaseError::InvalidParameter(_))));
    }

    #[test]
    fn test_invalid_chunk_size() {
        let result = generate_lengths(MixedCaseId::Staircase8, 8, 0);
        assert!(matches!(result, Err(MixedCaseError::InvalidParameter(_))));
    }

    #[test]
    fn test_all_mixed_case_ids() {
        let ids = all_mixed_case_ids();
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&"staircase_8"));
        assert!(ids.contains(&"bimodal_half"));
        assert!(ids.contains(&"one_long_rest_short"));
        assert!(ids.contains(&"realistic_chat_scaled"));
    }

    #[test]
    fn test_total_tokens() {
        let lengths = vec![100, 200, 300];
        assert_eq!(total_tokens(&lengths), 600);
    }

    #[test]
    fn test_scale_pattern_to_batch_preserves_order() {
        // When repeating, the pattern should cycle in order
        let base = vec![1, 2, 3, 4];
        let scaled = scale_pattern_to_batch(&base, 10);
        assert_eq!(scaled, vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2]);
    }

    #[test]
    fn test_bimodal_small_chunk() {
        // With very small C, short_len = C/8 could be 0, should be clamped to 1
        let lengths = generate_lengths(MixedCaseId::BimodalHalf, 4, 4).unwrap();
        assert!(lengths.iter().all(|&l| l >= 1));
    }

    // =========================================================================
    // TTFT TRACKING TESTS
    // =========================================================================

    #[test]
    fn test_prefill_mixed_result_from_ttft() {
        let ttft_ms = vec![10.0, 15.0, 25.0, 40.0];
        let lengths = vec![100, 200, 300, 400];
        let result = PrefillMixedResult::from_ttft(ttft_ms.clone(), &lengths, 4);

        assert_eq!(result.total_prompt_tokens, 1000); // 100+200+300+400
        assert_eq!(result.prefill_total_ms, 40.0); // max TTFT
        assert_eq!(result.num_infer_calls, 4);
        assert_eq!(result.ttft_ms_local, ttft_ms);
        assert_eq!(result.ttft_min_ms, 10.0);
        assert_eq!(result.ttft_max_ms, 40.0);

        // p50 = (15 + 25) / 2 = 20.0
        assert_eq!(result.ttft_p50_ms, 20.0);

        // prefill_tok_per_s = 1000 / (40ms / 1000) = 1000 / 0.04 = 25000
        assert!((result.prefill_tok_per_s - 25000.0).abs() < 0.1);
    }

    #[test]
    fn test_prefill_mixed_result_single_batch() {
        let ttft_ms = vec![50.0];
        let lengths = vec![256];
        let result = PrefillMixedResult::from_ttft(ttft_ms, &lengths, 1);

        assert_eq!(result.total_prompt_tokens, 256);
        assert_eq!(result.prefill_total_ms, 50.0);
        assert_eq!(result.ttft_min_ms, 50.0);
        assert_eq!(result.ttft_p50_ms, 50.0);
        assert_eq!(result.ttft_max_ms, 50.0);
    }

    #[test]
    fn test_prefill_mixed_result_odd_batch() {
        // Test with odd number of batches (median is middle element)
        let ttft_ms = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let lengths = vec![10, 20, 30, 40, 50];
        let result = PrefillMixedResult::from_ttft(ttft_ms, &lengths, 5);

        assert_eq!(result.total_prompt_tokens, 150);
        assert_eq!(result.ttft_p50_ms, 30.0); // Middle element of sorted array
    }

    #[test]
    fn test_mixed_ttft_tracker_new() {
        let lengths = vec![100, 200, 300, 400];
        let tracker = MixedTtftTracker::new(lengths.clone());

        assert_eq!(tracker.batch_size(), 4);
        assert_eq!(tracker.lengths(), &lengths[..]);
        assert_eq!(tracker.num_infer_calls(), 0);
        assert!(!tracker.all_done());
    }

    #[test]
    fn test_mixed_ttft_tracker_record_infer() {
        let lengths = vec![100, 200, 300, 400];
        let mut tracker = MixedTtftTracker::new(lengths);
        tracker.start();

        // First infer: batches 0 and 1 produce output (shorter sequences finish first)
        let done = tracker.record_infer(|i| i < 2);
        assert!(!done);
        assert_eq!(tracker.num_infer_calls(), 1);

        // Second infer: batches 0, 1, 2 produce output
        let done = tracker.record_infer(|i| i < 3);
        assert!(!done);
        assert_eq!(tracker.num_infer_calls(), 2);

        // Third infer: all batches produce output
        let done = tracker.record_infer(|_| true);
        assert!(done);
        assert_eq!(tracker.num_infer_calls(), 3);

        // All should be done
        assert!(tracker.all_done());
    }

    #[test]
    fn test_mixed_ttft_tracker_finalize() {
        let lengths = vec![100, 200, 300, 400];
        let mut tracker = MixedTtftTracker::new(lengths);
        tracker.start();

        // Simulate all batches finishing at once
        let done = tracker.record_infer(|_| true);
        assert!(done);

        let result = tracker.finalize();
        assert_eq!(result.total_prompt_tokens, 1000);
        assert_eq!(result.num_infer_calls, 1);
        assert_eq!(result.ttft_ms_local.len(), 4);
    }

    #[test]
    fn test_mixed_ttft_tracker_total_time() {
        let lengths = vec![100, 200];
        let mut tracker = MixedTtftTracker::new(lengths);
        tracker.start();

        // First batch finishes
        tracker.record_infer(|i| i == 0);

        // Second batch finishes
        tracker.record_infer(|_| true);

        // Total time should be the max TTFT
        let total = tracker.total_time();
        assert!(total > Duration::ZERO);
    }

    #[test]
    fn test_prefill_mixed_config_default() {
        let config = PrefillMixedConfig::default();
        assert_eq!(config.mixed_case_id, MixedCaseId::Staircase8);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.token_chunk_size, 256);
        assert_eq!(config.seed, 42);
        assert_eq!(config.vocab_size, 65536);
    }

    #[test]
    fn test_prefill_mixed_config_builder() {
        let config = PrefillMixedConfig::new(MixedCaseId::BimodalHalf, 4, 128)
            .with_seed(12345)
            .with_vocab_size(50257);

        assert_eq!(config.mixed_case_id, MixedCaseId::BimodalHalf);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.token_chunk_size, 128);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.vocab_size, 50257);
    }

    #[test]
    fn test_prefill_mixed_config_generate_lengths() {
        let config = PrefillMixedConfig::new(MixedCaseId::Staircase8, 8, 256);
        let lengths = config.generate_lengths().unwrap();
        assert_eq!(lengths.len(), 8);
        assert_eq!(lengths, vec![16, 32, 64, 128, 192, 256, 512, 1024]);
    }

    #[test]
    fn test_prefill_mixed_config_generate_tokens() {
        let config = PrefillMixedConfig::new(MixedCaseId::Staircase8, 4, 128).with_seed(42);
        let tokens = config.generate_tokens().unwrap();

        assert_eq!(tokens.len(), 4);
        // Each batch should have the right length from staircase pattern
        // For C=128, B=4: [8, 16, 32, 64] (first 4 elements of staircase)
        let lengths = config.generate_lengths().unwrap();
        for (i, batch_tokens) in tokens.iter().enumerate() {
            assert_eq!(
                batch_tokens.len(),
                lengths[i] as usize,
                "Batch {} should have {} tokens",
                i,
                lengths[i]
            );
        }

        // Should be deterministic
        let tokens2 = config.generate_tokens().unwrap();
        assert_eq!(tokens, tokens2);
    }

    #[test]
    fn test_prefill_mixed_config_total_prompt_tokens() {
        let config = PrefillMixedConfig::new(MixedCaseId::Staircase8, 8, 256);
        let total = config.total_prompt_tokens().unwrap();
        // Sum of [16, 32, 64, 128, 192, 256, 512, 1024] = 2224
        assert_eq!(total, 2224);
    }

    #[test]
    fn test_prefill_mixed_config_create_tracker() {
        let config = PrefillMixedConfig::new(MixedCaseId::BimodalHalf, 4, 256);
        let tracker = config.create_tracker().unwrap();

        assert_eq!(tracker.batch_size(), 4);
        // BimodalHalf with B=4: [1024, 1024, 32, 32]
        let expected_lengths = vec![1024, 1024, 32, 32];
        assert_eq!(tracker.lengths(), &expected_lengths[..]);
    }
}
