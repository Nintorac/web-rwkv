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
}
