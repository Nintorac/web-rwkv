//! FLA (Flash Linear Attention) chunked prefill module for RWKV7 HIP backend.
//!
//! This module provides:
//! - Chunk index precomputation ([`prepare_chunk_indices`], [`prepare_chunk_offsets`])
//!   for mapping flat chunk IDs to `(sequence_id, local_chunk_id)` pairs.
//! - The [`FlaChunkedWkv`] kernel skeleton, which implements the
//!   [`WkvKernel`] trait for sequences longer than [`FLA_CHUNK_THRESHOLD`].
//!
//! The chunk index functions run on the CPU and produce small buffers uploaded
//! to the GPU once per forward pass. They support variable-length batches via
//! cumulative sequence lengths (`cu_seqlens`).
//!
//! The kernel implementation is currently a skeleton that delegates to the
//! existing [`WaveReduceWkv`](super::prefill::WaveReduceWkv) kernel. It will
//! be replaced with the real 5-stage FLA pipeline in later tickets.
//!
//! Reference: `repos/flash-linear-attention/fla/ops/utils/index.py`

use half::f16;

use crate::hip::device::Stream;
use crate::hip::ffi::Result;
use crate::hip::tensor::TensorHip;

use super::prefill::{WkvInput, WkvKernel};

/// Threshold sequence length for dispatching to FLA chunked prefill.
/// Sequences with T >= this value use FLA; shorter sequences use WaveReduceWkv.
pub const FLA_CHUNK_THRESHOLD: usize = 32;

/// FLA chunked WKV7 kernel for efficient prefill.
///
/// Skeleton implementation: delegates to the existing WaveReduceWkv kernel.
/// Will be replaced with the real 5-stage FLA pipeline in later tickets.
pub struct FlaChunkedWkv;

impl WkvKernel for FlaChunkedWkv {
    fn compute(
        &self,
        input: &WkvInput<'_>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        stream: &Stream,
    ) -> Result<()> {
        // Skeleton: delegate to WaveReduceWkv until real FLA kernels are implemented
        use crate::hip::kernels::wkv7_wave_reduce;
        wkv7_wave_reduce(
            input.w_decay,
            input.r,
            input.k,
            input.v,
            input.a,
            input.b,
            state,
            output,
            input.lengths,
            stream,
        )
    }

    fn supports_multi_token(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "fla_chunked"
    }
}

// ---------------------------------------------------------------------------
// Chunk index precomputation
// ---------------------------------------------------------------------------

/// Integer ceiling division: `ceil(a / b)`.
pub(crate) fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Compute flat chunk-to-sequence mapping from cumulative sequence lengths.
///
/// Each sequence in the batch is divided into `ceil_div(seq_len, chunk_size)`
/// chunks. This function returns a flat list of `[seq_id, local_chunk_id]`
/// pairs, one per chunk across all sequences.
///
/// # Arguments
/// * `cu_seqlens` -- Cumulative sequence lengths, length `N+1` for `N` sequences.
///   For example, `[0, 100, 230, 280]` encodes three sequences of lengths 100, 130, 50.
/// * `chunk_size` -- Number of tokens per chunk.
///
/// # Returns
/// A `Vec<[u32; 2]>` where each entry is `[seq_id, local_chunk_id]`.
///
/// # Example
/// ```ignore
/// let cu = [0, 100, 230, 280];
/// let idx = prepare_chunk_indices(&cu, 64);
/// // chunks_per_seq = [2, 3, 1]
/// // idx = [[0,0], [0,1], [1,0], [1,1], [1,2], [2,0]]
/// ```
pub fn prepare_chunk_indices(cu_seqlens: &[u32], chunk_size: usize) -> Vec<[u32; 2]> {
    assert!(chunk_size > 0, "chunk_size must be positive");
    assert!(
        cu_seqlens.len() >= 2,
        "cu_seqlens must have at least 2 elements (start and end)"
    );

    let n_seqs = cu_seqlens.len() - 1;

    // Pre-compute total chunks for allocation
    let total_chunks: usize = (0..n_seqs)
        .map(|i| {
            let seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]) as usize;
            ceil_div(seq_len, chunk_size)
        })
        .sum();

    let mut indices = Vec::with_capacity(total_chunks);

    for seq_id in 0..n_seqs {
        let seq_len = (cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]) as usize;
        let n_chunks = ceil_div(seq_len, chunk_size);
        for local_chunk in 0..n_chunks {
            indices.push([seq_id as u32, local_chunk as u32]);
        }
    }

    indices
}

/// Compute cumulative chunk counts per sequence.
///
/// Returns a vector of length `N+1` (same length as `cu_seqlens`) where
/// entry `i` is the total number of chunks in sequences `0..i`. This is
/// the prefix sum of `ceil_div(seq_len_i, chunk_size)`.
///
/// # Arguments
/// * `cu_seqlens` -- Cumulative sequence lengths, length `N+1`.
/// * `chunk_size` -- Number of tokens per chunk.
///
/// # Returns
/// A `Vec<u32>` of cumulative chunk counts.
///
/// # Example
/// ```ignore
/// let cu = [0, 100, 230, 280];
/// let off = prepare_chunk_offsets(&cu, 64);
/// // chunks_per_seq = [2, 3, 1]
/// // off = [0, 2, 5, 6]
/// ```
pub fn prepare_chunk_offsets(cu_seqlens: &[u32], chunk_size: usize) -> Vec<u32> {
    assert!(chunk_size > 0, "chunk_size must be positive");
    assert!(
        cu_seqlens.len() >= 2,
        "cu_seqlens must have at least 2 elements (start and end)"
    );

    let n_seqs = cu_seqlens.len() - 1;
    let mut offsets = Vec::with_capacity(n_seqs + 1);
    offsets.push(0u32);

    let mut cumulative = 0u32;
    for i in 0..n_seqs {
        let seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]) as usize;
        cumulative += ceil_div(seq_len, chunk_size) as u32;
        offsets.push(cumulative);
    }

    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // ceil_div
    // ------------------------------------------------------------------

    #[test]
    fn test_ceil_div_exact() {
        assert_eq!(ceil_div(64, 16), 4);
        assert_eq!(ceil_div(128, 64), 2);
    }

    #[test]
    fn test_ceil_div_remainder() {
        assert_eq!(ceil_div(100, 64), 2);
        assert_eq!(ceil_div(130, 64), 3);
        assert_eq!(ceil_div(50, 64), 1);
    }

    #[test]
    fn test_ceil_div_one() {
        assert_eq!(ceil_div(1, 1), 1);
        assert_eq!(ceil_div(7, 1), 7);
    }

    // ------------------------------------------------------------------
    // prepare_chunk_indices
    // ------------------------------------------------------------------

    /// Example from the ticket: cu_seqlens = [0, 100, 230, 280], chunk_size = 64.
    #[test]
    fn test_chunk_indices_variable_lengths() {
        let cu = [0u32, 100, 230, 280];
        let idx = prepare_chunk_indices(&cu, 64);

        // chunks_per_seq = [2, 3, 1]  (100/64=2, 130/64=3, 50/64=1)
        let expected: Vec<[u32; 2]> = vec![
            [0, 0],
            [0, 1], // seq 0: 2 chunks
            [1, 0],
            [1, 1],
            [1, 2], // seq 1: 3 chunks
            [2, 0], // seq 2: 1 chunk
        ];
        assert_eq!(idx, expected);
    }

    /// T divisible by C: each sequence has an exact number of chunks.
    #[test]
    fn test_chunk_indices_divisible() {
        // Single sequence: T=64, C=16 -> 4 chunks
        let cu = [0u32, 64];
        let idx = prepare_chunk_indices(&cu, 16);

        let expected: Vec<[u32; 2]> = vec![[0, 0], [0, 1], [0, 2], [0, 3]];
        assert_eq!(idx, expected);
    }

    /// T not divisible by C: last chunk is partial.
    #[test]
    fn test_chunk_indices_not_divisible() {
        // Single sequence: T=100, C=64 -> 2 chunks (last is 36 tokens)
        let cu = [0u32, 100];
        let idx = prepare_chunk_indices(&cu, 64);

        let expected: Vec<[u32; 2]> = vec![[0, 0], [0, 1]];
        assert_eq!(idx, expected);
    }

    /// Single sequence.
    #[test]
    fn test_chunk_indices_single_seq() {
        let cu = [0u32, 50];
        let idx = prepare_chunk_indices(&cu, 16);

        // ceil(50 / 16) = 4
        let expected: Vec<[u32; 2]> = vec![[0, 0], [0, 1], [0, 2], [0, 3]];
        assert_eq!(idx, expected);
    }

    /// Edge case: chunk_size equals sequence length (exactly 1 chunk).
    #[test]
    fn test_chunk_indices_chunk_equals_seqlen() {
        let cu = [0u32, 64];
        let idx = prepare_chunk_indices(&cu, 64);

        let expected: Vec<[u32; 2]> = vec![[0, 0]];
        assert_eq!(idx, expected);
    }

    /// Equal-length batches: cu_seqlens = [0, T, 2T, ..., B*T].
    #[test]
    fn test_chunk_indices_equal_length_batch() {
        // B=3, T=128, C=64 -> 2 chunks each
        let cu = [0u32, 128, 256, 384];
        let idx = prepare_chunk_indices(&cu, 64);

        let expected: Vec<[u32; 2]> = vec![
            [0, 0],
            [0, 1], // seq 0
            [1, 0],
            [1, 1], // seq 1
            [2, 0],
            [2, 1], // seq 2
        ];
        assert_eq!(idx, expected);
    }

    /// chunk_size larger than sequence length: 1 chunk per sequence.
    #[test]
    fn test_chunk_indices_chunk_larger_than_seq() {
        let cu = [0u32, 10, 25, 30];
        let idx = prepare_chunk_indices(&cu, 64);

        let expected: Vec<[u32; 2]> = vec![
            [0, 0], // seq 0: 10 tokens, 1 chunk
            [1, 0], // seq 1: 15 tokens, 1 chunk
            [2, 0], // seq 2: 5 tokens, 1 chunk
        ];
        assert_eq!(idx, expected);
    }

    // ------------------------------------------------------------------
    // prepare_chunk_offsets
    // ------------------------------------------------------------------

    /// Example from the ticket: cu_seqlens = [0, 100, 230, 280], chunk_size = 64.
    #[test]
    fn test_chunk_offsets_variable_lengths() {
        let cu = [0u32, 100, 230, 280];
        let off = prepare_chunk_offsets(&cu, 64);

        // chunks_per_seq = [2, 3, 1]
        // cumulative = [0, 2, 5, 6]
        assert_eq!(off, vec![0u32, 2, 5, 6]);
    }

    /// T divisible by C.
    #[test]
    fn test_chunk_offsets_divisible() {
        let cu = [0u32, 64];
        let off = prepare_chunk_offsets(&cu, 16);

        // 64/16 = 4 chunks -> [0, 4]
        assert_eq!(off, vec![0u32, 4]);
    }

    /// T not divisible by C.
    #[test]
    fn test_chunk_offsets_not_divisible() {
        let cu = [0u32, 100];
        let off = prepare_chunk_offsets(&cu, 64);

        // ceil(100/64) = 2 chunks -> [0, 2]
        assert_eq!(off, vec![0u32, 2]);
    }

    /// Single sequence.
    #[test]
    fn test_chunk_offsets_single_seq() {
        let cu = [0u32, 50];
        let off = prepare_chunk_offsets(&cu, 16);

        // ceil(50/16) = 4 chunks -> [0, 4]
        assert_eq!(off, vec![0u32, 4]);
    }

    /// Edge case: chunk_size equals sequence length.
    #[test]
    fn test_chunk_offsets_chunk_equals_seqlen() {
        let cu = [0u32, 64];
        let off = prepare_chunk_offsets(&cu, 64);

        // 64/64 = 1 chunk -> [0, 1]
        assert_eq!(off, vec![0u32, 1]);
    }

    /// Equal-length batches.
    #[test]
    fn test_chunk_offsets_equal_length_batch() {
        // B=3, T=128, C=64 -> 2 chunks each
        let cu = [0u32, 128, 256, 384];
        let off = prepare_chunk_offsets(&cu, 64);

        assert_eq!(off, vec![0u32, 2, 4, 6]);
    }

    /// chunk_size larger than all sequences.
    #[test]
    fn test_chunk_offsets_chunk_larger_than_seq() {
        let cu = [0u32, 10, 25, 30];
        let off = prepare_chunk_offsets(&cu, 64);

        // 1 chunk each -> [0, 1, 2, 3]
        assert_eq!(off, vec![0u32, 1, 2, 3]);
    }

    // ------------------------------------------------------------------
    // Consistency: offsets agree with indices
    // ------------------------------------------------------------------

    /// The last element of chunk_offsets must equal the total number of chunk_indices.
    #[test]
    fn test_offsets_match_indices_count() {
        let cu = [0u32, 100, 230, 280];
        let chunk_size = 64;

        let idx = prepare_chunk_indices(&cu, chunk_size);
        let off = prepare_chunk_offsets(&cu, chunk_size);

        assert_eq!(*off.last().unwrap() as usize, idx.len());
    }

    /// For each sequence, the chunk_indices slice between offsets[i] and offsets[i+1]
    /// must contain the correct seq_id and sequential local_chunk_ids.
    #[test]
    fn test_offsets_index_into_indices() {
        let cu = [0u32, 100, 230, 280];
        let chunk_size = 64;

        let idx = prepare_chunk_indices(&cu, chunk_size);
        let off = prepare_chunk_offsets(&cu, chunk_size);

        let n_seqs = cu.len() - 1;
        for seq_id in 0..n_seqs {
            let start = off[seq_id] as usize;
            let end = off[seq_id + 1] as usize;
            let seq_chunks = &idx[start..end];

            for (local, pair) in seq_chunks.iter().enumerate() {
                assert_eq!(pair[0], seq_id as u32, "seq_id mismatch at chunk {local}");
                assert_eq!(
                    pair[1], local as u32,
                    "local_chunk_id mismatch at seq {seq_id}, chunk {local}"
                );
            }
        }
    }
}
