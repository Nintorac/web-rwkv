//! FLA (Flash Linear Attention) chunked prefill module for RWKV7 HIP backend.
//!
//! This module provides:
//! - Chunk index precomputation ([`prepare_chunk_indices`], [`prepare_chunk_offsets`])
//!   for mapping flat chunk IDs to `(sequence_id, local_chunk_id)` pairs.
//! - The [`FlaChunkedWkv`] struct, which holds pre-sized scratch buffer views
//!   and exposes a `compute()` method for the 5-stage FLA pipeline.
//!
//! The 5-stage FLA pipeline:
//!   1. Convert raw att_w (f16) to gk (f32) = -exp(att_w), then cumulative sum
//!   2. Intra-chunk attention matrices
//!   3. WY representation (matrix inversion + w/u computation)
//!   4. Inter-chunk state recurrence
//!   5. Output combination
//!
//! The chunk index functions run on the CPU and produce small buffers uploaded
//! to the GPU once per forward pass. They support variable-length batches via
//! cumulative sequence lengths (`cu_seqlens`).
//!
//! Reference: `repos/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk.py`

use half::f16;

use crate::hip::device::Stream;
use crate::hip::ffi::Result;
use crate::hip::kernels::fla::{
    fla_chunk_h, fla_chunk_o, fla_cumsum, fla_neg_exp_f16_to_f32, fla_intra, fla_wy_repr,
};
use crate::hip::scratch::PrefillScratch;
use crate::hip::tensor::{TensorHip, TensorShape};

/// Threshold sequence length for dispatching to FLA chunked prefill.
/// Sequences with T >= this value use FLA; shorter sequences use WaveReduceWkv.
/// FLA handles all prefill (T>1), so the threshold is set to 2.
pub const FLA_CHUNK_THRESHOLD: usize = 2;

/// FLA chunked WKV7 kernel for efficient prefill.
///
/// Holds pre-sized views of the FLA scratch buffers from [`PrefillScratch`] plus
/// configuration parameters for the current forward pass. Constructed at the
/// dispatch point in `dispatch()` when T >= [`FLA_CHUNK_THRESHOLD`].
///
/// The 5-stage pipeline executes entirely on GPU with no allocations:
/// all intermediate buffers come from the pre-allocated scratch pool.
///
/// Unlike the recurrent WKV kernels (which implement the [`super::prefill::WkvKernel`]
/// trait), FLA uses a direct `compute()` method that takes `&mut self` and the
/// raw `att_w` tensor (pre-exponentiation). This avoids:
/// - The precision-losing round-trip through f16 `exp(-exp(w))` then `log`
/// - The `ptr::read` hack needed to get `&mut` access from `&self`
#[allow(non_snake_case)]
pub struct FlaChunkedWkv {
    /// FLA chunk size (C, typically 16)
    pub chunk_size: usize,
    /// Number of sequences in the batch
    pub batch_size: usize,
    /// Sequence length (all sequences same length in the current batch)
    pub seq_len: usize,
    /// Head size (K, typically 64)
    pub head_size: usize,
    /// Number of heads (H)
    pub n_head: usize,

    // ---- FLA scratch buffer views (non-owning, pre-sized for current T, B) ----

    /// f32 log-decay / inclusive cumsum output, shape [K, H, T, B]
    pub fla_gi: TensorHip<f32>,
    /// f32 exclusive cumsum output, shape [K, H, T, B]
    pub fla_ge: TensorHip<f32>,
    /// Decay-scaled query, shape [K, H, T, B]
    pub fla_qg: TensorHip<f32>,
    /// Decay-scaled key, shape [K, H, T, B]
    pub fla_kg: TensorHip<f32>,
    /// Decay-scaled a, shape [K, H, T, B]
    pub fla_ag: TensorHip<f32>,
    /// Decay-scaled b, shape [K, H, T, B]
    pub fla_bg: TensorHip<f32>,
    /// Intra-chunk attention Q@K^T, shape [C, C, H, total_chunks]
    pub fla_A_qk: TensorHip<f32>,
    /// Query-bias attention Q@B^T, shape [C, C, H, total_chunks]
    pub fla_A_qb: TensorHip<f32>,
    /// Adapt-bias attention A@B^T, shape [C, C, H, total_chunks]
    pub fla_A_ab: TensorHip<f32>,
    /// Adapt-key attention A@K^T, shape [C, C, H, total_chunks]
    pub fla_A_ak: TensorHip<f32>,
    /// Inverse of A_ab, shape [C, C, H, total_chunks]
    pub fla_A_ab_inv: TensorHip<f32>,
    /// WY w output, shape [K, H, T, B]
    pub fla_w_wy: TensorHip<f32>,
    /// WY u output, shape [K, H, T, B]
    pub fla_u_wy: TensorHip<f32>,
    /// Per-chunk recurrent states, shape [K, K, H, total_chunks]
    pub fla_h: TensorHip<f32>,
    /// Corrected values, shape [K, H, T, B]
    pub fla_v_new: TensorHip<f32>,
}

impl FlaChunkedWkv {
    /// Create an `FlaChunkedWkv` instance with pre-sized views of the FLA
    /// scratch buffers for the given forward-pass dimensions.
    ///
    /// This does not allocate GPU memory -- it creates lightweight non-owning
    /// views into the pre-allocated scratch pool.
    ///
    /// # Arguments
    /// * `scratch` - Mutable reference to the scratch pool
    /// * `head_size` - Head size (K, typically 64)
    /// * `n_head` - Number of heads (H)
    /// * `seq_len` - Total packed sequence length (T_total)
    /// * `batch_size` - Tensor batch dimension (1 for packed layout)
    /// * `n_seq` - Actual number of sequences (for chunk count calculation)
    #[allow(non_snake_case)]
    pub fn new(
        scratch: &mut PrefillScratch,
        head_size: usize,
        n_head: usize,
        seq_len: usize,
        batch_size: usize,
        n_seq: usize,
    ) -> Result<Self> {
        let chunk_size = scratch.config.fla_chunk_size;
        // Packed layout: each sequence boundary can start a new chunk,
        // so total_chunks = ceil_div(seq_len, C) + n_seq (safe upper bound).
        let total_chunks = ceil_div(seq_len, chunk_size) + n_seq;

        // Per-token shape: [K, H, T_total, 1] for packed layout
        let per_token_shape = TensorShape::new(head_size, n_head, seq_len, batch_size);
        // Per-chunk attention matrix shape: [C, C, H, total_chunks]
        let chunk_mat_shape = TensorShape::new(chunk_size, chunk_size, n_head, total_chunks);
        // Per-chunk state shape: [K, K, H, total_chunks]
        let chunk_state_shape = TensorShape::new(head_size, head_size, n_head, total_chunks);

        Ok(Self {
            chunk_size,
            batch_size,
            seq_len,
            head_size,
            n_head,

            fla_gi: scratch.fla_gi.resized_view_mut(per_token_shape)?,
            fla_ge: scratch.fla_ge.resized_view_mut(per_token_shape)?,
            fla_qg: scratch.fla_qg.resized_view_mut(per_token_shape)?,
            fla_kg: scratch.fla_kg.resized_view_mut(per_token_shape)?,
            fla_ag: scratch.fla_ag.resized_view_mut(per_token_shape)?,
            fla_bg: scratch.fla_bg.resized_view_mut(per_token_shape)?,
            fla_A_qk: scratch.fla_A_qk.resized_view_mut(chunk_mat_shape)?,
            fla_A_qb: scratch.fla_A_qb.resized_view_mut(chunk_mat_shape)?,
            fla_A_ab: scratch.fla_A_ab.resized_view_mut(chunk_mat_shape)?,
            fla_A_ak: scratch.fla_A_ak.resized_view_mut(chunk_mat_shape)?,
            fla_A_ab_inv: scratch.fla_A_ab_inv.resized_view_mut(chunk_mat_shape)?,
            fla_w_wy: scratch.fla_w_wy.resized_view_mut(per_token_shape)?,
            fla_u_wy: scratch.fla_u_wy.resized_view_mut(per_token_shape)?,
            fla_h: scratch.fla_h.resized_view_mut(chunk_state_shape)?,
            fla_v_new: scratch.fla_v_new.resized_view_mut(per_token_shape)?,
        })
    }

    /// Run the 5-stage FLA chunked pipeline.
    ///
    /// Takes raw log-domain decay (`att_w`, f16) directly instead of
    /// going through `exp(-exp(w))` then `log`, avoiding precision loss.
    ///
    /// # Arguments
    /// * `att_w` - Raw log-domain decay (f16), shape `[K, H, T, B]`.
    ///   This is the output of `softplus_decay_f16`: `w = -softplus(...) - 0.5`.
    /// * `r` - Query (receptance), shape `[K, H, T, B]` (f16)
    /// * `k` - Key (controlled), shape `[K, H, T, B]` (f16)
    /// * `v` - Value, shape `[K, H, T, B]` (f16)
    /// * `a` - Negative normalized key for state update (wkv_a), shape `[K, H, T, B]` (f16)
    /// * `b` - Adaptation-weighted normalized key (wkv_b), shape `[K, H, T, B]` (f16)
    /// * `state` - Per-layer recurrent state `[K, K, H, B]` (f32), updated in-place
    /// * `output` - Output tensor `[K, H, T, B]` (f16)
    /// * `lengths` - Per-batch real sequence lengths (CPU-side, one entry per batch element).
    ///   Used to build cu_seqlens (packed convention) for chunk counting and bounds
    ///   checking: `cu_seqlens[b] = sum(lengths[0..b])`. With packed/concatenated
    ///   layout, `batch_offsets[b] = cu_seqlens[b]` (no padding gaps).
    /// * `stream` - HIP stream for kernel launches
    #[allow(non_snake_case)]
    pub fn compute(
        &mut self,
        att_w: &TensorHip<f16>,
        r: &TensorHip<f16>,
        k: &TensorHip<f16>,
        v: &TensorHip<f16>,
        a: &TensorHip<f16>,
        b: &TensorHip<f16>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        lengths: &[usize],
        stream: &Stream,
    ) -> Result<()> {
        let t = self.seq_len;
        let c = self.chunk_size;
        let head_size = self.head_size;
        let n_head = self.n_head;
        // n_seq = actual number of sequences from lengths, not the tensor batch dim.
        // With packed layout, self.batch_size=1 (buffer dim) but lengths has the real count.
        let n_seq = lengths.len();
        let bb = n_seq;

        // ================================================================
        // Step 0: Compute chunk indices on CPU and upload to GPU
        // ================================================================

        // Build cu_seqlens from actual per-batch lengths (packed convention).
        // cu_seqlens[b] = sum(lengths[0..b]), cu_seqlens[b+1] - cu_seqlens[b] = lengths[b].
        // This is used for chunk counting and bounds checking (sequence length T).
        let cu_seqlens_host: Vec<u32> = {
            let mut cs = Vec::with_capacity(bb + 1);
            cs.push(0u32);
            let mut accum = 0u32;
            for &l in lengths.iter() {
                accum += l as u32;
                cs.push(accum);
            }
            cs
        };

        // Build batch_offsets for data addressing.
        // Packed layout: batch_offsets = cu_seqlens (no padding gaps).
        // Each sequence starts at cu_seqlens[b] in the packed time dimension.
        let batch_offsets_host: Vec<i32> = cu_seqlens_host[..bb]
            .iter()
            .map(|&x| x as i32)
            .collect();

        // Compute chunk indices and offsets on CPU
        let chunk_indices_host = prepare_chunk_indices(&cu_seqlens_host, c);
        let chunk_offsets_host = prepare_chunk_offsets(&cu_seqlens_host, c);
        let total_chunks = chunk_indices_host.len();

        if total_chunks == 0 {
            return Ok(());
        }

        // Flatten chunk_indices from Vec<[u32; 2]> to Vec<i32> for GPU upload
        let chunk_indices_flat: Vec<i32> = chunk_indices_host
            .iter()
            .flat_map(|pair| [pair[0] as i32, pair[1] as i32])
            .collect();

        // Upload small index buffers to GPU (temporary allocations -- these are
        // tiny: O(total_chunks) and O(B) elements respectively)
        let ci_shape = TensorShape::new(total_chunks * 2, 1, 1, 1);
        let chunk_indices_gpu = TensorHip::<i32>::from_slice(&chunk_indices_flat, ci_shape, stream)?;

        let co_shape = TensorShape::new(n_seq + 1, 1, 1, 1);
        let chunk_offsets_flat: Vec<i32> = chunk_offsets_host.iter().map(|&x| x as i32).collect();
        let chunk_offsets_gpu = TensorHip::<i32>::from_slice(&chunk_offsets_flat, co_shape, stream)?;

        // cu_seqlens on GPU
        let cu_shape = TensorShape::new(bb + 1, 1, 1, 1);
        let cu_seqlens_flat: Vec<i32> = cu_seqlens_host.iter().map(|&x| x as i32).collect();
        let cu_seqlens_gpu = TensorHip::<i32>::from_slice(&cu_seqlens_flat, cu_shape, stream)?;

        // batch_offsets on GPU
        let bo_shape = TensorShape::new(bb, 1, 1, 1);
        let batch_offsets_gpu = TensorHip::<i32>::from_slice(&batch_offsets_host, bo_shape, stream)?;

        // ================================================================
        // Stage 0.5: Convert raw att_w (f16) to gk (f32) = -exp(att_w)
        // ================================================================
        // The FLA cumsum kernel needs gk = -exp(w) in f32. We compute this
        // directly from the raw log-domain decay (att_w), avoiding the
        // precision-losing round-trip through f16 exp(-exp(w)) then log.
        //
        // We reuse fla_gi as temporary storage for gk since Stage 1 will
        // overwrite fla_gi anyway. After cumsum, fla_gi holds the inclusive
        // cumsum result.

        // Use a temporary view for gk that shares memory with fla_gi
        // per_token_shape uses self.batch_size (=1 for packed) since buffers are [K,H,T,1]
        let per_token_shape = TensorShape::new(head_size, n_head, t, self.batch_size);
        let mut gk = self.fla_gi.resized_view_mut(per_token_shape)?;

        // att_w is [K, H, T, B] in f16, gk is [K, H, T, B] in f32
        fla_neg_exp_f16_to_f32(att_w, &mut gk, stream)?;

        // ================================================================
        // Stage 1: Cumulative decay scan
        // ================================================================
        // gk currently lives in fla_gi memory. fla_cumsum reads gk and
        // writes gi (inclusive) and ge (exclusive). Since gk and gi share
        // the same memory, the kernel reads the original gk value for each
        // element before writing the cumsum. The cumsum kernel processes
        // each chunk sequentially (C=16 loop), reading gk[t] then writing
        // gi[t], so the read-before-write is safe.
        fla_cumsum(
            &gk,
            &mut self.fla_gi,
            &mut self.fla_ge,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
            &batch_offsets_gpu,
            c,
            total_chunks,
            stream,
        )?;
        // fla_gi and gk alias the same memory, and now fla_gi holds the
        // inclusive cumsum. Drop the gk alias to avoid confusion.
        drop(gk);

        // ================================================================
        // Stage 2: Intra-chunk attention matrices
        // ================================================================
        // Input: q(f16), k(f16), a(f16), b(f16), gi(f32), ge(f32)
        // Output: qg, kg, ag, bg (f32 per-token), A_qk, A_qb, A_ak, A_ab (f32 CxC matrices)
        //
        // In the WKV pipeline:
        //   r = receptance (query in RWKV7 = q in FLA)
        //   k = controlled key (k in FLA)
        //   v = value (v in FLA)
        //   a = wkv_a = -kk (a in FLA)
        //   b = wkv_b = kk * att_a (b in FLA)
        fla_intra(
            r, // q in FLA (receptance)
            k, // k in FLA (controlled key)
            a, // a in FLA (wkv_a = -kk)
            b, // b in FLA (wkv_b = kk * att_a)
            &self.fla_gi,
            &self.fla_ge,
            &mut self.fla_qg,
            &mut self.fla_kg,
            &mut self.fla_ag,
            &mut self.fla_bg,
            &mut self.fla_A_qk,
            &mut self.fla_A_qb,
            &mut self.fla_A_ak,
            &mut self.fla_A_ab,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
            &batch_offsets_gpu,
            c,
            total_chunks,
            stream,
        )?;

        // ================================================================
        // Stage 3: WY representation
        // ================================================================
        // Input: A_ab(f32), A_ak(f32), ag(f32), v(f16)
        // Output: A_ab_inv(f32), w_wy(f32), u_wy(f32)
        fla_wy_repr(
            &self.fla_A_ab,
            &self.fla_A_ak,
            &mut self.fla_A_ab_inv,
            &self.fla_ag,
            v,
            &mut self.fla_w_wy,
            &mut self.fla_u_wy,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
            &batch_offsets_gpu,
            c,
            total_chunks,
            stream,
        )?;

        // ================================================================
        // Stage 4: Inter-chunk state recurrence
        // ================================================================
        // Input: kg, bg (f32), v(f16), w_wy, u_wy (f32), gi(f32),
        //        state (f32 in/out), chunk_offsets, cu_seqlens
        // Output: h (per-chunk states), v_new (corrected values), state (updated)
        //
        // The kernel reads state_in at start and writes state_out at end.
        // State is [K, K, H, B] = [head_size, head_size, n_head, batch_size].
        fla_chunk_h(
            &self.fla_kg,
            &self.fla_bg,
            v,
            &self.fla_w_wy,
            &self.fla_u_wy,
            &self.fla_gi,
            state,
            &mut self.fla_h,
            &mut self.fla_v_new,
            &chunk_offsets_gpu,
            &cu_seqlens_gpu,
            &batch_offsets_gpu,
            c,
            n_seq,
            stream,
        )?;

        // ================================================================
        // Stage 5: Output combination
        // ================================================================
        // Input: qg(f32), v(f16), v_new(f32), A_qk(f32), A_qb(f32), h(f32)
        // Output: o(f16) = qg @ h + A_qk @ v + A_qb @ v_new
        fla_chunk_o(
            &self.fla_qg,
            v,
            &self.fla_v_new,
            &self.fla_A_qk,
            &self.fla_A_qb,
            &self.fla_h,
            output,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
            &batch_offsets_gpu,
            c,
            total_chunks,
            stream,
        )?;

        Ok(())
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
    // FLA dispatch threshold
    // ------------------------------------------------------------------

    /// Verify FLA_CHUNK_THRESHOLD is set to 2, meaning FLA handles all T>1.
    #[test]
    fn test_fla_threshold_handles_all_prefill() {
        assert_eq!(
            FLA_CHUNK_THRESHOLD, 2,
            "FLA_CHUNK_THRESHOLD should be 2 (FLA handles all T>1)"
        );
    }

    /// Verify the dispatch boundary values.
    /// T < 2 should NOT use FLA (T=1 is recurrent), T >= 2 should use FLA.
    #[test]
    fn test_fla_dispatch_boundary() {
        // T=1 is recurrent, not FLA
        assert!(
            1 < FLA_CHUNK_THRESHOLD,
            "T=1 should be below threshold (recurrent)"
        );
        // At threshold: yes FLA
        assert!(
            2 >= FLA_CHUNK_THRESHOLD,
            "T=2 should be at or above threshold"
        );
        // Well above threshold: yes FLA
        assert!(
            256 >= FLA_CHUNK_THRESHOLD,
            "T=256 should be at or above threshold"
        );
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
