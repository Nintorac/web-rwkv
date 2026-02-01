//! FLA (Flash Linear Attention) chunked prefill module for RWKV7 HIP backend.
//!
//! This module provides:
//! - Chunk index precomputation ([`prepare_chunk_indices`], [`prepare_chunk_offsets`])
//!   for mapping flat chunk IDs to `(sequence_id, local_chunk_id)` pairs.
//! - The [`FlaChunkedWkv`] kernel, which implements the [`WkvKernel`] trait for
//!   sequences longer than [`FLA_CHUNK_THRESHOLD`].
//!
//! The 5-stage FLA pipeline:
//!   1. Convert w_decay (f16) to gk (f32 log-decay), then cumulative sum
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
    fla_chunk_h, fla_chunk_o, fla_cumsum, fla_decay_to_log, fla_intra, fla_wy_repr,
};
use crate::hip::scratch::HipScratch;
use crate::hip::tensor::{TensorHip, TensorShape};

use super::prefill::{WkvInput, WkvKernel};

/// Threshold sequence length for dispatching to FLA chunked prefill.
/// Sequences with T >= this value use FLA; shorter sequences use WaveReduceWkv.
pub const FLA_CHUNK_THRESHOLD: usize = 32;

/// FLA chunked WKV7 kernel for efficient prefill.
///
/// Holds pre-sized views of the FLA scratch buffers from [`HipScratch`] plus
/// configuration parameters for the current forward pass. Constructed at the
/// dispatch point in `step_inner()` when T >= [`FLA_CHUNK_THRESHOLD`].
///
/// The 5-stage pipeline executes entirely on GPU with no allocations:
/// all intermediate buffers come from the pre-allocated scratch pool.
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
    /// * `seq_len` - Sequence length (T)
    /// * `batch_size` - Batch size (B)
    #[allow(non_snake_case)]
    pub fn new(
        scratch: &mut HipScratch,
        head_size: usize,
        n_head: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let chunk_size = scratch.config.fla_chunk_size;
        let total_chunks = batch_size * ceil_div(seq_len, chunk_size);

        // Per-token shape: [K, H, T, B]
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
}

impl WkvKernel for FlaChunkedWkv {
    #[allow(non_snake_case)]
    fn compute(
        &self,
        input: &WkvInput<'_>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        stream: &Stream,
    ) -> Result<()> {
        let t = self.seq_len;
        let b = self.batch_size;
        let c = self.chunk_size;
        let head_size = self.head_size;
        let n_head = self.n_head;
        let n_seq = b; // for equal-length batches, n_seq == batch_size

        // We need mutable access to the scratch buffer views.
        // Since the views are non-owning copies of pointers, we can safely
        // create mutable aliases for kernel dispatch. The kernel wrappers
        // require &mut TensorHip but the underlying GPU memory is the same
        // as in self. This is safe because:
        //   1. Each buffer is written only by one kernel stage before being
        //      read by later stages (no aliasing writes).
        //   2. GPU execution is serialized on the same stream.
        //
        // We use ptr::read to create non-owning copies (owned=false means
        // Drop is a no-op, so no double-free).
        let mut fla_gi = unsafe { std::ptr::read(&self.fla_gi) };
        let mut fla_ge = unsafe { std::ptr::read(&self.fla_ge) };
        let mut fla_qg = unsafe { std::ptr::read(&self.fla_qg) };
        let mut fla_kg = unsafe { std::ptr::read(&self.fla_kg) };
        let mut fla_ag = unsafe { std::ptr::read(&self.fla_ag) };
        let mut fla_bg = unsafe { std::ptr::read(&self.fla_bg) };
        let mut fla_A_qk = unsafe { std::ptr::read(&self.fla_A_qk) };
        let mut fla_A_qb = unsafe { std::ptr::read(&self.fla_A_qb) };
        let mut fla_A_ab = unsafe { std::ptr::read(&self.fla_A_ab) };
        let mut fla_A_ak = unsafe { std::ptr::read(&self.fla_A_ak) };
        let mut fla_A_ab_inv = unsafe { std::ptr::read(&self.fla_A_ab_inv) };
        let mut fla_w_wy = unsafe { std::ptr::read(&self.fla_w_wy) };
        let mut fla_u_wy = unsafe { std::ptr::read(&self.fla_u_wy) };
        let mut fla_h = unsafe { std::ptr::read(&self.fla_h) };
        let mut fla_v_new = unsafe { std::ptr::read(&self.fla_v_new) };

        // ================================================================
        // Step 0: Compute chunk indices on CPU and upload to GPU
        // ================================================================

        // Build cu_seqlens for equal-length batches: [0, T, 2T, ..., B*T]
        let cu_seqlens_host: Vec<u32> = (0..=b).map(|i| (i * t) as u32).collect();

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
        let cu_shape = TensorShape::new(b + 1, 1, 1, 1);
        let cu_seqlens_flat: Vec<i32> = cu_seqlens_host.iter().map(|&x| x as i32).collect();
        let cu_seqlens_gpu = TensorHip::<i32>::from_slice(&cu_seqlens_flat, cu_shape, stream)?;

        // ================================================================
        // Stage 0.5: Convert w_decay (f16) to gk (f32 log-decay)
        // ================================================================
        // The FLA cumsum kernel needs gk = log(w_decay) in f32.
        // We reuse fla_gi as temporary storage for gk since Stage 1 will
        // overwrite fla_gi anyway. After cumsum, fla_gi holds the inclusive
        // cumsum result.

        // Use a temporary view for gk that shares memory with fla_gi
        let per_token_shape = TensorShape::new(head_size, n_head, t, b);
        let mut gk = fla_gi.resized_view_mut(per_token_shape)?;

        // w_decay is [K, H, T, B] in f16, gk is [K, H, T, B] in f32
        fla_decay_to_log(input.w_decay, &mut gk, stream)?;

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
            &mut fla_gi,
            &mut fla_ge,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
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
        // Note: The FLA intra kernel reads q, k, a, b from the WkvInput.
        // In the WKV pipeline:
        //   input.r = receptance (query in RWKV7 = q in FLA)
        //   input.k = controlled key (k in FLA)
        //   input.v = value (v in FLA)
        //   input.a = wkv_a = -kk (a in FLA)
        //   input.b = wkv_b = kk * a (b in FLA)
        //   input.w_decay = exp(-exp(w)) (decay factor, already converted to gk above)
        fla_intra(
            input.r, // q in FLA (receptance)
            input.k, // k in FLA (controlled key)
            input.a, // a in FLA (wkv_a = -kk)
            input.b, // b in FLA (wkv_b = kk * att_a)
            &fla_gi,
            &fla_ge,
            &mut fla_qg,
            &mut fla_kg,
            &mut fla_ag,
            &mut fla_bg,
            &mut fla_A_qk,
            &mut fla_A_qb,
            &mut fla_A_ak,
            &mut fla_A_ab,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
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
            &fla_A_ab,
            &fla_A_ak,
            &mut fla_A_ab_inv,
            &fla_ag,
            input.v,
            &mut fla_w_wy,
            &mut fla_u_wy,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
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
            &fla_kg,
            &fla_bg,
            input.v,
            &fla_w_wy,
            &fla_u_wy,
            &fla_gi,
            state,
            &mut fla_h,
            &mut fla_v_new,
            &chunk_offsets_gpu,
            &cu_seqlens_gpu,
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
            &fla_qg,
            input.v,
            &fla_v_new,
            &fla_A_qk,
            &fla_A_qb,
            &fla_h,
            output,
            &chunk_indices_gpu,
            &cu_seqlens_gpu,
            c,
            total_chunks,
            stream,
        )?;

        Ok(())
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
