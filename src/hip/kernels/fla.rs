//! HIP kernel wrappers for FLA (Flash Linear Attention) operations.
//!
//! Stage 1: [`fla_cumsum`] — cumulative decay scan within each chunk.
//! Stage 2: [`fla_intra`] — intra-chunk attention matrices.
//! Stage 3: [`fla_wy_repr`] — WY representation (matrix inversion + w/u computation).
//! Stage 4: [`fla_chunk_h`] — inter-chunk state recurrence.
//! Stage 5: [`fla_chunk_o`] — output combination.

use std::ffi::c_int;

use half::f16;

use crate::hip::device::Stream;
use crate::hip::ffi::{
    check, launch_fla_chunk_h, launch_fla_chunk_o, launch_fla_cumsum,
    launch_fla_neg_exp_f16_to_f32, launch_fla_intra, launch_fla_wy_repr, HipErrorKind, Result,
};
use crate::hip::tensor::TensorHip;

/// Launch the FLA cumulative decay scan kernel (Stage 1).
///
/// Computes intra-chunk inclusive and exclusive cumulative sums of log-decay `gk`:
/// - `gi[t] = sum(gk[chunk_start..=t])` — inclusive (add current value, then store)
/// - `ge[t] = sum(gk[chunk_start..t])` — exclusive (store, then add current value)
///
/// # Arguments
/// * `gk` - Input log-decay tensor, shape `[K, H, T, B]` (f32)
/// * `gi` - Output inclusive cumsum tensor, shape `[K, H, T, B]` (f32)
/// * `ge` - Output exclusive cumsum tensor, shape `[K, H, T, B]` (f32)
/// * `chunk_indices` - Flat `[total_chunks * 2]` mapping: `(seq_id, local_chunk_id)` pairs
/// * `cu_seqlens` - Cumulative sequence lengths `[N+1]` (i32)
/// * `chunk_size` - Number of tokens per chunk (C, typically 16)
/// * `total_chunks` - Total number of chunks across all sequences
/// * `stream` - HIP stream for async execution
///
/// # Memory Layout
/// All per-token tensors use column-major `[K, H, T, B]` where K is fastest.
/// For packed/varlen sequences, B=1 and T is total packed length.
///
/// # Errors
/// Returns error on shape mismatches or kernel launch failure.
pub fn fla_cumsum(
    gk: &TensorHip<f32>,
    gi: &mut TensorHip<f32>,
    ge: &mut TensorHip<f32>,
    chunk_indices: &TensorHip<i32>,
    cu_seqlens: &TensorHip<i32>,
    chunk_size: usize,
    total_chunks: usize,
    stream: &Stream,
) -> Result<()> {
    // Extract dimensions from gk shape [K, H, T, B]
    let k = gk.shape()[0]; // head_size
    let h = gk.shape()[1]; // n_heads

    // Validate gi and ge shapes match gk
    if gi.shape() != gk.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_cumsum: gi shape mismatch: expected {}, got {}",
                gk.shape(),
                gi.shape()
            ),
        });
    }
    if ge.shape() != gk.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_cumsum: ge shape mismatch: expected {}, got {}",
                gk.shape(),
                ge.shape()
            ),
        });
    }

    // Validate chunk_indices: should have total_chunks * 2 elements
    if chunk_indices.len() != total_chunks * 2 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_cumsum: chunk_indices length mismatch: expected {} (total_chunks={} * 2), got {}",
                total_chunks * 2,
                total_chunks,
                chunk_indices.len()
            ),
        });
    }

    // Validate contiguity
    if !gk.is_contiguous() || !gi.is_contiguous() || !ge.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_cumsum: all tensors must be contiguous".to_string(),
        });
    }

    if !chunk_indices.is_contiguous() || !cu_seqlens.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_cumsum: chunk_indices and cu_seqlens must be contiguous".to_string(),
        });
    }

    if total_chunks == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_cumsum(
            gk.as_ptr(),
            gi.as_mut_ptr(),
            ge.as_mut_ptr(),
            chunk_indices.as_ptr(),
            cu_seqlens.as_ptr(),
            k as c_int,
            h as c_int,
            chunk_size as c_int,
            total_chunks as c_int,
            stream.handle(),
        ))
    }
}

/// Launch the FLA intra-chunk attention kernel (Stage 2).
///
/// This kernel has two parts:
/// - **Part A** computes decay-scaled vectors from the cumulative decays:
///   - `qg[t] = q[t] * exp(gi[t])`
///   - `kg[t] = k[t] * exp(-gi[t] + g_last)`
///   - `ag[t] = a[t] * exp(ge[t])`
///   - `bg[t] = b[t] * exp(-gi[t] + g_last)`
///
/// - **Part B** computes 4 CxC attention matrices per chunk:
///   - `A_qk[i,j] = sum_k(qg[i,k] * kg[j,k])` for `j <= i` (lower triangular)
///   - `A_qb[i,j] = sum_k(qg[i,k] * bg[j,k])` for `j <= i`
///   - `A_ak[i,j] = sum_k(ag[i,k] * kg[j,k])` for `j < i` (strict lower triangular)
///   - `A_ab[i,j] = sum_k(ag[i,k] * bg[j,k])` for `j < i`
///
/// # Arguments
/// * `q` - Query tensor, shape `[K, H, T, B]` (f16)
/// * `k` - Key tensor, shape `[K, H, T, B]` (f16)
/// * `a` - Adaptation tensor, shape `[K, H, T, B]` (f16)
/// * `b` - Bias tensor, shape `[K, H, T, B]` (f16)
/// * `gi` - Inclusive cumsum from Stage 1, shape `[K, H, T, B]` (f32)
/// * `ge` - Exclusive cumsum from Stage 1, shape `[K, H, T, B]` (f32)
/// * `qg` - Output decay-scaled query, shape `[K, H, T, B]` (f32)
/// * `kg` - Output decay-scaled key, shape `[K, H, T, B]` (f32)
/// * `ag` - Output decay-scaled a, shape `[K, H, T, B]` (f32)
/// * `bg` - Output decay-scaled b, shape `[K, H, T, B]` (f32)
/// * `a_qk` - Output attention matrix Q@K^T, shape `[C, C, H, total_chunks]` (f32)
/// * `a_qb` - Output attention matrix Q@B^T, shape `[C, C, H, total_chunks]` (f32)
/// * `a_ak` - Output attention matrix A@K^T, shape `[C, C, H, total_chunks]` (f32)
/// * `a_ab` - Output attention matrix A@B^T, shape `[C, C, H, total_chunks]` (f32)
/// * `chunk_indices` - Flat `[total_chunks * 2]` mapping: `(seq_id, local_chunk_id)` pairs
/// * `cu_seqlens` - Cumulative sequence lengths `[N+1]` (i32)
/// * `chunk_size` - Number of tokens per chunk (C, typically 16)
/// * `total_chunks` - Total number of chunks across all sequences
/// * `stream` - HIP stream for async execution
///
/// # Memory Layout
/// - Per-token tensors (q,k,a,b,gi,ge,qg,kg,ag,bg): column-major `[K, H, T, B]`
/// - Attention matrices: column-major `[C, C, H, total_chunks]`
///
/// # Errors
/// Returns error on shape mismatches or kernel launch failure.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn fla_intra(
    q: &TensorHip<f16>,
    k: &TensorHip<f16>,
    a: &TensorHip<f16>,
    b: &TensorHip<f16>,
    gi: &TensorHip<f32>,
    ge: &TensorHip<f32>,
    qg: &mut TensorHip<f32>,
    kg: &mut TensorHip<f32>,
    ag: &mut TensorHip<f32>,
    bg: &mut TensorHip<f32>,
    A_qk: &mut TensorHip<f32>,
    A_qb: &mut TensorHip<f32>,
    A_ak: &mut TensorHip<f32>,
    A_ab: &mut TensorHip<f32>,
    chunk_indices: &TensorHip<i32>,
    cu_seqlens: &TensorHip<i32>,
    chunk_size: usize,
    total_chunks: usize,
    stream: &Stream,
) -> Result<()> {
    // Extract dimensions from gi shape [K, H, T, B]
    let k_dim = gi.shape()[0]; // head_size
    let h = gi.shape()[1]; // n_heads

    // Validate f16 input shapes match [K, H, T, B]
    // The f16 inputs have the same shape as the f32 gi/ge tensors
    // (same per-token layout, just different element type)
    let expected_len = gi.len();
    for (name, tensor_len) in [
        ("q", q.len()),
        ("k", k.len()),
        ("a", a.len()),
        ("b", b.len()),
    ] {
        if tensor_len != expected_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_intra: {} length mismatch: expected {}, got {}",
                    name, expected_len, tensor_len
                ),
            });
        }
    }

    // Validate ge shape matches gi
    if ge.shape() != gi.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_intra: ge shape mismatch: expected {}, got {}",
                gi.shape(),
                ge.shape()
            ),
        });
    }

    // Validate f32 per-token output shapes match gi
    for (name, tensor) in [
        ("qg", qg as &TensorHip<f32>),
        ("kg", kg as &TensorHip<f32>),
        ("ag", ag as &TensorHip<f32>),
        ("bg", bg as &TensorHip<f32>),
    ] {
        if tensor.shape() != gi.shape() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_intra: {} shape mismatch: expected {}, got {}",
                    name,
                    gi.shape(),
                    tensor.shape()
                ),
            });
        }
    }

    // Validate attention matrix shapes: [C, C, H, total_chunks]
    let expected_mat_len = chunk_size * chunk_size * h * total_chunks;
    for (name, tensor) in [
        ("A_qk", A_qk as &TensorHip<f32>),
        ("A_qb", A_qb as &TensorHip<f32>),
        ("A_ak", A_ak as &TensorHip<f32>),
        ("A_ab", A_ab as &TensorHip<f32>),
    ] {
        if tensor.len() < expected_mat_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_intra: {} too small: need {} elements (C={}, H={}, chunks={}), got {}",
                    name,
                    expected_mat_len,
                    chunk_size,
                    h,
                    total_chunks,
                    tensor.len()
                ),
            });
        }
    }

    // Validate chunk_indices length
    if chunk_indices.len() != total_chunks * 2 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_intra: chunk_indices length mismatch: expected {} (total_chunks={} * 2), got {}",
                total_chunks * 2,
                total_chunks,
                chunk_indices.len()
            ),
        });
    }

    // Validate contiguity
    if !q.is_contiguous()
        || !k.is_contiguous()
        || !a.is_contiguous()
        || !b.is_contiguous()
        || !gi.is_contiguous()
        || !ge.is_contiguous()
        || !qg.is_contiguous()
        || !kg.is_contiguous()
        || !ag.is_contiguous()
        || !bg.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_intra: all per-token tensors must be contiguous".to_string(),
        });
    }
    if !A_qk.is_contiguous()
        || !A_qb.is_contiguous()
        || !A_ak.is_contiguous()
        || !A_ab.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_intra: all attention matrix tensors must be contiguous".to_string(),
        });
    }
    if !chunk_indices.is_contiguous() || !cu_seqlens.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_intra: chunk_indices and cu_seqlens must be contiguous".to_string(),
        });
    }

    if total_chunks == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_intra(
            q.as_ptr(),
            k.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            gi.as_ptr(),
            ge.as_ptr(),
            qg.as_mut_ptr(),
            kg.as_mut_ptr(),
            ag.as_mut_ptr(),
            bg.as_mut_ptr(),
            A_qk.as_mut_ptr(),
            A_qb.as_mut_ptr(),
            A_ak.as_mut_ptr(),
            A_ab.as_mut_ptr(),
            chunk_indices.as_ptr(),
            cu_seqlens.as_ptr(),
            k_dim as c_int,
            h as c_int,
            chunk_size as c_int,
            total_chunks as c_int,
            stream.handle(),
        ))
    }
}

/// Launch the FLA WY representation kernel (Stage 3).
///
/// This kernel has two parts:
/// - **Part A** (`kernel_fla_wy_inv`) inverts the strict lower-triangular `A_ab` matrix
///   per chunk via forward substitution, producing `A_ab_inv = (I - A_ab)^{-1}`.
///
/// - **Part B** (`kernel_fla_wy_wu`) computes:
///   - `w = A_ab_inv @ ag` — WY representation of decay-scaled adaptation
///   - `u = (A_ab_inv @ A_ak) @ v` — WY representation of state contribution
///
///   These encode how chunk-internal recurrence affects the state, enabling the
///   inter-chunk propagation (Stage 4) to incorporate within-chunk dynamics.
///
/// # Arguments
/// * `a_ab` - Intra-chunk attention matrix A@B^T from Stage 2, shape `[C, C, H, total_chunks]` (f32)
/// * `a_ak` - Intra-chunk attention matrix A@K^T from Stage 2, shape `[C, C, H, total_chunks]` (f32)
/// * `a_ab_inv` - Output inverse matrix, shape `[C, C, H, total_chunks]` (f32)
/// * `ag` - Decay-scaled adaptation from Stage 2, shape `[K, H, T, B]` (f32)
/// * `v` - Value tensor (original), shape `[K, H, T, B]` (f16)
/// * `w_wy` - Output WY w tensor, shape `[K, H, T, B]` (f32)
/// * `u_wy` - Output WY u tensor, shape `[K, H, T, B]` (f32)
/// * `chunk_indices` - Flat `[total_chunks * 2]` mapping: `(seq_id, local_chunk_id)` pairs
/// * `cu_seqlens` - Cumulative sequence lengths `[N+1]` (i32)
/// * `chunk_size` - Number of tokens per chunk (C, typically 16)
/// * `total_chunks` - Total number of chunks across all sequences
/// * `stream` - HIP stream for async execution
///
/// # Memory Layout
/// - Attention matrices: column-major `[C, C, H, total_chunks]`
/// - Per-token tensors: column-major `[K, H, T, B]`
///
/// # Errors
/// Returns error on shape mismatches or kernel launch failure.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn fla_wy_repr(
    A_ab: &TensorHip<f32>,
    A_ak: &TensorHip<f32>,
    A_ab_inv: &mut TensorHip<f32>,
    ag: &TensorHip<f32>,
    v: &TensorHip<f16>,
    w_wy: &mut TensorHip<f32>,
    u_wy: &mut TensorHip<f32>,
    chunk_indices: &TensorHip<i32>,
    cu_seqlens: &TensorHip<i32>,
    chunk_size: usize,
    total_chunks: usize,
    stream: &Stream,
) -> Result<()> {
    // Extract dimensions from ag shape [K, H, T, B]
    let k_dim = ag.shape()[0]; // head_size
    let h = ag.shape()[1]; // n_heads

    // Validate attention matrix sizes: [C, C, H, total_chunks]
    let expected_mat_len = chunk_size * chunk_size * h * total_chunks;
    for (name, tensor) in [
        ("A_ab", A_ab as &TensorHip<f32>),
        ("A_ak", A_ak as &TensorHip<f32>),
        ("A_ab_inv", A_ab_inv as &TensorHip<f32>),
    ] {
        if tensor.len() < expected_mat_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_wy_repr: {} too small: need {} elements (C={}, H={}, chunks={}), got {}",
                    name,
                    expected_mat_len,
                    chunk_size,
                    h,
                    total_chunks,
                    tensor.len()
                ),
            });
        }
    }

    // Validate ag shape matches expected per-token layout [K, H, T, B]
    // v (f16) should have the same length as ag (f32)
    if v.len() != ag.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_wy_repr: v length mismatch: expected {} (same as ag), got {}",
                ag.len(),
                v.len()
            ),
        });
    }

    // Validate w_wy and u_wy shapes match ag
    for (name, tensor) in [
        ("w_wy", w_wy as &TensorHip<f32>),
        ("u_wy", u_wy as &TensorHip<f32>),
    ] {
        if tensor.shape() != ag.shape() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_wy_repr: {} shape mismatch: expected {}, got {}",
                    name,
                    ag.shape(),
                    tensor.shape()
                ),
            });
        }
    }

    // Validate chunk_indices length
    if chunk_indices.len() != total_chunks * 2 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_wy_repr: chunk_indices length mismatch: expected {} (total_chunks={} * 2), got {}",
                total_chunks * 2,
                total_chunks,
                chunk_indices.len()
            ),
        });
    }

    // Validate contiguity
    if !A_ab.is_contiguous()
        || !A_ak.is_contiguous()
        || !A_ab_inv.is_contiguous()
        || !ag.is_contiguous()
        || !v.is_contiguous()
        || !w_wy.is_contiguous()
        || !u_wy.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_wy_repr: all tensors must be contiguous".to_string(),
        });
    }
    if !chunk_indices.is_contiguous() || !cu_seqlens.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_wy_repr: chunk_indices and cu_seqlens must be contiguous".to_string(),
        });
    }

    if total_chunks == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_wy_repr(
            A_ab.as_ptr(),
            A_ak.as_ptr(),
            A_ab_inv.as_mut_ptr(),
            ag.as_ptr(),
            v.as_ptr(),
            w_wy.as_mut_ptr(),
            u_wy.as_mut_ptr(),
            chunk_indices.as_ptr(),
            cu_seqlens.as_ptr(),
            k_dim as c_int,
            h as c_int,
            chunk_size as c_int,
            total_chunks as c_int,
            stream.handle(),
        ))
    }
}

/// Launch the FLA inter-chunk state recurrence kernel (Stage 4).
///
/// This is the sequential bottleneck: it propagates state across chunks
/// within each sequence. One thread-block processes ALL chunks for a
/// single (head, sequence) pair.
///
/// Per chunk, the recurrence is:
/// ```text
/// // Store current state to per-chunk buffer
/// fla_h[chunk] = h
///
/// // For each position in chunk:
/// b_v2 = w_wy @ h + u_wy        // corrected values [C, V]
/// hc  += kg^T @ v + bg^T @ b_v2  // chunk contribution [K, V]
/// v_new = b_v2                    // store for output stage
///
/// // Decay and update
/// g_last = gi[last_position_in_chunk]
/// h = h * exp(g_last) + hc
/// ```
///
/// # Arguments
/// * `kg` - Decay-scaled key from Stage 2, shape `[K, H, T, B]` (f32)
/// * `bg` - Decay-scaled b from Stage 2, shape `[K, H, T, B]` (f32)
/// * `v` - Original value tensor, shape `[K, H, T, B]` (f16)
/// * `w_wy` - WY w output from Stage 3, shape `[K, H, T, B]` (f32)
/// * `u_wy` - WY u output from Stage 3, shape `[K, H, T, B]` (f32)
/// * `gi` - Inclusive cumsum from Stage 1, shape `[K, H, T, B]` (f32)
/// * `state` - State tensor (in/out), shape `[K, K, H, B]` (f32)
/// * `h_out` - Per-chunk intermediate states, shape `[K, K, H, total_chunks]` (f32)
/// * `v_new` - Corrected values output, shape `[K, H, T, B]` (f32)
/// * `chunk_offsets` - Cumulative chunk counts per sequence `[N+1]` (i32)
/// * `cu_seqlens` - Cumulative sequence lengths `[N+1]` (i32)
/// * `chunk_size` - Number of tokens per chunk (C, typically 16)
/// * `n_seq` - Number of sequences in the batch
/// * `stream` - HIP stream for async execution
///
/// # Memory Layout
/// - State: column-major `[K, K, H, B]` (K fastest)
/// - Per-chunk states: column-major `[K, K, H, total_chunks]`
/// - Per-token tensors: column-major `[K, H, T, B]`
///
/// # Errors
/// Returns error on shape mismatches or kernel launch failure.
#[allow(clippy::too_many_arguments)]
pub fn fla_chunk_h(
    kg: &TensorHip<f32>,
    bg: &TensorHip<f32>,
    v: &TensorHip<f16>,
    w_wy: &TensorHip<f32>,
    u_wy: &TensorHip<f32>,
    gi: &TensorHip<f32>,
    state: &mut TensorHip<f32>,
    h_out: &mut TensorHip<f32>,
    v_new: &mut TensorHip<f32>,
    chunk_offsets: &TensorHip<i32>,
    cu_seqlens: &TensorHip<i32>,
    chunk_size: usize,
    n_seq: usize,
    stream: &Stream,
) -> Result<()> {
    // Extract dimensions from kg shape [K, H, T, B]
    let k_dim = kg.shape()[0]; // head_size
    let h = kg.shape()[1]; // n_heads

    // Validate per-token f32 tensor shapes match kg
    for (name, tensor) in [
        ("bg", bg as &TensorHip<f32>),
        ("w_wy", w_wy as &TensorHip<f32>),
        ("u_wy", u_wy as &TensorHip<f32>),
        ("gi", gi as &TensorHip<f32>),
        ("v_new", v_new as &TensorHip<f32>),
    ] {
        if tensor.shape() != kg.shape() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_chunk_h: {} shape mismatch: expected {}, got {}",
                    name,
                    kg.shape(),
                    tensor.shape()
                ),
            });
        }
    }

    // Validate v (f16) has the same length as kg (f32)
    if v.len() != kg.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_h: v length mismatch: expected {} (same as kg), got {}",
                kg.len(),
                v.len()
            ),
        });
    }

    // Validate state shape: [K, K, H, B]
    // B = n_seq for varlen (packed) sequences
    let expected_state_len = k_dim * k_dim * h * n_seq;
    if state.len() < expected_state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_h: state too small: need {} elements (K={}, H={}, N={}), got {}",
                expected_state_len,
                k_dim,
                h,
                n_seq,
                state.len()
            ),
        });
    }

    // Validate chunk_offsets: should have n_seq + 1 elements
    if chunk_offsets.len() != n_seq + 1 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_h: chunk_offsets length mismatch: expected {} (n_seq={} + 1), got {}",
                n_seq + 1,
                n_seq,
                chunk_offsets.len()
            ),
        });
    }

    // Validate cu_seqlens: should have n_seq + 1 elements
    if cu_seqlens.len() != n_seq + 1 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_h: cu_seqlens length mismatch: expected {} (n_seq={} + 1), got {}",
                n_seq + 1,
                n_seq,
                cu_seqlens.len()
            ),
        });
    }

    // Validate contiguity
    if !kg.is_contiguous()
        || !bg.is_contiguous()
        || !v.is_contiguous()
        || !w_wy.is_contiguous()
        || !u_wy.is_contiguous()
        || !gi.is_contiguous()
        || !v_new.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_h: all per-token tensors must be contiguous".to_string(),
        });
    }
    if !state.is_contiguous() || !h_out.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_h: state and h_out must be contiguous".to_string(),
        });
    }
    if !chunk_offsets.is_contiguous() || !cu_seqlens.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_h: chunk_offsets and cu_seqlens must be contiguous".to_string(),
        });
    }

    if n_seq == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_chunk_h(
            kg.as_ptr(),
            bg.as_ptr(),
            v.as_ptr(),
            w_wy.as_ptr(),
            u_wy.as_ptr(),
            gi.as_ptr(),
            state.as_mut_ptr(),
            h_out.as_mut_ptr(),
            v_new.as_mut_ptr(),
            chunk_offsets.as_ptr(),
            cu_seqlens.as_ptr(),
            k_dim as c_int,
            h as c_int,
            chunk_size as c_int,
            n_seq as c_int,
            stream.handle(),
        ))
    }
}

/// Launch the FLA output combination kernel (Stage 5).
///
/// Combines intra-chunk attention with inter-chunk state contributions to
/// produce the final WKV7 output:
///
/// ```text
/// o[i, :] = qg[i, :] @ h[chunk]               -- state contribution
///         + sum_j(A_qk[i,j] * v[j, :])          -- local attention (original values)
///         + sum_j(A_qb[i,j] * v_new[j, :])      -- WY correction (corrected values)
/// ```
///
/// Where `v_new = w_wy @ h + u_wy` was pre-computed by Stage 4.
///
/// Both `A_qk` and `A_qb` are applied with causal mask (`i >= j`).
/// Positions beyond sequence length produce zero output.
///
/// # Arguments
/// * `qg` - Decay-scaled query from Stage 2, shape `[K, H, T, B]` (f32)
/// * `v` - Original value tensor, shape `[K, H, T, B]` (f16)
/// * `v_new` - Corrected values from Stage 4, shape `[K, H, T, B]` (f32)
/// * `a_qk` - Intra-chunk attention matrix Q@K^T from Stage 2, shape `[C, C, H, total_chunks]` (f32)
/// * `a_qb` - Intra-chunk attention matrix Q@B^T from Stage 2, shape `[C, C, H, total_chunks]` (f32)
/// * `h` - Per-chunk intermediate states from Stage 4, shape `[K, K, H, total_chunks]` (f32)
/// * `o` - Output tensor, shape `[K, H, T, B]` (f16)
/// * `chunk_indices` - Flat `[total_chunks * 2]` mapping: `(seq_id, local_chunk_id)` pairs
/// * `cu_seqlens` - Cumulative sequence lengths `[N+1]` (i32)
/// * `chunk_size` - Number of tokens per chunk (C, typically 16)
/// * `total_chunks` - Total number of chunks across all sequences
/// * `stream` - HIP stream for async execution
///
/// # Memory Layout
/// - Per-token tensors (qg, v, v_new, o): column-major `[K, H, T, B]`
/// - Attention matrices: column-major `[C, C, H, total_chunks]`
/// - Per-chunk states: column-major `[K, K, H, total_chunks]`
///
/// # Errors
/// Returns error on shape mismatches or kernel launch failure.
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn fla_chunk_o(
    qg: &TensorHip<f32>,
    v: &TensorHip<f16>,
    v_new: &TensorHip<f32>,
    A_qk: &TensorHip<f32>,
    A_qb: &TensorHip<f32>,
    h: &TensorHip<f32>,
    o: &mut TensorHip<f16>,
    chunk_indices: &TensorHip<i32>,
    cu_seqlens: &TensorHip<i32>,
    chunk_size: usize,
    total_chunks: usize,
    stream: &Stream,
) -> Result<()> {
    // Extract dimensions from qg shape [K, H, T, B]
    let k_dim = qg.shape()[0]; // head_size
    let h_dim = qg.shape()[1]; // n_heads

    // Validate v (f16) has the same length as qg (f32)
    if v.len() != qg.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_o: v length mismatch: expected {} (same as qg), got {}",
                qg.len(),
                v.len()
            ),
        });
    }

    // Validate v_new shape matches qg
    if v_new.shape() != qg.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_o: v_new shape mismatch: expected {}, got {}",
                qg.shape(),
                v_new.shape()
            ),
        });
    }

    // Validate output shape: o (f16) has same length as qg (f32)
    if o.len() != qg.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_o: o length mismatch: expected {} (same as qg), got {}",
                qg.len(),
                o.len()
            ),
        });
    }

    // Validate attention matrix sizes: [C, C, H, total_chunks]
    let expected_mat_len = chunk_size * chunk_size * h_dim * total_chunks;
    for (name, tensor) in [
        ("A_qk", A_qk as &TensorHip<f32>),
        ("A_qb", A_qb as &TensorHip<f32>),
    ] {
        if tensor.len() < expected_mat_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "fla_chunk_o: {} too small: need {} elements (C={}, H={}, chunks={}), got {}",
                    name,
                    expected_mat_len,
                    chunk_size,
                    h_dim,
                    total_chunks,
                    tensor.len()
                ),
            });
        }
    }

    // Validate per-chunk state size: [K, K, H, total_chunks]
    let expected_h_len = k_dim * k_dim * h_dim * total_chunks;
    if h.len() < expected_h_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_o: h too small: need {} elements (K={}, H={}, chunks={}), got {}",
                expected_h_len,
                k_dim,
                h_dim,
                total_chunks,
                h.len()
            ),
        });
    }

    // Validate chunk_indices length
    if chunk_indices.len() != total_chunks * 2 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_chunk_o: chunk_indices length mismatch: expected {} (total_chunks={} * 2), got {}",
                total_chunks * 2,
                total_chunks,
                chunk_indices.len()
            ),
        });
    }

    // Validate contiguity
    if !qg.is_contiguous()
        || !v.is_contiguous()
        || !v_new.is_contiguous()
        || !o.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_o: all per-token tensors must be contiguous".to_string(),
        });
    }
    if !A_qk.is_contiguous() || !A_qb.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_o: attention matrix tensors must be contiguous".to_string(),
        });
    }
    if !h.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_o: h (per-chunk state) must be contiguous".to_string(),
        });
    }
    if !chunk_indices.is_contiguous() || !cu_seqlens.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_chunk_o: chunk_indices and cu_seqlens must be contiguous".to_string(),
        });
    }

    if total_chunks == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_chunk_o(
            qg.as_ptr(),
            v.as_ptr(),
            v_new.as_ptr(),
            A_qk.as_ptr(),
            A_qb.as_ptr(),
            h.as_ptr(),
            o.as_mut_ptr(),
            chunk_indices.as_ptr(),
            cu_seqlens.as_ptr(),
            k_dim as c_int,
            h_dim as c_int,
            chunk_size as c_int,
            total_chunks as c_int,
            stream.handle(),
        ))
    }
}

/// Convert f16 raw log-decay att_w to f32 gk = -exp(att_w).
///
/// Computes `gk[i] = -expf(att_w[i])` where `att_w` holds the raw log-domain
/// decay (`-softplus(...) - 0.5`) before the `exp(-exp(w))` conversion.
/// This avoids the precision-losing round-trip through f16 exp then log.
///
/// # Arguments
/// * `att_w` - Input f16 raw log-domain decay tensor, any contiguous shape
/// * `gk` - Output f32 tensor, same number of elements
/// * `stream` - HIP stream for async execution
///
/// # Errors
/// Returns error on length mismatch, non-contiguity, or kernel launch failure.
pub fn fla_neg_exp_f16_to_f32(
    att_w: &TensorHip<f16>,
    gk: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    if att_w.len() != gk.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "fla_neg_exp_f16_to_f32: length mismatch: att_w={}, gk={}",
                att_w.len(),
                gk.len()
            ),
        });
    }

    if !att_w.is_contiguous() || !gk.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "fla_neg_exp_f16_to_f32: both tensors must be contiguous".to_string(),
        });
    }

    let n = att_w.len();
    if n == 0 {
        return Ok(());
    }

    unsafe {
        check(launch_fla_neg_exp_f16_to_f32(
            att_w.as_ptr(),
            gk.as_mut_ptr(),
            n as c_int,
            stream.handle(),
        ))
    }
}
