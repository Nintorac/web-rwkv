//! HIP kernel wrappers for FLA (Flash Linear Attention) operations.
//!
//! Stage 1: [`fla_cumsum`] — cumulative decay scan within each chunk.
//! Stage 2: [`fla_intra`] — intra-chunk attention matrices.

use std::ffi::c_int;

use half::f16;

use crate::hip::device::Stream;
use crate::hip::ffi::{check, launch_fla_cumsum, launch_fla_intra, HipErrorKind, Result};
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
