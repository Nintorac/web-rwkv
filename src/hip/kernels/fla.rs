//! HIP kernel wrappers for FLA (Flash Linear Attention) operations.
//!
//! Stage 1: [`fla_cumsum`] — cumulative decay scan within each chunk.

use std::ffi::c_int;

use crate::hip::device::Stream;
use crate::hip::ffi::{check, launch_fla_cumsum, HipErrorKind, Result};
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
