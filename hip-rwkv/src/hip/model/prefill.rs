//! WKV kernel abstraction for swappable prefill/decode implementations.
//!
//! This module defines the [`WkvKernel`] trait, which allows the T>1 prefill
//! kernel to be swapped for alternative implementations (e.g., a future FLA
//! module) without changing the inference loop in `step_inner()`.
//!
//! Two built-in implementations are provided:
//! - [`FusedT1Wkv`] -- optimized for T=1 decode (in-place state update)
//! - [`WaveReduceWkv`] -- wave-cooperative reduction for T>1 prefill

use half::f16;

use crate::hip::device::Stream;
use crate::hip::ffi::Result;
use crate::hip::kernels::{wkv7_fused_t1, wkv7_wave_reduce};
use crate::hip::tensor::TensorHip;

/// Inputs to a WKV kernel invocation.
///
/// All tensor views are already reshaped to WKV layout `[head_size, n_head, T, B]`
/// before being passed here.
pub struct WkvInput<'a> {
    /// Decay weights, shape `[head_size, n_head, T, B]`
    pub w_decay: &'a TensorHip<f16>,
    /// Query (receptance), shape `[head_size, n_head, T, B]`
    pub r: &'a TensorHip<f16>,
    /// Key (controlled), shape `[head_size, n_head, T, B]`
    pub k: &'a TensorHip<f16>,
    /// Value, shape `[head_size, n_head, T, B]`
    pub v: &'a TensorHip<f16>,
    /// Negative normalized key for state update, shape `[head_size, n_head, T, B]`
    pub a: &'a TensorHip<f16>,
    /// Adaptation-weighted normalized key, shape `[head_size, n_head, T, B]`
    pub b: &'a TensorHip<f16>,
    /// Per-batch sequence lengths, shape `[B, 1, 1, 1]`
    pub lengths: &'a TensorHip<i32>,
}

/// Trait abstracting the WKV7 attention kernel.
///
/// Implementations must handle both the output computation and state update.
/// The caller provides the current per-layer recurrent state which is updated
/// in-place by all implementations.
///
/// # Adding a new implementation
///
/// To add a new WKV kernel (e.g., FLA):
///
/// 1. Create a unit struct (e.g., `pub struct FlaWkv;`)
/// 2. Implement `WkvKernel` for it
/// 3. Select it in `step_inner()` based on configuration or token count
pub trait WkvKernel: Send + Sync {
    /// Execute the WKV attention kernel.
    ///
    /// # Arguments
    /// * `input` -- Pre-shaped WKV input tensors
    /// * `state` -- Per-layer recurrent state `[head_size, head_size, n_head, B]` (f32).
    ///   Updated in-place by all implementations.
    /// * `output` -- Output tensor `[head_size, n_head, T, B]` (f16)
    /// * `stream` -- HIP stream for kernel launches
    fn compute(
        &self,
        input: &WkvInput<'_>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        stream: &Stream,
    ) -> Result<()>;

    /// Whether this kernel supports multi-token (T>1) sequences.
    fn supports_multi_token(&self) -> bool;

    /// Human-readable name for logging / profiling.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Built-in implementations
// ---------------------------------------------------------------------------

/// Wave-cooperative WKV7 kernel for T>=1 prefill.
///
/// Uses wave-shuffle reduction with in-place state update. The HIP kernel
/// loads state into LDS before writing, so aliasing is safe.
pub struct WaveReduceWkv;

impl WkvKernel for WaveReduceWkv {
    fn compute(
        &self,
        input: &WkvInput<'_>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        stream: &Stream,
    ) -> Result<()> {
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
        "wave_reduce"
    }
}

/// Fused WKV7 kernel optimized for T=1 decode.
///
/// Updates state in-place (single mutable tensor, no separate in/out).
pub struct FusedT1Wkv;

impl WkvKernel for FusedT1Wkv {
    fn compute(
        &self,
        input: &WkvInput<'_>,
        state: &mut TensorHip<f32>,
        output: &mut TensorHip<f16>,
        stream: &Stream,
    ) -> Result<()> {
        wkv7_fused_t1(
            input.w_decay,
            input.r,
            input.k,
            input.v,
            input.a,
            input.b,
            state,  // in-place state update
            output,
            input.lengths,
            stream,
        )
    }

    fn supports_multi_token(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "fused_t1"
    }
}
