//! Reusable dispatch helper functions for RWKV7 HIP forward pass.
//!
//! These helpers extract the core computational logic shared between
//! `HipDecode::dispatch_decode()` (decode.rs) and `HipPrefill::dispatch_fla()`
//! (hip_prefill.rs). Each function takes explicit tensor references rather than
//! scratch structs, so it works with any scratch type (PrefillScratch, DecodeScratch).
//!
//! The WKV kernel call is parameterized via a closure in `attention_block`,
//! allowing callers to provide either FLA (prefill) or FusedT1Wkv (decode).
//!
//! When the `hip-probes` feature is enabled, the helpers accept an optional
//! `ProbeState` to fire probe hooks at the same points the old monolithic
//! `dispatch()` used to. When the feature is disabled, the probe parameters
//! compile away entirely (zero overhead).

use half::f16;

use crate::hip::blas::HipBlasContext;
use crate::hip::device::Stream;
use crate::hip::ffi::Result;
use crate::hip::kernels::{
    add_f16, broadcast_add_f16, broadcast_mul_f16, channel_mix_state_f16, control_k_f16,
    copy_f16_to_f32, copy_tensor_f16, decay_exp_f16, group_norm_f16, l2_norm_f16, layer_norm_f16,
    lerp_f16, mul_f16, negate_f16, sigmoid_f16, softplus_decay_f16, squared_relu_f16, tanh_f16,
    wkv_bonus_f16,
};
use crate::hip::pinned::PinnedBuffer;
use crate::hip::tensor::{TensorHip, TensorShape};

use super::weights::{AttentionHip, EmbedHip, FfnHip, HeadHip, LayerHip, LayerNormHip};

#[cfg(feature = "hip-probes")]
use crate::hip::probe::{self, HipProbeMapRef, ProbeContext};

// ---------------------------------------------------------------------------
// Probe support (compiles away without `hip-probes` feature)
// ---------------------------------------------------------------------------

/// Bundles the probe map reference and mutable context so helpers can fire
/// probes without knowing about `Rwkv7Hip`.
///
/// When `hip-probes` is disabled this is a zero-size unit struct; callers
/// always pass `&mut None::<ProbeState>` when they don't have probes.
#[cfg(feature = "hip-probes")]
pub struct ProbeState<'a> {
    pub probes: &'a HipProbeMapRef,
    pub ctx: &'a mut ProbeContext,
    /// Padded chunk size (GPU tensor T dimension) -- needed for stacking
    pub t_stride: usize,
}

#[cfg(not(feature = "hip-probes"))]
pub struct ProbeState<'a>(std::marker::PhantomData<&'a ()>);

/// Download a GPU f16 tensor to a CPU f32 Vec.
/// Only used when probes are enabled; the cost is acceptable for validation.
#[cfg(feature = "hip-probes")]
#[inline]
fn download_f16_as_f32(tensor: &TensorHip<f16>, stream: &Stream) -> Result<Vec<f32>> {
    let f16_data = tensor.to_vec(stream)?;
    Ok(f16_data.iter().map(|v| v.to_f32()).collect())
}

/// Download a GPU f32 tensor to a CPU f32 Vec.
#[cfg(feature = "hip-probes")]
#[inline]
fn download_f32(tensor: &TensorHip<f32>, stream: &Stream) -> Result<Vec<f32>> {
    tensor.to_vec(stream)
}

/// Fire a single probe hook, trimming padded data if needed.
#[cfg(feature = "hip-probes")]
#[inline]
fn fire_probe(
    probes: &HipProbeMapRef,
    ctx: &mut ProbeContext,
    hook: probe::HipHook,
    data: &[f32],
    shape: &[usize],
) {
    if let Some(f) = probes.get(&hook) {
        ctx.set_shape(shape);
        let expected_len: usize = shape.iter().product();
        if data.len() == expected_len {
            f(data, ctx);
        } else {
            let trimmed = probe::trim_padded_data(data, ctx.chunk_size, shape);
            f(&trimmed, ctx);
        }
    }
}

/// Build a stacked tensor from multiple downloaded tensors, interleaved by
/// (batch, token) position. This replicates the stacking logic from step.rs.
#[cfg(feature = "hip-probes")]
fn stack_tensors(
    tensors: &[&[f32]],
    n_embd: usize,
    actual_t: usize,
    t_stride: usize,
    b: usize,
) -> Vec<f32> {
    let n_stack = tensors.len();
    let mut stacked = Vec::with_capacity(n_embd * n_stack * actual_t * b);
    for b_idx in 0..b {
        for t_idx in 0..actual_t {
            let base = n_embd * (t_idx + t_stride * b_idx);
            for tensor in tensors {
                stacked.extend_from_slice(&tensor[base..base + n_embd]);
            }
        }
    }
    stacked
}

/// Inputs for the WKV kernel call within `attention_block`.
///
/// These are the reshaped tensors (wkv_data_shape) that both FLA and
/// FusedT1Wkv need. The `att_w_wkv` field contains raw att_w for FLA
/// or is unused for decode (which uses `w_decay_wkv` instead).
pub struct WkvCallInputs<'a> {
    /// Raw attention decay (pre-exponentiation), reshaped to wkv_data_shape.
    /// Used by FLA for better precision; decode uses w_decay_wkv instead.
    pub att_w_wkv: &'a TensorHip<f16>,
    /// Exponentiated decay: exp(-exp(att_w)), reshaped to wkv_data_shape.
    /// Used by FusedT1Wkv (decode); FLA ignores this.
    pub w_decay_wkv: &'a TensorHip<f16>,
    /// Receptance, reshaped to wkv_data_shape.
    pub r_wkv: &'a TensorHip<f16>,
    /// Controlled key, reshaped to wkv_data_shape.
    pub k_ctrl_wkv: &'a TensorHip<f16>,
    /// Value, reshaped to wkv_data_shape.
    pub v_wkv: &'a TensorHip<f16>,
    /// WKV A (negated kk), reshaped to wkv_data_shape.
    pub wkv_a_wkv: &'a TensorHip<f16>,
    /// WKV B (kk * a), reshaped to wkv_data_shape.
    pub wkv_b_wkv: &'a TensorHip<f16>,
}

/// Perform embedding table lookup and upload to GPU.
///
/// Looks up token embeddings from the CPU-side embedding table, copies them
/// into the pinned staging buffer, then performs an async H2D transfer to
/// the GPU tensor `x`.
///
/// For layer 0, also applies ln0 (embedding layer norm) in-place on `x`.
///
/// # Arguments
/// * `tokens` - Batch of token sequences, shape `[b][t]`
/// * `embed` - Embedding weights (CPU-side table + layer norm)
/// * `n_embd` - Embedding dimension
/// * `emb_staging` - Pinned host staging buffer for async upload
/// * `x` - GPU tensor to receive embedded tokens, shape `[n_embd, t, b]`
/// * `x_ln` - Temporary buffer for layer norm output
/// * `stream` - HIP stream for async operations
pub fn embed_lookup(
    tokens: &[&[u32]],
    embed: &EmbedHip,
    n_embd: usize,
    emb_staging: &mut PinnedBuffer<f16>,
    x: &mut TensorHip<f16>,
    x_ln: &mut TensorHip<f16>,
    stream: &Stream,
    probe: &mut Option<ProbeState<'_>>,
) -> Result<()> {
    let b = tokens.len();
    let t = tokens[0].len();

    // Embedding lookup: tokens[b][t] -> x[c, t, b]
    // Embedding table is kept on CPU - use pinned staging buffer for async upload
    let emb_data = &embed.w;
    let emb_stride = embed.n_embd;
    let x_host = emb_staging.as_slice_mut();
    for batch_idx in 0..b {
        for time_idx in 0..t {
            let token = tokens[batch_idx][time_idx] as usize;
            let src_offset = token * emb_stride;
            let dst_offset = batch_idx * t * n_embd + time_idx * n_embd;
            x_host[dst_offset..dst_offset + n_embd]
                .copy_from_slice(&emb_data[src_offset..src_offset + n_embd]);
        }
    }
    // Async copy from pinned host memory to GPU (truly non-blocking)
    unsafe {
        emb_staging.copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
    }

    // PostEmbed probe: raw embedding before ln0
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostEmbed) {
                let data = download_f16_as_f32(x, stream)?;
                let actual_t = ps.ctx.seq_len;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostEmbed, &data, &[n_embd, actual_t, b]);
            }
        }
    }
    let _ = &probe; // suppress unused warning when hip-probes is off

    // Apply ln0 (embedding layer norm)
    layer_norm_f16(x, &embed.ln.weight, &embed.ln.bias, x_ln, 1e-5, stream)?;
    copy_tensor_f16(x_ln, x, stream)?;

    // PostEmbedLayerNorm probe: after ln0
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostEmbedLayerNorm) {
                let data = download_f16_as_f32(x, stream)?;
                let actual_t = ps.ctx.seq_len;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostEmbedLayerNorm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    Ok(())
}

/// Run one attention (time-mix) block for a single layer.
///
/// Performs:
/// 1. Attention layer norm
/// 2. Token shifts (masked for x_r to extract correct state)
/// 3. Linear projections (r, k, v)
/// 4. Decay computation (LoRA w pathway)
/// 5. Adaptation (LoRA a pathway)
/// 6. Gate (LoRA g pathway)
/// 7. Value residual (layers > 0)
/// 8. L2 key normalization
/// 9. Control K
/// 10. WKV input preparation (negate kk, kk*a, decay_exp)
/// 11. WKV kernel call (via caller-provided closure)
/// 12. Group norm on WKV output
/// 13. WKV bonus
/// 14. Combine, gate, and output projection
/// 15. Residual addition
///
/// The WKV call is parameterized by `wkv_fn`, which receives the reshaped
/// WKV inputs, the per-layer state tensor, and the output tensor.
///
/// # Type parameter
/// * `F` - Closure that runs the WKV kernel. Signature:
///   `FnOnce(WkvCallInputs, &mut TensorHip<f32>, &mut TensorHip<f16>) -> Result<()>`
///   where the second arg is the per-layer wkv_state and the third is wkv_out (reshaped).
#[allow(clippy::too_many_arguments)]
pub fn attention_block<F>(
    layer_idx: usize,
    layer: &LayerHip,
    // Core tensors
    x: &mut TensorHip<f16>,
    x_ln: &mut TensorHip<f16>,
    // Token shift buffers
    att_xr: &mut TensorHip<f16>,
    att_xw: &mut TensorHip<f16>,
    att_xk: &mut TensorHip<f16>,
    att_xv: &mut TensorHip<f16>,
    att_xa: &mut TensorHip<f16>,
    att_xg: &mut TensorHip<f16>,
    // Shift state
    att_shift_state: &mut TensorHip<f16>,
    new_att_shift: &mut TensorHip<f16>,
    // Linear projection outputs
    att_r: &mut TensorHip<f16>,
    att_k: &mut TensorHip<f16>,
    att_v: &mut TensorHip<f16>,
    att_w: &mut TensorHip<f16>,
    att_a: &mut TensorHip<f16>,
    att_g: &mut TensorHip<f16>,
    // Key normalization
    att_kk: &mut TensorHip<f16>,
    att_k_ctrl: &mut TensorHip<f16>,
    // WKV intermediates
    wkv_a: &mut TensorHip<f16>,
    wkv_b: &mut TensorHip<f16>,
    w_decay: &mut TensorHip<f16>,
    wkv_out: &mut TensorHip<f16>,
    wkv_normed: &mut TensorHip<f16>,
    wkv_bonus: &mut TensorHip<f16>,
    // Output
    att_out: &mut TensorHip<f16>,
    // v_first for value residual
    v_first: &mut TensorHip<f16>,
    // LoRA buffers
    lora_w: &mut TensorHip<f16>,
    lora_w_tanh: &mut TensorHip<f16>,
    lora_a: &mut TensorHip<f16>,
    lora_a_proj: &mut TensorHip<f16>,
    lora_g: &mut TensorHip<f16>,
    lora_g_sig: &mut TensorHip<f16>,
    lora_v: &mut TensorHip<f16>,
    v_lora2: &mut TensorHip<f16>,
    // Temporaries
    temp1: &mut TensorHip<f16>,
    temp2: &mut TensorHip<f16>,
    // Lens GPU tensor for masked shift
    lens_gpu: &TensorHip<i32>,
    // Batch offsets GPU tensor for packed sequence addressing
    batch_offsets_gpu: &TensorHip<i32>,
    // WKV state for this layer
    wkv_state: &mut TensorHip<f32>,
    // Model dimensions
    head_size: usize,
    n_head: usize,
    // Shape for WKV reshape
    wkv_data_shape: TensorShape,
    // BLAS context
    ctx: &HipBlasContext,
    stream: &Stream,
    // WKV kernel closure
    wkv_fn: F,
    // Optional probe state (zero-cost when hip-probes feature is off)
    probe: &mut Option<ProbeState<'_>>,
) -> Result<()>
where
    F: FnOnce(WkvCallInputs<'_>, &mut TensorHip<f32>, &mut TensorHip<f16>) -> Result<()>,
{
    let n_embd = head_size * n_head;

    // ==== Time-Mix (Attention) ====
    {
        layer_norm_f16(
            x,
            &layer.att_ln.weight,
            &layer.att_ln.bias,
            x_ln,
            1e-5,
            stream,
        )?;
    }

    // PostAttLayerNorm probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttLayerNorm) {
                let data = download_f16_as_f32(x_ln, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttLayerNorm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Token shifts for attention - unified kernel uses lengths + batch_offsets
    // x_r shift extracts the correct state at lengths[b]-1
    {
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_r,
            att_xr,
            new_att_shift,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
        // Remaining shifts use same unified kernel (we only need outputs, not state)
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_w,
            att_xw,
            temp1,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_k,
            att_xk,
            temp1,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_v,
            att_xv,
            temp1,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_a,
            att_xa,
            temp1,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
        channel_mix_state_f16(
            x_ln,
            att_shift_state,
            &layer.att.x_g,
            att_xg,
            temp1,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
    }

    // Update shift state - copy from new_att_shift into the layer's state buffer.
    copy_tensor_f16(new_att_shift, att_shift_state, stream)?;

    // PostAttTokenShift probe (stacked: xr, xw, xk, xv, xa, xg)
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttTokenShift) {
                let xr_data = download_f16_as_f32(att_xr, stream)?;
                let xw_data = download_f16_as_f32(att_xw, stream)?;
                let xk_data = download_f16_as_f32(att_xk, stream)?;
                let xv_data = download_f16_as_f32(att_xv, stream)?;
                let xa_data = download_f16_as_f32(att_xa, stream)?;
                let xg_data = download_f16_as_f32(att_xg, stream)?;
                let tensors: &[&[f32]] = &[&xr_data, &xw_data, &xk_data, &xv_data, &xa_data, &xg_data];
                let n_stack = tensors.len();
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                let stacked = stack_tensors(tensors, n_embd, actual_t, ps.t_stride, b);
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttTokenShift, &stacked, &[n_embd * n_stack, actual_t, b, 1]);
            }
        }
    }

    // Linear projections: r, k, v
    {
        ctx.hgemm_into(&layer.att.w_r, att_xr, att_r)?;
        ctx.hgemm_into(&layer.att.w_k, att_xk, att_k)?;
        ctx.hgemm_into(&layer.att.w_v, att_xv, att_v)?;
    }

    // PostAttLinear probe (stacked: r, k, v)
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttLinear) {
                let r_data = download_f16_as_f32(att_r, stream)?;
                let k_data = download_f16_as_f32(att_k, stream)?;
                let v_data = download_f16_as_f32(att_v, stream)?;
                let tensors: &[&[f32]] = &[&r_data, &k_data, &v_data];
                let n_stack = tensors.len();
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                let stacked = stack_tensors(tensors, n_embd, actual_t, ps.t_stride, b);
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttLinear, &stacked, &[n_embd * n_stack, actual_t, b, 1]);
            }
        }
    }

    // Decay: w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
    {
        ctx.hgemm_into(&layer.att.w1, att_xw, lora_w)?;
        tanh_f16(lora_w, lora_w_tanh, stream)?;
        ctx.hgemm_into(&layer.att.w2, lora_w_tanh, att_w)?;
        broadcast_add_f16(att_w, &layer.att.w0, temp1, stream)?;
        softplus_decay_f16(temp1, att_w, stream)?;
    }

    // PostAttDecay probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttDecay) {
                let data = download_f16_as_f32(att_w, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttDecay, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
    {
        ctx.hgemm_into(&layer.att.a1, att_xa, lora_a)?;
        ctx.hgemm_into(&layer.att.a2, lora_a, lora_a_proj)?;
        broadcast_add_f16(lora_a_proj, &layer.att.a0, temp1, stream)?;
        sigmoid_f16(temp1, att_a, stream)?;
    }

    // PostAttAdapt probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttAdapt) {
                let data = download_f16_as_f32(att_a, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttAdapt, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Gate: g = sigmoid(xg @ g1) @ g2
    {
        ctx.hgemm_into(&layer.att.g1, att_xg, lora_g)?;
        sigmoid_f16(lora_g, lora_g_sig, stream)?;
        ctx.hgemm_into(&layer.att.g2, lora_g_sig, att_g)?;
    }

    // PostAttGate probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttGate) {
                let data = download_f16_as_f32(att_g, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttGate, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Value residual (layers > 0)
    if layer_idx > 0 {
        if let (Some(v0), Some(v1), Some(v2)) =
            (&layer.att.v0, &layer.att.v1, &layer.att.v2)
        {
            {
                ctx.hgemm_into(v1, att_xv, lora_v)?;
                ctx.hgemm_into(v2, lora_v, v_lora2)?;
                broadcast_add_f16(v_lora2, v0, temp1, stream)?;
                sigmoid_f16(temp1, temp2, stream)?;
                lerp_f16(att_v, v_first, temp2, temp1, stream)?;
                copy_tensor_f16(temp1, att_v, stream)?;
            }
        }
        // PostAttValueResidual probe (only layers > 0)
        #[cfg(feature = "hip-probes")]
        {
            if let Some(ref mut ps) = probe {
                if ps.probes.contains_key(&probe::HipHook::PostAttValueResidual) {
                    let data = download_f16_as_f32(att_v, stream)?;
                    let actual_t = ps.ctx.seq_len;
                    let b = ps.ctx.batch_size;
                    fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttValueResidual, &data, &[n_embd, actual_t, b]);
                }
            }
        }
    } else {
        copy_tensor_f16(att_v, v_first, stream)?;
    }

    // L2 normalize k
    {
        broadcast_mul_f16(att_k, &layer.att.k_k, temp1, stream)?;
        l2_norm_f16(temp1, att_kk, head_size, 1e-12, stream)?;
    }

    // PostAttL2Norm probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttL2Norm) {
                let data = download_f16_as_f32(att_kk, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttL2Norm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Control K
    {
        control_k_f16(&layer.att.k_a, att_a, att_k, att_k_ctrl, stream)?;
    }

    // PostAttControlK probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttControlK) {
                let data = download_f16_as_f32(att_k_ctrl, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttControlK, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // WKV inputs
    {
        negate_f16(att_kk, wkv_a, stream)?;
        mul_f16(att_kk, att_a, wkv_b, stream)?;
        // Decay: exp(-exp(w)) where w = log(sigmoid(d)) - 0.5
        // This gives decay = exp(-sigmoid(d) * 0.606531) in range (0.545, 1)
        decay_exp_f16(att_w, w_decay, stream)?;
    }

    // Reshape for WKV
    let att_w_wkv = att_w.reshape_view(wkv_data_shape)?;
    let w_decay_wkv = w_decay.reshape_view(wkv_data_shape)?;
    let r_wkv = att_r.reshape_view(wkv_data_shape)?;
    let k_ctrl_wkv = att_k_ctrl.reshape_view(wkv_data_shape)?;
    let v_wkv = att_v.reshape_view(wkv_data_shape)?;
    let wkv_a_wkv = wkv_a.reshape_view(wkv_data_shape)?;
    let wkv_b_wkv = wkv_b.reshape_view(wkv_data_shape)?;
    let mut wkv_out_wkv = wkv_out.reshape_view_mut(wkv_data_shape)?;

    // PreWkv + PreWkvState probes
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PreWkv) {
                let wd = download_f16_as_f32(att_w, stream)?;
                let r_d = download_f16_as_f32(att_r, stream)?;
                let kc = download_f16_as_f32(att_k_ctrl, stream)?;
                let v_d = download_f16_as_f32(att_v, stream)?;
                let wa = download_f16_as_f32(wkv_a, stream)?;
                let wb = download_f16_as_f32(wkv_b, stream)?;
                let tensors: &[&[f32]] = &[&wd, &r_d, &kc, &v_d, &wa, &wb];
                let n_stack = tensors.len();
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                let stacked = stack_tensors(tensors, n_embd, actual_t, ps.t_stride, b);
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PreWkv, &stacked, &[n_embd * n_stack, actual_t, b, 1]);
            }
            if ps.probes.contains_key(&probe::HipHook::PreWkvState) {
                let data = download_f32(wkv_state, stream)?;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PreWkvState, &data, &[head_size, head_size, n_head, b]);
            }
        }
    }

    // Run WKV kernel via caller-provided closure
    let inputs = WkvCallInputs {
        att_w_wkv: &att_w_wkv,
        w_decay_wkv: &w_decay_wkv,
        r_wkv: &r_wkv,
        k_ctrl_wkv: &k_ctrl_wkv,
        v_wkv: &v_wkv,
        wkv_a_wkv: &wkv_a_wkv,
        wkv_b_wkv: &wkv_b_wkv,
    };
    wkv_fn(inputs, wkv_state, &mut wkv_out_wkv)?;

    // PostWkv + PostWkvState probes
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostWkv) {
                let data = download_f16_as_f32(wkv_out, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostWkv, &data, &[n_embd, actual_t, b]);
            }
            if ps.probes.contains_key(&probe::HipHook::PostWkvState) {
                let data = download_f32(wkv_state, stream)?;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostWkvState, &data, &[head_size, head_size, n_head, b]);
            }
        }
    }

    // Group norm on WKV output
    {
        group_norm_f16(
            wkv_out,
            &layer.att.gn.weight,
            &layer.att.gn.bias,
            wkv_normed,
            n_head,
            64e-5,
            stream,
        )?;
    }

    // PostAttGroupNorm probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttGroupNorm) {
                let data = download_f16_as_f32(wkv_normed, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttGroupNorm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // WKV bonus
    let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
    let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
    let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
    {
        wkv_bonus_f16(
            &r_wkv,
            &k_ctrl_wkv,
            &v_wkv,
            &r_k_wkv,
            &mut wkv_bonus_wkv,
            stream,
        )?;
    }

    // PostWkvBonus probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostWkvBonus) {
                let data = download_f16_as_f32(wkv_bonus, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostWkvBonus, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Combine and gate
    {
        add_f16(wkv_normed, wkv_bonus, temp1, stream)?;
        mul_f16(temp1, att_g, temp2, stream)?;
    }

    // PostAttGated probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttGated) {
                let data = download_f16_as_f32(temp2, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttGated, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Output projection
    {
        ctx.hgemm_into(&layer.att.w_o, temp2, att_out)?;
    }

    // PostAttOut probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAttOut) {
                let data = download_f16_as_f32(att_out, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAttOut, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Residual
    {
        add_f16(x, att_out, temp1, stream)?;
        copy_tensor_f16(temp1, x, stream)?;
    }

    // PostAtt probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostAtt) {
                let data = download_f16_as_f32(x, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostAtt, &data, &[n_embd, actual_t, b]);
            }
        }
    }
    let _ = &probe; // suppress unused warning when hip-probes is off

    Ok(())
}

/// Run one FFN (channel-mix) block for a single layer.
///
/// Performs:
/// 1. FFN layer norm
/// 2. Token shift (masked for correct state extraction)
/// 3. Key projection
/// 4. Squared ReLU activation
/// 5. Value projection
/// 6. Residual addition
#[allow(clippy::too_many_arguments)]
pub fn ffn_block(
    layer: &LayerHip,
    n_embd: usize,
    // Core tensors
    x: &mut TensorHip<f16>,
    x_ln: &mut TensorHip<f16>,
    // Token shift
    ffn_xk: &mut TensorHip<f16>,
    ffn_shift_state: &mut TensorHip<f16>,
    new_ffn_shift: &mut TensorHip<f16>,
    // FFN intermediates
    ffn_k: &mut TensorHip<f16>,
    ffn_k_sq: &mut TensorHip<f16>,
    ffn_out: &mut TensorHip<f16>,
    // Temporaries
    temp1: &mut TensorHip<f16>,
    // Lens GPU tensor for masked shift
    lens_gpu: &TensorHip<i32>,
    // Batch offsets GPU tensor for packed sequence addressing
    batch_offsets_gpu: &TensorHip<i32>,
    // BLAS context
    ctx: &HipBlasContext,
    stream: &Stream,
    // Optional probe state
    probe: &mut Option<ProbeState<'_>>,
) -> Result<()> {
    let ffn_hidden = layer.ffn.w_k.shape().dim(0); // ffn hidden dimension

    // ==== Channel-Mix (FFN) ====
    {
        layer_norm_f16(
            x,
            &layer.ffn_ln.weight,
            &layer.ffn_ln.bias,
            x_ln,
            1e-5,
            stream,
        )?;
    }

    // PostFfnLayerNorm probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfnLayerNorm) {
                let data = download_f16_as_f32(x_ln, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfnLayerNorm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Token shift for FFN - unified kernel with lengths + batch_offsets
    {
        channel_mix_state_f16(
            x_ln,
            ffn_shift_state,
            &layer.ffn.x_k,
            ffn_xk,
            new_ffn_shift,
            lens_gpu,
            batch_offsets_gpu,
            stream,
        )?;
    }

    // Update FFN shift state - copy instead of swap (same aliasing reason as att_shift)
    copy_tensor_f16(new_ffn_shift, ffn_shift_state, stream)?;

    // PostFfnTokenShift probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfnTokenShift) {
                let data = download_f16_as_f32(ffn_xk, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfnTokenShift, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Key projection
    {
        ctx.hgemm_into(&layer.ffn.w_k, ffn_xk, ffn_k)?;
    }

    // PostFfnLinear probe (ffn_k has hidden dim)
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfnLinear) {
                let data = download_f16_as_f32(ffn_k, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfnLinear, &data, &[ffn_hidden, actual_t, b]);
            }
        }
    }

    // Squared ReLU
    {
        squared_relu_f16(ffn_k, ffn_k_sq, stream)?;
    }

    // PostFfnActivate probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfnActivate) {
                let data = download_f16_as_f32(ffn_k_sq, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfnActivate, &data, &[ffn_hidden, actual_t, b]);
            }
        }
    }

    // Value projection
    {
        ctx.hgemm_into(&layer.ffn.w_v, ffn_k_sq, ffn_out)?;
    }

    // PostFfnOut probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfnOut) {
                let data = download_f16_as_f32(ffn_out, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfnOut, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    // Residual
    {
        add_f16(x, ffn_out, temp1, stream)?;
        copy_tensor_f16(temp1, x, stream)?;
    }

    // PostFfn probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostFfn) {
                let data = download_f16_as_f32(x, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostFfn, &data, &[n_embd, actual_t, b]);
            }
        }
    }
    let _ = &probe; // suppress unused warning when hip-probes is off

    Ok(())
}

/// Run the output head: layer norm, projection, and logits staging.
///
/// Performs:
/// 1. Head layer norm
/// 2. Head weight projection (produces f16 logits)
/// 3. Convert f16 logits to f32
/// 4. Async copy to CPU staging buffer
#[allow(clippy::too_many_arguments)]
pub fn output_head(
    head: &HeadHip,
    n_embd: usize,
    n_vocab: usize,
    x: &TensorHip<f16>,
    x_ln: &mut TensorHip<f16>,
    logits: &mut TensorHip<f16>,
    logits_f32: &mut TensorHip<f32>,
    logits_staging: &mut PinnedBuffer<f32>,
    ctx: &HipBlasContext,
    stream: &Stream,
    probe: &mut Option<ProbeState<'_>>,
) -> Result<()> {
    // ==== Output Head ====
    layer_norm_f16(
        x,
        &head.ln.weight,
        &head.ln.bias,
        x_ln,
        1e-5,
        stream,
    )?;

    // PostHeadLayerNorm probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostHeadLayerNorm) {
                let data = download_f16_as_f32(x_ln, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostHeadLayerNorm, &data, &[n_embd, actual_t, b]);
            }
        }
    }

    ctx.hgemm_into(&head.w, x_ln, logits)?;

    // PostHead probe
    #[cfg(feature = "hip-probes")]
    {
        if let Some(ref mut ps) = probe {
            if ps.probes.contains_key(&probe::HipHook::PostHead) {
                let data = download_f16_as_f32(logits, stream)?;
                let actual_t = ps.ctx.seq_len;
                let b = ps.ctx.batch_size;
                fire_probe(ps.probes, ps.ctx, probe::HipHook::PostHead, &data, &[n_vocab, actual_t, b]);
            }
        }
    }
    let _ = &probe; // suppress unused warning when hip-probes is off
    let _ = n_embd; // used only by probes
    let _ = n_vocab; // used only by probes

    // Convert f16 logits to f32 on GPU, then download asynchronously.
    // The staging buffer is pre-allocated for max_prefill_chunk but the
    // logits tensor is sized for the actual T (max_len). Take a sub-slice.
    copy_f16_to_f32(logits, logits_f32, stream)?;
    let logits_len = logits_f32.len();
    logits_f32.copy_to_slice_async(
        &mut logits_staging.as_slice_mut()[..logits_len],
        stream,
    )?;

    Ok(())
}
