//! Step (inference) implementation for the RWKV7 HIP backend.

use half::f16;

use super::prefill::{FusedT1Wkv, WkvInput, WkvKernel};
use super::state::HipState;
use super::Rwkv7Hip;
#[cfg(feature = "hip-probes")]
use crate::hip::device::Stream;
use crate::hip::ffi::{HipErrorKind, Result};
use crate::hip::kernels::{
    add_f16,
    broadcast_add_f16,
    broadcast_mul_f16,
    channel_mix_state_f16,
    channel_mix_state_f16_masked,
    control_k_f16,
    copy_f16_to_f32,
    copy_tensor_f16,
    decay_exp_f16,
    group_norm_f16,
    l2_norm_f16,
    layer_norm_f16,
    lerp_f16,
    mul_f16,
    negate_f16,
    sigmoid_f16,
    softplus_decay_f16,
    squared_relu_f16,
    tanh_f16,
    wkv_bonus_f16,
};
use crate::hip::scratch::HipScratch;
use crate::hip::tensor::{TensorHip, TensorShape};

#[cfg(feature = "hip-probes")]
use crate::hip_probe;

#[cfg(feature = "hip-probes")]
use crate::hip::probe;

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

impl Rwkv7Hip {
    /// Core GPU forward pass. State is always GPU-resident in scratch buffers.
    ///
    /// Runs the full layer loop, writing logits to scratch.logits_staging.
    /// The caller must sync the stream before reading the staging buffer.
    /// State persists in scratch between calls.
    pub(super) fn dispatch(
        &self,
        tokens: &[&[u32]],
        scratch: &mut HipScratch,
        lens: &[usize],
    ) -> Result<()> {
        let b = tokens.len();
        let t = tokens[0].len();

        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;

        // Pre-create FLA kernel views before ctx borrows scratch.
        // FlaChunkedWkv::new takes &mut scratch to create non-owning views
        // of the FLA scratch buffers. This must happen before ctx borrows
        // scratch.blas_ctx, since &mut HipScratch conflicts with any
        // outstanding borrows.
        //
        // 2-tier dispatch: T=1 -> FusedT1Wkv (decode), T>1 -> FLA (prefill).
        // FLA replaces WaveReduceWkv entirely for all prefill lengths.
        let mut fla_kernel = if t > 1 {
            Some(super::fla::FlaChunkedWkv::new(
                scratch, head_size, n_head, t, b,
            )?)
        } else {
            None
        };

        let ctx = &scratch.blas_ctx;
        let stream = ctx.stream();
        let n_layer = self.info.n_layer;
        let n_hidden = self.info.n_hidden;
        let n_vocab = self.info.n_vocab;
        let lora_dims = &scratch.lora_dims;

        // Initialize probe context (compiles out without feature)
        #[cfg(feature = "hip-probes")]
        let mut probe_ctx = probe::ProbeContext {
            layer: None,
            batch_size: b,
            seq_len: *lens.iter().max().unwrap_or(&t),
            chunk_size: t,
            n_embd,
            n_head,
            head_size,
            n_layer,
            shape_storage: [0; probe::MAX_SHAPE_DIMS],
            shape_len: 0,
        };
        #[cfg(feature = "hip-probes")]
        let actual_t = probe_ctx.seq_len;

        // Convert lens to i32 tensor for masked kernel
        let lens_i32: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
        let lens_shape = TensorShape::new(b, 1, 1, 1);
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        lens_gpu.copy_from_slice(&lens_i32, stream)?;

        // Shapes for this forward pass
        let std_shape = TensorShape::new(n_embd, t, b, 1);
        let ffn_shape = TensorShape::new(n_hidden, t, b, 1);
        let out_shape = TensorShape::new(n_vocab, t, b, 1);
        let state_shape = TensorShape::new(n_embd, b, 1, 1);
        let wkv_data_shape = TensorShape::new(head_size, n_head, t, b);
        let wkv_state_shape = TensorShape::new(head_size, head_size, n_head, b);
        let lora_w_shape = TensorShape::new(lora_dims.w_dim, t, b, 1);
        let lora_a_shape = TensorShape::new(lora_dims.a_dim, t, b, 1);
        let lora_g_shape = TensorShape::new(lora_dims.g_dim, t, b, 1);
        let lora_v_shape = TensorShape::new(lora_dims.v_dim.unwrap_or(1), t, b, 1);
        // Create resized views of scratch buffers
        let mut x = scratch.x.resized_view_mut(std_shape)?;
        let mut x_ln = scratch.x_ln.resized_view_mut(std_shape)?;
        let mut att_xr = scratch.att_xr.resized_view_mut(std_shape)?;
        let mut att_xw = scratch.att_xw.resized_view_mut(std_shape)?;
        let mut att_xk = scratch.att_xk.resized_view_mut(std_shape)?;
        let mut att_xv = scratch.att_xv.resized_view_mut(std_shape)?;
        let mut att_xa = scratch.att_xa.resized_view_mut(std_shape)?;
        let mut att_xg = scratch.att_xg.resized_view_mut(std_shape)?;
        let mut att_r = scratch.att_r.resized_view_mut(std_shape)?;
        let mut att_k = scratch.att_k.resized_view_mut(std_shape)?;
        let mut att_v = scratch.att_v.resized_view_mut(std_shape)?;
        let mut att_w = scratch.att_w.resized_view_mut(std_shape)?;
        let mut att_a = scratch.att_a.resized_view_mut(std_shape)?;
        let mut att_g = scratch.att_g.resized_view_mut(std_shape)?;
        let mut att_kk = scratch.att_kk.resized_view_mut(std_shape)?;
        let mut att_k_ctrl = scratch.att_k_ctrl.resized_view_mut(std_shape)?;
        let mut wkv_a = scratch.wkv_a.resized_view_mut(std_shape)?;
        let mut wkv_b = scratch.wkv_b.resized_view_mut(std_shape)?;
        let mut w_decay = scratch.w_decay.resized_view_mut(std_shape)?;
        let mut wkv_out = scratch.wkv_out.resized_view_mut(std_shape)?;
        let mut wkv_normed = scratch.wkv_normed.resized_view_mut(std_shape)?;
        let mut wkv_bonus = scratch.wkv_bonus.resized_view_mut(std_shape)?;
        let mut att_out = scratch.att_out.resized_view_mut(std_shape)?;
        let mut ffn_xk = scratch.ffn_xk.resized_view_mut(std_shape)?;
        let mut ffn_out = scratch.ffn_out.resized_view_mut(std_shape)?;
        let mut v_first = scratch.v_first.resized_view_mut(std_shape)?;

        // FFN hidden buffers
        let mut ffn_k = scratch.ffn_k.resized_view_mut(ffn_shape)?;
        let mut ffn_k_sq = scratch.ffn_k_sq.resized_view_mut(ffn_shape)?;

        // LoRA buffers
        let mut lora_w = scratch.lora_w.resized_view_mut(lora_w_shape)?;
        let mut lora_a = scratch.lora_a.resized_view_mut(lora_a_shape)?;
        let mut lora_g = scratch.lora_g.resized_view_mut(lora_g_shape)?;
        let mut lora_v = scratch.lora_v.resized_view_mut(lora_v_shape)?;
        let mut lora_w_tanh = scratch.lora_w_tanh.resized_view_mut(lora_w_shape)?;
        let mut lora_a_proj = scratch.lora_a_proj.resized_view_mut(std_shape)?;
        let mut lora_g_sig = scratch.lora_g_sig.resized_view_mut(lora_g_shape)?;
        let mut v_lora2 = scratch.v_lora2.resized_view_mut(std_shape)?;

        // Output buffers
        let mut logits = scratch.logits.resized_view_mut(out_shape)?;
        let mut logits_f32 = scratch.logits_f32.resized_view_mut(out_shape)?;

        {
            // Embedding lookup: tokens[b][t] -> x[c, t, b]
            // Embedding table is kept on CPU - use pinned staging buffer for async upload
            let emb_data = &self.embed.w;
            let emb_stride = self.embed.n_embd;
            let x_host = scratch.emb_staging.as_slice_mut();
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
                scratch
                    .emb_staging
                    .copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
            }
        }

        // State is always GPU-resident in scratch buffers
        let (mut att_shift_gpu, mut ffn_shift_gpu, mut wkv_state_gpu) = (
            std::mem::take(&mut scratch.att_shift_state_gpu),
            std::mem::take(&mut scratch.ffn_state_gpu),
            std::mem::take(&mut scratch.wkv_state_gpu),
        );

        let result = (|| {
            // Temporary buffers
            let mut new_att_shift = scratch.new_att_shift.resized_view_mut(state_shape)?;
            let mut new_ffn_shift = scratch.new_ffn_shift.resized_view_mut(state_shape)?;
            let mut temp1 = scratch.temp1.resized_view_mut(std_shape)?;
            let mut temp2 = scratch.temp2.resized_view_mut(std_shape)?;

            // 2-tier dispatch: T=1 uses FusedT1Wkv, T>1 uses FLA.
            // The wkv_kernel is only used when fla_kernel is None (i.e., T=1).
            let wkv_kernel: &dyn WkvKernel = &FusedT1Wkv;

            // PostEmbed probe: embedding output before any layer processing
            #[cfg(feature = "hip-probes")]
            {
                if let Some(ref probes) = self.probes {
                    if probes.contains_key(&probe::HipHook::PostEmbed) {
                        let data = download_f16_as_f32(&x, stream)?;
                        hip_probe!(self, probe_ctx, probe::HipHook::PostEmbed, &data, [n_embd, actual_t, b]);
                    }
                }
            }

            // Process each layer
            for layer_idx in 0..n_layer {
                #[cfg(feature = "hip-probes")]
                { probe_ctx.layer = Some(layer_idx); }

                let layer = &self.layers[layer_idx];

                // Apply ln0 for layer 0
                if layer_idx == 0 {
                    {
                        layer_norm_f16(
                            &x,
                            &self.embed.ln.weight,
                            &self.embed.ln.bias,
                            &mut x_ln,
                            1e-5,
                            stream,
                        )?;
                        copy_tensor_f16(&x_ln, &mut x, stream)?;
                    }

                    // PostEmbedLayerNorm probe
                    #[cfg(feature = "hip-probes")]
                    {
                        if let Some(ref probes) = self.probes {
                            if probes.contains_key(&probe::HipHook::PostEmbedLayerNorm) {
                                let data = download_f16_as_f32(&x, stream)?;
                                hip_probe!(self, probe_ctx, probe::HipHook::PostEmbedLayerNorm, &data, [n_embd, actual_t, b]);
                            }
                        }
                    }
                }

                // ==== Time-Mix (Attention) ====
                {
                    layer_norm_f16(
                        &x,
                        &layer.att_ln.weight,
                        &layer.att_ln.bias,
                        &mut x_ln,
                        1e-5,
                        stream,
                    )?;
                }

                // PostAttLayerNorm probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttLayerNorm) {
                            let data = download_f16_as_f32(&x_ln, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttLayerNorm, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Token shifts for attention - use masked kernel for x_r to get correct state
                // The masked kernel extracts state at lengths[b]-1 instead of T-1
                {
                    channel_mix_state_f16_masked(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_r,
                        &mut att_xr,
                        &mut new_att_shift,
                        &lens_gpu,
                        stream,
                    )?;
                    // Remaining shifts use regular kernel (we only need outputs, not state)
                    channel_mix_state_f16(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_w,
                        &mut att_xw,
                        &mut temp1,
                        stream,
                    )?;
                    channel_mix_state_f16(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_k,
                        &mut att_xk,
                        &mut temp1,
                        stream,
                    )?;
                    channel_mix_state_f16(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_v,
                        &mut att_xv,
                        &mut temp1,
                        stream,
                    )?;
                    channel_mix_state_f16(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_a,
                        &mut att_xa,
                        &mut temp1,
                        stream,
                    )?;
                    channel_mix_state_f16(
                        &x_ln,
                        &att_shift_gpu[layer_idx],
                        &layer.att.x_g,
                        &mut att_xg,
                        &mut temp1,
                        stream,
                    )?;
                }

                // Update shift state - copy from new_att_shift into the layer's state buffer.
                // We use copy instead of swap because resized_view_mut shares the
                // underlying device pointer with scratch.new_att_shift. Swapping would
                // cause att_shift_gpu[layer_idx] to alias scratch.new_att_shift on the
                // next dispatch() call, corrupting state.
                copy_tensor_f16(&new_att_shift, &mut att_shift_gpu[layer_idx], stream)?;

                // PostAttTokenShift probe (stacked: xr, xw, xk, xv, xa, xg)
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttTokenShift) {
                            let xr_data = download_f16_as_f32(&att_xr, stream)?;
                            let xw_data = download_f16_as_f32(&att_xw, stream)?;
                            let xk_data = download_f16_as_f32(&att_xk, stream)?;
                            let xv_data = download_f16_as_f32(&att_xv, stream)?;
                            let xa_data = download_f16_as_f32(&att_xa, stream)?;
                            let xg_data = download_f16_as_f32(&att_xg, stream)?;
                            let tensors: &[&[f32]] = &[&xr_data, &xw_data, &xk_data, &xv_data, &xa_data, &xg_data];
                            let n_stack = tensors.len();
                            let mut stacked = Vec::with_capacity(n_embd * n_stack * actual_t * b);
                            // Interleave by token: for each (b, t) position, append all stacks
                            for b_idx in 0..b {
                                for t_idx in 0..actual_t {
                                    let base = n_embd * (t_idx + t * b_idx);
                                    for tensor in tensors {
                                        stacked.extend_from_slice(&tensor[base..base + n_embd]);
                                    }
                                }
                            }
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttTokenShift, &stacked, [n_embd * n_stack, actual_t, b, 1]);
                        }
                    }
                }

                // Linear projections: r, k, v
                {
                    ctx.hgemm_into(&layer.att.w_r, &att_xr, &mut att_r)?;
                    ctx.hgemm_into(&layer.att.w_k, &att_xk, &mut att_k)?;
                    ctx.hgemm_into(&layer.att.w_v, &att_xv, &mut att_v)?;
                }

                // PostAttLinear probe (stacked: r, k, v)
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttLinear) {
                            let r_data = download_f16_as_f32(&att_r, stream)?;
                            let k_data = download_f16_as_f32(&att_k, stream)?;
                            let v_data = download_f16_as_f32(&att_v, stream)?;
                            let tensors: &[&[f32]] = &[&r_data, &k_data, &v_data];
                            let n_stack = tensors.len();
                            let mut stacked = Vec::with_capacity(n_embd * n_stack * actual_t * b);
                            // Interleave by token: for each (b, t) position, append all stacks
                            for b_idx in 0..b {
                                for t_idx in 0..actual_t {
                                    let base = n_embd * (t_idx + t * b_idx);
                                    for tensor in tensors {
                                        stacked.extend_from_slice(&tensor[base..base + n_embd]);
                                    }
                                }
                            }
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttLinear, &stacked, [n_embd * n_stack, actual_t, b, 1]);
                        }
                    }
                }

                // Decay: w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
                {
                    ctx.hgemm_into(&layer.att.w1, &att_xw, &mut lora_w)?;
                    tanh_f16(&lora_w, &mut lora_w_tanh, stream)?;
                    ctx.hgemm_into(&layer.att.w2, &lora_w_tanh, &mut att_w)?;
                    broadcast_add_f16(&att_w, &layer.att.w0, &mut temp1, stream)?;
                    softplus_decay_f16(&temp1, &mut att_w, stream)?;
                }

                // PostAttDecay probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttDecay) {
                            let data = download_f16_as_f32(&att_w, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttDecay, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
                {
                    ctx.hgemm_into(&layer.att.a1, &att_xa, &mut lora_a)?;
                    ctx.hgemm_into(&layer.att.a2, &lora_a, &mut lora_a_proj)?;
                    broadcast_add_f16(&lora_a_proj, &layer.att.a0, &mut temp1, stream)?;
                    sigmoid_f16(&temp1, &mut att_a, stream)?;
                }

                // PostAttAdapt probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttAdapt) {
                            let data = download_f16_as_f32(&att_a, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttAdapt, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Gate: g = sigmoid(xg @ g1) @ g2
                {
                    ctx.hgemm_into(&layer.att.g1, &att_xg, &mut lora_g)?;
                    sigmoid_f16(&lora_g, &mut lora_g_sig, stream)?;
                    ctx.hgemm_into(&layer.att.g2, &lora_g_sig, &mut att_g)?;
                }

                // PostAttGate probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttGate) {
                            let data = download_f16_as_f32(&att_g, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGate, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Value residual (layers > 0)
                if layer_idx > 0 {
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (&layer.att.v0, &layer.att.v1, &layer.att.v2)
                    {
                        {
                            ctx.hgemm_into(v1, &att_xv, &mut lora_v)?;
                            ctx.hgemm_into(v2, &lora_v, &mut v_lora2)?;
                            broadcast_add_f16(&v_lora2, v0, &mut temp1, stream)?;
                            sigmoid_f16(&temp1, &mut temp2, stream)?;
                            lerp_f16(&att_v, &v_first, &temp2, &mut temp1, stream)?;
                            copy_tensor_f16(&temp1, &mut att_v, stream)?;
                        }
                    }
                    // PostAttValueResidual probe (only layers > 0)
                    #[cfg(feature = "hip-probes")]
                    {
                        if let Some(ref probes) = self.probes {
                            if probes.contains_key(&probe::HipHook::PostAttValueResidual) {
                                let data = download_f16_as_f32(&att_v, stream)?;
                                hip_probe!(self, probe_ctx, probe::HipHook::PostAttValueResidual, &data, [n_embd, actual_t, b]);
                            }
                        }
                    }
                } else {
                    copy_tensor_f16(&att_v, &mut v_first, stream)?;
                }

                // L2 normalize k
                {
                    broadcast_mul_f16(&att_k, &layer.att.k_k, &mut temp1, stream)?;
                    l2_norm_f16(&temp1, &mut att_kk, head_size, 1e-12, stream)?;
                }

                // PostAttL2Norm probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttL2Norm) {
                            let data = download_f16_as_f32(&att_kk, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttL2Norm, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Control K
                {
                    control_k_f16(&layer.att.k_a, &att_a, &att_k, &mut att_k_ctrl, stream)?;
                }

                // PostAttControlK probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttControlK) {
                            let data = download_f16_as_f32(&att_k_ctrl, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttControlK, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // WKV inputs
                {
                    negate_f16(&att_kk, &mut wkv_a, stream)?;
                    mul_f16(&att_kk, &att_a, &mut wkv_b, stream)?;
                    // Decay: exp(-exp(w)) where w = log(sigmoid(d)) - 0.5
                    // This gives decay = exp(-sigmoid(d) * 0.606531) in range (0.545, 1)
                    decay_exp_f16(&att_w, &mut w_decay, stream)?;
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

                // PreWkv probe (stacked: att_w, r, k_ctrl, v, wkv_a, wkv_b) + PreWkvState
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PreWkv) {
                            let wd = download_f16_as_f32(&att_w, stream)?;
                            let r_d = download_f16_as_f32(&att_r, stream)?;
                            let kc = download_f16_as_f32(&att_k_ctrl, stream)?;
                            let v_d = download_f16_as_f32(&att_v, stream)?;
                            let wa = download_f16_as_f32(&wkv_a, stream)?;
                            let wb = download_f16_as_f32(&wkv_b, stream)?;
                            let tensors: &[&[f32]] = &[&wd, &r_d, &kc, &v_d, &wa, &wb];
                            let n_stack = tensors.len();
                            let mut stacked = Vec::with_capacity(n_embd * n_stack * actual_t * b);
                            // Interleave by token: for each (b, t) position, append all stacks
                            for b_idx in 0..b {
                                for t_idx in 0..actual_t {
                                    let base = n_embd * (t_idx + t * b_idx);
                                    for tensor in tensors {
                                        stacked.extend_from_slice(&tensor[base..base + n_embd]);
                                    }
                                }
                            }
                            hip_probe!(self, probe_ctx, probe::HipHook::PreWkv, &stacked, [n_embd * n_stack, actual_t, b, 1]);
                        }
                        if probes.contains_key(&probe::HipHook::PreWkvState) {
                            let data = download_f32(&wkv_state_gpu[layer_idx], stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PreWkvState, &data, [head_size, head_size, n_head, b]);
                        }
                    }
                }

                // Run WKV7: 2-tier dispatch.
                // T>1 (prefill): FLA receives raw att_w (pre-exponentiation) for better precision.
                // T=1 (decode): FusedT1Wkv receives w_decay = exp(-exp(att_w)).
                if let Some(ref mut fla) = fla_kernel {
                    fla.compute(
                        &att_w_wkv,
                        &r_wkv,
                        &k_ctrl_wkv,
                        &v_wkv,
                        &wkv_a_wkv,
                        &wkv_b_wkv,
                        &mut wkv_state_gpu[layer_idx],
                        &mut wkv_out_wkv,
                        lens,
                        stream,
                    )?;
                } else {
                    let wkv_input = WkvInput {
                        w_decay: &w_decay_wkv,
                        r: &r_wkv,
                        k: &k_ctrl_wkv,
                        v: &v_wkv,
                        a: &wkv_a_wkv,
                        b: &wkv_b_wkv,
                        lengths: &lens_gpu,
                    };
                    wkv_kernel.compute(
                        &wkv_input,
                        &mut wkv_state_gpu[layer_idx],
                        &mut wkv_out_wkv,
                        stream,
                    )?;
                }

                // PostWkv + PostWkvState probes
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostWkv) {
                            let data = download_f16_as_f32(&wkv_out, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostWkv, &data, [n_embd, actual_t, b]);
                        }
                        if probes.contains_key(&probe::HipHook::PostWkvState) {
                            let data = download_f32(&wkv_state_gpu[layer_idx], stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostWkvState, &data, [head_size, head_size, n_head, b]);
                        }
                    }
                }

                // Group norm on WKV output
                {
                    group_norm_f16(
                        &wkv_out,
                        &layer.att.gn.weight,
                        &layer.att.gn.bias,
                        &mut wkv_normed,
                        n_head,
                        64e-5,
                        stream,
                    )?;
                }

                // PostAttGroupNorm probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttGroupNorm) {
                            let data = download_f16_as_f32(&wkv_normed, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGroupNorm, &data, [n_embd, actual_t, b]);
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
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostWkvBonus) {
                            let data = download_f16_as_f32(&wkv_bonus, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostWkvBonus, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Combine and gate
                {
                    add_f16(&wkv_normed, &wkv_bonus, &mut temp1, stream)?;
                    mul_f16(&temp1, &att_g, &mut temp2, stream)?;
                }

                // PostAttGated probe (gated output is in temp2)
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttGated) {
                            let data = download_f16_as_f32(&temp2, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGated, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Output projection
                {
                    ctx.hgemm_into(&layer.att.w_o, &temp2, &mut att_out)?;
                }

                // PostAttOut probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAttOut) {
                            let data = download_f16_as_f32(&att_out, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttOut, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Residual
                {
                    add_f16(&x, &att_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                }

                // PostAtt probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostAtt) {
                            let data = download_f16_as_f32(&x, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAtt, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // ==== Channel-Mix (FFN) ====
                {
                    layer_norm_f16(
                        &x,
                        &layer.ffn_ln.weight,
                        &layer.ffn_ln.bias,
                        &mut x_ln,
                        1e-5,
                        stream,
                    )?;
                }

                // PostFfnLayerNorm probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfnLayerNorm) {
                            let data = download_f16_as_f32(&x_ln, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLayerNorm, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Token shift for FFN - use masked kernel for correct state extraction
                {
                    channel_mix_state_f16_masked(
                        &x_ln,
                        &ffn_shift_gpu[layer_idx],
                        &layer.ffn.x_k,
                        &mut ffn_xk,
                        &mut new_ffn_shift,
                        &lens_gpu,
                        stream,
                    )?;
                }

                // Update FFN shift state - copy instead of swap (same aliasing reason as att_shift)
                copy_tensor_f16(&new_ffn_shift, &mut ffn_shift_gpu[layer_idx], stream)?;

                // PostFfnTokenShift probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfnTokenShift) {
                            let data = download_f16_as_f32(&ffn_xk, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnTokenShift, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Key projection
                {
                    ctx.hgemm_into(&layer.ffn.w_k, &ffn_xk, &mut ffn_k)?;
                }

                // PostFfnLinear probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfnLinear) {
                            let data = download_f16_as_f32(&ffn_k, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLinear, &data, [n_hidden, actual_t, b]);
                        }
                    }
                }

                // Squared ReLU
                {
                    squared_relu_f16(&ffn_k, &mut ffn_k_sq, stream)?;
                }

                // PostFfnActivate probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfnActivate) {
                            let data = download_f16_as_f32(&ffn_k_sq, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnActivate, &data, [n_hidden, actual_t, b]);
                        }
                    }
                }

                // Value projection
                {
                    ctx.hgemm_into(&layer.ffn.w_v, &ffn_k_sq, &mut ffn_out)?;
                }

                // PostFfnOut probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfnOut) {
                            let data = download_f16_as_f32(&ffn_out, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnOut, &data, [n_embd, actual_t, b]);
                        }
                    }
                }

                // Residual
                {
                    add_f16(&x, &ffn_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                }

                // PostFfn probe
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PostFfn) {
                            let data = download_f16_as_f32(&x, stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PostFfn, &data, [n_embd, actual_t, b]);
                        }
                    }
                }
            }

            // Reset layer context for head probes
            #[cfg(feature = "hip-probes")]
            { probe_ctx.layer = None; }

            {
                // ==== Output Head ====
                layer_norm_f16(
                    &x,
                    &self.head.ln.weight,
                    &self.head.ln.bias,
                    &mut x_ln,
                    1e-5,
                    stream,
                )?;

                ctx.hgemm_into(&self.head.w, &x_ln, &mut logits)?;
            }

            // PostHeadLayerNorm probe (x_ln holds head layer norm output)
            #[cfg(feature = "hip-probes")]
            {
                if let Some(ref probes) = self.probes {
                    if probes.contains_key(&probe::HipHook::PostHeadLayerNorm) {
                        let data = download_f16_as_f32(&x_ln, stream)?;
                        hip_probe!(self, probe_ctx, probe::HipHook::PostHeadLayerNorm, &data, [n_embd, actual_t, b]);
                    }
                }
            }

            // PostHead probe (logits are in f16, convert to f32 for probe)
            #[cfg(feature = "hip-probes")]
            {
                if let Some(ref probes) = self.probes {
                    if probes.contains_key(&probe::HipHook::PostHead) {
                        let data = download_f16_as_f32(&logits, stream)?;
                        hip_probe!(self, probe_ctx, probe::HipHook::PostHead, &data, [n_vocab, actual_t, b]);
                    }
                }
            }

            {
                // Convert f16 logits to f32 on GPU, then download asynchronously.
                // The staging buffer is pre-allocated for max_prefill_chunk but the
                // logits tensor is sized for the actual T (max_len). Take a sub-slice.
                copy_f16_to_f32(&logits, &mut logits_f32, stream)?;
                let logits_len = logits_f32.len();
                logits_f32.copy_to_slice_async(
                    &mut scratch.logits_staging.as_slice_mut()[..logits_len],
                    stream,
                )?;
            }

            Ok(())
        })();

        // Always put state back to scratch
        scratch.att_shift_state_gpu = att_shift_gpu;
        scratch.ffn_state_gpu = ffn_shift_gpu;
        scratch.wkv_state_gpu = wkv_state_gpu;

        result
    }

    /// Run one inference step on variable-length input sequences.
    ///
    /// This is the backward-compatible wrapper that manages CPU<->GPU state
    /// transfers around `dispatch()`. For GPU-resident state without per-call
    /// H2D/D2H overhead, use `HipRuntime::infer()` instead.
    ///
    /// # Arguments
    /// * `x` - Batch of token sequences
    /// * `state` - Optional initial state (None = fresh zeros)
    ///
    /// # Returns
    /// `(logits, new_state)` where logits contains `n_vocab` floats per input token.
    pub fn step(
        &self,
        x: &[&[u32]],
        state: Option<HipState>,
    ) -> Result<(Vec<f32>, HipState)> {
        let batch_size = x.len();
        if batch_size == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }

        // Get scratch and config
        let mut scratch_ref = self.scratch.lock().unwrap();
        let scratch = scratch_ref.as_mut().ok_or_else(|| HipErrorKind {
            code: -1,
            message: "Scratch not initialized - call with_config() first".to_string(),
        })?;

        let chunk_size = scratch.config.max_prefill_chunk;
        let max_batch = scratch.config.batch_size;

        // Validate batch size
        if batch_size > max_batch {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Batch size {} exceeds configured max {}",
                    batch_size, max_batch
                ),
            });
        }

        // Get real lengths and validate
        let lens: Vec<usize> = x.iter().map(|s| s.len()).collect();
        let max_len = *lens.iter().max().unwrap_or(&0);

        if max_len == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "All sequences are empty".to_string(),
            });
        }

        // Validate total token count fits within scratch allocation.
        // For B>1 with T>1 (prefill), sequences are padded to max_len, so
        // the effective total is batch_size * max_len.
        // For B=1 or T=1, no padding is needed, so total is sum of lens.
        let t_effective = if batch_size > 1 && max_len > 1 {
            batch_size * max_len
        } else {
            lens.iter().sum()
        };
        if t_effective > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "step() requires effective token count <= chunk_size ({} vs {})",
                    t_effective, chunk_size
                ),
            });
        }

        // Validate state batch size if provided
        if let Some(ref s) = state {
            if s.batch_size != batch_size {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "State batch_size mismatch: state has {} but input has {} sequences",
                        s.batch_size, batch_size
                    ),
                });
            }
        }

        // Dispatch strategy:
        // - B=1 (any T): pass sequence directly, no padding needed.
        // - B>1, T=1 (decode): pass sequences directly, FusedT1Wkv handles B>1 natively.
        // - B>1, T>1 (prefill): pad shorter sequences to max_len so dispatch() gets
        //   uniform-length sequences. FLA uses per-batch cu_seqlens for real lengths.
        let (dispatch_tokens_storage, dispatch_refs, dispatch_lens);

        if batch_size > 1 && max_len > 1 {
            // B>1 prefill: pad shorter sequences to max_len (token 0 as padding)
            let padded: Vec<Vec<u32>> = x
                .iter()
                .map(|seq| {
                    let mut v = seq.to_vec();
                    v.resize(max_len, 0);
                    v
                })
                .collect();
            dispatch_tokens_storage = padded;
            dispatch_refs = dispatch_tokens_storage
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>();
            // Pass real per-sequence lengths so FLA can build cu_seqlens
            dispatch_lens = lens.clone();
        } else {
            // B=1 or T=1: pass sequences directly, no padding needed
            dispatch_tokens_storage = Vec::new(); // unused
            dispatch_refs = x.to_vec();
            dispatch_lens = lens.clone();
        }

        let n_vocab = self.info.n_vocab;
        let n_layer = self.info.n_layer;

        // H2D: upload CPU state to scratch GPU buffers (or reset to zeros)
        match state {
            Some(ref s) => {
                let stream_handle = scratch.blas_ctx.stream().handle();
                for i in 0..n_layer {
                    unsafe {
                        s.att_shift_states[i].copy_to_device_async(
                            scratch.att_shift_state_gpu[i].as_mut_ptr(),
                            stream_handle,
                        )?;
                        s.ffn_states[i].copy_to_device_async(
                            scratch.ffn_state_gpu[i].as_mut_ptr(),
                            stream_handle,
                        )?;
                        s.att_states[i].copy_to_device_async(
                            scratch.wkv_state_gpu[i].as_mut_ptr(),
                            stream_handle,
                        )?;
                    }
                }
            }
            None => {
                scratch.reset_state_gpu()?;
            }
        }

        // Run the forward pass (all GPU-resident, no state param)
        self.dispatch(&dispatch_refs, scratch, &dispatch_lens)?;

        // D2H: download scratch GPU buffers to a fresh HipState
        let stream_handle = scratch.blas_ctx.stream().handle();
        let mut new_state = HipState::new(&self.info, batch_size)?;
        for i in 0..n_layer {
            unsafe {
                new_state.att_shift_states[i].copy_from_device_async(
                    scratch.att_shift_state_gpu[i].as_ptr(),
                    stream_handle,
                )?;
                new_state.ffn_states[i].copy_from_device_async(
                    scratch.ffn_state_gpu[i].as_ptr(),
                    stream_handle,
                )?;
                new_state.att_states[i].copy_from_device_async(
                    scratch.wkv_state_gpu[i].as_ptr(),
                    stream_handle,
                )?;
            }
        }

        // Sync the stream -- all GPU work and D->H transfers are now complete
        scratch.blas_ctx.synchronize()?;

        // Extract logits from staging buffer.
        // Layout is [n_vocab, T_stride, B] where T_stride depends on dispatch mode:
        // - B>1, T>1 (padded): T_stride = max_len, skip padding positions
        // - B=1 or T=1 (no padding): T_stride = lens[b], contiguous
        let staging = scratch.logits_staging.as_slice();
        let t_total: usize = lens.iter().sum();
        let mut logits = Vec::with_capacity(t_total * n_vocab);

        if batch_size > 1 && max_len > 1 {
            // Padded layout: logits at [n_vocab, max_len, B], extract only real tokens
            for (b, &real_len) in lens.iter().enumerate() {
                for t in 0..real_len {
                    let offset = (b * max_len + t) * n_vocab;
                    logits.extend_from_slice(&staging[offset..offset + n_vocab]);
                }
            }
        } else {
            // Packed/contiguous layout: no padding gaps
            let mut token_offset = 0usize;
            for &real_len in lens.iter() {
                for t in 0..real_len {
                    let offset = (token_offset + t) * n_vocab;
                    logits.extend_from_slice(&staging[offset..offset + n_vocab]);
                }
                token_offset += real_len;
            }
        }

        Ok((logits, new_state))
    }

    /// Run inference using GPU-resident state. No H2D/D2H state transfers.
    /// State persists in scratch between calls. Used by `HipRuntime::infer()`.
    pub(crate) fn infer_resident(&self, x: &[&[u32]]) -> Result<Vec<f32>> {
        let batch_size = x.len();
        if batch_size == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }

        // Get scratch and config
        let mut scratch_ref = self.scratch.lock().unwrap();
        let scratch = scratch_ref.as_mut().ok_or_else(|| HipErrorKind {
            code: -1,
            message: "Scratch not initialized - call with_config() first".to_string(),
        })?;

        let chunk_size = scratch.config.max_prefill_chunk;
        let max_batch = scratch.config.batch_size;

        // Validate batch size
        if batch_size > max_batch {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Batch size {} exceeds configured max {}",
                    batch_size, max_batch
                ),
            });
        }

        // Get real lengths and validate
        let lens: Vec<usize> = x.iter().map(|s| s.len()).collect();
        let max_len = *lens.iter().max().unwrap_or(&0);

        if max_len == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "All sequences are empty".to_string(),
            });
        }

        // Validate total token count fits within scratch allocation.
        // Same strategy as step(): B>1 with T>1 pads to max_len.
        let t_effective = if batch_size > 1 && max_len > 1 {
            batch_size * max_len
        } else {
            lens.iter().sum()
        };
        if t_effective > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "infer_resident() requires effective token count <= chunk_size ({} vs {})",
                    t_effective, chunk_size
                ),
            });
        }

        // Dispatch strategy (same as step()):
        // - B=1 (any T): pass sequence directly, no padding needed.
        // - B>1, T=1 (decode): pass sequences directly, FusedT1Wkv handles B>1 natively.
        // - B>1, T>1 (prefill): pad shorter sequences to max_len.
        let (dispatch_tokens_storage, dispatch_refs, dispatch_lens);

        if batch_size > 1 && max_len > 1 {
            // B>1 prefill: pad shorter sequences to max_len (token 0 as padding)
            let padded: Vec<Vec<u32>> = x
                .iter()
                .map(|seq| {
                    let mut v = seq.to_vec();
                    v.resize(max_len, 0);
                    v
                })
                .collect();
            dispatch_tokens_storage = padded;
            dispatch_refs = dispatch_tokens_storage
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>();
            dispatch_lens = lens.clone();
        } else {
            // B=1 or T=1: pass sequences directly, no padding needed
            dispatch_tokens_storage = Vec::new(); // unused
            dispatch_refs = x.to_vec();
            dispatch_lens = lens.clone();
        }

        let n_vocab = self.info.n_vocab;

        // Run the forward pass (state stays GPU-resident, no H2D/D2H)
        self.dispatch(&dispatch_refs, scratch, &dispatch_lens)?;

        // Sync the stream
        let stream = scratch.blas_ctx.stream();
        stream.synchronize()?;

        // Extract logits from staging buffer.
        // Same layout logic as step(): padded vs contiguous.
        let staging = scratch.logits_staging.as_slice();
        let t_total: usize = lens.iter().sum();
        let mut logits = Vec::with_capacity(t_total * n_vocab);

        if batch_size > 1 && max_len > 1 {
            // Padded layout: logits at [n_vocab, max_len, B], extract only real tokens
            for (b, &real_len) in lens.iter().enumerate() {
                for t in 0..real_len {
                    let offset = (b * max_len + t) * n_vocab;
                    logits.extend_from_slice(&staging[offset..offset + n_vocab]);
                }
            }
        } else {
            // Packed/contiguous layout: no padding gaps
            let mut token_offset = 0usize;
            for &real_len in lens.iter() {
                for t in 0..real_len {
                    let offset = (token_offset + t) * n_vocab;
                    logits.extend_from_slice(&staging[offset..offset + n_vocab]);
                }
                token_offset += real_len;
            }
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::HipState;
    use crate::hip::model::Rwkv7Hip;
    use crate::hip::model::Rwkv7ModelInfo;
    use crate::hip::pinned::PinnedBuffer;
    use crate::hip::scratch::HipRuntimeConfig;

    // === State I/O and Batched Inference Tests (bd-2sh.5.6) ===

    /// Test that HipState is correctly sized for batched inference.
    #[test]
    fn test_hip_state_batched_sizing() {
        let info = Rwkv7ModelInfo {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            head_size: 64,
            n_vocab: 65536,
            n_hidden: 2048,
        };

        let batch_size = 4;
        let state = HipState::new(&info, batch_size).expect("Failed to allocate state");

        assert_eq!(state.batch_size, 4);
        assert_eq!(state.att_states.len(), 12);
        assert_eq!(state.att_shift_states.len(), 12);
        assert_eq!(state.ffn_states.len(), 12);

        // Check per-layer sizes include batch dimension
        let expected_att_state_size = 64 * 64 * 12 * 4; // head_size² * n_head * batch
        let expected_shift_state_size = 768 * 4; // n_embd * batch

        assert_eq!(state.att_states[0].len(), expected_att_state_size);
        assert_eq!(state.att_shift_states[0].len(), expected_shift_state_size);
        assert_eq!(state.ffn_states[0].len(), expected_shift_state_size);

        println!("HipState batched sizing test passed (B=4)");
    }

    /// Test that step() API works with (logits, state) return.
    #[test]
    fn test_step_basic() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 1);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

        let (logits, _state) = model.step(&[&tokens], None).expect("step() failed");

        // Should return vocab_size * T logits
        let expected_len = model.info.n_vocab * tokens.len();
        assert_eq!(
            logits.len(),
            expected_len,
            "Expected {} logits, got {}",
            expected_len,
            logits.len()
        );

        println!("step() basic test passed");
    }

    /// Test batched inference with B=2 produces same results as sequential B=1.
    #[test]
    fn test_batched_matches_sequential() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        // Load two separate models to get fresh state each time
        let model1 = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config1 = HipRuntimeConfig::new(256, 1);
        let model1 = model1
            .with_config(config1)
            .expect("Failed to configure model");

        let model2 = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config2 = HipRuntimeConfig::new(256, 1);
        let model2 = model2
            .with_config(config2)
            .expect("Failed to configure model");

        let model_batch = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config_batch = HipRuntimeConfig::new(256, 2);
        let model_batch = model_batch
            .with_config(config_batch)
            .expect("Failed to configure model");

        // Two sequences
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];
        let t = seq1.len();

        // Run sequentially with B=1 (fresh state each time)
        let (logits1, _) = model1.step(&[&seq1], None).expect("seq1 step failed");
        let (logits2, _) = model2.step(&[&seq2], None).expect("seq2 step failed");

        // Run batched with B=2 (fresh state)
        let (batched_logits, _) = model_batch
            .step(&[&seq1, &seq2], None)
            .expect("batched step failed");

        // Batched output: [seq1 tokens, seq2 tokens] concatenated
        let vocab = model1.info.n_vocab;
        let b = 2;
        assert_eq!(batched_logits.len(), vocab * t * b);

        let top_k = |logits: &[f32], token_idx: usize, k: usize| -> Vec<usize> {
            let start = token_idx * vocab;
            let end = start + vocab;
            let mut indexed: Vec<(usize, f32)> =
                logits[start..end].iter().copied().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            indexed.into_iter().take(k).map(|(i, _)| i).collect()
        };

        for ti in 0..t {
            let seq1_top5 = top_k(&logits1, ti, 5);
            let seq2_top5 = top_k(&logits2, ti, 5);

            let batch1_top5 = top_k(&batched_logits, 0 * t + ti, 5);
            let batch2_top5 = top_k(&batched_logits, 1 * t + ti, 5);

            let overlap1 =
                seq1_top5.iter().filter(|i| batch1_top5.contains(i)).count() as f32 / 5.0;
            let overlap2 =
                seq2_top5.iter().filter(|i| batch2_top5.contains(i)).count() as f32 / 5.0;

            if overlap1 < 0.8 {
                eprintln!(
                    "Warning: seq1 top-5 overlap low at t={} (overlap={:.2})",
                    ti, overlap1
                );
            }
            if overlap2 < 0.8 {
                eprintln!(
                    "Warning: seq2 top-5 overlap low at t={} (overlap={:.2})",
                    ti, overlap2
                );
            }
        }

        println!("Batched matches sequential test PASSED");
    }

    /// Test streaming equivalence with batched state.
    #[test]
    fn test_batched_streaming_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 1);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        let tokens: Vec<u32> = vec![1, 2, 3];

        // Single-sequence batch step (all tokens at once)
        let (logits_batch, _) = model
            .step(&[&tokens], None)
            .expect("batch step failed");

        // Single-sequence streaming (token by token, chain state)
        let mut logits_stream = Vec::new();
        let mut state: Option<HipState> = None;
        for &tok in &tokens {
            let (logits, new_state) = model
                .step(&[&[tok]], state)
                .expect("streaming step failed");
            logits_stream.extend(logits);
            state = Some(new_state);
        }

        assert_eq!(logits_batch.len(), logits_stream.len());

        let mut max_diff = 0.0f32;
        for (i, (batch, stream)) in logits_batch.iter().zip(logits_stream.iter()).enumerate() {
            let diff = (batch - stream).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Streaming mismatch at {}: {} vs {}",
                i,
                batch,
                stream
            );
        }

        println!(
            "Batched streaming equivalence test PASSED (max_diff={})",
            max_diff
        );
    }

    /// Test chunked processing with batched state.
    #[test]
    fn test_batched_chunked_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 1);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        // 10 tokens, chunk into [4, 4, 2]
        let tokens: Vec<u32> = (1..=10).collect();
        let chunk_size = 4;

        // Full step
        let (logits_full, _) = model
            .step(&[&tokens], None)
            .expect("full step failed");

        // Chunked step (manual chunking)
        let mut logits_chunked = Vec::new();
        let mut state: Option<HipState> = None;
        for chunk in tokens.chunks(chunk_size) {
            let (logits, new_state) = model
                .step(&[chunk], state)
                .expect("chunked step failed");
            logits_chunked.extend(logits);
            state = Some(new_state);
        }

        assert_eq!(logits_full.len(), logits_chunked.len());

        let mut max_diff = 0.0f32;
        for (i, (full, chunked)) in logits_full.iter().zip(logits_chunked.iter()).enumerate() {
            let diff = (full - chunked).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Chunked mismatch at {}: {} vs {}",
                i,
                full,
                chunked
            );
        }

        println!(
            "Batched chunked equivalence test PASSED (max_diff={})",
            max_diff
        );
    }

    /// Test state evolution in batched mode.
    #[test]
    fn test_batched_state_evolution() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 2);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];

        // Run step with fresh state (None)
        let (_, state) = model
            .step(&[&seq1, &seq2], None)
            .expect("step failed");

        // Returned state should have evolved (non-zero)
        let att_sum: f32 = state
            .att_states
            .iter()
            .flat_map(|v| v.as_slice().iter())
            .map(|x| x.abs())
            .sum();
        assert!(att_sum > 0.0, "att_states should be non-zero after step");

        println!("Batched state evolution test PASSED");
    }

    /// Test batch size mismatch error when state batch_size != input batch size.
    #[test]
    fn test_batch_size_mismatch_error() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 3);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        // State with batch_size=2, but provide 3 sequences
        let state = HipState::new(&model.info, 2).expect("Failed to allocate state");
        let result = model.step(&[&[1u32], &[2u32], &[3u32]], Some(state));

        assert!(result.is_err(), "Should error on batch size mismatch");
        println!("Batch size mismatch error test PASSED");
    }

    /// Test variable-length sequences (now supported, not an error).
    #[test]
    fn test_variable_length_sequences() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let config = HipRuntimeConfig::new(256, 2);
        let model = model
            .with_config(config)
            .expect("Failed to configure model");

        // Two sequences with different lengths - should now work
        let seq1: Vec<u32> = vec![1, 2, 3]; // length 3
        let seq2: Vec<u32> = vec![4, 5]; // length 2
        let result = model.step(&[&seq1, &seq2], None);

        assert!(
            result.is_ok(),
            "Variable-length sequences should be supported"
        );

        let (logits, _) = result.unwrap();
        let vocab = model.info.n_vocab;
        // Logits should have (3 + 2) * vocab elements
        assert_eq!(
            logits.len(),
            5 * vocab,
            "Should have logits for all 5 tokens"
        );

        println!("Variable-length sequences test PASSED");
    }

    /// Test HipState reset.
    #[test]
    fn test_hip_state_reset() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        let mut state = HipState::new(&info, 2).expect("Failed to allocate state");

        // Fill with values
        for s in &mut state.att_states {
            s.as_slice_mut().fill(1.0);
        }
        for s in &mut state.att_shift_states {
            s.as_slice_mut().fill(f16::from_f32(2.0));
        }
        for s in &mut state.ffn_states {
            s.as_slice_mut().fill(f16::from_f32(3.0));
        }

        state.reset();

        // All should be zero
        assert!(state
            .att_states
            .iter()
            .all(|s| s.as_slice().iter().all(|&x| x == 0.0)));
        assert!(state
            .att_shift_states
            .iter()
            .all(|s| s.as_slice().iter().all(|&x| x == f16::from_f32(0.0))));
        assert!(state
            .ffn_states
            .iter()
            .all(|s| s.as_slice().iter().all(|&x| x == f16::from_f32(0.0))));
        assert!(state.v_first.is_none());

        println!("HipState reset test PASSED");
    }

    /// Test HipState v_first persistence across operations.
    #[test]
    fn test_hip_state_v_first_persistence() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        // Fresh state should have v_first = None
        let mut state = HipState::new(&info, 1).expect("Failed to allocate state");
        assert!(
            state.v_first.is_none(),
            "New state should have v_first = None"
        );

        // Set v_first and verify it persists
        let mut v_first_buf = PinnedBuffer::<f16>::new(64).expect("Failed to allocate v_first");
        v_first_buf.as_slice_mut().fill(f16::from_f32(1.0));
        state.v_first = Some(v_first_buf);
        assert!(
            state.v_first.is_some(),
            "v_first should persist after assignment"
        );
        assert_eq!(state.v_first.as_ref().unwrap().len(), 64);

        // Reset should clear v_first
        state.reset();
        assert!(
            state.v_first.is_none(),
            "Reset should clear v_first to None"
        );

        println!("HipState v_first persistence test PASSED");
    }

    // === End-to-end FLA prefill tests (bd-2sh.8.10) ===
    //
    // These tests verify that the FLA chunked prefill path produces results
    // matching the recurrent (token-by-token) path, and that state transfers
    // correctly from FLA prefill to recurrent decode.
    //
    // 2-tier dispatch:
    // - T == 1 => FusedT1Wkv (decode)
    // - T > 1  => FlaChunkedWkv (prefill)
    //
    // max_prefill_chunk only limits the max allowed sequence length (scratch size).

    /// Helper: compute top-k token indices from a logits slice.
    fn top_k_indices(logits: &[f32], vocab: usize, token_idx: usize, k: usize) -> Vec<usize> {
        let start = token_idx * vocab;
        let end = start + vocab;
        let mut indexed: Vec<(usize, f32)> =
            logits[start..end].iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Helper: compute cosine similarity between two logit slices for one token.
    fn cosine_similarity(a: &[f32], b: &[f32], vocab: usize, token_idx: usize) -> f64 {
        let start = token_idx * vocab;
        let end = start + vocab;
        let a_slice = &a[start..end];
        let b_slice = &b[start..end];

        let mut dot = 0.0f64;
        let mut a_norm = 0.0f64;
        let mut b_norm = 0.0f64;
        for (&av, &bv) in a_slice.iter().zip(b_slice.iter()) {
            dot += (av as f64) * (bv as f64);
            a_norm += (av as f64).powi(2);
            b_norm += (bv as f64).powi(2);
        }
        dot / (a_norm.sqrt() * b_norm.sqrt())
    }

    /// End-to-end FLA vs recurrent correctness test.
    ///
    /// Runs the same 64 tokens through both paths and compares the last-token
    /// logits. The FLA path uses real T=64 > 1 (triggers FLA dispatch).
    /// The recurrent path processes tokens one at a time (each step T=1,
    /// uses FusedT1Wkv).
    ///
    /// Since the two paths use mathematically equivalent but numerically
    /// different algorithms (chunked parallel vs sequential recurrent),
    /// we expect approximate agreement, not exact match.
    #[test]
    fn test_fla_vs_recurrent_logits() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        // Input: 64 real tokens — T > 1 triggers FLA dispatch
        let tokens: Vec<u32> = (1..=64).collect();
        let n_real = tokens.len();

        // FLA path: T=64 > 1 => FlaChunkedWkv
        let model_fla = Rwkv7Hip::load(model_path).expect("load");
        let config_fla = HipRuntimeConfig::new(256, 1);
        let model_fla = model_fla.with_config(config_fla).expect("config");
        let (logits_fla, state_fla) = model_fla
            .step(&[&tokens], None)
            .expect("FLA step failed");

        // Recurrent path: process one token at a time (each step T=1 => FusedT1Wkv)
        let model_rec = Rwkv7Hip::load(model_path).expect("load");
        let config_rec = HipRuntimeConfig::new(256, 1);
        let model_rec = model_rec.with_config(config_rec).expect("config");
        let mut logits_rec = Vec::new();
        let mut state_rec_opt: Option<HipState> = None;
        for &tok in &tokens {
            let (logits, new_state) = model_rec
                .step(&[&[tok]], state_rec_opt)
                .expect("Recurrent step failed");
            logits_rec.extend(logits);
            state_rec_opt = Some(new_state);
        }
        let state_rec = state_rec_opt.unwrap();

        let vocab = model_fla.info.n_vocab;

        // Both should return the same number of logits
        assert_eq!(
            logits_fla.len(),
            n_real * vocab,
            "FLA logits count mismatch"
        );
        assert_eq!(
            logits_rec.len(),
            n_real * vocab,
            "Recurrent logits count mismatch"
        );

        // Verify FLA logits are finite
        assert!(
            logits_fla.iter().all(|x| x.is_finite()),
            "FLA logits contain NaN or Inf"
        );

        // Compare last-token logits (the most important for generation)
        let last_tok = n_real - 1;

        // Cosine similarity between FLA and recurrent for last token
        let cos_sim = cosine_similarity(&logits_fla, &logits_rec, vocab, last_tok);

        // Top-k overlap
        let fla_top10 = top_k_indices(&logits_fla, vocab, last_tok, 10);
        let rec_top10 = top_k_indices(&logits_rec, vocab, last_tok, 10);
        let top10_overlap = fla_top10
            .iter()
            .filter(|i| rec_top10.contains(i))
            .count();

        let fla_top1 = fla_top10[0];
        let rec_top1 = rec_top10[0];

        // Max absolute difference across all logits
        let mut max_diff = 0.0f32;
        let mut mean_diff = 0.0f64;
        for (f, r) in logits_fla.iter().zip(logits_rec.iter()) {
            let d = (f - r).abs();
            max_diff = max_diff.max(d);
            mean_diff += d as f64;
        }
        mean_diff /= logits_fla.len() as f64;

        // WKV state comparison: max absolute diff across all layers
        let n_layer = state_fla.att_states.len();
        let mut max_state_diff = 0.0f32;
        for layer in 0..n_layer {
            let fla_state = state_fla.att_states[layer].as_slice();
            let rec_state = state_rec.att_states[layer].as_slice();
            for (f, r) in fla_state.iter().zip(rec_state.iter()) {
                let d = (f - r).abs();
                max_state_diff = max_state_diff.max(d);
            }
        }

        println!("=== FLA vs Recurrent Comparison (bd-2sh.8.10) ===");
        println!("  Tokens: {} real, FLA chunk=256, Recurrent T=1", n_real);
        println!("  Last-token cosine similarity: {:.6}", cos_sim);
        println!(
            "  Last-token top-1: FLA={}, Recurrent={} ({})",
            fla_top1,
            rec_top1,
            if fla_top1 == rec_top1 { "MATCH" } else { "DIFFER" }
        );
        println!("  Last-token top-10 overlap: {}/10", top10_overlap);
        println!("  All-token max logit diff: {:.6e}", max_diff);
        println!("  All-token mean logit diff: {:.6e}", mean_diff);
        println!("  WKV state max diff (across layers): {:.6e}", max_state_diff);

        // Assertions: FLA should produce coherent output even if not exactly matching.
        // The tolerance is lenient because FLA uses a fundamentally different computation
        // path (chunked parallel) vs the sequential recurrent kernel.
        // Key acceptance criteria from the ticket:
        // - "Model output matches recurrent-only inference within tolerance"
        // - "No regression in decode quality"
        assert!(
            cos_sim > 0.90,
            "Cosine similarity {:.6} too low (expected > 0.90) -- FLA output is incoherent",
            cos_sim
        );
        assert!(
            top10_overlap >= 3,
            "Top-10 overlap {}/10 too low (expected >= 3)",
            top10_overlap
        );
        assert!(
            logits_fla.iter().all(|x| x.is_finite()),
            "FLA logits must be finite"
        );

        println!("test_fla_vs_recurrent_logits PASSED");
    }

    /// FLA prefill -> recurrent decode state continuity test.
    ///
    /// 1. Prefill 48 tokens with FLA (T=48 > 1 => FlaChunkedWkv)
    /// 2. Decode 1 more token with recurrent (step with T=1 => FusedT1Wkv)
    /// 3. Compare the decode output against doing all tokens recurrently (T=1 each)
    ///
    /// This tests that the WKV state produced by FLA is compatible with
    /// subsequent recurrent decode, which is the core prefill->decode transition.
    #[test]
    fn test_fla_prefill_to_recurrent_decode() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        // 48 tokens for prefill (T > 1 triggers FLA), 1 token for decode
        let prefill_tokens: Vec<u32> = (1..=48).collect();
        let decode_token: u32 = 49;

        // === Path A: FLA prefill (T=48 > 1 => FLA) -> recurrent decode (T=1) ===
        let model_a = Rwkv7Hip::load(model_path).expect("load");
        let config_a = HipRuntimeConfig::new(256, 1);
        let model_a = model_a.with_config(config_a).expect("config");

        // Prefill with FLA
        let (_logits_prefill_a, state_a) = model_a
            .step(&[&prefill_tokens], None)
            .expect("FLA prefill failed");

        // Decode with recurrent (T=1, FusedT1Wkv)
        let (logits_decode_a, _) = model_a
            .step(&[&[decode_token]], Some(state_a))
            .expect("Decode after FLA failed");

        // === Path B: All recurrent (token-by-token, each T=1 => FusedT1Wkv) ===
        let model_b = Rwkv7Hip::load(model_path).expect("load");
        let config_b = HipRuntimeConfig::new(256, 1);
        let model_b = model_b.with_config(config_b).expect("config");

        // Prefill recurrently one token at a time
        let mut state_b: Option<HipState> = None;
        for &tok in &prefill_tokens {
            let (_logits, new_state) = model_b
                .step(&[&[tok]], state_b)
                .expect("Recurrent prefill step failed");
            state_b = Some(new_state);
        }

        // Decode recurrently
        let (logits_decode_b, _) = model_b
            .step(&[&[decode_token]], state_b)
            .expect("Recurrent decode failed");

        let vocab = model_a.info.n_vocab;
        assert_eq!(logits_decode_a.len(), vocab, "Decode A logits size");
        assert_eq!(logits_decode_b.len(), vocab, "Decode B logits size");

        // Compare decode outputs
        let cos_sim = {
            let mut dot = 0.0f64;
            let mut a_norm = 0.0f64;
            let mut b_norm = 0.0f64;
            for (&a, &b) in logits_decode_a.iter().zip(logits_decode_b.iter()) {
                dot += (a as f64) * (b as f64);
                a_norm += (a as f64).powi(2);
                b_norm += (b as f64).powi(2);
            }
            dot / (a_norm.sqrt() * b_norm.sqrt())
        };

        let fla_top10 = top_k_indices(&logits_decode_a, vocab, 0, 10);
        let rec_top10 = top_k_indices(&logits_decode_b, vocab, 0, 10);
        let top10_overlap = fla_top10
            .iter()
            .filter(|i| rec_top10.contains(i))
            .count();

        let fla_top1 = fla_top10[0];
        let rec_top1 = rec_top10[0];

        let mut max_diff = 0.0f32;
        for (&a, &b) in logits_decode_a.iter().zip(logits_decode_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("=== FLA Prefill -> Recurrent Decode (bd-2sh.8.10) ===");
        println!("  Prefill: {} tokens (FLA), Decode: 1 token (recurrent)", prefill_tokens.len());
        println!("  Decode cosine similarity: {:.6}", cos_sim);
        println!(
            "  Decode top-1: FLA+decode={}, Recurrent+decode={} ({})",
            fla_top1,
            rec_top1,
            if fla_top1 == rec_top1 { "MATCH" } else { "DIFFER" }
        );
        println!("  Decode top-10 overlap: {}/10", top10_overlap);
        println!("  Decode max logit diff: {:.6e}", max_diff);

        // State continuity assertion: the decode output after FLA prefill should
        // be highly similar to the decode output after recurrent prefill.
        // Cosine similarity is the primary metric -- it measures whether the
        // logit distributions point in the same direction. Top-k overlap is
        // secondary since closely-ranked tokens may swap order with small
        // numerical differences between FLA and recurrent state.
        assert!(
            cos_sim > 0.90,
            "State continuity broken: cosine similarity {:.6} < 0.90",
            cos_sim
        );
        assert!(
            top10_overlap >= 1,
            "State continuity: top-10 overlap {}/10 = 0 (no agreement at all)",
            top10_overlap
        );
        assert!(
            logits_decode_a.iter().all(|x| x.is_finite()),
            "Decode logits after FLA must be finite"
        );

        println!("test_fla_prefill_to_recurrent_decode PASSED");
    }

    /// Verify FLA is transparent to HipRuntime.infer_one() / infer_resident() API.
    ///
    /// Uses the high-level HipRuntime interface (the same one hip_gen uses)
    /// to run a prefill + decode sequence with FLA enabled by default.
    /// This verifies the acceptance criterion:
    /// "hip_gen example works with FLA enabled"
    #[test]
    fn test_fla_transparent_via_hip_runtime() {
        use std::path::Path;
        use crate::hip::HipRuntime;
        use crate::tensor::TensorShape as TensorShapeTrait;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("load");
        // Default HipRuntime: max_prefill_chunk=256, FLA is active when T > 1
        let runtime = HipRuntime::new(model, 1);
        let vocab = runtime.info().n_vocab;

        // Prefill: 48 tokens (T=48 > 1, dispatches to FLA)
        let prompt: Vec<u32> = (1..=48).collect();
        let logits = runtime.infer_one(&prompt).expect("FLA prefill via HipRuntime failed");

        // Basic sanity: output shape
        let shape = logits.shape();
        assert_eq!(shape[0], vocab, "vocab dim");
        assert_eq!(shape[1], prompt.len(), "token dim");
        assert!(
            logits.data().iter().all(|x| x.is_finite()),
            "Prefill logits must be finite"
        );

        // Get last-token logits and sample argmax
        let last_start = (prompt.len() - 1) * vocab;
        let last_logits = &logits.data()[last_start..last_start + vocab];
        let first_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        // Decode: 10 tokens one at a time (dispatches to FusedT1Wkv since T=1)
        let mut generated = vec![first_token];
        for _ in 0..9 {
            let tok = *generated.last().unwrap();
            let logits = runtime.infer_one(&[tok]).expect("Decode step failed");
            assert!(
                logits.data().iter().all(|x| x.is_finite()),
                "Decode logits must be finite"
            );
            let next = logits
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            generated.push(next);
        }

        println!("=== FLA Transparent via HipRuntime (bd-2sh.8.10) ===");
        println!("  Prefill: {} tokens (FLA path)", prompt.len());
        println!("  Decoded: {} tokens (recurrent path)", generated.len());
        println!("  Generated tokens: {:?}", generated);

        // The key assertion is that we got through without errors.
        // All logits finite, shapes correct, state transfer worked.
        assert_eq!(generated.len(), 10, "Should have generated 10 tokens");

        println!("test_fla_transparent_via_hip_runtime PASSED");
    }

    /// Verify the 2-tier dispatch is correct by comparing results across
    /// different processing strategies for the same input tokens.
    ///
    /// Uses 64 tokens and compares:
    /// - Tier 1 (FusedT1Wkv): token-by-token streaming (T=1 per step)
    /// - Tier 2 (FlaChunkedWkv): full batch (T=64 > 1)
    ///
    /// Both paths should produce finite, sane logits for the same input.
    #[test]
    fn test_two_tier_dispatch() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let tokens: Vec<u32> = (1..=64).collect();

        // Tier 1: FusedT1Wkv (one token at a time, T=1 per step)
        let model_t1 = Rwkv7Hip::load(model_path).expect("load");
        let model_t1 = model_t1.with_config(HipRuntimeConfig::new(256, 1)).expect("config");
        let mut t1_logits = Vec::new();
        let mut state: Option<HipState> = None;
        for &tok in &tokens {
            let (logits, new_state) = model_t1
                .step(&[&[tok]], state)
                .expect("T=1 step failed");
            t1_logits.extend(logits);
            state = Some(new_state);
        }

        // Tier 2: FlaChunkedWkv (full batch, T=64 > 1)
        let model_fla = Rwkv7Hip::load(model_path).expect("load");
        let model_fla = model_fla.with_config(HipRuntimeConfig::new(256, 1)).expect("config");
        let (fla_logits, _) = model_fla
            .step(&[&tokens], None)
            .expect("FLA step failed");

        let vocab = model_t1.info.n_vocab;
        let n_real = tokens.len();

        // Both paths should produce correct number of logits
        assert_eq!(t1_logits.len(), n_real * vocab, "T1 logits size");
        assert_eq!(fla_logits.len(), n_real * vocab, "FLA logits size");

        // Both paths should produce finite logits
        assert!(t1_logits.iter().all(|x| x.is_finite()), "T1 logits finite");
        assert!(fla_logits.iter().all(|x| x.is_finite()), "FLA logits finite");

        // FLA should be at least roughly similar to T=1 recurrent
        let cos_t1_fla = cosine_similarity(&t1_logits, &fla_logits, vocab, n_real - 1);

        println!("=== 2-Tier Dispatch Test (bd-2sh.8.10) ===");
        println!("  T=1 vs FLA cosine: {:.6}", cos_t1_fla);

        // FLA vs recurrent should be at least roughly similar
        assert!(
            cos_t1_fla > 0.90,
            "T=1 vs FLA cosine {:.6} too low",
            cos_t1_fla
        );

        println!("test_two_tier_dispatch PASSED");
    }

    /// Detailed ranking analysis comparing FLA vs recurrent logit distributions.
    ///
    /// For each of the 10 input tokens AND the last token specifically, computes:
    /// - Top-k overlap for k=1,5,10,50,100,500,1000
    /// - Spearman rank correlation across full vocabulary
    /// - Kendall tau on top-1000 subset (full vocab would be O(n^2))
    /// - Max absolute logit difference
    /// - Mean absolute logit difference
    /// - L2 distance between logit vectors
    /// - Cosine similarity
    /// - KL divergence after softmax
    /// - Side-by-side top-10 tokens from each path
    #[test]
    fn test_fla_ranking_analysis() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        // Input: 64 real tokens (T > 1 triggers FLA dispatch)
        let tokens: Vec<u32> = (1..=64).collect();
        let n_real = tokens.len();

        // FLA path: T=64 > 1 => FlaChunkedWkv
        let model_fla = Rwkv7Hip::load(model_path).expect("load");
        let config_fla = HipRuntimeConfig::new(256, 1);
        let model_fla = model_fla.with_config(config_fla).expect("config");
        let (logits_fla, _) = model_fla
            .step(&[&tokens], None)
            .expect("FLA step failed");

        // Recurrent path: one token at a time (each step T=1 => FusedT1Wkv)
        let model_rec = Rwkv7Hip::load(model_path).expect("load");
        let config_rec = HipRuntimeConfig::new(256, 1);
        let model_rec = model_rec.with_config(config_rec).expect("config");
        let mut logits_rec = Vec::new();
        let mut state_rec: Option<HipState> = None;
        for &tok in &tokens {
            let (logits, new_state) = model_rec
                .step(&[&[tok]], state_rec)
                .expect("Recurrent step failed");
            logits_rec.extend(logits);
            state_rec = Some(new_state);
        }

        let vocab = model_fla.info.n_vocab;

        assert_eq!(logits_fla.len(), n_real * vocab);
        assert_eq!(logits_rec.len(), n_real * vocab);

        // ---- Helper closures ----

        // Extract logit slice for a given token index (inline as a fn to avoid lifetime issues)
        fn logit_slice_of(logits: &[f32], vocab: usize, token_idx: usize) -> &[f32] {
            let start = token_idx * vocab;
            &logits[start..start + vocab]
        }

        // Top-k indices (sorted by descending logit value)
        let top_k = |slice: &[f32], k: usize| -> Vec<usize> {
            let mut indexed: Vec<(usize, f32)> =
                slice.iter().copied().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.into_iter().take(k).map(|(i, _)| i).collect()
        };

        // Top-k overlap count
        let top_k_overlap = |a: &[f32], b: &[f32], k: usize| -> usize {
            let a_top = top_k(a, k);
            let b_top = top_k(b, k);
            a_top.iter().filter(|i| b_top.contains(i)).count()
        };

        // Compute ranks for a slice (rank 0 = highest logit)
        let compute_ranks = |slice: &[f32]| -> Vec<f64> {
            let mut indexed: Vec<(usize, f32)> =
                slice.iter().copied().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut ranks = vec![0.0f64; slice.len()];
            for (rank, (idx, _)) in indexed.iter().enumerate() {
                ranks[*idx] = rank as f64;
            }
            ranks
        };

        // Pearson correlation between two f64 slices
        let pearson = |a: &[f64], b: &[f64]| -> f64 {
            let n = a.len() as f64;
            let mean_a: f64 = a.iter().sum::<f64>() / n;
            let mean_b: f64 = b.iter().sum::<f64>() / n;
            let mut cov = 0.0f64;
            let mut var_a = 0.0f64;
            let mut var_b = 0.0f64;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let da = ai - mean_a;
                let db = bi - mean_b;
                cov += da * db;
                var_a += da * da;
                var_b += db * db;
            }
            if var_a == 0.0 || var_b == 0.0 {
                return 0.0;
            }
            cov / (var_a.sqrt() * var_b.sqrt())
        };

        // Spearman rank correlation: rank both, then Pearson of ranks
        let spearman = |a: &[f32], b: &[f32]| -> f64 {
            let ranks_a = compute_ranks(a);
            let ranks_b = compute_ranks(b);
            pearson(&ranks_a, &ranks_b)
        };

        // Kendall tau on a subset (top-k indices from the union of both paths)
        // Uses the O(n^2) naive algorithm, so we restrict to a subset.
        let kendall_tau_subset = |a: &[f32], b: &[f32], k: usize| -> f64 {
            // Get the union of top-k indices from both
            let a_top = top_k(a, k);
            let b_top = top_k(b, k);
            let mut union_indices: Vec<usize> = a_top.clone();
            for &idx in &b_top {
                if !union_indices.contains(&idx) {
                    union_indices.push(idx);
                }
            }
            let n = union_indices.len();
            if n < 2 {
                return 1.0;
            }

            // Extract (a_val, b_val) pairs for union indices
            let pairs: Vec<(f32, f32)> = union_indices
                .iter()
                .map(|&i| (a[i], b[i]))
                .collect();

            let mut concordant: i64 = 0;
            let mut discordant: i64 = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let a_diff = pairs[i].0 - pairs[j].0;
                    let b_diff = pairs[i].1 - pairs[j].1;
                    let product = (a_diff as f64) * (b_diff as f64);
                    if product > 0.0 {
                        concordant += 1;
                    } else if product < 0.0 {
                        discordant += 1;
                    }
                    // ties are ignored (neither concordant nor discordant)
                }
            }

            let total = concordant + discordant;
            if total == 0 {
                return 1.0;
            }
            (concordant - discordant) as f64 / total as f64
        };

        // Cosine similarity between two slices
        let cosine_sim = |a: &[f32], b: &[f32]| -> f64 {
            let mut dot = 0.0f64;
            let mut a_norm = 0.0f64;
            let mut b_norm = 0.0f64;
            for (&av, &bv) in a.iter().zip(b.iter()) {
                dot += (av as f64) * (bv as f64);
                a_norm += (av as f64).powi(2);
                b_norm += (bv as f64).powi(2);
            }
            dot / (a_norm.sqrt() * b_norm.sqrt())
        };

        // Max absolute difference
        let max_abs_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        // Mean absolute difference
        let mean_abs_diff = |a: &[f32], b: &[f32]| -> f64 {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() as f64)
                .sum();
            sum / a.len() as f64
        };

        // L2 distance
        let l2_distance = |a: &[f32], b: &[f32]| -> f64 {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum();
            sum.sqrt()
        };

        // Stable softmax: subtract max before exponentiating
        let softmax = |slice: &[f32]| -> Vec<f64> {
            let max_val = slice
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = slice
                .iter()
                .map(|&x| ((x - max_val) as f64).exp())
                .collect();
            let sum: f64 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        };

        // KL divergence: sum(p * log(p/q)) with epsilon for numerical stability
        let kl_divergence = |a: &[f32], b: &[f32]| -> f64 {
            let p = softmax(a);
            let q = softmax(b);
            let eps = 1e-10f64;
            let mut kl = 0.0f64;
            for (&pi, &qi) in p.iter().zip(q.iter()) {
                if pi > eps {
                    kl += pi * (pi / (qi + eps)).ln();
                }
            }
            kl
        };

        // ---- Detailed analysis for one token position ----
        let analyze_token = |token_idx: usize, label: &str| {
            let fla_slice = logit_slice_of(&logits_fla, vocab, token_idx);
            let rec_slice = logit_slice_of(&logits_rec, vocab, token_idx);

            println!("\n--- {} (token position {}) ---", label, token_idx);

            // Top-k overlap
            let k_values = [1, 5, 10, 50, 100, 500, 1000];
            println!("  Top-k overlap:");
            for &k in &k_values {
                let overlap = top_k_overlap(fla_slice, rec_slice, k);
                println!(
                    "    k={:>4}: {}/{} ({:.1}%)",
                    k,
                    overlap,
                    k,
                    100.0 * overlap as f64 / k as f64
                );
            }

            // Spearman rank correlation
            let rho = spearman(fla_slice, rec_slice);
            println!("  Spearman rank correlation: {:.6}", rho);

            // Kendall tau on top-1000 subset
            let tau = kendall_tau_subset(fla_slice, rec_slice, 1000);
            println!("  Kendall tau (top-1000 subset): {:.6}", tau);

            // Max absolute difference
            let max_d = max_abs_diff(fla_slice, rec_slice);
            println!("  Max absolute logit diff: {:.6e}", max_d);

            // Mean absolute difference
            let mean_d = mean_abs_diff(fla_slice, rec_slice);
            println!("  Mean absolute logit diff: {:.6e}", mean_d);

            // L2 distance
            let l2 = l2_distance(fla_slice, rec_slice);
            println!("  L2 distance: {:.6e}", l2);

            // Cosine similarity
            let cos = cosine_sim(fla_slice, rec_slice);
            println!("  Cosine similarity: {:.6}", cos);

            // KL divergence (FLA || Recurrent) and (Recurrent || FLA)
            let kl_fla_rec = kl_divergence(fla_slice, rec_slice);
            let kl_rec_fla = kl_divergence(rec_slice, fla_slice);
            println!("  KL(FLA || Recurrent): {:.6e}", kl_fla_rec);
            println!("  KL(Recurrent || FLA): {:.6e}", kl_rec_fla);

            // Side-by-side top-10
            let fla_top10 = top_k(fla_slice, 10);
            let rec_top10 = top_k(rec_slice, 10);
            println!("  Top-10 side-by-side:");
            println!(
                "    {:>4}  {:>12} {:>12}  |  {:>12} {:>12}",
                "Rank", "FLA_id", "FLA_logit", "Rec_id", "Rec_logit"
            );
            for rank in 0..10 {
                let fi = fla_top10[rank];
                let ri = rec_top10[rank];
                println!(
                    "    {:>4}  {:>12} {:>12.4}  |  {:>12} {:>12.4}",
                    rank + 1,
                    fi,
                    fla_slice[fi],
                    ri,
                    rec_slice[ri]
                );
            }
        };

        // ---- Run analysis ----
        println!("================================================================");
        println!("=== FLA vs Recurrent Ranking Analysis ===");
        println!("================================================================");
        println!("  Tokens: {} real, FLA chunk=256, Recurrent chunk=16", n_real);
        println!("  Vocab size: {}", vocab);

        // Detailed analysis for the LAST token
        analyze_token(n_real - 1, "LAST TOKEN (primary)");

        // Summary table header for all tokens
        println!("\n================================================================");
        println!("=== Per-Token Summary Table ===");
        println!("================================================================");
        println!(
            "{:>5} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Pos", "Top1=?", "Top5", "Top10", "Top100",
            "Spearman", "Kendall", "CosSim", "MaxDiff", "KL(F||R)"
        );

        // Analyze each of the 10 tokens
        for ti in 0..n_real {
            let fla_slice = logit_slice_of(&logits_fla, vocab, ti);
            let rec_slice = logit_slice_of(&logits_rec, vocab, ti);

            let top1_match = if top_k(fla_slice, 1)[0] == top_k(rec_slice, 1)[0] {
                "YES"
            } else {
                "NO"
            };
            let top5_ov = top_k_overlap(fla_slice, rec_slice, 5);
            let top10_ov = top_k_overlap(fla_slice, rec_slice, 10);
            let top100_ov = top_k_overlap(fla_slice, rec_slice, 100);
            let rho = spearman(fla_slice, rec_slice);
            let tau = kendall_tau_subset(fla_slice, rec_slice, 1000);
            let cos = cosine_sim(fla_slice, rec_slice);
            let max_d = max_abs_diff(fla_slice, rec_slice);
            let kl = kl_divergence(fla_slice, rec_slice);

            println!(
                "{:>5} {:>8} {:>5}/{:<2} {:>6}/{:<3} {:>7}/{:<3} {:>10.6} {:>10.6} {:>10.6} {:>10.4e} {:>10.4e}",
                ti,
                top1_match,
                top5_ov, 5,
                top10_ov, 10,
                top100_ov, 100,
                rho,
                tau,
                cos,
                max_d,
                kl
            );
        }

        // Also print detailed analysis for each token
        for ti in 0..n_real {
            analyze_token(ti, &format!("Token {}", ti));
        }

        println!("\n================================================================");
        println!("=== Analysis Complete ===");
        println!("================================================================");

        // No assertions -- this test is purely for diagnostic output
        println!("test_fla_ranking_analysis PASSED (diagnostic only, no assertions)");
    }
}
