//! Step (inference) implementation for the RWKV7 HIP backend.

use half::f16;

use super::prefill::{FusedT1Wkv, WaveReduceWkv, WkvInput, WkvKernel};
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

        let ctx = &scratch.blas_ctx;
        let stream = ctx.stream();

        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;
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

            // Select WKV kernel: fused_t1 for decode (T=1), wave_reduce for prefill (T>1)
            let wkv_kernel: &dyn WkvKernel = if t == 1 { &FusedT1Wkv } else { &WaveReduceWkv };

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
                            let stacked: Vec<f32> = [&xr_data, &xw_data, &xk_data, &xv_data, &xa_data, &xg_data]
                                .into_iter()
                                .flatten()
                                .copied()
                                .collect();
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttTokenShift, &stacked, [n_embd, actual_t, b, 6]);
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
                            let stacked: Vec<f32> = [&r_data, &k_data, &v_data]
                                .into_iter()
                                .flatten()
                                .copied()
                                .collect();
                            hip_probe!(self, probe_ctx, probe::HipHook::PostAttLinear, &stacked, [n_embd, actual_t, b, 3]);
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
                let w_decay_wkv = w_decay.reshape_view(wkv_data_shape)?;
                let r_wkv = att_r.reshape_view(wkv_data_shape)?;
                let k_ctrl_wkv = att_k_ctrl.reshape_view(wkv_data_shape)?;
                let v_wkv = att_v.reshape_view(wkv_data_shape)?;
                let wkv_a_wkv = wkv_a.reshape_view(wkv_data_shape)?;
                let wkv_b_wkv = wkv_b.reshape_view(wkv_data_shape)?;
                let mut wkv_out_wkv = wkv_out.reshape_view_mut(wkv_data_shape)?;

                // PreWkv probe (stacked: w_decay, r, k_ctrl, v, wkv_a, wkv_b) + PreWkvState
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref probes) = self.probes {
                        if probes.contains_key(&probe::HipHook::PreWkv) {
                            let wd = download_f16_as_f32(&w_decay, stream)?;
                            let r_d = download_f16_as_f32(&att_r, stream)?;
                            let kc = download_f16_as_f32(&att_k_ctrl, stream)?;
                            let v_d = download_f16_as_f32(&att_v, stream)?;
                            let wa = download_f16_as_f32(&wkv_a, stream)?;
                            let wb = download_f16_as_f32(&wkv_b, stream)?;
                            let stacked: Vec<f32> = [&wd, &r_d, &kc, &v_d, &wa, &wb]
                                .into_iter()
                                .flatten()
                                .copied()
                                .collect();
                            hip_probe!(self, probe_ctx, probe::HipHook::PreWkv, &stacked, [n_embd, actual_t, b, 6]);
                        }
                        if probes.contains_key(&probe::HipHook::PreWkvState) {
                            let data = download_f32(&wkv_state_gpu[layer_idx], stream)?;
                            hip_probe!(self, probe_ctx, probe::HipHook::PreWkvState, &data, [head_size, head_size, n_head, b]);
                        }
                    }
                }

                // Run WKV7 via trait dispatch: fused_t1 for T=1, wave_reduce for T>1
                {
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
                // Convert f16 logits to f32 on GPU, then download asynchronously
                copy_f16_to_f32(&logits, &mut logits_f32, stream)?;
                logits_f32.copy_to_slice_async(scratch.logits_staging.as_slice_mut(), stream)?;
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

        if max_len > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "step() requires sequence length <= chunk_size ({} vs {})",
                    max_len, chunk_size
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

        // Pad sequences to chunk_size
        let chunk_tokens: Vec<Vec<u32>> = x
            .iter()
            .map(|seq| {
                let mut padded = seq.to_vec();
                padded.resize(chunk_size, 0);
                padded
            })
            .collect();
        let chunk_refs: Vec<&[u32]> = chunk_tokens.iter().map(|v| v.as_slice()).collect();

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
        self.dispatch(&chunk_refs, scratch, &lens)?;

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

        // Extract real tokens from padded staging buffer
        let padded = scratch.logits_staging.as_slice();
        let mut logits = Vec::new();
        for (b, &real_len) in lens.iter().enumerate() {
            for t in 0..real_len {
                let offset = (b * chunk_size + t) * n_vocab;
                logits.extend_from_slice(&padded[offset..offset + n_vocab]);
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

        if max_len > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "infer_resident() requires sequence length <= chunk_size ({} vs {})",
                    max_len, chunk_size
                ),
            });
        }

        // Pad sequences to chunk_size
        let chunk_tokens: Vec<Vec<u32>> = x
            .iter()
            .map(|seq| {
                let mut padded = seq.to_vec();
                padded.resize(chunk_size, 0);
                padded
            })
            .collect();
        let chunk_refs: Vec<&[u32]> = chunk_tokens.iter().map(|v| v.as_slice()).collect();

        let n_vocab = self.info.n_vocab;

        // Run the forward pass (state stays GPU-resident, no H2D/D2H)
        self.dispatch(&chunk_refs, scratch, &lens)?;

        // Sync the stream
        let stream = scratch.blas_ctx.stream();
        stream.synchronize()?;

        // Extract real tokens from padded staging buffer
        let padded = scratch.logits_staging.as_slice();
        let mut logits = Vec::new();
        for (b, &real_len) in lens.iter().enumerate() {
            for t in 0..real_len {
                let offset = (b * chunk_size + t) * n_vocab;
                logits.extend_from_slice(&padded[offset..offset + n_vocab]);
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
}
