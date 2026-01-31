//! Step (inference) implementation for the RWKV7 HIP backend.

use half::f16;

use super::state::HipState;
use super::Rwkv7Hip;
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
    // GPU-native kernels used in forward
    layer_norm_f16,
    lerp_f16,
    mul_f16,
    negate_f16,
    sigmoid_f16,
    softplus_decay_f16,
    squared_relu_f16,
    tanh_f16,
    wkv7_wave_reduce,
    wkv7_fused_t1,
    wkv_bonus_f16,
};
use crate::hip::scratch::HipScratch;
use crate::hip::tensor::{TensorHip, TensorShape};
use crate::hip::HipProf;

#[cfg(feature = "hip-probes")]
use crate::hip_probe;

#[cfg(feature = "hip-probes")]
use crate::hip::probe::{self, HipProbeMap, HipProbeMapRef};

/// Sync stream only when hip-prof feature is enabled.
/// This gives accurate per-operation GPU timings at the cost of serialization.
#[cfg(feature = "hip-prof")]
#[inline]
fn prof_sync(stream: &Stream) -> Result<()> {
    stream.synchronize()
}

#[cfg(not(feature = "hip-prof"))]
#[inline]
fn prof_sync(_stream: &Stream) -> Result<()> {
    Ok(())
}

impl Rwkv7Hip {
    /// Internal step implementation with async logits copy.
    ///
    /// Instead of synchronously downloading logits, this copies to a provided pinned
    /// buffer asynchronously. The caller must sync the stream before reading the buffer.
    pub(super) fn step_inner(
        &self,
        tokens: &[&[u32]],
        state: &mut HipState,
        scratch: &mut HipScratch,
        lens: &[usize],
    ) -> Result<()> {
        let use_resident_state = scratch.config.resident_state;
        let b = tokens.len();
        let t = tokens[0].len();
        let mut prof = HipProf::new("step_inner");

        let ctx = &scratch.blas_ctx;
        let stream = ctx.stream();

        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;
        let n_layer = self.info.n_layer;
        let n_hidden = self.info.n_hidden;
        let n_vocab = self.info.n_vocab;
        let lora_dims = &scratch.lora_dims;

        // Convert lens to i32 tensor for masked kernel
        let lens_i32: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
        let lens_shape = TensorShape::new(b, 1, 1, 1);
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        prof.time("lens_upload", || {
            lens_gpu.copy_from_slice(&lens_i32, stream)
        })?;

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

        prof.time("embedding", || {
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
            prof_sync(stream)?;
            Ok(())
        })?;

        let (mut att_shift_gpu, mut ffn_shift_gpu, mut wkv_state_gpu) = if use_resident_state {
            (
                std::mem::take(&mut scratch.att_shift_state_gpu),
                std::mem::take(&mut scratch.ffn_state_gpu),
                std::mem::take(&mut scratch.wkv_state_gpu),
            )
        } else {
            prof.time("state_upload", || {
                // Upload state to GPU using pinned async transfers
                let mut att_shift_gpu = Vec::with_capacity(n_layer);
                for s in &state.att_shift_states {
                    let mut gpu_tensor = TensorHip::<f16>::new(state_shape)?;
                    unsafe {
                        s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
                    }
                    att_shift_gpu.push(gpu_tensor);
                }

                let mut ffn_shift_gpu = Vec::with_capacity(n_layer);
                for s in &state.ffn_states {
                    let mut gpu_tensor = TensorHip::<f16>::new(state_shape)?;
                    unsafe {
                        s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
                    }
                    ffn_shift_gpu.push(gpu_tensor);
                }

                let mut wkv_state_gpu = Vec::with_capacity(n_layer);
                for s in &state.att_states {
                    let mut gpu_tensor = TensorHip::<f32>::new(wkv_state_shape)?;
                    unsafe {
                        s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
                    }
                    wkv_state_gpu.push(gpu_tensor);
                }
                Ok((att_shift_gpu, ffn_shift_gpu, wkv_state_gpu))
            })?
        };

        let result = (|| {
            // Temporary buffers
            let mut new_att_shift = scratch.new_att_shift.resized_view_mut(state_shape)?;
            let mut new_ffn_shift = scratch.new_ffn_shift.resized_view_mut(state_shape)?;
            let mut new_wkv_state = scratch.new_wkv_state.resized_view_mut(wkv_state_shape)?;
            let mut temp1 = scratch.temp1.resized_view_mut(std_shape)?;
            let mut temp2 = scratch.temp2.resized_view_mut(std_shape)?;

            // Process each layer
            for layer_idx in 0..n_layer {
                let layer = &self.layers[layer_idx];

                // Apply ln0 for layer 0
                if layer_idx == 0 {
                    prof.time("ln0", || {
                        layer_norm_f16(
                            &x,
                            &self.embed.ln.weight,
                            &self.embed.ln.bias,
                            &mut x_ln,
                            1e-5,
                            stream,
                        )?;
                        copy_tensor_f16(&x_ln, &mut x, stream)?;
                        Ok(())
                    })?;
                }

                // ==== Time-Mix (Attention) ====
                prof.time("att_ln", || {
                    layer_norm_f16(
                        &x,
                        &layer.att_ln.weight,
                        &layer.att_ln.bias,
                        &mut x_ln,
                        1e-5,
                        stream,
                    )?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Token shifts for attention - use masked kernel for x_r to get correct state
                // The masked kernel extracts state at lengths[b]-1 instead of T-1
                prof.time("att_shift", || {
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
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Update shift state - new_att_shift has correct state from masked kernel
                std::mem::swap(&mut new_att_shift, &mut att_shift_gpu[layer_idx]);

                // Linear projections: r, k, v
                prof.time("att_proj", || {
                    ctx.hgemm_into(&layer.att.w_r, &att_xr, &mut att_r)?;
                    ctx.hgemm_into(&layer.att.w_k, &att_xk, &mut att_k)?;
                    ctx.hgemm_into(&layer.att.w_v, &att_xv, &mut att_v)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Decay: w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
                prof.time("att_decay", || {
                    ctx.hgemm_into(&layer.att.w1, &att_xw, &mut lora_w)?;
                    tanh_f16(&lora_w, &mut lora_w_tanh, stream)?;
                    ctx.hgemm_into(&layer.att.w2, &lora_w_tanh, &mut att_w)?;
                    broadcast_add_f16(&att_w, &layer.att.w0, &mut temp1, stream)?;
                    softplus_decay_f16(&temp1, &mut att_w, stream)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
                prof.time("att_adapt", || {
                    ctx.hgemm_into(&layer.att.a1, &att_xa, &mut lora_a)?;
                    ctx.hgemm_into(&layer.att.a2, &lora_a, &mut lora_a_proj)?;
                    broadcast_add_f16(&lora_a_proj, &layer.att.a0, &mut temp1, stream)?;
                    sigmoid_f16(&temp1, &mut att_a, stream)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Gate: g = sigmoid(xg @ g1) @ g2
                prof.time("att_gate", || {
                    ctx.hgemm_into(&layer.att.g1, &att_xg, &mut lora_g)?;
                    sigmoid_f16(&lora_g, &mut lora_g_sig, stream)?;
                    ctx.hgemm_into(&layer.att.g2, &lora_g_sig, &mut att_g)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Value residual (layers > 0)
                if layer_idx > 0 {
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (&layer.att.v0, &layer.att.v1, &layer.att.v2)
                    {
                        prof.time("att_vres", || {
                            ctx.hgemm_into(v1, &att_xv, &mut lora_v)?;
                            ctx.hgemm_into(v2, &lora_v, &mut v_lora2)?;
                            broadcast_add_f16(&v_lora2, v0, &mut temp1, stream)?;
                            sigmoid_f16(&temp1, &mut temp2, stream)?;
                            lerp_f16(&att_v, &v_first, &temp2, &mut temp1, stream)?;
                            copy_tensor_f16(&temp1, &mut att_v, stream)?;
                            Ok(())
                        })?;
                    }
                } else {
                    copy_tensor_f16(&att_v, &mut v_first, stream)?;
                }

                // L2 normalize k
                prof.time("att_norm_k", || {
                    broadcast_mul_f16(&att_k, &layer.att.k_k, &mut temp1, stream)?;
                    l2_norm_f16(&temp1, &mut att_kk, head_size, 1e-12, stream)?;
                    Ok(())
                })?;

                // Control K
                prof.time("att_ctrl_k", || {
                    control_k_f16(&layer.att.k_a, &att_a, &att_k, &mut att_k_ctrl, stream)?;
                    Ok(())
                })?;

                // WKV inputs
                prof.time("att_wkv_in", || {
                    negate_f16(&att_kk, &mut wkv_a, stream)?;
                    mul_f16(&att_kk, &att_a, &mut wkv_b, stream)?;
                    // Decay: exp(-exp(w)) where w = log(sigmoid(d)) - 0.5
                    // This gives decay = exp(-sigmoid(d) * 0.606531) in range (0.545, 1)
                    decay_exp_f16(&att_w, &mut w_decay, stream)?;
                    Ok(())
                })?;

                // Reshape for WKV
                let w_decay_wkv = w_decay.reshape_view(wkv_data_shape)?;
                let r_wkv = att_r.reshape_view(wkv_data_shape)?;
                let k_ctrl_wkv = att_k_ctrl.reshape_view(wkv_data_shape)?;
                let v_wkv = att_v.reshape_view(wkv_data_shape)?;
                let wkv_a_wkv = wkv_a.reshape_view(wkv_data_shape)?;
                let wkv_b_wkv = wkv_b.reshape_view(wkv_data_shape)?;
                let mut wkv_out_wkv = wkv_out.reshape_view_mut(wkv_data_shape)?;

                // Run WKV7: fused_t1 for decode (T=1), wave_reduce for prefill (T>1)
                prof.time("wkv", || {
                    if t == 1 {
                        // Fused T=1 decode kernel (in-place state)
                        wkv7_fused_t1(
                            &w_decay_wkv,
                            &r_wkv,
                            &k_ctrl_wkv,
                            &v_wkv,
                            &wkv_a_wkv,
                            &wkv_b_wkv,
                            &mut wkv_state_gpu[layer_idx],
                            &mut wkv_out_wkv,
                            &lens_gpu,
                            stream,
                        )?;
                    } else {
                        // Wave-reduce prefill kernel (state_in -> state_out)
                        wkv7_wave_reduce(
                            &w_decay_wkv,
                            &r_wkv,
                            &k_ctrl_wkv,
                            &v_wkv,
                            &wkv_a_wkv,
                            &wkv_b_wkv,
                            &wkv_state_gpu[layer_idx],
                            &mut wkv_out_wkv,
                            &mut new_wkv_state,
                            &lens_gpu,
                            stream,
                        )?;
                        std::mem::swap(&mut wkv_state_gpu[layer_idx], &mut new_wkv_state);
                    }
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Group norm on WKV output
                prof.time("wkv_norm", || {
                    group_norm_f16(
                        &wkv_out,
                        &layer.att.gn.weight,
                        &layer.att.gn.bias,
                        &mut wkv_normed,
                        n_head,
                        64e-5,
                        stream,
                    )?;
                    Ok(())
                })?;

                // WKV bonus
                let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
                let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
                let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
                prof.time("wkv_bonus", || {
                    wkv_bonus_f16(
                        &r_wkv,
                        &k_ctrl_wkv,
                        &v_wkv,
                        &r_k_wkv,
                        &mut wkv_bonus_wkv,
                        stream,
                    )?;
                    Ok(())
                })?;

                // Combine and gate
                prof.time("att_gate_out", || {
                    add_f16(&wkv_normed, &wkv_bonus, &mut temp1, stream)?;
                    mul_f16(&temp1, &att_g, &mut temp2, stream)?;
                    Ok(())
                })?;

                // Output projection
                prof.time("att_out", || {
                    ctx.hgemm_into(&layer.att.w_o, &temp2, &mut att_out)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Residual
                prof.time("att_resid", || {
                    add_f16(&x, &att_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                    Ok(())
                })?;

                // ==== Channel-Mix (FFN) ====
                prof.time("ffn_ln", || {
                    layer_norm_f16(
                        &x,
                        &layer.ffn_ln.weight,
                        &layer.ffn_ln.bias,
                        &mut x_ln,
                        1e-5,
                        stream,
                    )?;
                    Ok(())
                })?;

                // Token shift for FFN - use masked kernel for correct state extraction
                prof.time("ffn_shift", || {
                    channel_mix_state_f16_masked(
                        &x_ln,
                        &ffn_shift_gpu[layer_idx],
                        &layer.ffn.x_k,
                        &mut ffn_xk,
                        &mut new_ffn_shift,
                        &lens_gpu,
                        stream,
                    )?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Update FFN shift state - new_ffn_shift has correct state from masked kernel
                std::mem::swap(&mut new_ffn_shift, &mut ffn_shift_gpu[layer_idx]);

                // Key projection
                prof.time("ffn_k", || {
                    ctx.hgemm_into(&layer.ffn.w_k, &ffn_xk, &mut ffn_k)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Squared ReLU
                prof.time("ffn_relu2", || {
                    squared_relu_f16(&ffn_k, &mut ffn_k_sq, stream)?;
                    Ok(())
                })?;

                // Value projection
                prof.time("ffn_v", || {
                    ctx.hgemm_into(&layer.ffn.w_v, &ffn_k_sq, &mut ffn_out)?;
                    prof_sync(stream)?;
                    Ok(())
                })?;

                // Residual
                prof.time("ffn_resid", || {
                    add_f16(&x, &ffn_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                    Ok(())
                })?;
            }

            prof.time("head", || {
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
                prof_sync(stream)?;
                Ok(())
            })?;

            prof.time("logits_download", || {
                // Convert f16 logits to f32 on GPU, then download asynchronously
                copy_f16_to_f32(&logits, &mut logits_f32, stream)?;
                logits_f32.copy_to_slice_async(scratch.logits_staging.as_slice_mut(), stream)?;
                Ok(())
            })?;

            Ok(())
        })();

        if use_resident_state {
            scratch.att_shift_state_gpu = att_shift_gpu;
            scratch.ffn_state_gpu = ffn_shift_gpu;
            scratch.wkv_state_gpu = wkv_state_gpu;
        } else if result.is_ok() {
            prof.time("state_download", || {
                // Download state back to host using pinned async transfers
                for (i, gpu_state) in att_shift_gpu.iter().enumerate() {
                    unsafe {
                        state.att_shift_states[i]
                            .copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
                    }
                }
                for (i, gpu_state) in ffn_shift_gpu.iter().enumerate() {
                    unsafe {
                        state.ffn_states[i]
                            .copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
                    }
                }
                for (i, gpu_state) in wkv_state_gpu.iter().enumerate() {
                    unsafe {
                        state.att_states[i]
                            .copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
                    }
                }
                Ok(())
            })?;
        }

        #[cfg(feature = "hip-prof")]
        prof.print(&format!("b={b} t={t} layers={n_layer}"));

        result
    }

    /// Run one inference step on variable-length input sequences.
    ///
    /// This method queues all GPU work and data transfers
    /// without waiting for them to complete. The caller can check completion
    /// status or wait for results using the returned `ForwardCompletion`.
    ///
    /// This enables overlapping GPU computation with CPU work:
    ///
    /// ```ignore
    /// let completion = model.step(&[&tokens], None)?;
    ///
    /// // Do CPU work while GPU computes...
    /// process_other_data();
    ///
    /// // Wait for results when needed
    /// let (logits, state) = completion.wait()?;
    /// ```
    ///
    /// # Note
    ///
    /// This uses HIP events for synchronization. On some ROCm versions,
    /// stream creation may fail; in that case this falls back to the null
    /// stream which provides less overlap but still works correctly.
    ///
    /// # Arguments
    /// * `x` - Batch of token sequences
    /// * `state` - Optional initial state (None = fresh zeros)
    ///
    /// # Returns
    /// A `ForwardCompletion` handle that can be used to check status or wait for results.
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

        // For async, we only support single-chunk processing for now
        // Multi-chunk async would require more complex state management
        if max_len > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "step() requires sequence length <= chunk_size ({} vs {})",
                    max_len, chunk_size
                ),
            });
        }

        // Initialize state
        let current_state = match state {
            Some(s) => {
                if s.batch_size != batch_size {
                    return Err(HipErrorKind {
                        code: -1,
                        message: format!(
                            "State batch_size mismatch: state has {} but input has {} sequences",
                            s.batch_size, batch_size
                        ),
                    });
                }
                s
            }
            None => HipState::new(&self.info, batch_size)?,
        };

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

        // Run the step (queues work but doesn't sync)
        let mut current_state = current_state;
        self.step_inner(
            &chunk_refs,
            &mut current_state,
            scratch,
            &lens,
        )?;

        // Sync the stream -- all GPU work and D->H transfers are now complete
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

        Ok((logits, current_state))
    }
}
