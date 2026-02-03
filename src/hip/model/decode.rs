//! Standalone HIP decode module for T=1 token generation.
//!
//! This module provides [`HipDecode`], a standalone inference module that uses
//! the FusedT1Wkv kernel for single-token decode. It holds shared model weights
//! via `Arc<Rwkv7Model>` and its own [`DecodeScratch`] buffers sized for T=1.
//!
//! # Architecture
//!
//! ```text
//! Arc<Rwkv7Model> (shared weights)
//!   └── HipDecode
//!         ├── DecodeScratch (T=1 buffers, no FLA)
//!         ├── wkv_state in decode layout [V_row, K_col]
//!         ├── load_state(HipState) — transpose + copy from CPU
//!         └── decode(tokens) → logits
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let model = Rwkv7Model::load("model.st")?;
//! let decode = HipDecode::new(model.clone(), DecodeConfig::default())?;
//!
//! // Load state from a prefill pass
//! decode.load_state(&state)?;
//!
//! // Decode tokens one at a time
//! let logits = decode.decode(&[&[next_token]])?;
//! ```

use std::sync::Arc;

use super::prefill::{FusedT1Wkv, WkvInput, WkvKernel};
use super::state::{HipState, StateLayout};
use super::{Rwkv7Model, Rwkv7ModelInfo};
use crate::hip::ffi::{check, hip_memcpy_h2d, HipErrorKind, Result};
use crate::hip::kernels::{
    add_f16, broadcast_add_f16, broadcast_mul_f16, channel_mix_state_f16,
    channel_mix_state_f16_masked, control_k_f16, copy_f16_to_f32, copy_tensor_f16, decay_exp_f16,
    group_norm_f16, l2_norm_f16, layer_norm_f16, lerp_f16, mul_f16, negate_f16, sigmoid_f16,
    softplus_decay_f16, squared_relu_f16, tanh_f16, wkv_bonus_f16,
};
use crate::hip::kernels::fla::state_transpose;
use crate::hip::scratch::{DecodeConfig, DecodeScratch};
use crate::hip::tensor::{TensorHip, TensorShape};

/// Standalone decode module for T=1 token generation.
///
/// Holds shared model weights via `Arc<Rwkv7Model>` and owns its own
/// `DecodeScratch` with T=1 buffers. Uses only the FusedT1Wkv kernel
/// (no FLA, no WaveReduce).
///
/// This module is independent of `Rwkv7Hip` and `HipPrefill` -- it can be
/// created and used standalone with just an `Arc<Rwkv7Model>`.
pub struct HipDecode {
    /// Shared model weights
    model: Arc<Rwkv7Model>,
    /// T=1 scratch buffers (no FLA buffers)
    scratch: DecodeScratch,
}

impl std::fmt::Debug for HipDecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipDecode")
            .field("model", &self.model.info)
            .field("batch_size", &self.scratch.config.batch_size)
            .finish()
    }
}

impl HipDecode {
    /// Create a new decode module with T=1 scratch buffers.
    ///
    /// # Arguments
    /// * `model` - Shared model weights (can be shared with HipPrefill)
    /// * `config` - Decode configuration (batch_size)
    ///
    /// # Errors
    /// Returns error if GPU memory allocation fails.
    pub fn new(model: Arc<Rwkv7Model>, config: DecodeConfig) -> Result<Self> {
        let lora_dims = model.lora_dims();
        let scratch = DecodeScratch::new(&model.info, lora_dims, config)?;
        Ok(Self { model, scratch })
    }

    /// Get a reference to the model info.
    pub fn info(&self) -> &Rwkv7ModelInfo {
        &self.model.info
    }

    /// Get the batch size this decode module was configured for.
    pub fn batch_size(&self) -> usize {
        self.scratch.config.batch_size
    }

    /// Load state from an `HipState` (CPU pinned buffers) into the GPU scratch.
    ///
    /// If the source state has `layout == StateLayout::Fla`, the WKV state
    /// matrices are transposed to decode layout during the copy. Attention
    /// shift and FFN states are always direct copies (no layout difference).
    ///
    /// # Arguments
    /// * `state` - CPU-side state to load. Must have `batch_size == 1` or match
    ///   the configured batch size.
    ///
    /// # Errors
    /// Returns error on batch size mismatch, layer count mismatch, or GPU errors.
    pub fn load_state(&mut self, state: &HipState) -> Result<()> {
        let n_layer = self.model.info.n_layer;
        let n_head = self.model.info.n_head;
        let batch_size = self.scratch.config.batch_size;

        // Validate layer counts
        if state.att_states.len() != n_layer {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "load_state: state has {} layers but model has {}",
                    state.att_states.len(),
                    n_layer
                ),
            });
        }
        if state.att_shift_states.len() != n_layer {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "load_state: att_shift has {} layers but model has {}",
                    state.att_shift_states.len(),
                    n_layer
                ),
            });
        }
        if state.ffn_states.len() != n_layer {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "load_state: ffn has {} layers but model has {}",
                    state.ffn_states.len(),
                    n_layer
                ),
            });
        }

        // Validate batch size
        if state.batch_size != batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "load_state: state batch_size {} != decode batch_size {}",
                    state.batch_size, batch_size
                ),
            });
        }

        let stream = self.scratch.blas_ctx.stream();
        let stream_handle = stream.handle();

        // Copy att_shift and ffn states (always direct copy, no layout difference)
        for i in 0..n_layer {
            unsafe {
                state.att_shift_states[i].copy_to_device_async(
                    self.scratch.att_shift_state_gpu[i].as_mut_ptr(),
                    stream_handle,
                )?;
                state.ffn_states[i].copy_to_device_async(
                    self.scratch.ffn_state_gpu[i].as_mut_ptr(),
                    stream_handle,
                )?;
            }
        }

        // Copy WKV state with optional transpose
        if state.layout == StateLayout::Fla {
            // Source is in FLA layout [K_row, V_col], need to transpose to
            // decode layout [V_row, K_col]. Upload to a temp GPU tensor,
            // then transpose into the scratch state tensor.
            let wkv_shape = self.scratch.wkv_state_gpu[0].shape();
            let mut temp_wkv = TensorHip::<f32>::new(wkv_shape)?;

            for i in 0..n_layer {
                // Upload CPU pinned -> temp GPU tensor
                let src_data = state.att_states[i].as_slice();
                let byte_count = src_data.len() * std::mem::size_of::<f32>();
                unsafe {
                    check(hip_memcpy_h2d(
                        temp_wkv.as_mut_ptr() as *mut std::ffi::c_void,
                        src_data.as_ptr() as *const std::ffi::c_void,
                        byte_count,
                        stream_handle,
                    ))?;
                }

                // Out-of-place transpose: temp (Fla layout) -> scratch (Decode layout)
                state_transpose(
                    &temp_wkv,
                    &mut self.scratch.wkv_state_gpu[i],
                    n_head,
                    batch_size,
                    stream,
                )?;
            }
        } else {
            // Source is already in decode layout, direct copy
            for i in 0..n_layer {
                unsafe {
                    state.att_states[i].copy_to_device_async(
                        self.scratch.wkv_state_gpu[i].as_mut_ptr(),
                        stream_handle,
                    )?;
                }
            }
        }

        // Synchronize to ensure all transfers are complete
        stream.synchronize()?;

        Ok(())
    }

    /// Reset the decode state to zeros.
    ///
    /// Clears all GPU-resident state (att_shift, ffn, wkv) to zero.
    pub fn reset_state(&mut self) -> Result<()> {
        self.scratch.reset_state_gpu()
    }

    /// Run a single decode step (T=1 per sequence) using FusedT1Wkv.
    ///
    /// Each inner slice must contain exactly one token. Returns logits as
    /// a flat `Vec<f32>` of length `batch_size * n_vocab`.
    ///
    /// # Arguments
    /// * `tokens` - Batch of single-token sequences: `&[&[u32]]` where each
    ///   inner slice has length 1.
    ///
    /// # Errors
    /// Returns error on batch size mismatch, empty input, or GPU errors.
    pub fn decode(&mut self, tokens: &[&[u32]]) -> Result<Vec<f32>> {
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "decode: empty batch".to_string(),
            });
        }
        if batch_size > self.scratch.config.batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "decode: batch size {} exceeds configured max {}",
                    batch_size, self.scratch.config.batch_size
                ),
            });
        }

        // Validate all sequences have exactly 1 token
        for (i, seq) in tokens.iter().enumerate() {
            if seq.len() != 1 {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "decode: sequence {} has {} tokens, expected 1",
                        i,
                        seq.len()
                    ),
                });
            }
        }

        // Run the T=1 dispatch
        let lens: Vec<usize> = vec![1; batch_size];
        self.dispatch_decode(tokens, &lens)?;

        // Sync and extract logits
        self.scratch.blas_ctx.synchronize()?;

        let n_vocab = self.model.info.n_vocab;
        let total_logits = batch_size * n_vocab;
        let staging = self.scratch.logits_staging.as_slice();
        Ok(staging[..total_logits].to_vec())
    }

    /// Core GPU forward pass for T=1 decode. Simplified from `Rwkv7Hip::dispatch()`.
    ///
    /// Uses only FusedT1Wkv (no FLA, no WaveReduce). State is GPU-resident
    /// in scratch buffers and updated in-place.
    fn dispatch_decode(&mut self, tokens: &[&[u32]], lens: &[usize]) -> Result<()> {
        let b = tokens.len();
        let t = 1; // Always T=1 for decode

        let n_embd = self.model.info.n_embd;
        let n_head = self.model.info.n_head;
        let head_size = self.model.info.head_size;
        let n_layer = self.model.info.n_layer;
        let n_hidden = self.model.info.n_hidden;
        let n_vocab = self.model.info.n_vocab;

        let scratch = &mut self.scratch;
        let lora_dims = &scratch.lora_dims;

        let ctx = &scratch.blas_ctx;
        let stream = ctx.stream();

        // Convert lens to i32 tensor for masked kernel
        let lens_i32: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
        let lens_shape = TensorShape::new(b, 1, 1, 1);
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        lens_gpu.copy_from_slice(&lens_i32, stream)?;

        // Shapes for this forward pass (all T=1)
        let std_shape = TensorShape::new(n_embd, t, b, 1);
        let ffn_shape = TensorShape::new(n_hidden, t, b, 1);
        let out_shape = TensorShape::new(n_vocab, t, b, 1);
        let state_shape = TensorShape::new(n_embd, b, 1, 1);
        let wkv_data_shape = TensorShape::new(head_size, n_head, t, b);
        let _wkv_state_shape = TensorShape::new(head_size, head_size, n_head, b);
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
            // Embedding lookup: tokens[b][0] -> x[c, 1, b]
            let emb_data = &self.model.embed.w;
            let emb_stride = self.model.embed.n_embd;
            let x_host = scratch.emb_staging.as_slice_mut();
            for batch_idx in 0..b {
                let token = tokens[batch_idx][0] as usize;
                let src_offset = token * emb_stride;
                let dst_offset = batch_idx * n_embd;
                x_host[dst_offset..dst_offset + n_embd]
                    .copy_from_slice(&emb_data[src_offset..src_offset + n_embd]);
            }
            // Async copy from pinned host memory to GPU
            unsafe {
                scratch
                    .emb_staging
                    .copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
            }
        }

        // Take state vectors out of scratch for the dispatch
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

            // FusedT1Wkv kernel for decode
            let wkv_kernel: &dyn WkvKernel = &FusedT1Wkv;

            // Process each layer
            for layer_idx in 0..n_layer {
                let layer = &self.model.layers[layer_idx];

                // Apply ln0 for layer 0
                if layer_idx == 0 {
                    layer_norm_f16(
                        &x,
                        &self.model.embed.ln.weight,
                        &self.model.embed.ln.bias,
                        &mut x_ln,
                        1e-5,
                        stream,
                    )?;
                    copy_tensor_f16(&x_ln, &mut x, stream)?;
                }

                // ==== Time-Mix (Attention) ====
                layer_norm_f16(
                    &x,
                    &layer.att_ln.weight,
                    &layer.att_ln.bias,
                    &mut x_ln,
                    1e-5,
                    stream,
                )?;

                // Token shifts for attention
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

                // Update shift state
                copy_tensor_f16(&new_att_shift, &mut att_shift_gpu[layer_idx], stream)?;

                // Linear projections: r, k, v
                {
                    ctx.hgemm_into(&layer.att.w_r, &att_xr, &mut att_r)?;
                    ctx.hgemm_into(&layer.att.w_k, &att_xk, &mut att_k)?;
                    ctx.hgemm_into(&layer.att.w_v, &att_xv, &mut att_v)?;
                }

                // Decay: w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
                {
                    ctx.hgemm_into(&layer.att.w1, &att_xw, &mut lora_w)?;
                    tanh_f16(&lora_w, &mut lora_w_tanh, stream)?;
                    ctx.hgemm_into(&layer.att.w2, &lora_w_tanh, &mut att_w)?;
                    broadcast_add_f16(&att_w, &layer.att.w0, &mut temp1, stream)?;
                    softplus_decay_f16(&temp1, &mut att_w, stream)?;
                }

                // Adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
                {
                    ctx.hgemm_into(&layer.att.a1, &att_xa, &mut lora_a)?;
                    ctx.hgemm_into(&layer.att.a2, &lora_a, &mut lora_a_proj)?;
                    broadcast_add_f16(&lora_a_proj, &layer.att.a0, &mut temp1, stream)?;
                    sigmoid_f16(&temp1, &mut att_a, stream)?;
                }

                // Gate: g = sigmoid(xg @ g1) @ g2
                {
                    ctx.hgemm_into(&layer.att.g1, &att_xg, &mut lora_g)?;
                    sigmoid_f16(&lora_g, &mut lora_g_sig, stream)?;
                    ctx.hgemm_into(&layer.att.g2, &lora_g_sig, &mut att_g)?;
                }

                // Value residual (layers > 0)
                if layer_idx > 0 {
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (&layer.att.v0, &layer.att.v1, &layer.att.v2)
                    {
                        ctx.hgemm_into(v1, &att_xv, &mut lora_v)?;
                        ctx.hgemm_into(v2, &lora_v, &mut v_lora2)?;
                        broadcast_add_f16(&v_lora2, v0, &mut temp1, stream)?;
                        sigmoid_f16(&temp1, &mut temp2, stream)?;
                        lerp_f16(&att_v, &v_first, &temp2, &mut temp1, stream)?;
                        copy_tensor_f16(&temp1, &mut att_v, stream)?;
                    }
                } else {
                    copy_tensor_f16(&att_v, &mut v_first, stream)?;
                }

                // L2 normalize k
                {
                    broadcast_mul_f16(&att_k, &layer.att.k_k, &mut temp1, stream)?;
                    l2_norm_f16(&temp1, &mut att_kk, head_size, 1e-12, stream)?;
                }

                // Control K
                control_k_f16(&layer.att.k_a, &att_a, &att_k, &mut att_k_ctrl, stream)?;

                // WKV inputs
                {
                    negate_f16(&att_kk, &mut wkv_a, stream)?;
                    mul_f16(&att_kk, &att_a, &mut wkv_b, stream)?;
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

                // Run FusedT1Wkv (decode only, no FLA path)
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

                // Group norm on WKV output
                group_norm_f16(
                    &wkv_out,
                    &layer.att.gn.weight,
                    &layer.att.gn.bias,
                    &mut wkv_normed,
                    n_head,
                    64e-5,
                    stream,
                )?;

                // WKV bonus
                let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
                let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
                let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
                wkv_bonus_f16(
                    &r_wkv,
                    &k_ctrl_wkv,
                    &v_wkv,
                    &r_k_wkv,
                    &mut wkv_bonus_wkv,
                    stream,
                )?;

                // Combine and gate
                {
                    add_f16(&wkv_normed, &wkv_bonus, &mut temp1, stream)?;
                    mul_f16(&temp1, &att_g, &mut temp2, stream)?;
                }

                // Output projection
                ctx.hgemm_into(&layer.att.w_o, &temp2, &mut att_out)?;

                // Residual
                {
                    add_f16(&x, &att_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                }

                // ==== Channel-Mix (FFN) ====
                layer_norm_f16(
                    &x,
                    &layer.ffn_ln.weight,
                    &layer.ffn_ln.bias,
                    &mut x_ln,
                    1e-5,
                    stream,
                )?;

                // Token shift for FFN
                channel_mix_state_f16_masked(
                    &x_ln,
                    &ffn_shift_gpu[layer_idx],
                    &layer.ffn.x_k,
                    &mut ffn_xk,
                    &mut new_ffn_shift,
                    &lens_gpu,
                    stream,
                )?;

                // Update FFN shift state
                copy_tensor_f16(&new_ffn_shift, &mut ffn_shift_gpu[layer_idx], stream)?;

                // Key projection
                ctx.hgemm_into(&layer.ffn.w_k, &ffn_xk, &mut ffn_k)?;

                // Squared ReLU
                squared_relu_f16(&ffn_k, &mut ffn_k_sq, stream)?;

                // Value projection
                ctx.hgemm_into(&layer.ffn.w_v, &ffn_k_sq, &mut ffn_out)?;

                // Residual
                {
                    add_f16(&x, &ffn_out, &mut temp1, stream)?;
                    copy_tensor_f16(&temp1, &mut x, stream)?;
                }
            }

            // ==== Output Head ====
            layer_norm_f16(
                &x,
                &self.model.head.ln.weight,
                &self.model.head.ln.bias,
                &mut x_ln,
                1e-5,
                stream,
            )?;

            ctx.hgemm_into(&self.model.head.w, &x_ln, &mut logits)?;

            // Convert f16 logits to f32 on GPU, then download asynchronously.
            copy_f16_to_f32(&logits, &mut logits_f32, stream)?;
            let logits_len = logits_f32.len();
            logits_f32.copy_to_slice_async(
                &mut scratch.logits_staging.as_slice_mut()[..logits_len],
                stream,
            )?;

            Ok(())
        })();

        // Always put state back to scratch
        scratch.att_shift_state_gpu = att_shift_gpu;
        scratch.ffn_state_gpu = ffn_shift_gpu;
        scratch.wkv_state_gpu = wkv_state_gpu;

        result
    }
}
