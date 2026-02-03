//! Standalone prefill module for RWKV7 HIP backend.
//!
//! `HipPrefill` runs FLA chunked prefill only (no decode/FusedT1Wkv).
//! It holds an `Arc<Rwkv7Model>` for shared weight access and owns its
//! own `PrefillScratch` buffers. State is GPU-resident between calls.
//!
//! # Usage
//!
//! ```rust,ignore
//! let model = Rwkv7Model::load("model.st")?;
//! let config = PrefillConfig::new(256, 1);
//! let mut prefill = HipPrefill::new(model, config)?;
//!
//! let logits = prefill.prefill(&[&tokens])?;
//! let state = prefill.get_state()?;
//! ```

use std::sync::Arc;

use half::f16;

use super::fla::FlaChunkedWkv;
use super::state::{HipState, StateLayout};
use super::{Rwkv7Model, Rwkv7ModelInfo};
use crate::hip::ffi::{HipErrorKind, Result};
use crate::hip::kernels::{
    add_f16, broadcast_add_f16, broadcast_mul_f16, channel_mix_state_f16,
    channel_mix_state_f16_masked, control_k_f16, copy_f16_to_f32, copy_tensor_f16,
    group_norm_f16, l2_norm_f16, layer_norm_f16, lerp_f16, mul_f16, negate_f16, sigmoid_f16,
    softplus_decay_f16, squared_relu_f16, tanh_f16, wkv_bonus_f16,
};
use crate::hip::pinned::PinnedBuffer;
use crate::hip::scratch::{PrefillConfig, PrefillScratch};
use crate::hip::tensor::TensorShape;

/// Standalone prefill module for RWKV7 on HIP.
///
/// Runs the FLA chunked prefill pipeline only. State is GPU-resident
/// between calls. Use [`get_state()`](HipPrefill::get_state) to extract
/// state for handoff to a decode module.
///
/// # Architecture
///
/// ```text
/// HipPrefill
///   ├── Arc<Rwkv7Model>     (shared weights)
///   ├── PrefillScratch      (owned FLA + intermediate buffers)
///   └── wkv_state in FLA layout [K_row, V_col]
/// ```
pub struct HipPrefill {
    /// Shared model weights (immutable, reference-counted).
    model: Arc<Rwkv7Model>,

    /// Owned scratch buffers including all FLA intermediates and GPU state.
    scratch: PrefillScratch,
}

impl std::fmt::Debug for HipPrefill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipPrefill")
            .field("info", &self.model.info)
            .field("config", &self.scratch.config)
            .finish()
    }
}

impl HipPrefill {
    /// Create a new standalone prefill module.
    ///
    /// Allocates `PrefillScratch` buffers on the GPU and initializes state
    /// to zeros.
    ///
    /// # Arguments
    /// * `model` - Shared model weights (via `Arc`)
    /// * `config` - Prefill configuration (chunk size, batch size, FLA chunk size)
    ///
    /// # Errors
    /// Returns error if GPU memory allocation fails.
    pub fn new(model: Arc<Rwkv7Model>, config: PrefillConfig) -> Result<Self> {
        let lora_dims = model.lora_dims();
        let runtime_config = config.to_runtime_config();
        let scratch = PrefillScratch::new(&model.info, lora_dims, runtime_config)?;
        Ok(Self { model, scratch })
    }

    /// Access model info (dimensions, vocab size, etc.).
    pub fn info(&self) -> &Rwkv7ModelInfo {
        &self.model.info
    }

    /// Get a clone of the shared model Arc.
    pub fn model(&self) -> Arc<Rwkv7Model> {
        self.model.clone()
    }

    /// Reset GPU-resident state to zeros.
    pub fn reset_state(&mut self) -> Result<()> {
        self.scratch.reset_state_gpu()
    }

    /// Extract the current GPU-resident state as a CPU-side `HipState`.
    ///
    /// Copies all per-layer state tensors (att_shift, ffn_shift, wkv_state)
    /// from GPU to pinned host memory. The returned state is tagged with
    /// [`StateLayout::Fla`] since the FLA kernel stores WKV state in
    /// K-row, V-col layout.
    ///
    /// # Errors
    /// Returns error if GPU-to-host copy fails.
    pub fn get_state(&mut self) -> Result<HipState> {
        let batch_size = self.scratch.config.batch_size;
        let n_layer = self.model.info.n_layer;
        let stream = self.scratch.blas_ctx.stream();

        // Allocate pinned buffers for each layer
        let mut att_states = Vec::with_capacity(n_layer);
        let mut att_shift_states = Vec::with_capacity(n_layer);
        let mut ffn_states = Vec::with_capacity(n_layer);

        for layer_idx in 0..n_layer {
            // WKV state: [head_size, head_size, n_head, batch_size]
            let wkv_gpu = &self.scratch.wkv_state_gpu[layer_idx];
            let wkv_elems = wkv_gpu.shape().len();
            let mut wkv_buf = PinnedBuffer::<f32>::new(wkv_elems)?;
            wkv_gpu.copy_to_slice_async(wkv_buf.as_slice_mut(), stream)?;
            att_states.push(wkv_buf);

            // Attention shift state: [n_embd, batch_size]
            let att_shift_gpu = &self.scratch.att_shift_state_gpu[layer_idx];
            let att_shift_elems = att_shift_gpu.shape().len();
            let mut att_shift_buf = PinnedBuffer::<f16>::new(att_shift_elems)?;
            att_shift_gpu.copy_to_slice_async(att_shift_buf.as_slice_mut(), stream)?;
            att_shift_states.push(att_shift_buf);

            // FFN shift state: [n_embd, batch_size]
            let ffn_gpu = &self.scratch.ffn_state_gpu[layer_idx];
            let ffn_elems = ffn_gpu.shape().len();
            let mut ffn_buf = PinnedBuffer::<f16>::new(ffn_elems)?;
            ffn_gpu.copy_to_slice_async(ffn_buf.as_slice_mut(), stream)?;
            ffn_states.push(ffn_buf);
        }

        // Synchronize to ensure all D2H copies complete
        stream.synchronize()?;

        Ok(HipState {
            batch_size,
            att_states,
            att_shift_states,
            ffn_states,
            v_first: None,
            layout: StateLayout::Fla,
        })
    }

    /// Run FLA-only prefill on a batch of token sequences.
    ///
    /// Processes the input tokens through the full model using the FLA chunked
    /// attention kernel (no FusedT1Wkv decode path). State is updated in-place
    /// on the GPU.
    ///
    /// # Arguments
    /// * `tokens` - Batch of token sequences. All sequences in the batch are
    ///   padded to the length of the longest sequence. The batch size must not
    ///   exceed the configured maximum.
    ///
    /// # Returns
    /// Flattened f32 logits for all real tokens (padding positions excluded).
    /// Layout: `[batch_0_tokens..., batch_1_tokens..., ...]` where each token
    /// contributes `n_vocab` logits.
    ///
    /// # Errors
    /// Returns error if:
    /// - The batch is empty or all sequences are empty
    /// - The batch size exceeds the configured maximum
    /// - The effective token count exceeds the configured chunk size
    /// - A GPU kernel fails
    pub fn prefill(&mut self, tokens: &[&[u32]]) -> Result<Vec<f32>> {
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }

        // Read config values without holding a long-lived &mut borrow
        let chunk_size = self.scratch.config.max_prefill_chunk;
        let max_batch = self.scratch.config.batch_size;

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
        let lens: Vec<usize> = tokens.iter().map(|s| s.len()).collect();
        let max_len = *lens.iter().max().unwrap_or(&0);

        if max_len == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "All sequences are empty".to_string(),
            });
        }

        // FLA requires T > 1 for chunked prefill.
        // For T=1, we still use FLA (it handles T=1 correctly via single-chunk path).
        // But the primary use case is T>1.

        // Validate effective token count
        let t_effective = if batch_size > 1 && max_len > 1 {
            batch_size * max_len
        } else {
            lens.iter().sum()
        };
        if t_effective > chunk_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "prefill() requires effective token count <= chunk_size ({} vs {})",
                    t_effective, chunk_size
                ),
            });
        }

        // Pad sequences for B>1 prefill
        let (dispatch_tokens_storage, dispatch_refs, dispatch_lens);
        if batch_size > 1 && max_len > 1 {
            let padded: Vec<Vec<u32>> = tokens
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
            dispatch_tokens_storage = vec![]; // unused but needed for lifetime
            let _ = &dispatch_tokens_storage;
            dispatch_refs = tokens.to_vec();
            dispatch_lens = lens.clone();
        }

        // Run FLA-only dispatch (borrows &mut self)
        self.dispatch_fla(&dispatch_refs, &dispatch_lens)?;

        // Sync and extract logits (re-borrow scratch after dispatch returns)
        let scratch = &self.scratch;
        let stream = scratch.blas_ctx.stream();
        stream.synchronize()?;

        let n_vocab = self.model.info.n_vocab;
        let staging = scratch.logits_staging.as_slice();
        let t_total: usize = lens.iter().sum();
        let mut logits = Vec::with_capacity(t_total * n_vocab);

        if batch_size > 1 && max_len > 1 {
            for (b, &real_len) in lens.iter().enumerate() {
                for t in 0..real_len {
                    let offset = (b * max_len + t) * n_vocab;
                    logits.extend_from_slice(&staging[offset..offset + n_vocab]);
                }
            }
        } else {
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

    /// Core FLA-only GPU forward pass.
    ///
    /// Runs all layers using the FLA chunked attention kernel exclusively
    /// (never dispatches to FusedT1Wkv). State is GPU-resident in scratch.
    fn dispatch_fla(&mut self, tokens: &[&[u32]], lens: &[usize]) -> Result<()> {
        let b = tokens.len();
        let t = tokens[0].len();

        let n_embd = self.model.info.n_embd;
        let n_head = self.model.info.n_head;
        let head_size = self.model.info.head_size;

        let scratch = &mut self.scratch;

        // Always create FLA kernel (this is prefill-only, always FLA).
        let mut fla_kernel = FlaChunkedWkv::new(scratch, head_size, n_head, t, b)?;

        let ctx = &scratch.blas_ctx;
        let stream = ctx.stream();
        let n_layer = self.model.info.n_layer;
        let n_hidden = self.model.info.n_hidden;
        let n_vocab = self.model.info.n_vocab;
        let lora_dims = &scratch.lora_dims;

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
        // Note: w_decay is not used by FLA (FLA receives raw att_w for better precision).
        // We still allocate and compute it because removing it would change scratch
        // buffer semantics. The cost is negligible.
        let mut _w_decay = scratch.w_decay.resized_view_mut(std_shape)?;
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
            let emb_data = &self.model.embed.w;
            let emb_stride = self.model.embed.n_embd;
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
            unsafe {
                scratch
                    .emb_staging
                    .copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
            }
        }

        // Take state out of scratch for the layer loop
        let (mut att_shift_gpu, mut ffn_shift_gpu, mut wkv_state_gpu) = (
            std::mem::take(&mut scratch.att_shift_state_gpu),
            std::mem::take(&mut scratch.ffn_state_gpu),
            std::mem::take(&mut scratch.wkv_state_gpu),
        );

        let result = (|| {
            let mut new_att_shift = scratch.new_att_shift.resized_view_mut(state_shape)?;
            let mut new_ffn_shift = scratch.new_ffn_shift.resized_view_mut(state_shape)?;
            let mut temp1 = scratch.temp1.resized_view_mut(std_shape)?;
            let mut temp2 = scratch.temp2.resized_view_mut(std_shape)?;

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
                // Note: decay_exp_f16 is NOT called here because FLA receives
                // raw att_w directly (pre-exponentiation) for better precision.
                {
                    negate_f16(&att_kk, &mut wkv_a, stream)?;
                    mul_f16(&att_kk, &att_a, &mut wkv_b, stream)?;
                }

                // Reshape for WKV
                let att_w_wkv = att_w.reshape_view(wkv_data_shape)?;
                let r_wkv = att_r.reshape_view(wkv_data_shape)?;
                let k_ctrl_wkv = att_k_ctrl.reshape_view(wkv_data_shape)?;
                let v_wkv = att_v.reshape_view(wkv_data_shape)?;
                let wkv_a_wkv = wkv_a.reshape_view(wkv_data_shape)?;
                let wkv_b_wkv = wkv_b.reshape_view(wkv_data_shape)?;
                let mut wkv_out_wkv = wkv_out.reshape_view_mut(wkv_data_shape)?;

                // Run FLA chunked WKV (always FLA, never FusedT1Wkv)
                fla_kernel.compute(
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

            // Convert f16 logits to f32, then download asynchronously
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
