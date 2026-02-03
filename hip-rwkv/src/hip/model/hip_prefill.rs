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

use super::dispatch_helpers::{self, ProbeState};
use super::fla::FlaChunkedWkv;
use super::state::{HipState, StateLayout};
use super::{Rwkv7Model, Rwkv7ModelInfo};
use crate::hip::ffi::{check, hip_memcpy_h2d, HipErrorKind, Result};
use crate::hip::kernels::fla::state_transpose;
use crate::hip::pinned::PinnedBuffer;
use crate::hip::scratch::{PrefillConfig, PrefillScratch};
use crate::hip::tensor::{TensorHip, TensorShape};

#[cfg(feature = "hip-probes")]
use crate::hip::probe::{self, HipProbeMapRef};

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

    /// Optional probe hooks for capturing intermediate values.
    #[cfg(feature = "hip-probes")]
    probes: Option<HipProbeMapRef>,
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
        Ok(Self {
            model,
            scratch,
            #[cfg(feature = "hip-probes")]
            probes: None,
        })
    }

    /// Attach probes for capturing intermediate values during prefill.
    ///
    /// Only available with `hip-probes` feature.
    #[cfg(feature = "hip-probes")]
    pub fn with_probes(mut self, probes: HipProbeMapRef) -> Self {
        self.probes = Some(probes);
        self
    }

    /// Set probes on an existing instance (for use by HipRuntime).
    #[cfg(feature = "hip-probes")]
    pub fn set_probes(&mut self, probes: Option<HipProbeMapRef>) {
        self.probes = probes;
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

    /// Load state from a CPU-side `HipState` into the GPU scratch.
    ///
    /// Copies all per-layer state tensors (att_shift, ffn_shift, wkv_state)
    /// from pinned host memory to GPU. This is the inverse of `get_state()`.
    ///
    /// Layout conversion is handled automatically:
    /// - FLA layout: direct copy (same layout as prefill scratch)
    /// - Decode layout: WKV state is transposed from [V_row, K_col] to [K_row, V_col]
    ///
    /// # Errors
    /// Returns error if host-to-GPU copy fails or batch size mismatches.
    pub fn load_state(&mut self, state: &HipState) -> Result<()> {
        let n_layer = self.model.info.n_layer;
        let n_head = self.model.info.n_head;
        let batch_size = self.scratch.config.batch_size;

        if state.att_states.len() != n_layer
            || state.att_shift_states.len() != n_layer
            || state.ffn_states.len() != n_layer
        {
            return Err(crate::hip::ffi::HipErrorKind {
                code: -1,
                message: format!(
                    "load_state: layer count mismatch (state has {}/{}/{}, model has {})",
                    state.att_states.len(),
                    state.att_shift_states.len(),
                    state.ffn_states.len(),
                    n_layer
                ),
            });
        }

        let stream = self.scratch.blas_ctx.stream();
        let stream_handle = stream.handle();

        // Copy att_shift and ffn states (layout-independent)
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
        if state.layout == StateLayout::Decode {
            // Source is in Decode layout [V_row, K_col], need to transpose to
            // FLA layout [K_row, V_col]. Upload to temp GPU tensor, then transpose.
            let wkv_shape = self.scratch.wkv_state_gpu[0].shape();
            let mut temp_wkv = TensorHip::<f32>::new(wkv_shape)?;

            for i in 0..n_layer {
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

                // Transpose: Decode layout -> FLA layout
                state_transpose(
                    &temp_wkv,
                    &mut self.scratch.wkv_state_gpu[i],
                    n_head,
                    batch_size,
                    stream,
                )?;
            }
        } else {
            // Source is already in FLA layout, direct copy
            for i in 0..n_layer {
                unsafe {
                    state.att_states[i].copy_to_device_async(
                        self.scratch.wkv_state_gpu[i].as_mut_ptr(),
                        stream_handle,
                    )?;
                }
            }
        }

        // Synchronize to ensure all H2D copies complete
        stream.synchronize()?;

        Ok(())
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
    /// Uses the shared dispatch helpers with FLA closures for the WKV
    /// kernel call. State is GPU-resident in scratch and updated in-place.
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

        // Build ProbeState if probes are attached
        #[cfg(feature = "hip-probes")]
        let probes_ref = self.probes.as_ref();

        #[cfg(feature = "hip-probes")]
        let mut probe_state: Option<ProbeState<'_>> = probes_ref.map(|p| ProbeState {
            probes: p,
            ctx: &mut probe_ctx,
            t_stride: t,
        });

        #[cfg(not(feature = "hip-probes"))]
        let mut probe_state: Option<ProbeState<'_>> = None;

        // Convert lens to i32 tensor for masked kernel
        let lens_i32: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
        let lens_shape = TensorShape::new(b, 1, 1, 1);
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        lens_gpu.copy_from_slice(&lens_i32, stream)?;

        // Rectangular batch_offsets: [0, T, 2T, ...]
        let offsets_i32: Vec<i32> = (0..b).map(|i| (i * t) as i32).collect();
        let batch_offsets_gpu = TensorHip::from_slice(&offsets_i32, lens_shape, stream)?;

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
        // Note: w_decay is computed inside attention_block (decay_exp_f16) but
        // FLA ignores it and uses raw att_w instead. The cost is negligible.
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

        // Embedding lookup + ln0
        dispatch_helpers::embed_lookup(
            tokens,
            &self.model.embed,
            n_embd,
            &mut scratch.emb_staging,
            &mut x,
            &mut x_ln,
            stream,
            &mut probe_state,
        )?;

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

                // Update probe layer context
                #[cfg(feature = "hip-probes")]
                {
                    if let Some(ref mut ps) = probe_state {
                        ps.ctx.layer = Some(layer_idx);
                    }
                }

                // Attention block with FLA closure
                dispatch_helpers::attention_block(
                    layer_idx,
                    layer,
                    &mut x,
                    &mut x_ln,
                    &mut att_xr,
                    &mut att_xw,
                    &mut att_xk,
                    &mut att_xv,
                    &mut att_xa,
                    &mut att_xg,
                    &mut att_shift_gpu[layer_idx],
                    &mut new_att_shift,
                    &mut att_r,
                    &mut att_k,
                    &mut att_v,
                    &mut att_w,
                    &mut att_a,
                    &mut att_g,
                    &mut att_kk,
                    &mut att_k_ctrl,
                    &mut wkv_a,
                    &mut wkv_b,
                    &mut w_decay,
                    &mut wkv_out,
                    &mut wkv_normed,
                    &mut wkv_bonus,
                    &mut att_out,
                    &mut v_first,
                    &mut lora_w,
                    &mut lora_w_tanh,
                    &mut lora_a,
                    &mut lora_a_proj,
                    &mut lora_g,
                    &mut lora_g_sig,
                    &mut lora_v,
                    &mut v_lora2,
                    &mut temp1,
                    &mut temp2,
                    &lens_gpu,
                    &batch_offsets_gpu,
                    &mut wkv_state_gpu[layer_idx],
                    head_size,
                    n_head,
                    wkv_data_shape,
                    ctx,
                    stream,
                    // FLA closure: uses raw att_w (pre-exponentiation) for better precision
                    |inputs, wkv_state, wkv_out_wkv| {
                        fla_kernel.compute(
                            inputs.att_w_wkv,
                            inputs.r_wkv,
                            inputs.k_ctrl_wkv,
                            inputs.v_wkv,
                            inputs.wkv_a_wkv,
                            inputs.wkv_b_wkv,
                            wkv_state,
                            wkv_out_wkv,
                            lens,
                            stream,
                        )
                    },
                    &mut probe_state,
                )?;

                // FFN block
                dispatch_helpers::ffn_block(
                    layer,
                    n_embd,
                    &mut x,
                    &mut x_ln,
                    &mut ffn_xk,
                    &mut ffn_shift_gpu[layer_idx],
                    &mut new_ffn_shift,
                    &mut ffn_k,
                    &mut ffn_k_sq,
                    &mut ffn_out,
                    &mut temp1,
                    &lens_gpu,
                    &batch_offsets_gpu,
                    ctx,
                    stream,
                    &mut probe_state,
                )?;
            }

            // Reset probe layer context for output head
            #[cfg(feature = "hip-probes")]
            {
                if let Some(ref mut ps) = probe_state {
                    ps.ctx.layer = None;
                }
            }

            // Output head
            dispatch_helpers::output_head(
                &self.model.head,
                n_embd,
                n_vocab,
                &x,
                &mut x_ln,
                &mut logits,
                &mut logits_f32,
                &mut scratch.logits_staging,
                ctx,
                stream,
                &mut probe_state,
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
