//! State types for HIP RWKV7 inference.

use half::f16;

use super::Rwkv7ModelInfo;
use crate::hip::device::Stream;
use crate::hip::ffi::{check, hip_memcpy_d2h, hip_memcpy_h2d, HipErrorKind, Result};
use crate::hip::pinned::PinnedBuffer;
use crate::hip::tensor::TensorHip;

/// Layout tag for WKV state matrices.
///
/// FLA (prefill) stores WKV state as `state[k * K + v]` (K-rows, V-cols).
/// FusedT1Wkv (decode) reads `state[v * K + k]` (V-rows, K-cols).
///
/// When transferring state between prefill and decode modules, the WKV state
/// must be transposed if the layouts differ. The `layout` field on [`HipState`]
/// tracks which layout the state is currently in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateLayout {
    /// FLA layout: `state[k * K + v]` — K-rows, V-cols.
    /// Used by the chunked FLA prefill kernel.
    Fla,
    /// Decode layout: `state[v * K + k]` — V-rows, K-cols.
    /// Used by the FusedT1Wkv single-token decode kernel.
    Decode,
}

/// State for HIP RWKV7 batched inference.
///
/// Holds the recurrent state needed to continue inference from a previous position.
/// Supports batch sizes > 1 for high-throughput inference.
///
/// # State Layout
/// All state tensors include a batch dimension:
/// - `att_states`: WKV recurrent state `[n_layer][head_size * head_size * n_head * batch]`
/// - `att_shift_states`: Attention token shift state `[n_layer][n_embd * batch]`
/// - `ffn_states`: FFN token shift state `[n_layer][n_embd * batch]`
///
/// # Usage
/// ```ignore
/// // Batched inference (B=4 sequences)
/// let tokens: Vec<&[u32]> = vec![&seq1, &seq2, &seq3, &seq4];
/// let (logits, state) = model.step(&tokens, None)?;
///
/// // Streaming with batching: process one token per sequence
/// let next_tokens: Vec<&[u32]> = vec![&[t1], &[t2], &[t3], &[t4]];
/// let (logits, state) = model.step(&next_tokens, Some(state))?;
///
/// // Single sequence (B=1) for simple use cases
/// let (logits, state) = model.step(&[&tokens], None)?;
/// ```
#[derive(Debug, Clone)]
pub struct HipState {
    /// Batch size this state was created for
    pub batch_size: usize,
    /// WKV state per layer: [head_size * head_size * n_head * batch] per layer
    /// Stored in pinned memory for fast GPU transfers.
    pub att_states: Vec<PinnedBuffer<f32>>,
    /// Attention token shift state per layer: [n_embd * batch] per layer
    /// Stored in pinned memory for fast GPU transfers.
    pub att_shift_states: Vec<PinnedBuffer<f16>>,
    /// FFN token shift state per layer: [n_embd * batch] per layer
    /// Stored in pinned memory for fast GPU transfers.
    pub ffn_states: Vec<PinnedBuffer<f16>>,
    /// Value residual from first layer for RWKV7, persisted across chunks: [n_embd * batch]
    /// Stored in pinned memory for fast GPU transfers.
    pub v_first: Option<PinnedBuffer<f16>>,
    /// Layout of the WKV state matrices.
    ///
    /// Tracks whether `att_states` is in FLA layout (K-rows, V-cols) or
    /// decode layout (V-rows, K-cols). Used by `load_state()` to determine
    /// whether a transpose is needed.
    pub layout: StateLayout,
}

impl HipState {
    /// Create a fresh state initialized to zeros.
    ///
    /// # Arguments
    /// * `info` - Model info containing dimensions
    /// * `batch_size` - Number of sequences to process in parallel
    ///
    /// # Errors
    /// Returns error if pinned memory allocation fails.
    pub fn new(info: &Rwkv7ModelInfo, batch_size: usize) -> Result<Self> {
        let n_layer = info.n_layer;
        let n_embd = info.n_embd;
        let head_size = info.head_size;
        let n_head = info.n_head;

        // Allocate pinned buffers for each layer
        let mut att_states = Vec::with_capacity(n_layer);
        let mut att_shift_states = Vec::with_capacity(n_layer);
        let mut ffn_states = Vec::with_capacity(n_layer);

        for _ in 0..n_layer {
            let mut att = PinnedBuffer::new(head_size * head_size * n_head * batch_size)?;
            att.as_slice_mut().fill(0.0);
            att_states.push(att);

            let mut att_shift = PinnedBuffer::new(n_embd * batch_size)?;
            att_shift.as_slice_mut().fill(f16::from_f32(0.0));
            att_shift_states.push(att_shift);

            let mut ffn = PinnedBuffer::new(n_embd * batch_size)?;
            ffn.as_slice_mut().fill(f16::from_f32(0.0));
            ffn_states.push(ffn);
        }

        Ok(HipState {
            batch_size,
            att_states,
            att_shift_states,
            ffn_states,
            v_first: None,
            layout: StateLayout::Decode,
        })
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        for state in &mut self.att_states {
            state.as_slice_mut().fill(0.0);
        }
        for state in &mut self.att_shift_states {
            state.as_slice_mut().fill(f16::from_f32(0.0));
        }
        for state in &mut self.ffn_states {
            state.as_slice_mut().fill(f16::from_f32(0.0));
        }
        self.v_first = None;
        self.layout = StateLayout::Decode;
    }

    // ========== Per-batch GPU state read/write ==========

    /// Read WKV state for a single batch item from GPU to pinned buffers.
    ///
    /// Returns one `PinnedBuffer<f32>` per layer, each of size
    /// `head_size * head_size * n_head` (elements for one batch item).
    ///
    /// The GPU state layout is `[head_size, head_size, n_head, batch_size]`
    /// in column-major order, so elements per batch = `head_size * head_size * n_head`.
    ///
    /// # Arguments
    /// * `gpu_states` - Per-layer GPU WKV state tensors
    /// * `batch_idx` - Index of the batch item to read
    /// * `batch_size` - Total batch size
    /// * `stream` - HIP stream for async copy
    pub fn read_batch_wkv(
        gpu_states: &[TensorHip<f32>],
        batch_idx: usize,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Vec<PinnedBuffer<f32>>> {
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }

        let mut buffers = Vec::with_capacity(gpu_states.len());
        for gpu_state in gpu_states {
            let total_elems = gpu_state.shape().len();
            let elems_per_batch = total_elems / batch_size;
            let byte_offset = batch_idx * elems_per_batch * std::mem::size_of::<f32>();
            let byte_count = elems_per_batch * std::mem::size_of::<f32>();

            let mut buf = PinnedBuffer::<f32>::new(elems_per_batch)?;
            unsafe {
                let src = (gpu_state.as_ptr() as *const u8).add(byte_offset);
                check(hip_memcpy_d2h(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    byte_count,
                    stream.handle(),
                ))?;
            }
            buffers.push(buf);
        }
        Ok(buffers)
    }

    /// Write WKV state for a single batch item from pinned buffers to GPU.
    ///
    /// Each buffer should contain `head_size * head_size * n_head` f32 elements.
    ///
    /// # Arguments
    /// * `gpu_states` - Per-layer GPU WKV state tensors
    /// * `batch_idx` - Index of the batch item to write
    /// * `batch_size` - Total batch size
    /// * `buffers` - Per-layer pinned buffers with the state data
    /// * `stream` - HIP stream for async copy
    pub fn write_batch_wkv(
        gpu_states: &mut [TensorHip<f32>],
        batch_idx: usize,
        batch_size: usize,
        buffers: &[PinnedBuffer<f32>],
        stream: &Stream,
    ) -> Result<()> {
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }
        if buffers.len() != gpu_states.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "buffer count {} != gpu_state count {}",
                    buffers.len(),
                    gpu_states.len()
                ),
            });
        }

        for (gpu_state, buf) in gpu_states.iter_mut().zip(buffers.iter()) {
            let total_elems = gpu_state.shape().len();
            let elems_per_batch = total_elems / batch_size;
            let byte_offset = batch_idx * elems_per_batch * std::mem::size_of::<f32>();
            let byte_count = elems_per_batch * std::mem::size_of::<f32>();

            if buf.len() != elems_per_batch {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "buffer len {} != expected elems_per_batch {}",
                        buf.len(),
                        elems_per_batch
                    ),
                });
            }

            unsafe {
                let dst = (gpu_state.as_mut_ptr() as *mut u8).add(byte_offset);
                check(hip_memcpy_h2d(
                    dst as *mut std::ffi::c_void,
                    buf.as_ptr() as *const std::ffi::c_void,
                    byte_count,
                    stream.handle(),
                ))?;
            }
        }
        Ok(())
    }

    /// Read attention shift state for a single batch item from GPU to pinned buffers.
    ///
    /// Returns one `PinnedBuffer<f16>` per layer, each of size `n_embd`.
    /// The GPU state layout is `[n_embd, batch_size]` in column-major order.
    ///
    /// # Arguments
    /// * `gpu_states` - Per-layer GPU attention shift state tensors
    /// * `batch_idx` - Index of the batch item to read
    /// * `batch_size` - Total batch size
    /// * `stream` - HIP stream for async copy
    pub fn read_batch_att_shift(
        gpu_states: &[TensorHip<f16>],
        batch_idx: usize,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Vec<PinnedBuffer<f16>>> {
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }

        let mut buffers = Vec::with_capacity(gpu_states.len());
        for gpu_state in gpu_states {
            let total_elems = gpu_state.shape().len();
            let elems_per_batch = total_elems / batch_size;
            let byte_offset = batch_idx * elems_per_batch * std::mem::size_of::<f16>();
            let byte_count = elems_per_batch * std::mem::size_of::<f16>();

            let mut buf = PinnedBuffer::<f16>::new(elems_per_batch)?;
            unsafe {
                let src = (gpu_state.as_ptr() as *const u8).add(byte_offset);
                check(hip_memcpy_d2h(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    byte_count,
                    stream.handle(),
                ))?;
            }
            buffers.push(buf);
        }
        Ok(buffers)
    }

    /// Write attention shift state for a single batch item from pinned buffers to GPU.
    ///
    /// Each buffer should contain `n_embd` f16 elements.
    pub fn write_batch_att_shift(
        gpu_states: &mut [TensorHip<f16>],
        batch_idx: usize,
        batch_size: usize,
        buffers: &[PinnedBuffer<f16>],
        stream: &Stream,
    ) -> Result<()> {
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }
        if buffers.len() != gpu_states.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "buffer count {} != gpu_state count {}",
                    buffers.len(),
                    gpu_states.len()
                ),
            });
        }

        for (gpu_state, buf) in gpu_states.iter_mut().zip(buffers.iter()) {
            let total_elems = gpu_state.shape().len();
            let elems_per_batch = total_elems / batch_size;
            let byte_offset = batch_idx * elems_per_batch * std::mem::size_of::<f16>();
            let byte_count = elems_per_batch * std::mem::size_of::<f16>();

            if buf.len() != elems_per_batch {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "buffer len {} != expected elems_per_batch {}",
                        buf.len(),
                        elems_per_batch
                    ),
                });
            }

            unsafe {
                let dst = (gpu_state.as_mut_ptr() as *mut u8).add(byte_offset);
                check(hip_memcpy_h2d(
                    dst as *mut std::ffi::c_void,
                    buf.as_ptr() as *const std::ffi::c_void,
                    byte_count,
                    stream.handle(),
                ))?;
            }
        }
        Ok(())
    }

    /// Read FFN state for a single batch item from GPU to pinned buffers.
    ///
    /// Returns one `PinnedBuffer<f16>` per layer, each of size `n_embd`.
    /// The GPU state layout is `[n_embd, batch_size]` in column-major order.
    pub fn read_batch_ffn(
        gpu_states: &[TensorHip<f16>],
        batch_idx: usize,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Vec<PinnedBuffer<f16>>> {
        // Same layout as att_shift: [n_embd, batch_size]
        Self::read_batch_att_shift(gpu_states, batch_idx, batch_size, stream)
    }

    /// Write FFN state for a single batch item from pinned buffers to GPU.
    ///
    /// Each buffer should contain `n_embd` f16 elements.
    pub fn write_batch_ffn(
        gpu_states: &mut [TensorHip<f16>],
        batch_idx: usize,
        batch_size: usize,
        buffers: &[PinnedBuffer<f16>],
        stream: &Stream,
    ) -> Result<()> {
        // Same layout as att_shift: [n_embd, batch_size]
        Self::write_batch_att_shift(gpu_states, batch_idx, batch_size, buffers, stream)
    }
}
