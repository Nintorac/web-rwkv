//! State types for HIP RWKV7 inference.

use half::f16;

use super::Rwkv7ModelInfo;
use crate::hip::device::Event;
use crate::hip::device::Stream;
use crate::hip::ffi::Result;
use crate::hip::pinned::PinnedBuffer;

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
    }
}

/// Completion handle for an asynchronous step (inference) pass.
///
/// This struct is returned by `step()` and allows the caller to:
/// - Check if the GPU computation is complete without blocking (`is_ready()`)
/// - Wait for completion and retrieve results (`wait()`)
///
/// The async step enables overlapping GPU computation with CPU work:
///
/// ```ignore
/// let completion = model.step(&[&tokens], None)?;
/// // Do CPU work while GPU computes...
/// let (logits, state) = completion.wait()?;
/// ```
#[allow(dead_code)]
pub struct ForwardCompletion {
    /// Event that signals when GPU work is complete
    pub(super) event: Event,
    /// Stream the work was submitted on
    pub(super) stream: Stream,
    /// Pre-allocated buffer for logits (D->H copy is queued but not complete)
    /// Uses f32 - GPU does f16->f32 conversion before download
    pub(super) logits_buffer: PinnedBuffer<f32>,
    /// Pre-allocated buffers for state (D->H copies are queued but not complete)
    pub(super) state_buffers: ForwardStateBuffers,
    /// Model info for reconstructing HipState
    pub(super) n_layer: usize,
    pub(super) batch_size: usize,
    /// Sequence lengths for extracting real tokens from padded output
    pub(super) lens: Vec<usize>,
    /// Padded chunk size
    pub(super) chunk_size: usize,
    /// Vocabulary size
    pub(super) n_vocab: usize,
}

/// Internal buffers for async state download
pub(super) struct ForwardStateBuffers {
    pub(super) att_states: Vec<PinnedBuffer<f32>>,
    pub(super) att_shift_states: Vec<PinnedBuffer<f16>>,
    pub(super) ffn_states: Vec<PinnedBuffer<f16>>,
    pub(super) v_first: Option<PinnedBuffer<f16>>,
}

impl ForwardCompletion {
    /// Check if the GPU computation has completed (non-blocking).
    ///
    /// Returns `Ok(true)` if all GPU work and data transfers are done,
    /// `Ok(false)` if still in progress.
    pub fn is_ready(&self) -> Result<bool> {
        self.event.query()
    }

    /// Wait for completion and return the results.
    ///
    /// This blocks until all GPU work and data transfers are complete,
    /// then returns the logits and updated state.
    pub fn wait(self) -> Result<(Vec<f32>, HipState)> {
        // Block until all GPU work is done
        self.event.synchronize()?;

        // Extract only real tokens from padded output
        // Layout: [n_vocab, chunk_size, batch_size] column-major
        // For batch b, token t: offset = (b * chunk_size + t) * n_vocab
        // Data is already f32 (GPU did f16->f32 conversion before download)
        let padded = self.logits_buffer.as_slice();
        let mut logits = Vec::new();
        for (b, &real_len) in self.lens.iter().enumerate() {
            for t in 0..real_len {
                let offset = (b * self.chunk_size + t) * self.n_vocab;
                logits.extend_from_slice(&padded[offset..offset + self.n_vocab]);
            }
        }

        let state = HipState {
            batch_size: self.batch_size,
            att_states: self.state_buffers.att_states,
            att_shift_states: self.state_buffers.att_shift_states,
            ffn_states: self.state_buffers.ffn_states,
            v_first: self.state_buffers.v_first,
        };

        Ok((logits, state))
    }
}
