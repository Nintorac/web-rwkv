//! RWKV7 HIP model loading and forward pass implementation.

use half::f16;
use std::sync::Mutex;
use std::path::Path;

use super::ffi::{HipErrorKind, Result};
use super::device::{Stream, Event};
use super::pinned::PinnedBuffer;
use super::tensor::{TensorShape, TensorHip};
use super::scratch::{LoraDims, HipRuntimeConfig};
use super::kernels::{
    // GPU-native kernels used in forward
    layer_norm_f32, group_norm_f32, l2_norm_f32,
    sigmoid_f32, tanh_f32, softplus_decay_f32, squared_relu_f32,
    channel_mix_state_f32, control_k_f32, wkv7_f32_masked, wkv_bonus_f32,
    add_f32, mul_f32, negate_f32, exp_f32, broadcast_add_f32, broadcast_mul_f32,
    lerp_f32, copy_tensor_f32,
    // CPU helper for masked state extraction
    extract_shift_state_at_lengths,
};
use super::blas::HipBlasContext;
use super::scratch::HipScratch;

#[cfg(feature = "hip-probes")]
use crate::hip_probe;

#[cfg(feature = "hip-probes")]
use super::probe::{self, HipProbeMap, HipProbeMapRef};

// ============================================================================
// Model Loading for RWKV7 HIP Backend
// ============================================================================


/// Information about a loaded RWKV7 model.
#[derive(Debug, Clone)]
pub struct Rwkv7ModelInfo {
    /// Number of transformer layers
    pub n_layer: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Head size (n_embd / n_head)
    pub head_size: usize,
    /// Vocabulary size
    pub n_vocab: usize,
    /// Hidden dimension for FFN
    pub n_hidden: usize,
}

/// A single layer's layer normalization weights.
#[derive(Debug)]
pub struct LayerNormHip {
    pub weight: TensorHip<f32>,
    pub bias: TensorHip<f32>,
}

/// Attention weights for a single layer.
#[derive(Debug)]
pub struct AttentionHip {
    // Token shift mix weights
    pub x_r: TensorHip<f32>,
    pub x_w: TensorHip<f32>,
    pub x_k: TensorHip<f32>,
    pub x_v: TensorHip<f32>,
    pub x_a: TensorHip<f32>,
    pub x_g: TensorHip<f32>,

    // Decay LoRA
    pub w0: TensorHip<f32>,
    pub w1: TensorHip<f32>,
    pub w2: TensorHip<f32>,

    // Learning rate LoRA
    pub a0: TensorHip<f32>,
    pub a1: TensorHip<f32>,
    pub a2: TensorHip<f32>,

    // Gate LoRA
    pub g1: TensorHip<f32>,
    pub g2: TensorHip<f32>,

    // Value residual LoRA (layers > 0)
    pub v0: Option<TensorHip<f32>>,
    pub v1: Option<TensorHip<f32>>,
    pub v2: Option<TensorHip<f32>>,

    // Key normalization weights
    pub r_k: TensorHip<f32>,
    pub k_k: TensorHip<f32>,
    pub k_a: TensorHip<f32>,

    // Projection matrices (column-major for rocBLAS)
    pub w_r: TensorHip<f32>,  // Receptance: [n_embd, n_embd]
    pub w_k: TensorHip<f32>,  // Key: [n_embd, n_embd]
    pub w_v: TensorHip<f32>,  // Value: [n_embd, n_embd]
    pub w_o: TensorHip<f32>,  // Output: [n_embd, n_embd]

    // Group normalization
    pub gn: LayerNormHip,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnHip {
    // Token shift mix weight
    pub x_k: TensorHip<f32>,

    // Projection matrices
    pub w_k: TensorHip<f32>,  // Key (expand): [n_hidden, n_embd]
    pub w_v: TensorHip<f32>,  // Value (contract): [n_embd, n_hidden]
}

/// A single transformer layer's weights.
#[derive(Debug)]
pub struct LayerHip {
    pub att_ln: LayerNormHip,
    pub ffn_ln: LayerNormHip,
    pub att: AttentionHip,
    pub ffn: FfnHip,
}

/// Embedding weights.
#[derive(Debug)]
pub struct EmbedHip {
    pub ln: LayerNormHip,
    pub w: TensorHip<f32>,  // [n_vocab, n_embd]
}

/// Output head weights.
#[derive(Debug)]
pub struct HeadHip {
    pub ln: LayerNormHip,
    pub w: TensorHip<f32>,  // [n_vocab, n_embd]
}

/// RWKV7 model loaded into HIP memory.
///
/// Weights are stored in managed memory for zero-copy APU access.
/// All tensors use FP32 internally (converted from FP16 at load time).
pub struct Rwkv7Hip {
    pub info: Rwkv7ModelInfo,
    pub embed: EmbedHip,
    pub head: HeadHip,
    pub layers: Vec<LayerHip>,

    /// Lazily initialized scratch buffers for GPU-native forward pass.
    scratch: Mutex<Option<HipScratch>>,

    #[cfg(feature = "hip-probes")]
    pub(crate) probes: Option<HipProbeMapRef>,
}

impl std::fmt::Debug for Rwkv7Hip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Rwkv7Hip");
        s.field("info", &self.info)
            .field("embed", &self.embed)
            .field("head", &self.head)
            .field("layers", &self.layers)
            .field("scratch", &self.scratch.lock().unwrap().as_ref().map(|_| "initialized"));
        #[cfg(feature = "hip-probes")]
        s.field("probes", &self.probes.as_ref().map(|p| format!("{} hooks", p.len())));
        s.finish()
    }
}

/// Error type for model loading.
#[derive(Debug)]
pub enum ModelLoadError {
    Hip(HipErrorKind),
    Io(std::io::Error),
    SafeTensor(String),
    InvalidModel(String),
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::Hip(e) => write!(f, "HIP error: {}", e),
            ModelLoadError::Io(e) => write!(f, "IO error: {}", e),
            ModelLoadError::SafeTensor(e) => write!(f, "SafeTensor error: {}", e),
            ModelLoadError::InvalidModel(e) => write!(f, "Invalid model: {}", e),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<HipErrorKind> for ModelLoadError {
    fn from(e: HipErrorKind) -> Self {
        ModelLoadError::Hip(e)
    }
}

impl From<std::io::Error> for ModelLoadError {
    fn from(e: std::io::Error) -> Self {
        ModelLoadError::Io(e)
    }
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
/// let mut state = HipState::new(&model.info, 4);
/// let tokens: Vec<&[u32]> = vec![&seq1, &seq2, &seq3, &seq4];
/// let logits = model.forward_with_state(&tokens, &mut state)?;
///
/// // Streaming with batching: process one token per sequence
/// let next_tokens: Vec<&[u32]> = vec![&[t1], &[t2], &[t3], &[t4]];
/// let logits = model.forward_with_state(&next_tokens, &mut state)?;
///
/// // Single sequence (B=1) for simple use cases
/// let mut state = HipState::new(&model.info, 1);
/// let logits = model.forward_with_state(&[&tokens], &mut state)?;
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
    pub att_shift_states: Vec<PinnedBuffer<f32>>,
    /// FFN token shift state per layer: [n_embd * batch] per layer
    /// Stored in pinned memory for fast GPU transfers.
    pub ffn_states: Vec<PinnedBuffer<f32>>,
    /// Value residual from first layer for RWKV7, persisted across chunks: [n_embd * batch]
    /// Stored in pinned memory for fast GPU transfers.
    pub v_first: Option<PinnedBuffer<f32>>,
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
            att_shift.as_slice_mut().fill(0.0);
            att_shift_states.push(att_shift);

            let mut ffn = PinnedBuffer::new(n_embd * batch_size)?;
            ffn.as_slice_mut().fill(0.0);
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
            state.as_slice_mut().fill(0.0);
        }
        for state in &mut self.ffn_states {
            state.as_slice_mut().fill(0.0);
        }
        self.v_first = None;
    }
}

/// Completion handle for an asynchronous forward pass.
///
/// This struct is returned by `forward_async()` and allows the caller to:
/// - Check if the GPU computation is complete without blocking (`is_ready()`)
/// - Wait for completion and retrieve results (`wait()`)
///
/// The async forward enables overlapping GPU computation with CPU work:
///
/// ```ignore
/// let completion = model.forward_async(&[&tokens], None)?;
/// // Do CPU work while GPU computes...
/// let (logits, state) = completion.wait()?;
/// ```
#[allow(dead_code)]
pub struct ForwardCompletion {
    /// Event that signals when GPU work is complete
    event: Event,
    /// Stream the work was submitted on
    stream: Stream,
    /// Pre-allocated buffer for logits (D→H copy is queued but not complete)
    logits_buffer: Vec<f32>,
    /// Pre-allocated buffers for state (D→H copies are queued but not complete)
    state_buffers: ForwardStateBuffers,
    /// Model info for reconstructing HipState
    n_layer: usize,
    batch_size: usize,
}

/// Internal buffers for async state download
struct ForwardStateBuffers {
    att_states: Vec<PinnedBuffer<f32>>,
    att_shift_states: Vec<PinnedBuffer<f32>>,
    ffn_states: Vec<PinnedBuffer<f32>>,
    v_first: Option<PinnedBuffer<f32>>,
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

        // Now the buffers are safe to read
        let state = HipState {
            batch_size: self.batch_size,
            att_states: self.state_buffers.att_states,
            att_shift_states: self.state_buffers.att_shift_states,
            ffn_states: self.state_buffers.ffn_states,
            v_first: self.state_buffers.v_first,
        };

        Ok((self.logits_buffer, state))
    }
}

/// Transpose a 2D matrix from row-major to column-major layout.
///
/// Row-major [M, K]: element (i, j) at index i * K + j
/// Column-major [M, K]: element (i, j) at index j * M + i
///
/// This is used at load time to convert SafeTensors (row-major) weights
/// to rocBLAS-native column-major format, per the plan:
/// "Use rocBLAS-native column-major storage for GEMM/GEMV...
///  This avoids per-call row/col mapping in rocBLAS"
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            // row-major index: i * cols + j
            // column-major index: j * rows + i
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

/// Load a tensor from SafeTensors, converting f16 to f32 and loading into managed HIP memory.
fn load_tensor_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f32>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    // Convert f16 bytes to f32 vec
    let f32_data: Vec<f32> = match dtype {
        safetensors::Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        safetensors::Dtype::F32 => {
            bytemuck::cast_slice(data).to_vec()
        }
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    // Convert shape to TensorShape (web-rwkv convention: shape[0] is fastest axis)
    // SafeTensors stores shape as [slow, ..., fast], so we need to reverse
    let hip_shape = match shape_st.len() {
        1 => TensorShape::new(shape_st[0], 1, 1, 1),
        2 => TensorShape::new(shape_st[1], shape_st[0], 1, 1),
        3 => TensorShape::new(shape_st[2], shape_st[1], shape_st[0], 1),
        4 => TensorShape::new(shape_st[3], shape_st[2], shape_st[1], shape_st[0]),
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported shape {:?} for tensor {}", shape_st, name
        ))),
    };

    TensorHip::from_slice_managed(&f32_data, hip_shape, stream).map_err(ModelLoadError::from)
}

/// Load a weight matrix from SafeTensors, transposing to column-major for rocBLAS.
///
/// SafeTensors stores weights as row-major [out_features, in_features].
/// rocBLAS requires column-major storage for efficient GEMM.
/// This function transposes at load time to avoid per-inference transpose.
///
/// Per docs/RWKV7_HIP_BACKEND_PLAN.md:
/// "Use rocBLAS-native column-major storage for GEMM/GEMV...
///  This avoids per-call row/col mapping in rocBLAS"
///
/// Shape is stored as [M, K] where M=out_features, K=in_features.
/// Use dim(0) to get M (out_features) and dim(1) to get K (in_features).
fn load_weight_matrix_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f32>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    if shape_st.len() != 2 {
        return Err(ModelLoadError::InvalidModel(format!(
            "Weight matrix {} must be 2D, got {:?}", name, shape_st
        )));
    }

    let rows = shape_st[0];  // out_features (M)
    let cols = shape_st[1];  // in_features (K)

    // Convert f16 bytes to f32 vec
    let f32_data: Vec<f32> = match dtype {
        safetensors::Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        safetensors::Dtype::F32 => {
            bytemuck::cast_slice(data).to_vec()
        }
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    // Transpose from row-major to column-major
    let transposed = transpose_2d(&f32_data, rows, cols);

    // Shape is [M, K] where M=rows (out_features), K=cols (in_features)
    // This is the natural column-major representation for rocBLAS
    // Note: This differs from load_tensor_f32 which reverses dimensions
    let hip_shape = TensorShape::new(rows, cols, 1, 1);

    TensorHip::from_slice_managed(&transposed, hip_shape, stream).map_err(ModelLoadError::from)
}

/// Load layer normalization weights.
fn load_layer_norm(
    st: &safetensors::SafeTensors,
    prefix: &str,
    stream: &Stream,
) -> std::result::Result<LayerNormHip, ModelLoadError> {
    let weight = load_tensor_f32(st, &format!("{}.weight", prefix), stream)?;
    let bias = load_tensor_f32(st, &format!("{}.bias", prefix), stream)?;
    Ok(LayerNormHip { weight, bias })
}

impl Rwkv7Hip {
    /// Load an RWKV7 model from a SafeTensors file.
    ///
    /// Weights are loaded into managed (unified) memory for efficient APU access.
    /// All weights are converted to FP32 for computation.
    ///
    /// # Arguments
    /// * `path` - Path to the .st (SafeTensors) file
    ///
    /// # Returns
    /// The loaded model with all weights in HIP memory, or an error.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, ModelLoadError> {
        let data = std::fs::read(path.as_ref())?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| ModelLoadError::SafeTensor(format!("Failed to parse SafeTensors: {}", e)))?;

        let stream = Stream::null();

        // Detect model dimensions from tensor shapes
        let embed_tensor = st.tensor("emb.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("emb.weight: {}", e)))?;
        let embed_shape = embed_tensor.shape();  // [n_vocab, n_embd]
        let n_vocab = embed_shape[0];
        let n_embd = embed_shape[1];

        // Get n_head from r_k tensor (RWKV7-specific)
        let r_k_tensor = st.tensor("blocks.0.att.r_k")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.att.r_k: {}", e)))?;
        let n_head = r_k_tensor.shape()[0];
        let head_size = n_embd / n_head;

        // Get n_hidden from FFN key weight
        let ffn_k_tensor = st.tensor("blocks.0.ffn.key.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.ffn.key.weight: {}", e)))?;
        let n_hidden = ffn_k_tensor.shape()[0];

        // Count layers
        let n_layer = st.names().iter()
            .filter_map(|name| {
                if name.starts_with("blocks.") {
                    let rest = name.strip_prefix("blocks.")?;
                    let layer_num: usize = rest.split('.').next()?.parse().ok()?;
                    Some(layer_num + 1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);

        if n_layer == 0 {
            return Err(ModelLoadError::InvalidModel("No layers found".to_string()));
        }

        let info = Rwkv7ModelInfo {
            n_layer,
            n_embd,
            n_head,
            head_size,
            n_vocab,
            n_hidden,
        };

        log::info!("Loading RWKV7 model: {} layers, {} embd, {} heads, {} vocab",
            n_layer, n_embd, n_head, n_vocab);

        // Load embedding (no transpose - it's a lookup table, not GEMM)
        let embed = EmbedHip {
            ln: load_layer_norm(&st, "blocks.0.ln0", &stream)?,
            w: load_tensor_f32(&st, "emb.weight", &stream)?,
        };

        // Load output head (transpose to column-major for GEMM)
        let head = HeadHip {
            ln: load_layer_norm(&st, "ln_out", &stream)?,
            w: load_weight_matrix_f32(&st, "head.weight", &stream)?,
        };

        // Load layers
        let mut layers = Vec::with_capacity(n_layer);
        for layer_idx in 0..n_layer {
            let prefix = format!("blocks.{}", layer_idx);

            // Attention layer norm
            let att_ln = load_layer_norm(&st, &format!("{}.ln1", prefix), &stream)?;

            // FFN layer norm
            let ffn_ln = load_layer_norm(&st, &format!("{}.ln2", prefix), &stream)?;

            // Attention weights
            let att = AttentionHip {
                x_r: load_tensor_f32(&st, &format!("{}.att.x_r", prefix), &stream)?,
                x_w: load_tensor_f32(&st, &format!("{}.att.x_w", prefix), &stream)?,
                x_k: load_tensor_f32(&st, &format!("{}.att.x_k", prefix), &stream)?,
                x_v: load_tensor_f32(&st, &format!("{}.att.x_v", prefix), &stream)?,
                x_a: load_tensor_f32(&st, &format!("{}.att.x_a", prefix), &stream)?,
                x_g: load_tensor_f32(&st, &format!("{}.att.x_g", prefix), &stream)?,

                w0: load_tensor_f32(&st, &format!("{}.att.w0", prefix), &stream)?,
                w1: load_weight_matrix_f32(&st, &format!("{}.att.w1", prefix), &stream)?,
                w2: load_weight_matrix_f32(&st, &format!("{}.att.w2", prefix), &stream)?,

                a0: load_tensor_f32(&st, &format!("{}.att.a0", prefix), &stream)?,
                a1: load_weight_matrix_f32(&st, &format!("{}.att.a1", prefix), &stream)?,
                a2: load_weight_matrix_f32(&st, &format!("{}.att.a2", prefix), &stream)?,

                g1: load_weight_matrix_f32(&st, &format!("{}.att.g1", prefix), &stream)?,
                g2: load_weight_matrix_f32(&st, &format!("{}.att.g2", prefix), &stream)?,

                // Value residual LoRA (only for layers > 0)
                v0: if layer_idx > 0 {
                    Some(load_tensor_f32(&st, &format!("{}.att.v0", prefix), &stream)?)
                } else {
                    None
                },
                v1: if layer_idx > 0 {
                    Some(load_weight_matrix_f32(&st, &format!("{}.att.v1", prefix), &stream)?)
                } else {
                    None
                },
                v2: if layer_idx > 0 {
                    Some(load_weight_matrix_f32(&st, &format!("{}.att.v2", prefix), &stream)?)
                } else {
                    None
                },

                r_k: load_tensor_f32(&st, &format!("{}.att.r_k", prefix), &stream)?,
                k_k: load_tensor_f32(&st, &format!("{}.att.k_k", prefix), &stream)?,
                k_a: load_tensor_f32(&st, &format!("{}.att.k_a", prefix), &stream)?,

                w_r: load_weight_matrix_f32(&st, &format!("{}.att.receptance.weight", prefix), &stream)?,
                w_k: load_weight_matrix_f32(&st, &format!("{}.att.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f32(&st, &format!("{}.att.value.weight", prefix), &stream)?,
                w_o: load_weight_matrix_f32(&st, &format!("{}.att.output.weight", prefix), &stream)?,

                gn: load_layer_norm(&st, &format!("{}.att.ln_x", prefix), &stream)?,
            };

            // FFN weights (transpose weight matrices to column-major for GEMM)
            let ffn = FfnHip {
                x_k: load_tensor_f32(&st, &format!("{}.ffn.x_k", prefix), &stream)?,
                w_k: load_weight_matrix_f32(&st, &format!("{}.ffn.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f32(&st, &format!("{}.ffn.value.weight", prefix), &stream)?,
            };

            layers.push(LayerHip { att_ln, ffn_ln, att, ffn });
        }

        // Synchronize to ensure all transfers are complete
        stream.synchronize()?;

        log::info!("RWKV7 model loaded successfully");

        Ok(Self {
            info,
            embed,
            head,
            layers,
            scratch: Mutex::new(None),
            #[cfg(feature = "hip-probes")]
            probes: None,
        })
    }

    /// Attach probes for capturing intermediate values.
    ///
    /// Only available with `hip-probes` feature. Probes are called at each hook point
    /// during forward pass, receiving the tensor data and context information.
    ///
    /// # Example
    /// ```ignore
    /// use web_rwkv::hip::{HipHook, HipProbeBuilder};
    ///
    /// let probes = HipProbeBuilder::new()
    ///     .on(HipHook::PostAttLayerNorm, |data, ctx| {
    ///         println!("Layer {:?}: PostAttLayerNorm shape {:?}", ctx.layer, ctx.shape);
    ///     })
    ///     .build();
    ///
    /// let model = Rwkv7Hip::load("model.st")?.with_probes(probes);
    /// ```
    #[cfg(feature = "hip-probes")]
    pub fn with_probes(mut self, probes: HipProbeMap) -> Self {
        self.probes = Some(std::sync::Arc::new(probes));
        self
    }

    /// Get a reference to a specific weight tensor by name (for spot-checking).
    ///
    /// Name format examples:
    /// - "emb.weight" - embedding weights
    /// - "blocks.0.att.receptance.weight" - layer 0 attention receptance
    /// - "blocks.5.ffn.key.weight" - layer 5 FFN key weights
    /// - "head.weight" - output head weights
    pub fn get_weight(&self, name: &str) -> Option<&TensorHip<f32>> {
        if name == "emb.weight" {
            return Some(&self.embed.w);
        }
        if name == "head.weight" {
            return Some(&self.head.w);
        }
        if name.starts_with("blocks.") {
            let parts: Vec<&str> = name.strip_prefix("blocks.")?.split('.').collect();
            if parts.is_empty() {
                return None;
            }
            let layer_idx: usize = parts[0].parse().ok()?;
            if layer_idx >= self.layers.len() {
                return None;
            }
            let layer = &self.layers[layer_idx];
            let rest = parts[1..].join(".");

            return match rest.as_str() {
                "ln1.weight" => Some(&layer.att_ln.weight),
                "ln1.bias" => Some(&layer.att_ln.bias),
                "ln2.weight" => Some(&layer.ffn_ln.weight),
                "ln2.bias" => Some(&layer.ffn_ln.bias),
                "att.x_r" => Some(&layer.att.x_r),
                "att.x_w" => Some(&layer.att.x_w),
                "att.x_k" => Some(&layer.att.x_k),
                "att.x_v" => Some(&layer.att.x_v),
                "att.x_a" => Some(&layer.att.x_a),
                "att.x_g" => Some(&layer.att.x_g),
                "att.w0" => Some(&layer.att.w0),
                "att.w1" => Some(&layer.att.w1),
                "att.w2" => Some(&layer.att.w2),
                "att.a0" => Some(&layer.att.a0),
                "att.a1" => Some(&layer.att.a1),
                "att.a2" => Some(&layer.att.a2),
                "att.g1" => Some(&layer.att.g1),
                "att.g2" => Some(&layer.att.g2),
                "att.r_k" => Some(&layer.att.r_k),
                "att.k_k" => Some(&layer.att.k_k),
                "att.k_a" => Some(&layer.att.k_a),
                "att.receptance.weight" => Some(&layer.att.w_r),
                "att.key.weight" => Some(&layer.att.w_k),
                "att.value.weight" => Some(&layer.att.w_v),
                "att.output.weight" => Some(&layer.att.w_o),
                "att.ln_x.weight" => Some(&layer.att.gn.weight),
                "att.ln_x.bias" => Some(&layer.att.gn.bias),
                "ffn.x_k" => Some(&layer.ffn.x_k),
                "ffn.key.weight" => Some(&layer.ffn.w_k),
                "ffn.value.weight" => Some(&layer.ffn.w_v),
                _ => None,
            };
        }
        None
    }

    /// Read the first N elements of a weight tensor back to the host.
    ///
    /// Useful for spot-checking loaded weights against reference values.
    pub fn read_weight_head(&self, name: &str, n: usize) -> std::result::Result<Vec<f32>, ModelLoadError> {
        let tensor = self.get_weight(name)
            .ok_or_else(|| ModelLoadError::InvalidModel(format!("Weight not found: {}", name)))?;

        let stream = Stream::null();
        let all_data = tensor.to_vec(&stream)?;
        let n = n.min(all_data.len());
        Ok(all_data[..n].to_vec())
    }

    /// Extract LoRA dimensions from the model weights.
    ///
    /// These dimensions are needed to allocate scratch buffers for the forward pass.
    /// LoRA dimensions are consistent across layers, so we read them from layer 0.
    ///
    /// # Returns
    /// `LoraDims` struct containing:
    /// - `w_dim`: Decay LoRA rank
    /// - `a_dim`: Adaptation LoRA rank
    /// - `g_dim`: Gate LoRA rank
    /// - `v_dim`: Value residual LoRA rank (Some for layers > 0, None if absent)
    pub fn lora_dims(&self) -> LoraDims {
        let layer = &self.layers[0];
        LoraDims {
            w_dim: layer.att.w1.shape().dim(0),
            a_dim: layer.att.a1.shape().dim(0),
            g_dim: layer.att.g1.shape().dim(0),
            // v1/v2 are only present on layers > 0, check layer 1 if it exists
            v_dim: if self.layers.len() > 1 {
                self.layers[1].att.v1.as_ref().map(|v| v.shape().dim(0))
            } else {
                None
            },
        }
    }

    /// Configure the model with fixed scratch buffers.
    ///
    /// This pre-allocates all GPU buffers sized for the given configuration.
    /// Must be called before `forward()` to enable inference.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration specifying max chunk size and batch size
    ///
    /// # Example
    /// ```ignore
    /// let config = HipRuntimeConfig::new(256, 4);  // chunk_size=256, batch=4
    /// let model = Rwkv7Hip::load("model.st")?.with_config(config)?;
    /// ```
    pub fn with_config(self, config: HipRuntimeConfig) -> Result<Self> {
        let lora_dims = self.lora_dims();
        let scratch = HipScratch::new(&self.info, lora_dims, config)?;
        *self.scratch.lock().unwrap() = Some(scratch);
        Ok(self)
    }

    /// Get the configured chunk size, if scratch buffers are initialized.
    pub fn chunk_size(&self) -> Option<usize> {
        self.scratch.lock().unwrap().as_ref().map(|s| s.config.max_prefill_chunk)
    }

    /// Get the configured batch size, if scratch buffers are initialized.
    pub fn max_batch_size(&self) -> Option<usize> {
        self.scratch.lock().unwrap().as_ref().map(|s| s.config.batch_size)
    }

    /// Run a forward pass on variable-length input sequences.
    ///
    /// This is the main inference API. It handles:
    /// - **Variable-length sequences**: Each sequence can have any length
    /// - **Automatic chunking**: Long sequences are processed in fixed-size chunks
    /// - **Batched inference**: Process multiple sequences in parallel
    /// - **Zero-copy token staging**: Tokens are copied directly to GPU staging buffer
    ///
    /// # Requirements
    /// You must call `with_config()` before using this method to initialize scratch buffers.
    ///
    /// # Arguments
    /// * `x` - Batch of token sequences (can have different lengths)
    /// * `state` - Optional state; if `None`, creates fresh zero state
    ///
    /// # Returns
    /// Tuple of (logits, state):
    /// - `logits`: Flattened logits for all real tokens across all batches
    ///   Layout: `[batch_0_token_0..batch_0_token_n, batch_1_token_0..batch_1_token_m, ...]`
    ///   Each token has `vocab_size` logits.
    /// - `state`: Updated state for subsequent forward calls
    ///
    /// # Example
    /// ```ignore
    /// let config = HipRuntimeConfig::new(256, 4);  // chunk_size=256, batch=4
    /// let model = Rwkv7Hip::load("model.st")?.with_config(config)?;
    ///
    /// // Variable-length batch (no padding required)
    /// let seq1 = vec![1, 2, 3];        // 3 tokens
    /// let seq2 = vec![4, 5, 6, 7, 8];  // 5 tokens
    /// let (logits, state) = model.forward(&[&seq1, &seq2], None)?;
    ///
    /// // logits contains: [3 * vocab_size for seq1] ++ [5 * vocab_size for seq2]
    /// ```
    pub fn forward(
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

        // Get real lengths and compute chunking
        let lens: Vec<usize> = x.iter().map(|s| s.len()).collect();
        let max_len = *lens.iter().max().unwrap_or(&0);

        if max_len == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "All sequences are empty".to_string(),
            });
        }

        let num_chunks = (max_len + chunk_size - 1) / chunk_size;

        // Initialize state
        let mut current_state = match state {
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

        // Accumulate logits for each batch
        let n_vocab = self.info.n_vocab;
        let mut all_logits: Vec<Vec<f32>> = vec![Vec::new(); batch_size];

        // Get stream for token staging
        let stream = Stream::null();

        // Process chunks
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;

            // Compute real lengths within this chunk
            let chunk_lens: Vec<usize> = lens
                .iter()
                .map(|&l| l.saturating_sub(start).min(chunk_size))
                .collect();

            // Skip if all sequences exhausted
            if chunk_lens.iter().all(|&l| l == 0) {
                break;
            }

            // Zero the token staging buffer and copy token slices
            scratch.token_staging.fill_zero()?;
            for (b, seq) in x.iter().enumerate() {
                let seq_start = start.min(seq.len());
                let seq_end = (start + chunk_size).min(seq.len());
                if seq_start < seq_end {
                    let slice = &seq[seq_start..seq_end];
                    let offset = b * chunk_size;
                    scratch.token_staging.copy_from_slice_at(slice, offset, &stream)?;
                }
            }

            // Build padded token slices from staging buffer
            // For now, we need to read back and create slices (will optimize later)
            let staged_tokens = scratch.token_staging.to_vec(&stream)?;
            let chunk_tokens: Vec<Vec<u32>> = (0..batch_size)
                .map(|b| {
                    let offset = b * chunk_size;
                    staged_tokens[offset..offset + chunk_size].to_vec()
                })
                .collect();
            let chunk_refs: Vec<&[u32]> = chunk_tokens.iter().map(|v| v.as_slice()).collect();

            // Forward on this chunk
            let chunk_logits = self.forward_chunk(&chunk_refs, &mut current_state, &chunk_lens, scratch)?;

            // Extract logits for real tokens only (skip padding positions)
            for (b, &real_len) in chunk_lens.iter().enumerate() {
                if real_len > 0 {
                    // logits layout: [n_vocab, chunk_size, batch_size] column-major
                    // For batch b, token t: offset = (b * chunk_size + t) * n_vocab
                    for t in 0..real_len {
                        let offset = (b * chunk_size + t) * n_vocab;
                        all_logits[b].extend_from_slice(&chunk_logits[offset..offset + n_vocab]);
                    }
                }
            }
        }

        // Flatten logits: concatenate all batches
        let flat_logits: Vec<f32> = all_logits.into_iter().flatten().collect();
        Ok((flat_logits, current_state))
    }

    /// Process a single chunk of tokens (internal method).
    ///
    /// This is used by `forward()` to process each chunk. It expects:
    /// - All sequences padded to exactly `chunk_size` tokens
    /// - `lens` specifying actual token counts within this chunk (for masking)
    /// - Scratch buffers already initialized via `with_config()`
    ///
    /// # Arguments
    /// * `x` - Batch of token sequences, each exactly `chunk_size` tokens
    /// * `state` - Current state (will be mutated)
    /// * `lens` - Actual lengths per batch element within this chunk
    /// * `scratch` - Pre-allocated scratch buffers
    ///
    /// # Returns
    /// Logits tensor of shape [vocab_size * chunk_size * batch] in column-major layout
    fn forward_chunk(
        &self,
        x: &[&[u32]],
        state: &mut HipState,
        lens: &[usize],
        scratch: &mut HipScratch,
    ) -> Result<Vec<f32>> {
        let b = x.len();
        if b == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }
        if lens.len() != b {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Lens size mismatch: expected {} (batch size), got {}",
                    b, lens.len()
                ),
            });
        }

        let t = x[0].len();
        if t == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty sequence".to_string(),
            });
        }

        // Verify all sequences have the same (padded) length
        for (i, seq) in x.iter().enumerate() {
            if seq.len() != t {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "Sequence length mismatch: sequence 0 has {} tokens but sequence {} has {}",
                        t, i, seq.len()
                    ),
                });
            }
        }

        // Verify lens[b] <= t for all batches
        for (i, &len) in lens.iter().enumerate() {
            if len > t {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "Length {} for batch {} exceeds padded sequence length {}",
                        len, i, t
                    ),
                });
            }
        }

        // Verify scratch supports this size
        if !scratch.supports(t, b) {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Scratch buffers too small: need ({}, {}) but have ({}, {})",
                    t, b, scratch.config.max_prefill_chunk, scratch.config.batch_size
                ),
            });
        }

        // Run forward pass with scratch buffers
        self.forward_inner(x, state, scratch, lens)
    }

    /// Internal forward pass using scratch buffers and masked WKV.
    fn forward_inner(
        &self,
        tokens: &[&[u32]],
        state: &mut HipState,
        scratch: &mut HipScratch,
        lens: &[usize],
    ) -> Result<Vec<f32>> {
        let b = tokens.len();
        let t = tokens[0].len();

        // Create BLAS context for all GEMM operations
        let ctx = HipBlasContext::with_null_stream()?;
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
        let lens_gpu = TensorHip::from_slice(&lens_i32, lens_shape, stream)?;

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

        // Output buffer
        let mut logits = scratch.logits.resized_view_mut(out_shape)?;

        // Embedding lookup: tokens[b][t] -> x[c, t, b]
        let emb_data = self.embed.w.to_vec(stream)?;
        let mut x_host = vec![0.0f32; n_embd * t * b];
        for batch_idx in 0..b {
            for time_idx in 0..t {
                let token = tokens[batch_idx][time_idx] as usize;
                for c in 0..n_embd {
                    let idx = batch_idx * t * n_embd + time_idx * n_embd + c;
                    x_host[idx] = emb_data[token * n_embd + c];
                }
            }
        }
        x.copy_from_slice(&x_host, stream)?;

        // Upload state to GPU using pinned async transfers
        let mut att_shift_gpu: Vec<TensorHip<f32>> = Vec::with_capacity(n_layer);
        for s in &state.att_shift_states {
            let mut gpu_tensor = TensorHip::<f32>::new(state_shape)?;
            unsafe {
                s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
            }
            att_shift_gpu.push(gpu_tensor);
        }

        let mut ffn_shift_gpu: Vec<TensorHip<f32>> = Vec::with_capacity(n_layer);
        for s in &state.ffn_states {
            let mut gpu_tensor = TensorHip::<f32>::new(state_shape)?;
            unsafe {
                s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
            }
            ffn_shift_gpu.push(gpu_tensor);
        }

        let mut wkv_state_gpu: Vec<TensorHip<f32>> = Vec::with_capacity(n_layer);
        for s in &state.att_states {
            let mut gpu_tensor = TensorHip::<f32>::new(wkv_state_shape)?;
            unsafe {
                s.copy_to_device_async(gpu_tensor.as_mut_ptr(), stream.handle())?;
            }
            wkv_state_gpu.push(gpu_tensor);
        }

        // Temporary buffers
        let mut new_att_shift = TensorHip::<f32>::new(state_shape)?;
        let mut new_ffn_shift = TensorHip::<f32>::new(state_shape)?;
        let mut new_wkv_state = TensorHip::<f32>::new(wkv_state_shape)?;
        let mut temp1 = TensorHip::<f32>::new(std_shape)?;
        let mut temp2 = TensorHip::<f32>::new(std_shape)?;

        // Process each layer
        for layer_idx in 0..n_layer {
            let layer = &self.layers[layer_idx];

            // Apply ln0 for layer 0
            if layer_idx == 0 {
                layer_norm_f32(
                    &x, &self.embed.ln.weight, &self.embed.ln.bias,
                    &mut x_ln, 1e-5, stream
                )?;
                copy_tensor_f32(&x_ln, &mut x, stream)?;
            }

            // ==== Time-Mix (Attention) ====
            layer_norm_f32(
                &x, &layer.att_ln.weight, &layer.att_ln.bias,
                &mut x_ln, 1e-5, stream
            )?;

            // Token shifts for attention
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_r,
                &mut att_xr, &mut new_att_shift, stream
            )?;
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_w,
                &mut att_xw, &mut temp1, stream
            )?;
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_k,
                &mut att_xk, &mut temp1, stream
            )?;
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_v,
                &mut att_xv, &mut temp1, stream
            )?;
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_a,
                &mut att_xa, &mut temp1, stream
            )?;
            channel_mix_state_f32(
                &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_g,
                &mut att_xg, &mut temp1, stream
            )?;

            // Update shift state - re-extract at correct positions for masked sequences
            // The kernel extracts from position T-1, but we need position lens[b]-1
            {
                let x_ln_host = x_ln.to_vec(stream)?;
                let correct_state = extract_shift_state_at_lengths(&x_ln_host, lens, n_embd, t, b);
                att_shift_gpu[layer_idx].copy_from_slice(&correct_state, stream)?;
            }

            // Linear projections: r, k, v
            ctx.sgemm_into(&layer.att.w_r, &att_xr, &mut att_r)?;
            ctx.sgemm_into(&layer.att.w_k, &att_xk, &mut att_k)?;
            ctx.sgemm_into(&layer.att.w_v, &att_xv, &mut att_v)?;

            // Decay: w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
            ctx.sgemm_into(&layer.att.w1, &att_xw, &mut lora_w)?;
            let mut lora_tanh = TensorHip::<f32>::new(lora_w_shape)?;
            tanh_f32(&lora_w, &mut lora_tanh, stream)?;
            ctx.sgemm_into(&layer.att.w2, &lora_tanh, &mut att_w)?;
            broadcast_add_f32(&att_w, &layer.att.w0, &mut temp1, stream)?;
            softplus_decay_f32(&temp1, &mut att_w, stream)?;

            // Adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
            ctx.sgemm_into(&layer.att.a1, &att_xa, &mut lora_a)?;
            let mut lora_a_proj = TensorHip::<f32>::new(std_shape)?;
            ctx.sgemm_into(&layer.att.a2, &lora_a, &mut lora_a_proj)?;
            broadcast_add_f32(&lora_a_proj, &layer.att.a0, &mut temp1, stream)?;
            sigmoid_f32(&temp1, &mut att_a, stream)?;

            // Gate: g = sigmoid(xg @ g1) @ g2
            ctx.sgemm_into(&layer.att.g1, &att_xg, &mut lora_g)?;
            let mut lora_g_sig = TensorHip::<f32>::new(lora_g_shape)?;
            sigmoid_f32(&lora_g, &mut lora_g_sig, stream)?;
            ctx.sgemm_into(&layer.att.g2, &lora_g_sig, &mut att_g)?;

            // Value residual (layers > 0)
            if layer_idx > 0 {
                if let (Some(v0), Some(v1), Some(v2)) =
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2) {
                    ctx.sgemm_into(v1, &att_xv, &mut lora_v)?;
                    let mut v_lora2 = TensorHip::<f32>::new(std_shape)?;
                    ctx.sgemm_into(v2, &lora_v, &mut v_lora2)?;
                    broadcast_add_f32(&v_lora2, v0, &mut temp1, stream)?;
                    sigmoid_f32(&temp1, &mut temp2, stream)?;
                    lerp_f32(&att_v, &v_first, &temp2, &mut temp1, stream)?;
                    copy_tensor_f32(&temp1, &mut att_v, stream)?;
                }
            } else {
                copy_tensor_f32(&att_v, &mut v_first, stream)?;
            }

            // L2 normalize k
            broadcast_mul_f32(&att_k, &layer.att.k_k, &mut temp1, stream)?;
            l2_norm_f32(&temp1, &mut att_kk, head_size, 1e-12, stream)?;

            // Control K
            control_k_f32(&layer.att.k_a, &att_a, &att_k, &mut att_k_ctrl, stream)?;

            // WKV inputs
            negate_f32(&att_kk, &mut wkv_a, stream)?;
            mul_f32(&att_kk, &att_a, &mut wkv_b, stream)?;
            exp_f32(&att_w, &mut w_decay, stream)?;

            // Reshape for WKV
            let w_decay_wkv = w_decay.reshape_view(wkv_data_shape)?;
            let r_wkv = att_r.reshape_view(wkv_data_shape)?;
            let k_ctrl_wkv = att_k_ctrl.reshape_view(wkv_data_shape)?;
            let v_wkv = att_v.reshape_view(wkv_data_shape)?;
            let wkv_a_wkv = wkv_a.reshape_view(wkv_data_shape)?;
            let wkv_b_wkv = wkv_b.reshape_view(wkv_data_shape)?;
            let mut wkv_out_wkv = wkv_out.reshape_view_mut(wkv_data_shape)?;

            // Run masked WKV7 (skips state updates for padding positions)
            wkv7_f32_masked(
                &w_decay_wkv, &r_wkv, &k_ctrl_wkv, &v_wkv, &wkv_a_wkv, &wkv_b_wkv,
                &wkv_state_gpu[layer_idx], &mut wkv_out_wkv, &mut new_wkv_state,
                &lens_gpu, stream
            )?;
            std::mem::swap(&mut wkv_state_gpu[layer_idx], &mut new_wkv_state);

            // Group norm on WKV output
            group_norm_f32(
                &wkv_out, &layer.att.gn.weight, &layer.att.gn.bias,
                &mut wkv_normed, n_head, 64e-5, stream
            )?;

            // WKV bonus
            let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
            let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
            let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
            wkv_bonus_f32(&r_wkv, &k_ctrl_wkv, &v_wkv, &r_k_wkv, &mut wkv_bonus_wkv, stream)?;

            // Combine and gate
            add_f32(&wkv_normed, &wkv_bonus, &mut temp1, stream)?;
            mul_f32(&temp1, &att_g, &mut temp2, stream)?;

            // Output projection
            ctx.sgemm_into(&layer.att.w_o, &temp2, &mut att_out)?;

            // Residual
            add_f32(&x, &att_out, &mut temp1, stream)?;
            copy_tensor_f32(&temp1, &mut x, stream)?;

            // ==== Channel-Mix (FFN) ====
            layer_norm_f32(
                &x, &layer.ffn_ln.weight, &layer.ffn_ln.bias,
                &mut x_ln, 1e-5, stream
            )?;

            // Token shift for FFN
            channel_mix_state_f32(
                &x_ln, &ffn_shift_gpu[layer_idx], &layer.ffn.x_k,
                &mut ffn_xk, &mut new_ffn_shift, stream
            )?;
            // Re-extract at correct positions for masked sequences
            {
                let x_ln_host = x_ln.to_vec(stream)?;
                let correct_state = extract_shift_state_at_lengths(&x_ln_host, lens, n_embd, t, b);
                ffn_shift_gpu[layer_idx].copy_from_slice(&correct_state, stream)?;
            }

            // Key projection
            ctx.sgemm_into(&layer.ffn.w_k, &ffn_xk, &mut ffn_k)?;

            // Squared ReLU
            squared_relu_f32(&ffn_k, &mut ffn_k_sq, stream)?;

            // Value projection
            ctx.sgemm_into(&layer.ffn.w_v, &ffn_k_sq, &mut ffn_out)?;

            // Residual
            add_f32(&x, &ffn_out, &mut temp1, stream)?;
            copy_tensor_f32(&temp1, &mut x, stream)?;
        }

        // ==== Output Head ====
        layer_norm_f32(
            &x, &self.head.ln.weight, &self.head.ln.bias,
            &mut x_ln, 1e-5, stream
        )?;

        ctx.sgemm_into(&self.head.w, &x_ln, &mut logits)?;

        // Download state back to host using pinned async transfers
        for (i, gpu_state) in att_shift_gpu.iter().enumerate() {
            unsafe {
                state.att_shift_states[i].copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
            }
        }
        for (i, gpu_state) in ffn_shift_gpu.iter().enumerate() {
            unsafe {
                state.ffn_states[i].copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
            }
        }
        for (i, gpu_state) in wkv_state_gpu.iter().enumerate() {
            unsafe {
                state.att_states[i].copy_from_device_async(gpu_state.as_ptr(), stream.handle())?;
            }
        }

        // Download logits and return
        logits.to_vec(stream)
    }

    /// Asynchronous forward pass that returns immediately with a completion handle.
    ///
    /// Unlike `forward()`, this method queues all GPU work and data transfers
    /// without waiting for them to complete. The caller can check completion
    /// status or wait for results using the returned `ForwardCompletion`.
    ///
    /// This enables overlapping GPU computation with CPU work:
    ///
    /// ```ignore
    /// let completion = model.forward_async(&[&tokens], None)?;
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
    pub fn forward_async(
        &self,
        x: &[&[u32]],
        state: Option<HipState>,
    ) -> Result<ForwardCompletion> {
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
                    "forward_async() requires sequence length <= chunk_size ({} vs {}). \
                     Use forward() for longer sequences.",
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

        // Create stream for async operations (with fallback to null)
        let stream = match Stream::new() {
            Ok(s) => s,
            Err(_) => {
                // Fall back to null stream if creation fails
                Stream::null()
            }
        };

        // Pad sequences to chunk_size
        let chunk_tokens: Vec<Vec<u32>> = x.iter().map(|seq| {
            let mut padded = seq.to_vec();
            padded.resize(chunk_size, 0);
            padded
        }).collect();
        // Note: chunk_refs prepared for future true async implementation
        let _chunk_refs: Vec<&[u32]> = chunk_tokens.iter().map(|v| v.as_slice()).collect();

        // For this initial implementation, we use the sync forward and wrap the result.
        // This validates the ForwardCompletion infrastructure while we incrementally
        // add true async behavior (non-blocking D→H copies, stream parallelism, etc.)

        // Drop scratch_ref so we can call forward() which also takes the lock
        drop(scratch_ref);

        // Run the sync forward pass
        let (logits, state) = self.forward(x, Some(current_state))?;

        // Create completion event (already complete since forward() is sync)
        let event = Event::new()?;
        event.record(&stream)?;

        // Build state buffers from the result
        let state_buffers = ForwardStateBuffers {
            att_states: state.att_states,
            att_shift_states: state.att_shift_states,
            ffn_states: state.ffn_states,
            v_first: state.v_first,
        };

        Ok(ForwardCompletion {
            event,
            stream,
            logits_buffer: logits,
            state_buffers,
            n_layer: self.info.n_layer,
            batch_size,
        })
    }
}

// Old forward_with_state, forward_with_state_masked, forward_with_scratch deleted.
// See forward() for the unified API.
