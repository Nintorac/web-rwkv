//! RWKV7 HIP model loading and forward pass implementation.

use half::f16;
use half::slice::HalfFloatSliceExt;
use std::sync::Mutex;
use std::path::Path;

use super::ffi::{HipErrorKind, Result};
use super::device::{Stream, Event};
use super::pinned::PinnedBuffer;
use super::tensor::{TensorShape, TensorHip};
use super::scratch::{LoraDims, HipRuntimeConfig};
use super::kernels::{
    // GPU-native kernels used in forward
    layer_norm_f16, group_norm_f16, l2_norm_f16,
    sigmoid_f16, tanh_f16, softplus_decay_f16, squared_relu_f16,
    channel_mix_state_f16, channel_mix_state_f16_masked,
    control_k_f16, wkv7_f16_masked, wkv_bonus_f16,
    add_f16, mul_f16, negate_f16, decay_exp_f16, broadcast_add_f16, broadcast_mul_f16,
    lerp_f16, copy_tensor_f16, exp_f16, copy_f16_to_f32,
};
use super::blas::HipBlasContext;
use super::scratch::HipScratch;
use super::HipProf;

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
    pub weight: TensorHip<f16>,
    pub bias: TensorHip<f16>,
}

/// Attention weights for a single layer.
#[derive(Debug)]
pub struct AttentionHip {
    // Token shift mix weights
    pub x_r: TensorHip<f16>,
    pub x_w: TensorHip<f16>,
    pub x_k: TensorHip<f16>,
    pub x_v: TensorHip<f16>,
    pub x_a: TensorHip<f16>,
    pub x_g: TensorHip<f16>,

    // Decay LoRA
    pub w0: TensorHip<f16>,
    pub w1: TensorHip<f16>,
    pub w2: TensorHip<f16>,

    // Learning rate LoRA
    pub a0: TensorHip<f16>,
    pub a1: TensorHip<f16>,
    pub a2: TensorHip<f16>,

    // Gate LoRA
    pub g1: TensorHip<f16>,
    pub g2: TensorHip<f16>,

    // Value residual LoRA (layers > 0)
    pub v0: Option<TensorHip<f16>>,
    pub v1: Option<TensorHip<f16>>,
    pub v2: Option<TensorHip<f16>>,

    // Key normalization weights
    pub r_k: TensorHip<f16>,
    pub k_k: TensorHip<f16>,
    pub k_a: TensorHip<f16>,

    // Projection matrices (column-major for rocBLAS)
    pub w_r: TensorHip<f16>,  // Receptance: [n_embd, n_embd]
    pub w_k: TensorHip<f16>,  // Key: [n_embd, n_embd]
    pub w_v: TensorHip<f16>,  // Value: [n_embd, n_embd]
    pub w_o: TensorHip<f16>,  // Output: [n_embd, n_embd]

    // Group normalization
    pub gn: LayerNormHip,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnHip {
    // Token shift mix weight
    pub x_k: TensorHip<f16>,

    // Projection matrices
    pub w_k: TensorHip<f16>,  // Key (expand): [n_hidden, n_embd]
    pub w_v: TensorHip<f16>,  // Value (contract): [n_embd, n_hidden]
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
    /// Embedding table kept on CPU to avoid GPU→CPU transfer per forward.
    /// Shape: [n_vocab, n_embd] in row-major order (token_id * n_embd + c).
    pub w: Vec<f16>,
    /// Embedding dimension (n_embd)
    pub n_embd: usize,
}

/// Output head weights.
#[derive(Debug)]
pub struct HeadHip {
    pub ln: LayerNormHip,
    pub w: TensorHip<f16>,  // [n_vocab, n_embd]
}

/// RWKV7 model loaded into HIP memory.
///
/// Weights are stored in managed memory for zero-copy APU access.
/// Weights and activations use FP16; recurrent WKV state remains FP32.
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
    logits_buffer: PinnedBuffer<f16>,
    /// Pre-allocated buffers for state (D→H copies are queued but not complete)
    state_buffers: ForwardStateBuffers,
    /// Model info for reconstructing HipState
    n_layer: usize,
    batch_size: usize,
    /// Sequence lengths for extracting real tokens from padded output
    lens: Vec<usize>,
    /// Padded chunk size
    chunk_size: usize,
    /// Vocabulary size
    n_vocab: usize,
}

/// Internal buffers for async state download
struct ForwardStateBuffers {
    att_states: Vec<PinnedBuffer<f32>>,
    att_shift_states: Vec<PinnedBuffer<f16>>,
    ffn_states: Vec<PinnedBuffer<f16>>,
    v_first: Option<PinnedBuffer<f16>>,
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

        // Extract only real tokens from padded output (matches sync forward behavior)
        // Layout: [n_vocab, chunk_size, batch_size] column-major
        // For batch b, token t: offset = (b * chunk_size + t) * n_vocab
        let padded = self.logits_buffer.as_slice();
        let mut logits = Vec::new();
        for (b, &real_len) in self.lens.iter().enumerate() {
            for t in 0..real_len {
                let offset = (b * self.chunk_size + t) * self.n_vocab;
                logits.extend(
                    padded[offset..offset + self.n_vocab]
                        .iter()
                        .map(|v| v.to_f32()),
                );
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

/// Transpose a 2D matrix from row-major to column-major layout.
///
/// Row-major [M, K]: element (i, j) at index i * K + j
/// Column-major [M, K]: element (i, j) at index j * M + i
///
/// This is used at load time to convert SafeTensors (row-major) weights
/// to rocBLAS-native column-major format, per the plan:
/// "Use rocBLAS-native column-major storage for GEMM/GEMV...
///  This avoids per-call row/col mapping in rocBLAS"
fn transpose_2d(data: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut transposed = vec![f16::from_f32(0.0); data.len()];
    for i in 0..rows {
        for j in 0..cols {
            // row-major index: i * cols + j
            // column-major index: j * rows + i
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

/// Load a tensor from SafeTensors, converting to f16 and loading into managed HIP memory.
fn load_tensor_f16(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f16>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    // Convert to f16 vec
    let f16_data: Vec<f16> = match dtype {
        safetensors::Dtype::F16 => bytemuck::cast_slice(data).to_vec(),
        safetensors::Dtype::F32 => bytemuck::cast_slice(data)
            .iter()
            .map(|x: &f32| f16::from_f32(*x))
            .collect(),
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| f16::from_f32(x.to_f32())).collect()
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

    TensorHip::from_slice_managed(&f16_data, hip_shape, stream).map_err(ModelLoadError::from)
}

/// Load a tensor from SafeTensors, converting to f16, keeping data on CPU.
///
/// This is used for embedding tables which are accessed on CPU during embedding lookup.
/// The data is returned in row-major order as stored in SafeTensors.
fn load_tensor_f16_cpu(
    st: &safetensors::SafeTensors,
    name: &str,
) -> std::result::Result<Vec<f16>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let dtype = tensor.dtype();
    let data = tensor.data();

    // Convert to f16 vec
    let f16_data: Vec<f16> = match dtype {
        safetensors::Dtype::F16 => bytemuck::cast_slice(data).to_vec(),
        safetensors::Dtype::F32 => bytemuck::cast_slice(data)
            .iter()
            .map(|x: &f32| f16::from_f32(*x))
            .collect(),
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| f16::from_f32(x.to_f32())).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    Ok(f16_data)
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
fn load_weight_matrix_f16(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f16>, ModelLoadError> {
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

    // Convert to f16 vec
    let f16_data: Vec<f16> = match dtype {
        safetensors::Dtype::F16 => bytemuck::cast_slice(data).to_vec(),
        safetensors::Dtype::F32 => bytemuck::cast_slice(data)
            .iter()
            .map(|x: &f32| f16::from_f32(*x))
            .collect(),
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| f16::from_f32(x.to_f32())).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    // Transpose from row-major to column-major
    let transposed = transpose_2d(&f16_data, rows, cols);

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
    let weight = load_tensor_f16(st, &format!("{}.weight", prefix), stream)?;
    let bias = load_tensor_f16(st, &format!("{}.bias", prefix), stream)?;
    Ok(LayerNormHip { weight, bias })
}

impl Rwkv7Hip {
    /// Load an RWKV7 model from a SafeTensors file.
    ///
    /// Weights are loaded into managed (unified) memory for efficient APU access.
    /// Weights are stored as FP16; WKV state remains FP32.
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

        // Load embedding to CPU (lookup table accessed on CPU, avoids GPU→CPU transfer per forward)
        let embed = EmbedHip {
            ln: load_layer_norm(&st, "blocks.0.ln0", &stream)?,
            w: load_tensor_f16_cpu(&st, "emb.weight")?,
            n_embd,
        };

        // Load output head (transpose to column-major for GEMM)
        let head = HeadHip {
            ln: load_layer_norm(&st, "ln_out", &stream)?,
            w: load_weight_matrix_f16(&st, "head.weight", &stream)?,
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
                x_r: load_tensor_f16(&st, &format!("{}.att.x_r", prefix), &stream)?,
                x_w: load_tensor_f16(&st, &format!("{}.att.x_w", prefix), &stream)?,
                x_k: load_tensor_f16(&st, &format!("{}.att.x_k", prefix), &stream)?,
                x_v: load_tensor_f16(&st, &format!("{}.att.x_v", prefix), &stream)?,
                x_a: load_tensor_f16(&st, &format!("{}.att.x_a", prefix), &stream)?,
                x_g: load_tensor_f16(&st, &format!("{}.att.x_g", prefix), &stream)?,

                w0: load_tensor_f16(&st, &format!("{}.att.w0", prefix), &stream)?,
                w1: load_weight_matrix_f16(&st, &format!("{}.att.w1", prefix), &stream)?,
                w2: load_weight_matrix_f16(&st, &format!("{}.att.w2", prefix), &stream)?,

                a0: load_tensor_f16(&st, &format!("{}.att.a0", prefix), &stream)?,
                a1: load_weight_matrix_f16(&st, &format!("{}.att.a1", prefix), &stream)?,
                a2: load_weight_matrix_f16(&st, &format!("{}.att.a2", prefix), &stream)?,

                g1: load_weight_matrix_f16(&st, &format!("{}.att.g1", prefix), &stream)?,
                g2: load_weight_matrix_f16(&st, &format!("{}.att.g2", prefix), &stream)?,

                // Value residual LoRA (only for layers > 0)
                v0: if layer_idx > 0 {
                    Some(load_tensor_f16(&st, &format!("{}.att.v0", prefix), &stream)?)
                } else {
                    None
                },
                v1: if layer_idx > 0 {
                    Some(load_weight_matrix_f16(&st, &format!("{}.att.v1", prefix), &stream)?)
                } else {
                    None
                },
                v2: if layer_idx > 0 {
                    Some(load_weight_matrix_f16(&st, &format!("{}.att.v2", prefix), &stream)?)
                } else {
                    None
                },

                r_k: load_tensor_f16(&st, &format!("{}.att.r_k", prefix), &stream)?,
                k_k: load_tensor_f16(&st, &format!("{}.att.k_k", prefix), &stream)?,
                k_a: load_tensor_f16(&st, &format!("{}.att.k_a", prefix), &stream)?,

                w_r: load_weight_matrix_f16(&st, &format!("{}.att.receptance.weight", prefix), &stream)?,
                w_k: load_weight_matrix_f16(&st, &format!("{}.att.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f16(&st, &format!("{}.att.value.weight", prefix), &stream)?,
                w_o: load_weight_matrix_f16(&st, &format!("{}.att.output.weight", prefix), &stream)?,

                gn: load_layer_norm(&st, &format!("{}.att.ln_x", prefix), &stream)?,
            };

            // FFN weights (transpose weight matrices to column-major for GEMM)
            let ffn = FfnHip {
                x_k: load_tensor_f16(&st, &format!("{}.ffn.x_k", prefix), &stream)?,
                w_k: load_weight_matrix_f16(&st, &format!("{}.ffn.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f16(&st, &format!("{}.ffn.value.weight", prefix), &stream)?,
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
    pub fn get_weight(&self, name: &str) -> Option<&TensorHip<f16>> {
        // Note: emb.weight is kept on CPU, use get_embedding() instead
        if name == "emb.weight" {
            return None;
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
        Ok(all_data[..n].iter().map(|v| v.to_f32()).collect())
    }

    /// Get a reference to the embedding table (CPU storage).
    ///
    /// The embedding table is kept on CPU to avoid GPU→CPU transfer overhead per forward call.
    /// Shape: [n_vocab, n_embd] in row-major order (token_id * n_embd + c).
    pub fn get_embedding(&self) -> &[f16] {
        &self.embed.w
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

        // Fast path: single-token decode (no staging round-trip)
        if chunk_size == 1 && num_chunks == 1 && lens.iter().all(|&l| l == 1) {
            let chunk_logits = self.forward_chunk(x, &mut current_state, &lens, scratch)?;
            for b in 0..batch_size {
                let offset = b * n_vocab;
                all_logits[b].extend_from_slice(&chunk_logits[offset..offset + n_vocab]);
            }
            let flat_logits: Vec<f32> = all_logits.into_iter().flatten().collect();
            return Ok((flat_logits, current_state));
        }

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

    /// Reset resident GPU state to zeros (when enabled).
    pub fn reset_resident_state(&self) -> Result<()> {
        let mut scratch_ref = self.scratch.lock().unwrap();
        let scratch = scratch_ref.as_mut().ok_or_else(|| HipErrorKind {
            code: -1,
            message: "Scratch not initialized - call with_config() first".to_string(),
        })?;
        scratch.reset_state_gpu()?;
        Ok(())
    }

    /// Internal forward pass using scratch buffers and masked WKV.
    fn forward_inner(
        &self,
        tokens: &[&[u32]],
        state: &mut HipState,
        scratch: &mut HipScratch,
        lens: &[usize],
    ) -> Result<Vec<f32>> {
        let use_resident_state = scratch.config.resident_state;
        let b = tokens.len();
        let t = tokens[0].len();
        let mut prof = HipProf::new("forward_inner");

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
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        prof.time("lens_upload", || lens_gpu.copy_from_slice(&lens_i32, stream))?;

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
                scratch.emb_staging.copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
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
                        &x, &self.embed.ln.weight, &self.embed.ln.bias,
                        &mut x_ln, 1e-5, stream
                    )?;
                    copy_tensor_f16(&x_ln, &mut x, stream)?;
                    Ok(())
                })?;
            }

            // ==== Time-Mix (Attention) ====
            prof.time("att_ln", || {
                layer_norm_f16(
                    &x, &layer.att_ln.weight, &layer.att_ln.bias,
                    &mut x_ln, 1e-5, stream
                )?;
                prof_sync(stream)?;
                Ok(())
            })?;

            // Token shifts for attention - use masked kernel for x_r to get correct state
            // The masked kernel extracts state at lengths[b]-1 instead of T-1
            prof.time("att_shift", || {
                channel_mix_state_f16_masked(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_r,
                    &mut att_xr, &mut new_att_shift, &lens_gpu, stream
                )?;
                // Remaining shifts use regular kernel (we only need outputs, not state)
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_w,
                    &mut att_xw, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_k,
                    &mut att_xk, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_v,
                    &mut att_xv, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_a,
                    &mut att_xa, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_g,
                    &mut att_xg, &mut temp1, stream
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
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2) {
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

            // Run masked WKV7 (skips state updates for padding positions)
            prof.time("wkv", || {
                wkv7_f16_masked(
                    &w_decay_wkv, &r_wkv, &k_ctrl_wkv, &v_wkv, &wkv_a_wkv, &wkv_b_wkv,
                    &wkv_state_gpu[layer_idx], &mut wkv_out_wkv, &mut new_wkv_state,
                    &lens_gpu, stream
                )?;
                prof_sync(stream)?;
                Ok(())
            })?;
            std::mem::swap(&mut wkv_state_gpu[layer_idx], &mut new_wkv_state);

            // Group norm on WKV output
            prof.time("wkv_norm", || {
                group_norm_f16(
                    &wkv_out, &layer.att.gn.weight, &layer.att.gn.bias,
                    &mut wkv_normed, n_head, 64e-5, stream
                )?;
                Ok(())
            })?;

            // WKV bonus
            let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
            let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
            let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
            prof.time("wkv_bonus", || {
                wkv_bonus_f16(&r_wkv, &k_ctrl_wkv, &v_wkv, &r_k_wkv, &mut wkv_bonus_wkv, stream)?;
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
                    &x, &layer.ffn_ln.weight, &layer.ffn_ln.bias,
                    &mut x_ln, 1e-5, stream
                )?;
                Ok(())
            })?;

            // Token shift for FFN - use masked kernel for correct state extraction
            prof.time("ffn_shift", || {
                channel_mix_state_f16_masked(
                    &x_ln, &ffn_shift_gpu[layer_idx], &layer.ffn.x_k,
                    &mut ffn_xk, &mut new_ffn_shift, &lens_gpu, stream
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
                &x, &self.head.ln.weight, &self.head.ln.bias,
                &mut x_ln, 1e-5, stream
            )?;

            // f16 GEMM for head layer
            ctx.hgemm_into(&self.head.w, &x_ln, &mut logits)?;
            prof_sync(stream)?;
            Ok(())
        })?;

            // Convert f16 logits to f32 on GPU, then download
            prof.time("logits_dl", || {
                copy_f16_to_f32(&logits, &mut logits_f32, stream)?;
                logits_f32.to_vec(stream)
            })
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
                Ok(())
            })?;
        }

        #[cfg(feature = "hip-prof")]
        prof.print(&format!("b={b} t={t} layers={n_layer}"));

        result
    }

    /// Internal async forward pass - same as forward_inner but with async logits copy.
    ///
    /// Instead of synchronously downloading logits, this copies to a provided pinned
    /// buffer asynchronously. The caller must sync the stream before reading the buffer.
    fn forward_inner_async(
        &self,
        tokens: &[&[u32]],
        state: &mut HipState,
        scratch: &mut HipScratch,
        lens: &[usize],
        logits_dst: &mut PinnedBuffer<f16>,
    ) -> Result<()> {
        let use_resident_state = scratch.config.resident_state;
        let b = tokens.len();
        let t = tokens[0].len();
        let mut prof = HipProf::new("forward_inner_async");

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
        let mut lens_gpu = scratch.lens_gpu.resized_view_mut(lens_shape)?;
        prof.time("lens_upload", || lens_gpu.copy_from_slice(&lens_i32, stream))?;

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

        // Output buffer
        let mut logits = scratch.logits.resized_view_mut(out_shape)?;

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
                scratch.emb_staging.copy_to_device_async(x.as_mut_ptr(), stream.handle())?;
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
                        &x, &self.embed.ln.weight, &self.embed.ln.bias,
                        &mut x_ln, 1e-5, stream
                    )?;
                    copy_tensor_f16(&x_ln, &mut x, stream)?;
                    Ok(())
                })?;
            }

            // ==== Time-Mix (Attention) ====
            prof.time("att_ln", || {
                layer_norm_f16(
                    &x, &layer.att_ln.weight, &layer.att_ln.bias,
                    &mut x_ln, 1e-5, stream
                )?;
                prof_sync(stream)?;
                Ok(())
            })?;

            // Token shifts for attention - use masked kernel for x_r to get correct state
            // The masked kernel extracts state at lengths[b]-1 instead of T-1
            prof.time("att_shift", || {
                channel_mix_state_f16_masked(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_r,
                    &mut att_xr, &mut new_att_shift, &lens_gpu, stream
                )?;
                // Remaining shifts use regular kernel (we only need outputs, not state)
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_w,
                    &mut att_xw, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_k,
                    &mut att_xk, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_v,
                    &mut att_xv, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_a,
                    &mut att_xa, &mut temp1, stream
                )?;
                channel_mix_state_f16(
                    &x_ln, &att_shift_gpu[layer_idx], &layer.att.x_g,
                    &mut att_xg, &mut temp1, stream
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
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2) {
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

            // Run masked WKV7 (skips state updates for padding positions)
            prof.time("wkv", || {
                wkv7_f16_masked(
                    &w_decay_wkv, &r_wkv, &k_ctrl_wkv, &v_wkv, &wkv_a_wkv, &wkv_b_wkv,
                    &wkv_state_gpu[layer_idx], &mut wkv_out_wkv, &mut new_wkv_state,
                    &lens_gpu, stream
                )?;
                prof_sync(stream)?;
                Ok(())
            })?;
            std::mem::swap(&mut wkv_state_gpu[layer_idx], &mut new_wkv_state);

            // Group norm on WKV output
            prof.time("wkv_norm", || {
                group_norm_f16(
                    &wkv_out, &layer.att.gn.weight, &layer.att.gn.bias,
                    &mut wkv_normed, n_head, 64e-5, stream
                )?;
                Ok(())
            })?;

            // WKV bonus
            let r_k_shape = TensorShape::new(head_size, n_head, 1, 1);
            let r_k_wkv = layer.att.r_k.reshape_view(r_k_shape)?;
            let mut wkv_bonus_wkv = wkv_bonus.reshape_view_mut(wkv_data_shape)?;
            prof.time("wkv_bonus", || {
                wkv_bonus_f16(&r_wkv, &k_ctrl_wkv, &v_wkv, &r_k_wkv, &mut wkv_bonus_wkv, stream)?;
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
                    &x, &layer.ffn_ln.weight, &layer.ffn_ln.bias,
                    &mut x_ln, 1e-5, stream
                )?;
                Ok(())
            })?;

            // Token shift for FFN - use masked kernel for correct state extraction
            prof.time("ffn_shift", || {
                channel_mix_state_f16_masked(
                    &x_ln, &ffn_shift_gpu[layer_idx], &layer.ffn.x_k,
                    &mut ffn_xk, &mut new_ffn_shift, &lens_gpu, stream
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
                &x, &self.head.ln.weight, &self.head.ln.bias,
                &mut x_ln, 1e-5, stream
            )?;

            ctx.hgemm_into(&self.head.w, &x_ln, &mut logits)?;
            prof_sync(stream)?;
            Ok(())
        })?;

            prof.time("logits_download", || {
                // Download logits asynchronously (no sync)
                logits.copy_to_slice_async(logits_dst.as_slice_mut(), stream)?;
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
                Ok(())
            })?;
        }

        #[cfg(feature = "hip-prof")]
        prof.print(&format!("b={b} t={t} layers={n_layer}"));

        result
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
        let chunk_refs: Vec<&[u32]> = chunk_tokens.iter().map(|v| v.as_slice()).collect();

        // Allocate pinned buffer for logits
        let n_vocab = self.info.n_vocab;
        let logits_size = n_vocab * chunk_size * batch_size;
        let mut logits_buffer = PinnedBuffer::new(logits_size)?;

        // Run the async forward pass (queues work but doesn't sync)
        let mut current_state = current_state;
        self.forward_inner_async(&chunk_refs, &mut current_state, scratch, &lens, &mut logits_buffer)?;

        // Record event after all GPU work and D→H transfers are queued
        let event = Event::new()?;
        event.record(&stream)?;

        // Build state buffers (already updated by forward_inner_async)
        let state_buffers = ForwardStateBuffers {
            att_states: current_state.att_states,
            att_shift_states: current_state.att_shift_states,
            ffn_states: current_state.ffn_states,
            v_first: current_state.v_first,
        };

        Ok(ForwardCompletion {
            event,
            stream,
            logits_buffer,
            state_buffers,
            n_layer: self.info.n_layer,
            batch_size,
            lens,
            chunk_size,
            n_vocab,
        })
    }
}

// Old forward_with_state, forward_with_state_masked, forward_with_scratch deleted.
// See forward() for the unified API.
