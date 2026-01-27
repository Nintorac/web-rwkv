//! RWKV7 HIP model loading and forward pass implementation.

use half::f16;
use std::path::Path;

use super::ffi::{HipErrorKind, Result};
use super::device::Stream;
use super::tensor::{TensorShape, TensorHip};
use super::kernels::{
    hip_layer_norm, hip_wkv7, hip_channel_mix_state, hip_tanh,
    hip_softplus_decay, hip_sigmoid, hip_squared_relu,
    hip_l2_norm, hip_group_norm, hip_wkv_bonus, hip_control_k,
    hip_wkv7_masked, extract_shift_state_at_lengths,
};
use super::blas::hip_sgemm;

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

    #[cfg(feature = "hip-probes")]
    pub(crate) probes: Option<HipProbeMapRef>,
}

impl std::fmt::Debug for Rwkv7Hip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Rwkv7Hip");
        s.field("info", &self.info)
            .field("embed", &self.embed)
            .field("head", &self.head)
            .field("layers", &self.layers);
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
    pub att_states: Vec<Vec<f32>>,
    /// Attention token shift state per layer: [n_embd * batch] per layer
    pub att_shift_states: Vec<Vec<f32>>,
    /// FFN token shift state per layer: [n_embd * batch] per layer
    pub ffn_states: Vec<Vec<f32>>,
    /// Value residual from first layer for RWKV7, persisted across chunks: [n_embd * batch]
    pub v_first: Option<Vec<f32>>,
}

impl HipState {
    /// Create a fresh state initialized to zeros.
    ///
    /// # Arguments
    /// * `info` - Model info containing dimensions
    /// * `batch_size` - Number of sequences to process in parallel
    pub fn new(info: &Rwkv7ModelInfo, batch_size: usize) -> Self {
        let n_layer = info.n_layer;
        let n_embd = info.n_embd;
        let head_size = info.head_size;
        let n_head = info.n_head;

        HipState {
            batch_size,
            att_states: (0..n_layer)
                .map(|_| vec![0.0f32; head_size * head_size * n_head * batch_size])
                .collect(),
            att_shift_states: (0..n_layer)
                .map(|_| vec![0.0f32; n_embd * batch_size])
                .collect(),
            ffn_states: (0..n_layer)
                .map(|_| vec![0.0f32; n_embd * batch_size])
                .collect(),
            v_first: None,
        }
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        for state in &mut self.att_states {
            state.fill(0.0);
        }
        for state in &mut self.att_shift_states {
            state.fill(0.0);
        }
        for state in &mut self.ffn_states {
            state.fill(0.0);
        }
        self.v_first = None;
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

    /// Run a full forward pass on input tokens (single sequence, fresh state).
    ///
    /// Convenience wrapper for `forward_with_state` with batch_size=1 and fresh state.
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs for a single sequence
    ///
    /// # Returns
    /// Logits tensor of shape [vocab_size * T] for each input token.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut state = HipState::new(&self.info, 1);
        self.forward_with_state(&[tokens], &mut state)
    }

    /// Run a batched forward pass with explicit state management.
    ///
    /// This is the core inference API supporting:
    /// - **Batched inference**: Process multiple sequences in parallel (B > 1)
    /// - **Streaming**: Token-by-token generation with state carryover
    /// - **Chunked processing**: Long context in pieces with state carryover
    ///
    /// # Arguments
    /// * `tokens` - Batch of sequences. Each `&[u32]` is one sequence.
    ///              All sequences must have the same length T.
    /// * `state` - Mutable reference to inference state. Must have `batch_size == tokens.len()`.
    ///
    /// # Returns
    /// Logits tensor of shape [vocab_size * T * B] in column-major layout [V, T, B].
    ///
    /// # Example
    /// ```ignore
    /// // Batched prefill: 4 sequences of 128 tokens each
    /// let mut state = HipState::new(&model.info, 4);
    /// let sequences: Vec<&[u32]> = vec![&seq1, &seq2, &seq3, &seq4];
    /// let logits = model.forward_with_state(&sequences, &mut state)?;
    ///
    /// // Batched decode: generate next token for all 4 sequences
    /// let next_tokens: Vec<&[u32]> = vec![&[t1], &[t2], &[t3], &[t4]];
    /// let logits = model.forward_with_state(&next_tokens, &mut state)?;
    /// ```
    pub fn forward_with_state(&self, tokens: &[&[u32]], state: &mut HipState) -> Result<Vec<f32>> {
        let b = tokens.len();
        if b == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }
        if b != state.batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Batch size mismatch: tokens has {} sequences but state has batch_size={}",
                    b, state.batch_size
                ),
            });
        }

        let t = tokens[0].len();
        if t == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty sequence".to_string(),
            });
        }
        // Verify all sequences have the same length
        for (i, seq) in tokens.iter().enumerate() {
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

        let stream = Stream::null();
        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;
        let n_vocab = self.info.n_vocab;
        let n_layer = self.info.n_layer;
        let n_hidden = self.info.n_hidden;

        // Initialize probe context (compiles out without feature)
        #[cfg(feature = "hip-probes")]
        let mut probe_ctx = probe::ProbeContext {
            layer: None,
            batch_size: b,
            seq_len: t,
            n_embd,
            n_head,
            head_size,
            n_layer,
            shape_storage: [0; probe::MAX_SHAPE_DIMS],
            shape_len: 0,
        };

        // Embedding lookup: tokens[b][t] -> x[c, t, b]
        // Layout: x[c, t, b] = x[b * T * C + t * C + c]
        // embed.w is row-major [n_vocab, n_embd], so embed[token, c] = emb_data[token * n_embd + c]
        let emb_data = self.embed.w.to_vec(&stream)?;
        let mut x = vec![0.0f32; n_embd * t * b];
        for batch_idx in 0..b {
            for time_idx in 0..t {
                let token = tokens[batch_idx][time_idx] as usize;
                for c in 0..n_embd {
                    let idx = batch_idx * t * n_embd + time_idx * n_embd + c;
                    x[idx] = emb_data[token * n_embd + c];
                }
            }
        }
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostEmbed, &x, [n_embd, t, b]);

        // v_first is computed fresh each forward call (not part of persistent state)
        let mut v_first: Option<Vec<f32>> = None;

        // Process each layer
        for layer_idx in 0..n_layer {
            #[cfg(feature = "hip-probes")]
            { probe_ctx.layer = Some(layer_idx); }
            let layer = &self.layers[layer_idx];

            // Apply ln0 for layer 0
            if layer_idx == 0 {
                let ln0_w = self.embed.ln.weight.to_vec(&stream)?;
                let ln0_b = self.embed.ln.bias.to_vec(&stream)?;
                x = hip_layer_norm(&x, &ln0_w, &ln0_b, n_embd, t * b, 1e-5)?;
                #[cfg(feature = "hip-probes")]
                hip_probe!(self, probe_ctx, probe::HipHook::PostEmbedLayerNorm, &x, [n_embd, t, b]);
            }

            // ==== Time-Mix (Attention) ====
            let ln1_w = layer.att_ln.weight.to_vec(&stream)?;
            let ln1_b = layer.att_ln.bias.to_vec(&stream)?;
            let x_ln1 = hip_layer_norm(&x, &ln1_w, &ln1_b, n_embd, t * b, 1e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttLayerNorm, &x_ln1, [n_embd, t, b]);

            // Token shift for attention - all shifts use the same old state
            let att_shift_state = &state.att_shift_states[layer_idx];
            let x_r = layer.att.x_r.to_vec(&stream)?;
            let x_w = layer.att.x_w.to_vec(&stream)?;
            let x_k = layer.att.x_k.to_vec(&stream)?;
            let x_v = layer.att.x_v.to_vec(&stream)?;
            let x_a = layer.att.x_a.to_vec(&stream)?;
            let x_g = layer.att.x_g.to_vec(&stream)?;

            let (xr, new_att_shift) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_r, n_embd, t, b)?;
            let (xw, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_w, n_embd, t, b)?;
            let (xk, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_k, n_embd, t, b)?;
            let (xv, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_v, n_embd, t, b)?;
            let (xa, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_a, n_embd, t, b)?;
            let (xg, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_g, n_embd, t, b)?;
            #[cfg(feature = "hip-probes")]
            {
                // Stack all token shifts: xr, xw, xk, xv, xa, xg
                let stacked: Vec<f32> = [&xr, &xw, &xk, &xv, &xa, &xg]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PostAttTokenShift, &stacked, [n_embd, t, b, 6]);
            }

            // Update shift state after all shifts are computed
            state.att_shift_states[layer_idx] = new_att_shift;

            // Linear projections: r, k, v
            let w_r = layer.att.w_r.to_vec(&stream)?;
            let w_k = layer.att.w_k.to_vec(&stream)?;
            let w_v = layer.att.w_v.to_vec(&stream)?;
            let r = hip_sgemm(&w_r, &xr, n_embd, n_embd, t * b)?;
            let k = hip_sgemm(&w_k, &xk, n_embd, n_embd, t * b)?;
            let mut v = hip_sgemm(&w_v, &xv, n_embd, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            {
                // Stack r, k, v for linear projection probe
                let stacked: Vec<f32> = [&r, &k, &v]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PostAttLinear, &stacked, [n_embd, t, b, 3]);
            }

            // w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
            let w0 = layer.att.w0.to_vec(&stream)?;
            let w1 = layer.att.w1.to_vec(&stream)?;
            let w2 = layer.att.w2.to_vec(&stream)?;
            // Column-major weights: dim(0) is output features (LoRA dim for w1)
            let w1_dim = layer.att.w1.shape().dim(0);
            let w_lora1 = hip_sgemm(&w1, &xw, w1_dim, n_embd, t * b)?;
            let w_lora1_tanh = hip_tanh(&w_lora1)?;
            let w_lora2 = hip_sgemm(&w2, &w_lora1_tanh, n_embd, w1_dim, t * b)?;
            let mut w: Vec<f32> = w0.iter().cycle().take(w_lora2.len())
                .zip(w_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            w = hip_softplus_decay(&w)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttDecay, &w, [n_embd, t, b]);

            // a = sigmoid(a0 + (xa @ a1) @ a2)
            let a0 = layer.att.a0.to_vec(&stream)?;
            let a1 = layer.att.a1.to_vec(&stream)?;
            let a2 = layer.att.a2.to_vec(&stream)?;
            let a1_dim = layer.att.a1.shape().dim(0);
            let a_lora1 = hip_sgemm(&a1, &xa, a1_dim, n_embd, t * b)?;
            let a_lora2 = hip_sgemm(&a2, &a_lora1, n_embd, a1_dim, t * b)?;
            let a_biased: Vec<f32> = a0.iter().cycle().take(a_lora2.len())
                .zip(a_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let a = hip_sigmoid(&a_biased)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttAdapt, &a, [n_embd, t, b]);

            // g = sigmoid(xg @ g1) @ g2
            let g1 = layer.att.g1.to_vec(&stream)?;
            let g2 = layer.att.g2.to_vec(&stream)?;
            let g1_dim = layer.att.g1.shape().dim(0);
            let g_lora1 = hip_sgemm(&g1, &xg, g1_dim, n_embd, t * b)?;
            let g_lora1_sigmoid = hip_sigmoid(&g_lora1)?;
            let g = hip_sgemm(&g2, &g_lora1_sigmoid, n_embd, g1_dim, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGate, &g, [n_embd, t, b]);

            // Value residual (layers > 0)
            if layer_idx > 0 {
                if let (Some(v0), Some(v1), Some(v2), Some(ref vf)) =
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2, &v_first) {
                    let v0_data = v0.to_vec(&stream)?;
                    let v1_data = v1.to_vec(&stream)?;
                    let v2_data = v2.to_vec(&stream)?;
                    let v1_dim = v1.shape().dim(0);
                    // Note: Use xv (token-shifted input), not v (projected value)
                    let v_lora1 = hip_sgemm(&v1_data, &xv, v1_dim, n_embd, t * b)?;
                    let v_lora2 = hip_sgemm(&v2_data, &v_lora1, n_embd, v1_dim, t * b)?;
                    let v_biased: Vec<f32> = v0_data.iter().cycle().take(v_lora2.len())
                        .zip(v_lora2.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    let v_residual = hip_sigmoid(&v_biased)?;
                    // v = v + (v_first - v) * v_residual
                    v = v.iter().zip(vf.iter()).zip(v_residual.iter())
                        .map(|((&vi, &vfi), &vri)| vi + (vfi - vi) * vri)
                        .collect();
                    #[cfg(feature = "hip-probes")]
                    hip_probe!(self, probe_ctx, probe::HipHook::PostAttValueResidual, &v, [n_embd, t, b]);
                }
            } else {
                v_first = Some(v.clone());
            }

            // L2 normalize k
            let k_k = layer.att.k_k.to_vec(&stream)?;
            let k_scaled: Vec<f32> = k.iter().zip(k_k.iter().cycle())
                .map(|(&ki, &kki)| ki * kki)
                .collect();
            let kk = hip_l2_norm(&k_scaled, n_embd, t * b, head_size, 1e-12)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttL2Norm, &kk, [n_embd, t, b]);

            // Control K: k = k * (1 + (a - 1) * k_a)
            let k_a = layer.att.k_a.to_vec(&stream)?;
            let k_ctrl = hip_control_k(&k_a, &a, &k, n_embd, t, b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttControlK, &k_ctrl, [n_embd, t, b]);

            // WKV inputs
            let wkv_a: Vec<f32> = kk.iter().map(|&x| -x).collect();
            let wkv_b: Vec<f32> = kk.iter().zip(a.iter()).map(|(&kki, &ai)| kki * ai).collect();
            let w_decay: Vec<f32> = w.iter().map(|&wi| wi.exp()).collect();

            #[cfg(feature = "hip-probes")]
            {
                // Stack WKV inputs: w_decay, r, k_ctrl, v, wkv_a, wkv_b
                let stacked: Vec<f32> = [&w_decay, &r, &k_ctrl, &v, &wkv_a, &wkv_b]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PreWkv, &stacked, [n_embd, t, b, 6]);
                hip_probe!(self, probe_ctx, probe::HipHook::PreWkvState, &state.att_states[layer_idx], [head_size, head_size, n_head, b]);
            }

            // Run WKV7
            let (wkv_output, new_att_state) = hip_wkv7(
                &w_decay, &r, &k_ctrl, &v, &wkv_a, &wkv_b,
                &state.att_states[layer_idx], head_size, n_head, t, b
            )?;
            state.att_states[layer_idx] = new_att_state;
            #[cfg(feature = "hip-probes")]
            {
                hip_probe!(self, probe_ctx, probe::HipHook::PostWkv, &wkv_output, [n_embd, t, b]);
                hip_probe!(self, probe_ctx, probe::HipHook::PostWkvState, &state.att_states[layer_idx], [head_size, head_size, n_head, b]);
            }

            // Group norm on WKV output (BEFORE adding bonus, per RWKV7 spec)
            let gn_w = layer.att.gn.weight.to_vec(&stream)?;
            let gn_b = layer.att.gn.bias.to_vec(&stream)?;
            let wkv_normed = hip_group_norm(&wkv_output, &gn_w, &gn_b, n_embd, t * b, n_head, 64e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGroupNorm, &wkv_normed, [n_embd, t, b]);

            // WKV bonus (time_first) - added AFTER group norm
            let r_k = layer.att.r_k.to_vec(&stream)?;
            let wkv_bonus = hip_wkv_bonus(&r, &k_ctrl, &v, &r_k, head_size, n_head, t, b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostWkvBonus, &wkv_bonus, [n_embd, t, b]);

            // Combine normalized output and bonus: p_t = LayerNorm(wkv) + u_t
            let x_att_combined: Vec<f32> = wkv_normed.iter().zip(wkv_bonus.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // Gate and output projection
            let x_att_gated: Vec<f32> = x_att_combined.iter().zip(g.iter())
                .map(|(&xi, &gi)| xi * gi)
                .collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGated, &x_att_gated, [n_embd, t, b]);
            let w_o = layer.att.w_o.to_vec(&stream)?;
            let x_att_out = hip_sgemm(&w_o, &x_att_gated, n_embd, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttOut, &x_att_out, [n_embd, t, b]);

            // Residual
            x = x.iter().zip(x_att_out.iter()).map(|(&a, &b)| a + b).collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAtt, &x, [n_embd, t, b]);

            // ==== Channel-Mix (FFN) ====
            let ln2_w = layer.ffn_ln.weight.to_vec(&stream)?;
            let ln2_b = layer.ffn_ln.bias.to_vec(&stream)?;
            let x_ln2 = hip_layer_norm(&x, &ln2_w, &ln2_b, n_embd, t * b, 1e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLayerNorm, &x_ln2, [n_embd, t, b]);

            // Token shift for FFN
            let ffn_x_k = layer.ffn.x_k.to_vec(&stream)?;
            let (xk_ffn, new_ffn_state) = hip_channel_mix_state(&x_ln2, &state.ffn_states[layer_idx], &ffn_x_k, n_embd, t, b)?;
            state.ffn_states[layer_idx] = new_ffn_state;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnTokenShift, &xk_ffn, [n_embd, t, b]);

            // Key projection + squared ReLU
            let ffn_w_k = layer.ffn.w_k.to_vec(&stream)?;
            let k_ffn = hip_sgemm(&ffn_w_k, &xk_ffn, n_hidden, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLinear, &k_ffn, [n_hidden, t, b]);
            let k_sq = hip_squared_relu(&k_ffn)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnActivate, &k_sq, [n_hidden, t, b]);

            // Value projection
            let ffn_w_v = layer.ffn.w_v.to_vec(&stream)?;
            let x_ffn_out = hip_sgemm(&ffn_w_v, &k_sq, n_embd, n_hidden, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnOut, &x_ffn_out, [n_embd, t, b]);

            // Residual
            x = x.iter().zip(x_ffn_out.iter()).map(|(&a, &b)| a + b).collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfn, &x, [n_embd, t, b]);
        }

        // Reset layer context for head probes
        #[cfg(feature = "hip-probes")]
        { probe_ctx.layer = None; }

        // ==== Output Head ====
        let ln_out_w = self.head.ln.weight.to_vec(&stream)?;
        let ln_out_b = self.head.ln.bias.to_vec(&stream)?;
        let x_ln_out = hip_layer_norm(&x, &ln_out_w, &ln_out_b, n_embd, t * b, 1e-5)?;
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostHeadLayerNorm, &x_ln_out, [n_embd, t, b]);

        // Head projection
        let head_w = self.head.w.to_vec(&stream)?;
        let logits = hip_sgemm(&head_w, &x_ln_out, n_vocab, n_embd, t * b)?;
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostHead, &logits, [n_vocab, t, b]);

        Ok(logits)
    }

    /// Forward pass with length masking for variable-length batched sequences.
    ///
    /// This method allows processing batches where sequences have different real lengths,
    /// with padding to a common max length. The masked WKV7 kernel skips state updates
    /// for padding positions, preserving the invariant: state(seq + padding) == state(seq).
    ///
    /// # Arguments
    /// * `tokens` - Padded token sequences, all same length T
    /// * `lengths` - Real sequence length per batch (lengths[b] <= T)
    /// * `state` - Mutable state, updated in place
    ///
    /// # Returns
    /// Logits tensor [vocab_size, T, B] - only positions < lengths[b] are valid for each batch
    ///
    /// # Example
    /// ```rust,ignore
    /// let model = Rwkv7Hip::load("model.safetensors")?;
    /// let mut state = HipState::new(&model.info, 2);
    ///
    /// // Batch with different lengths (padded with 0)
    /// let seq1 = vec![1, 2, 3, 0, 0];     // real length = 3
    /// let seq2 = vec![10, 20, 30, 40, 50]; // real length = 5
    /// let tokens: Vec<&[u32]> = vec![&seq1, &seq2];
    /// let lengths = vec![3, 5];
    ///
    /// let logits = model.forward_with_state_masked(&tokens, &lengths, &mut state)?;
    /// // State for batch 0 matches unbatched processing of [1, 2, 3]
    /// // State for batch 1 matches unbatched processing of [10, 20, 30, 40, 50]
    /// ```
    pub fn forward_with_state_masked(
        &self,
        tokens: &[&[u32]],
        lengths: &[usize],
        state: &mut HipState,
    ) -> Result<Vec<f32>> {
        let b = tokens.len();
        if b == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }
        if b != state.batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Batch size mismatch: tokens has {} sequences but state has batch_size={}",
                    b, state.batch_size
                ),
            });
        }
        if lengths.len() != b {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Lengths size mismatch: expected {} (batch size), got {}",
                    b, lengths.len()
                ),
            });
        }

        let t = tokens[0].len();
        if t == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty sequence".to_string(),
            });
        }
        // Verify all sequences have the same length (padded to max)
        for (i, seq) in tokens.iter().enumerate() {
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
        // Verify lengths[b] <= T for all batches
        for (i, &len) in lengths.iter().enumerate() {
            if len > t {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "Length {} for batch {} exceeds padded sequence length {}",
                        len, i, t
                    ),
                });
            }
            if len == 0 {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!("Length for batch {} is 0 (empty sequence)", i),
                });
            }
        }

        // Convert lengths to i32 for the kernel
        let lengths_i32: Vec<i32> = lengths.iter().map(|&l| l as i32).collect();

        let stream = Stream::null();
        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;
        let n_vocab = self.info.n_vocab;
        let n_layer = self.info.n_layer;
        let n_hidden = self.info.n_hidden;

        // Initialize probe context (compiles out without feature)
        #[cfg(feature = "hip-probes")]
        let mut probe_ctx = probe::ProbeContext {
            layer: None,
            batch_size: b,
            seq_len: t,
            n_embd,
            n_head,
            head_size,
            n_layer,
            shape_storage: [0; probe::MAX_SHAPE_DIMS],
            shape_len: 0,
        };

        // Embedding lookup: tokens[b][t] -> x[c, t, b]
        let emb_data = self.embed.w.to_vec(&stream)?;
        let mut x = vec![0.0f32; n_embd * t * b];
        for batch_idx in 0..b {
            for time_idx in 0..t {
                let token = tokens[batch_idx][time_idx] as usize;
                for c in 0..n_embd {
                    let idx = batch_idx * t * n_embd + time_idx * n_embd + c;
                    x[idx] = emb_data[token * n_embd + c];
                }
            }
        }
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostEmbed, &x, [n_embd, t, b]);

        // v_first is computed fresh each forward call (not part of persistent state)
        let mut v_first: Option<Vec<f32>> = None;

        // Process each layer
        for layer_idx in 0..n_layer {
            #[cfg(feature = "hip-probes")]
            { probe_ctx.layer = Some(layer_idx); }
            let layer = &self.layers[layer_idx];

            // Apply ln0 for layer 0
            if layer_idx == 0 {
                let ln0_w = self.embed.ln.weight.to_vec(&stream)?;
                let ln0_b = self.embed.ln.bias.to_vec(&stream)?;
                x = hip_layer_norm(&x, &ln0_w, &ln0_b, n_embd, t * b, 1e-5)?;
                #[cfg(feature = "hip-probes")]
                hip_probe!(self, probe_ctx, probe::HipHook::PostEmbedLayerNorm, &x, [n_embd, t, b]);
            }

            // ==== Time-Mix (Attention) ====
            let ln1_w = layer.att_ln.weight.to_vec(&stream)?;
            let ln1_b = layer.att_ln.bias.to_vec(&stream)?;
            let x_ln1 = hip_layer_norm(&x, &ln1_w, &ln1_b, n_embd, t * b, 1e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttLayerNorm, &x_ln1, [n_embd, t, b]);

            // Token shift for attention - all shifts use the same old state
            let att_shift_state = &state.att_shift_states[layer_idx];
            let x_r = layer.att.x_r.to_vec(&stream)?;
            let x_w = layer.att.x_w.to_vec(&stream)?;
            let x_k = layer.att.x_k.to_vec(&stream)?;
            let x_v = layer.att.x_v.to_vec(&stream)?;
            let x_a = layer.att.x_a.to_vec(&stream)?;
            let x_g = layer.att.x_g.to_vec(&stream)?;

            let (xr, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_r, n_embd, t, b)?;
            let (xw, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_w, n_embd, t, b)?;
            let (xk, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_k, n_embd, t, b)?;
            let (xv, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_v, n_embd, t, b)?;
            let (xa, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_a, n_embd, t, b)?;
            let (xg, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_g, n_embd, t, b)?;
            #[cfg(feature = "hip-probes")]
            {
                // Stack all token shifts: xr, xw, xk, xv, xa, xg
                let stacked: Vec<f32> = [&xr, &xw, &xk, &xv, &xa, &xg]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PostAttTokenShift, &stacked, [n_embd, t, b, 6]);
            }

            // Update shift state: use last valid position for each batch (not padding)
            state.att_shift_states[layer_idx] = extract_shift_state_at_lengths(&x_ln1, lengths, n_embd, t, b);

            // Linear projections: r, k, v
            let w_r = layer.att.w_r.to_vec(&stream)?;
            let w_k = layer.att.w_k.to_vec(&stream)?;
            let w_v = layer.att.w_v.to_vec(&stream)?;
            let r = hip_sgemm(&w_r, &xr, n_embd, n_embd, t * b)?;
            let k = hip_sgemm(&w_k, &xk, n_embd, n_embd, t * b)?;
            let mut v = hip_sgemm(&w_v, &xv, n_embd, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            {
                // Stack r, k, v for linear projection probe
                let stacked: Vec<f32> = [&r, &k, &v]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PostAttLinear, &stacked, [n_embd, t, b, 3]);
            }

            // w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
            let w0 = layer.att.w0.to_vec(&stream)?;
            let w1 = layer.att.w1.to_vec(&stream)?;
            let w2 = layer.att.w2.to_vec(&stream)?;
            let w1_dim = layer.att.w1.shape().dim(0);
            let w_lora1 = hip_sgemm(&w1, &xw, w1_dim, n_embd, t * b)?;
            let w_lora1_tanh = hip_tanh(&w_lora1)?;
            let w_lora2 = hip_sgemm(&w2, &w_lora1_tanh, n_embd, w1_dim, t * b)?;
            let mut w: Vec<f32> = w0.iter().cycle().take(w_lora2.len())
                .zip(w_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            w = hip_softplus_decay(&w)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttDecay, &w, [n_embd, t, b]);

            // a = sigmoid(a0 + (xa @ a1) @ a2)
            let a0 = layer.att.a0.to_vec(&stream)?;
            let a1 = layer.att.a1.to_vec(&stream)?;
            let a2 = layer.att.a2.to_vec(&stream)?;
            let a1_dim = layer.att.a1.shape().dim(0);
            let a_lora1 = hip_sgemm(&a1, &xa, a1_dim, n_embd, t * b)?;
            let a_lora2 = hip_sgemm(&a2, &a_lora1, n_embd, a1_dim, t * b)?;
            let a_biased: Vec<f32> = a0.iter().cycle().take(a_lora2.len())
                .zip(a_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let a = hip_sigmoid(&a_biased)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttAdapt, &a, [n_embd, t, b]);

            // g = sigmoid(xg @ g1) @ g2
            let g1 = layer.att.g1.to_vec(&stream)?;
            let g2 = layer.att.g2.to_vec(&stream)?;
            let g1_dim = layer.att.g1.shape().dim(0);
            let g_lora1 = hip_sgemm(&g1, &xg, g1_dim, n_embd, t * b)?;
            let g_lora1_sigmoid = hip_sigmoid(&g_lora1)?;
            let g = hip_sgemm(&g2, &g_lora1_sigmoid, n_embd, g1_dim, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGate, &g, [n_embd, t, b]);

            // Value residual (layers > 0)
            if layer_idx > 0 {
                if let (Some(v0), Some(v1), Some(v2), Some(ref vf)) =
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2, &v_first) {
                    let v0_data = v0.to_vec(&stream)?;
                    let v1_data = v1.to_vec(&stream)?;
                    let v2_data = v2.to_vec(&stream)?;
                    let v1_dim = v1.shape().dim(0);
                    // Note: Use xv (token-shifted input), not v (projected value)
                    let v_lora1 = hip_sgemm(&v1_data, &xv, v1_dim, n_embd, t * b)?;
                    let v_lora2 = hip_sgemm(&v2_data, &v_lora1, n_embd, v1_dim, t * b)?;
                    let v_biased: Vec<f32> = v0_data.iter().cycle().take(v_lora2.len())
                        .zip(v_lora2.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    let v_residual = hip_sigmoid(&v_biased)?;
                    // v = v + (v_first - v) * v_residual
                    v = v.iter().zip(vf.iter()).zip(v_residual.iter())
                        .map(|((&vi, &vfi), &vri)| vi + (vfi - vi) * vri)
                        .collect();
                    #[cfg(feature = "hip-probes")]
                    hip_probe!(self, probe_ctx, probe::HipHook::PostAttValueResidual, &v, [n_embd, t, b]);
                }
            } else {
                v_first = Some(v.clone());
            }

            // L2 normalize k
            let k_k = layer.att.k_k.to_vec(&stream)?;
            let k_scaled: Vec<f32> = k.iter().zip(k_k.iter().cycle())
                .map(|(&ki, &kki)| ki * kki)
                .collect();
            let kk = hip_l2_norm(&k_scaled, n_embd, t * b, head_size, 1e-12)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttL2Norm, &kk, [n_embd, t, b]);

            // Control K: k = k * (1 + (a - 1) * k_a)
            let k_a = layer.att.k_a.to_vec(&stream)?;
            let k_ctrl = hip_control_k(&k_a, &a, &k, n_embd, t, b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttControlK, &k_ctrl, [n_embd, t, b]);

            // WKV inputs
            let wkv_a: Vec<f32> = kk.iter().map(|&x| -x).collect();
            let wkv_b: Vec<f32> = kk.iter().zip(a.iter()).map(|(&kki, &ai)| kki * ai).collect();
            let w_decay: Vec<f32> = w.iter().map(|&wi| wi.exp()).collect();

            #[cfg(feature = "hip-probes")]
            {
                // Stack WKV inputs: w_decay, r, k_ctrl, v, wkv_a, wkv_b
                let stacked: Vec<f32> = [&w_decay, &r, &k_ctrl, &v, &wkv_a, &wkv_b]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                hip_probe!(self, probe_ctx, probe::HipHook::PreWkv, &stacked, [n_embd, t, b, 6]);
                hip_probe!(self, probe_ctx, probe::HipHook::PreWkvState, &state.att_states[layer_idx], [head_size, head_size, n_head, b]);
            }

            // Run WKV7 with length masking
            let (wkv_output, new_att_state) = hip_wkv7_masked(
                &w_decay, &r, &k_ctrl, &v, &wkv_a, &wkv_b,
                &state.att_states[layer_idx], &lengths_i32, head_size, n_head, t, b
            )?;
            state.att_states[layer_idx] = new_att_state;
            #[cfg(feature = "hip-probes")]
            {
                hip_probe!(self, probe_ctx, probe::HipHook::PostWkv, &wkv_output, [n_embd, t, b]);
                hip_probe!(self, probe_ctx, probe::HipHook::PostWkvState, &state.att_states[layer_idx], [head_size, head_size, n_head, b]);
            }

            // Group norm on WKV output (BEFORE adding bonus, per RWKV7 spec)
            let gn_w = layer.att.gn.weight.to_vec(&stream)?;
            let gn_b = layer.att.gn.bias.to_vec(&stream)?;
            let wkv_normed = hip_group_norm(&wkv_output, &gn_w, &gn_b, n_embd, t * b, n_head, 64e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGroupNorm, &wkv_normed, [n_embd, t, b]);

            // WKV bonus (time_first) - added AFTER group norm
            let r_k = layer.att.r_k.to_vec(&stream)?;
            let wkv_bonus = hip_wkv_bonus(&r, &k_ctrl, &v, &r_k, head_size, n_head, t, b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostWkvBonus, &wkv_bonus, [n_embd, t, b]);

            // Combine normalized output and bonus: p_t = LayerNorm(wkv) + u_t
            let x_att_combined: Vec<f32> = wkv_normed.iter().zip(wkv_bonus.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // Gate and output projection
            let x_att_gated: Vec<f32> = x_att_combined.iter().zip(g.iter())
                .map(|(&xi, &gi)| xi * gi)
                .collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttGated, &x_att_gated, [n_embd, t, b]);
            let w_o = layer.att.w_o.to_vec(&stream)?;
            let x_att_out = hip_sgemm(&w_o, &x_att_gated, n_embd, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAttOut, &x_att_out, [n_embd, t, b]);

            // Residual
            x = x.iter().zip(x_att_out.iter()).map(|(&a, &b)| a + b).collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostAtt, &x, [n_embd, t, b]);

            // ==== Channel-Mix (FFN) ====
            let ln2_w = layer.ffn_ln.weight.to_vec(&stream)?;
            let ln2_b = layer.ffn_ln.bias.to_vec(&stream)?;
            let x_ln2 = hip_layer_norm(&x, &ln2_w, &ln2_b, n_embd, t * b, 1e-5)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLayerNorm, &x_ln2, [n_embd, t, b]);

            // Token shift for FFN
            let ffn_x_k = layer.ffn.x_k.to_vec(&stream)?;
            let (xk_ffn, _) = hip_channel_mix_state(&x_ln2, &state.ffn_states[layer_idx], &ffn_x_k, n_embd, t, b)?;
            // Update FFN state: use last valid position for each batch (not padding)
            state.ffn_states[layer_idx] = extract_shift_state_at_lengths(&x_ln2, lengths, n_embd, t, b);
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnTokenShift, &xk_ffn, [n_embd, t, b]);

            // Key projection + squared ReLU
            let ffn_w_k = layer.ffn.w_k.to_vec(&stream)?;
            let k_ffn = hip_sgemm(&ffn_w_k, &xk_ffn, n_hidden, n_embd, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnLinear, &k_ffn, [n_hidden, t, b]);
            let k_sq = hip_squared_relu(&k_ffn)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnActivate, &k_sq, [n_hidden, t, b]);

            // Value projection
            let ffn_w_v = layer.ffn.w_v.to_vec(&stream)?;
            let x_ffn_out = hip_sgemm(&ffn_w_v, &k_sq, n_embd, n_hidden, t * b)?;
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfnOut, &x_ffn_out, [n_embd, t, b]);

            // Residual
            x = x.iter().zip(x_ffn_out.iter()).map(|(&a, &b)| a + b).collect();
            #[cfg(feature = "hip-probes")]
            hip_probe!(self, probe_ctx, probe::HipHook::PostFfn, &x, [n_embd, t, b]);
        }

        // Reset layer context for head probes
        #[cfg(feature = "hip-probes")]
        { probe_ctx.layer = None; }

        // ==== Output Head ====
        let ln_out_w = self.head.ln.weight.to_vec(&stream)?;
        let ln_out_b = self.head.ln.bias.to_vec(&stream)?;
        let x_ln_out = hip_layer_norm(&x, &ln_out_w, &ln_out_b, n_embd, t * b, 1e-5)?;
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostHeadLayerNorm, &x_ln_out, [n_embd, t, b]);

        // Head projection
        let head_w = self.head.w.to_vec(&stream)?;
        let logits = hip_sgemm(&head_w, &x_ln_out, n_vocab, n_embd, t * b)?;
        #[cfg(feature = "hip-probes")]
        hip_probe!(self, probe_ctx, probe::HipHook::PostHead, &logits, [n_vocab, t, b]);

        Ok(logits)
    }
}

