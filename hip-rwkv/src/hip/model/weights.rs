//! Weight structs and safetensors loading for the RWKV7 HIP backend.

use half::f16;

use super::{ModelLoadError, Rwkv7ModelInfo};
use crate::hip::device::Stream;
use crate::hip::tensor::{TensorHip, TensorShape};

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
    pub w_r: TensorHip<f16>, // Receptance: [n_embd, n_embd]
    pub w_k: TensorHip<f16>, // Key: [n_embd, n_embd]
    pub w_v: TensorHip<f16>, // Value: [n_embd, n_embd]
    pub w_o: TensorHip<f16>, // Output: [n_embd, n_embd]

    // Group normalization
    pub gn: LayerNormHip,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnHip {
    // Token shift mix weight
    pub x_k: TensorHip<f16>,

    // Projection matrices
    pub w_k: TensorHip<f16>, // Key (expand): [n_hidden, n_embd]
    pub w_v: TensorHip<f16>, // Value (contract): [n_embd, n_hidden]
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
    /// Embedding table kept on CPU to avoid GPU->CPU transfer per forward.
    /// Shape: [n_vocab, n_embd] in row-major order (token_id * n_embd + c).
    pub w: Vec<f16>,
    /// Embedding dimension (n_embd)
    pub n_embd: usize,
}

/// Output head weights.
#[derive(Debug)]
pub struct HeadHip {
    pub ln: LayerNormHip,
    pub w: TensorHip<f16>, // [n_vocab, n_embd]
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
    let tensor = st
        .tensor(name)
        .map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

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
            bf16_slice
                .iter()
                .map(|x| f16::from_f32(x.to_f32()))
                .collect()
        }
        _ => {
            return Err(ModelLoadError::InvalidModel(format!(
                "Unsupported dtype {:?} for tensor {}",
                dtype, name
            )))
        }
    };

    // Convert shape to TensorShape (web-rwkv convention: shape[0] is fastest axis)
    // SafeTensors stores shape as [slow, ..., fast], so we need to reverse
    let hip_shape = match shape_st.len() {
        1 => TensorShape::new(shape_st[0], 1, 1, 1),
        2 => TensorShape::new(shape_st[1], shape_st[0], 1, 1),
        3 => TensorShape::new(shape_st[2], shape_st[1], shape_st[0], 1),
        4 => TensorShape::new(shape_st[3], shape_st[2], shape_st[1], shape_st[0]),
        _ => {
            return Err(ModelLoadError::InvalidModel(format!(
                "Unsupported shape {:?} for tensor {}",
                shape_st, name
            )))
        }
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
    let tensor = st
        .tensor(name)
        .map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

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
            bf16_slice
                .iter()
                .map(|x| f16::from_f32(x.to_f32()))
                .collect()
        }
        _ => {
            return Err(ModelLoadError::InvalidModel(format!(
                "Unsupported dtype {:?} for tensor {}",
                dtype, name
            )))
        }
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
    let tensor = st
        .tensor(name)
        .map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    if shape_st.len() != 2 {
        return Err(ModelLoadError::InvalidModel(format!(
            "Weight matrix {} must be 2D, got {:?}",
            name, shape_st
        )));
    }

    let rows = shape_st[0]; // out_features (M)
    let cols = shape_st[1]; // in_features (K)

    // Convert to f16 vec
    let f16_data: Vec<f16> = match dtype {
        safetensors::Dtype::F16 => bytemuck::cast_slice(data).to_vec(),
        safetensors::Dtype::F32 => bytemuck::cast_slice(data)
            .iter()
            .map(|x: &f32| f16::from_f32(*x))
            .collect(),
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice
                .iter()
                .map(|x| f16::from_f32(x.to_f32()))
                .collect()
        }
        _ => {
            return Err(ModelLoadError::InvalidModel(format!(
                "Unsupported dtype {:?} for tensor {}",
                dtype, name
            )))
        }
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
pub(super) fn load_layer_norm(
    st: &safetensors::SafeTensors,
    prefix: &str,
    stream: &Stream,
) -> std::result::Result<LayerNormHip, ModelLoadError> {
    let weight = load_tensor_f16(st, &format!("{}.weight", prefix), stream)?;
    let bias = load_tensor_f16(st, &format!("{}.bias", prefix), stream)?;
    Ok(LayerNormHip { weight, bias })
}

/// Load all layers from SafeTensors.
pub(super) fn load_layers(
    st: &safetensors::SafeTensors,
    info: &Rwkv7ModelInfo,
    stream: &Stream,
) -> std::result::Result<Vec<LayerHip>, ModelLoadError> {
    let mut layers = Vec::with_capacity(info.n_layer);
    for layer_idx in 0..info.n_layer {
        let prefix = format!("blocks.{}", layer_idx);

        // Attention layer norm
        let att_ln = load_layer_norm(st, &format!("{}.ln1", prefix), stream)?;

        // FFN layer norm
        let ffn_ln = load_layer_norm(st, &format!("{}.ln2", prefix), stream)?;

        // Attention weights
        let att = AttentionHip {
            x_r: load_tensor_f16(st, &format!("{}.att.x_r", prefix), stream)?,
            x_w: load_tensor_f16(st, &format!("{}.att.x_w", prefix), stream)?,
            x_k: load_tensor_f16(st, &format!("{}.att.x_k", prefix), stream)?,
            x_v: load_tensor_f16(st, &format!("{}.att.x_v", prefix), stream)?,
            x_a: load_tensor_f16(st, &format!("{}.att.x_a", prefix), stream)?,
            x_g: load_tensor_f16(st, &format!("{}.att.x_g", prefix), stream)?,

            w0: load_tensor_f16(st, &format!("{}.att.w0", prefix), stream)?,
            w1: load_weight_matrix_f16(st, &format!("{}.att.w1", prefix), stream)?,
            w2: load_weight_matrix_f16(st, &format!("{}.att.w2", prefix), stream)?,

            a0: load_tensor_f16(st, &format!("{}.att.a0", prefix), stream)?,
            a1: load_weight_matrix_f16(st, &format!("{}.att.a1", prefix), stream)?,
            a2: load_weight_matrix_f16(st, &format!("{}.att.a2", prefix), stream)?,

            g1: load_weight_matrix_f16(st, &format!("{}.att.g1", prefix), stream)?,
            g2: load_weight_matrix_f16(st, &format!("{}.att.g2", prefix), stream)?,

            // Value residual LoRA (only for layers > 0)
            v0: if layer_idx > 0 {
                Some(load_tensor_f16(
                    st,
                    &format!("{}.att.v0", prefix),
                    stream,
                )?)
            } else {
                None
            },
            v1: if layer_idx > 0 {
                Some(load_weight_matrix_f16(
                    st,
                    &format!("{}.att.v1", prefix),
                    stream,
                )?)
            } else {
                None
            },
            v2: if layer_idx > 0 {
                Some(load_weight_matrix_f16(
                    st,
                    &format!("{}.att.v2", prefix),
                    stream,
                )?)
            } else {
                None
            },

            r_k: load_tensor_f16(st, &format!("{}.att.r_k", prefix), stream)?,
            k_k: load_tensor_f16(st, &format!("{}.att.k_k", prefix), stream)?,
            k_a: load_tensor_f16(st, &format!("{}.att.k_a", prefix), stream)?,

            w_r: load_weight_matrix_f16(
                st,
                &format!("{}.att.receptance.weight", prefix),
                stream,
            )?,
            w_k: load_weight_matrix_f16(st, &format!("{}.att.key.weight", prefix), stream)?,
            w_v: load_weight_matrix_f16(st, &format!("{}.att.value.weight", prefix), stream)?,
            w_o: load_weight_matrix_f16(
                st,
                &format!("{}.att.output.weight", prefix),
                stream,
            )?,

            gn: load_layer_norm(st, &format!("{}.att.ln_x", prefix), stream)?,
        };

        // FFN weights (transpose weight matrices to column-major for GEMM)
        let ffn = FfnHip {
            x_k: load_tensor_f16(st, &format!("{}.ffn.x_k", prefix), stream)?,
            w_k: load_weight_matrix_f16(st, &format!("{}.ffn.key.weight", prefix), stream)?,
            w_v: load_weight_matrix_f16(st, &format!("{}.ffn.value.weight", prefix), stream)?,
        };

        layers.push(LayerHip {
            att_ln,
            ffn_ln,
            att,
            ffn,
        });
    }
    Ok(layers)
}

/// Load embedding weights.
pub(super) fn load_embedding(
    st: &safetensors::SafeTensors,
    n_embd: usize,
    stream: &Stream,
) -> std::result::Result<EmbedHip, ModelLoadError> {
    Ok(EmbedHip {
        ln: load_layer_norm(st, "blocks.0.ln0", stream)?,
        w: load_tensor_f16_cpu(st, "emb.weight")?,
        n_embd,
    })
}

/// Load output head weights.
pub(super) fn load_head(
    st: &safetensors::SafeTensors,
    stream: &Stream,
) -> std::result::Result<HeadHip, ModelLoadError> {
    Ok(HeadHip {
        ln: load_layer_norm(st, "ln_out", stream)?,
        w: load_weight_matrix_f16(st, "head.weight", stream)?,
    })
}
