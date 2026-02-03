//! RWKV7 HIP model loading and step (inference) implementation.

pub mod decode;
pub mod fla;
pub mod hip_prefill;
pub mod prefill;
pub mod state;
mod step;
pub mod weights;

// Re-export all public types from submodules
pub use decode::HipDecode;
pub use fla::FlaChunkedWkv;
pub use hip_prefill::HipPrefill;
pub use prefill::{FusedT1Wkv, WaveReduceWkv, WkvInput, WkvKernel};
pub use state::{HipState, StateLayout};
pub use weights::{
    AttentionHip, EmbedHip, FfnHip, HeadHip, LayerHip, LayerNormHip,
};

use half::f16;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::device::Stream;
use super::ffi::{HipErrorKind, Result};
use super::scratch::PrefillScratch;
use super::scratch::{HipRuntimeConfig, LoraDims};
use super::tensor::TensorHip;

#[cfg(feature = "hip-probes")]
use super::probe::{self, HipProbeMap, HipProbeMapRef};

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

/// Shared model weights for RWKV7 on HIP.
///
/// Contains the immutable weight data (embedding, layers, output head) that
/// can be shared across multiple inference modules (e.g., HipPrefill, HipDecode)
/// via `Arc<Rwkv7Model>`. Weights are stored in managed memory for zero-copy
/// APU access. Weights use FP16; recurrent WKV state remains FP32.
pub struct Rwkv7Model {
    pub info: Rwkv7ModelInfo,
    pub embed: EmbedHip,
    pub head: HeadHip,
    pub layers: Vec<LayerHip>,
}

impl std::fmt::Debug for Rwkv7Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rwkv7Model")
            .field("info", &self.info)
            .field("embed", &self.embed)
            .field("head", &self.head)
            .field("layers", &self.layers)
            .finish()
    }
}

/// RWKV7 model loaded into HIP memory.
///
/// Convenience wrapper around `Arc<Rwkv7Model>` that also owns scratch buffers
/// and probe state. For shared access to model weights (e.g., from HipPrefill
/// and HipDecode), use the `model()` method to get the `Arc<Rwkv7Model>`.
///
/// Weights and activations use FP16; recurrent WKV state remains FP32.
pub struct Rwkv7Hip {
    /// Shared model weights wrapped in Arc for multi-module sharing.
    pub(crate) model: Arc<Rwkv7Model>,

    /// Lazily initialized scratch buffers for GPU-native forward pass.
    pub(crate) scratch: Mutex<Option<PrefillScratch>>,

    #[cfg(feature = "hip-probes")]
    pub(crate) probes: Option<HipProbeMapRef>,
}

impl std::fmt::Debug for Rwkv7Hip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Rwkv7Hip");
        s.field("model", &self.model)
            .field(
                "scratch",
                &self.scratch.lock().unwrap().as_ref().map(|_| "initialized"),
            );
        #[cfg(feature = "hip-probes")]
        s.field(
            "probes",
            &self.probes.as_ref().map(|p| format!("{} hooks", p.len())),
        );
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

impl Rwkv7Model {
    /// Load RWKV7 model weights from a SafeTensors file.
    ///
    /// Weights are loaded into managed (unified) memory for efficient APU access.
    /// Weights are stored as FP16; WKV state remains FP32.
    ///
    /// Returns `Arc<Rwkv7Model>` for shared ownership across inference modules.
    ///
    /// # Arguments
    /// * `path` - Path to the .st (SafeTensors) file
    ///
    /// # Returns
    /// The loaded model weights wrapped in Arc, or an error.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Arc<Self>, ModelLoadError> {
        let data = std::fs::read(path.as_ref())?;
        let st = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
            ModelLoadError::SafeTensor(format!("Failed to parse SafeTensors: {}", e))
        })?;

        let stream = Stream::null();

        // Detect model dimensions from tensor shapes
        let embed_tensor = st
            .tensor("emb.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("emb.weight: {}", e)))?;
        let embed_shape = embed_tensor.shape(); // [n_vocab, n_embd]
        let n_vocab = embed_shape[0];
        let n_embd = embed_shape[1];

        // Get n_head from r_k tensor (RWKV7-specific)
        let r_k_tensor = st
            .tensor("blocks.0.att.r_k")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.att.r_k: {}", e)))?;
        let n_head = r_k_tensor.shape()[0];
        let head_size = n_embd / n_head;

        // Get n_hidden from FFN key weight
        let ffn_k_tensor = st
            .tensor("blocks.0.ffn.key.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.ffn.key.weight: {}", e)))?;
        let n_hidden = ffn_k_tensor.shape()[0];

        // Count layers
        let n_layer = st
            .names()
            .iter()
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

        log::info!(
            "Loading RWKV7 model: {} layers, {} embd, {} heads, {} vocab",
            n_layer,
            n_embd,
            n_head,
            n_vocab
        );

        // Load embedding to CPU (lookup table accessed on CPU, avoids GPU->CPU transfer per forward)
        let embed = weights::load_embedding(&st, n_embd, &stream)?;

        // Load output head (transpose to column-major for GEMM)
        let head = weights::load_head(&st, &stream)?;

        // Load layers
        let layers = weights::load_layers(&st, &info, &stream)?;

        // Synchronize to ensure all transfers are complete
        stream.synchronize()?;

        log::info!("RWKV7 model loaded successfully");

        Ok(Arc::new(Self {
            info,
            embed,
            head,
            layers,
        }))
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
    pub fn read_weight_head(
        &self,
        name: &str,
        n: usize,
    ) -> std::result::Result<Vec<f32>, ModelLoadError> {
        let tensor = self
            .get_weight(name)
            .ok_or_else(|| ModelLoadError::InvalidModel(format!("Weight not found: {}", name)))?;

        let stream = Stream::null();
        let all_data = tensor.to_vec(&stream)?;
        let n = n.min(all_data.len());
        Ok(all_data[..n].iter().map(|v| v.to_f32()).collect())
    }

    /// Get a reference to the embedding table (CPU storage).
    ///
    /// The embedding table is kept on CPU to avoid GPU->CPU transfer overhead per step() call.
    /// Shape: [n_vocab, n_embd] in row-major order (token_id * n_embd + c).
    pub fn get_embedding(&self) -> &[f16] {
        &self.embed.w
    }
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
        let model = Rwkv7Model::load(path)?;
        Ok(Self {
            model,
            scratch: Mutex::new(None),
            #[cfg(feature = "hip-probes")]
            probes: None,
        })
    }

    /// Create an Rwkv7Hip from a pre-loaded Arc<Rwkv7Model>.
    ///
    /// Useful when sharing model weights across multiple inference modules.
    pub fn from_model(model: Arc<Rwkv7Model>) -> Self {
        Self {
            model,
            scratch: Mutex::new(None),
            #[cfg(feature = "hip-probes")]
            probes: None,
        }
    }

    /// Get a clone of the shared model Arc.
    ///
    /// Use this to share model weights with other inference modules
    /// (e.g., HipPrefill, HipDecode).
    pub fn model(&self) -> Arc<Rwkv7Model> {
        self.model.clone()
    }

    /// Access model info (dimensions, vocab size, etc.).
    ///
    /// Convenience accessor that delegates to `self.model.info`.
    pub fn info(&self) -> &Rwkv7ModelInfo {
        &self.model.info
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

    /// Delegate to `Rwkv7Model::get_weight()`.
    pub fn get_weight(&self, name: &str) -> Option<&TensorHip<f16>> {
        self.model.get_weight(name)
    }

    /// Delegate to `Rwkv7Model::read_weight_head()`.
    pub fn read_weight_head(
        &self,
        name: &str,
        n: usize,
    ) -> std::result::Result<Vec<f32>, ModelLoadError> {
        self.model.read_weight_head(name, n)
    }

    /// Delegate to `Rwkv7Model::get_embedding()`.
    pub fn get_embedding(&self) -> &[f16] {
        self.model.get_embedding()
    }

    /// Delegate to `Rwkv7Model::lora_dims()`.
    pub fn lora_dims(&self) -> LoraDims {
        self.model.lora_dims()
    }

    /// Configure the model with fixed scratch buffers.
    ///
    /// This pre-allocates all GPU buffers sized for the given configuration.
    /// Must be called before `step()` to enable inference.
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
        let lora_dims = self.model.lora_dims();
        let scratch = PrefillScratch::new(&self.model.info, lora_dims, config)?;
        *self.scratch.lock().unwrap() = Some(scratch);
        Ok(self)
    }

    /// Get the configured chunk size, if scratch buffers are initialized.
    pub fn chunk_size(&self) -> Option<usize> {
        self.scratch
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.config.max_prefill_chunk)
    }

    /// Get the configured batch size, if scratch buffers are initialized.
    pub fn max_batch_size(&self) -> Option<usize> {
        self.scratch
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.config.batch_size)
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

    /// Read the current GPU-resident state for a single batch item back to CPU.
    ///
    /// The returned `HipState` has `batch_size = 1` and contains the state
    /// for just the requested batch item.
    ///
    /// The caller must ensure no inference is in flight on this model when
    /// calling this method (the scratch mutex is held for the duration).
    ///
    /// Similar to WebGPU's `State::back(batch)` API.
    pub fn read_state(&self, batch_idx: usize) -> Result<HipState> {
        let mut scratch_ref = self.scratch.lock().unwrap();
        let scratch = scratch_ref.as_mut().ok_or_else(|| HipErrorKind {
            code: -1,
            message: "Scratch not initialized - call with_config() first".to_string(),
        })?;

        let batch_size = scratch.config.batch_size;
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }

        let stream = scratch.blas_ctx.stream();

        let att_states = HipState::read_batch_wkv(
            &scratch.wkv_state_gpu,
            batch_idx,
            batch_size,
            stream,
        )?;
        let att_shift_states = HipState::read_batch_att_shift(
            &scratch.att_shift_state_gpu,
            batch_idx,
            batch_size,
            stream,
        )?;
        let ffn_states = HipState::read_batch_ffn(
            &scratch.ffn_state_gpu,
            batch_idx,
            batch_size,
            stream,
        )?;

        // Synchronize to ensure all D2H copies complete
        stream.synchronize()?;

        Ok(HipState {
            batch_size: 1,
            att_states,
            att_shift_states,
            ffn_states,
            v_first: None,
            layout: state::StateLayout::Decode,
        })
    }

    /// Write a CPU state into the GPU-resident state for a single batch item.
    ///
    /// The provided `HipState` should have `batch_size = 1`.
    ///
    /// The caller must ensure no inference is in flight on this model when
    /// calling this method (the scratch mutex is held for the duration).
    ///
    /// Similar to WebGPU's `State::write(tensor, batch)` API.
    pub fn write_state(&self, batch_idx: usize, state: &HipState) -> Result<()> {
        let mut scratch_ref = self.scratch.lock().unwrap();
        let scratch = scratch_ref.as_mut().ok_or_else(|| HipErrorKind {
            code: -1,
            message: "Scratch not initialized - call with_config() first".to_string(),
        })?;

        let batch_size = scratch.config.batch_size;
        if batch_idx >= batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "batch_idx {} out of range for batch_size {}",
                    batch_idx, batch_size
                ),
            });
        }

        let stream = scratch.blas_ctx.stream();

        HipState::write_batch_wkv(
            &mut scratch.wkv_state_gpu,
            batch_idx,
            batch_size,
            &state.att_states,
            stream,
        )?;
        HipState::write_batch_att_shift(
            &mut scratch.att_shift_state_gpu,
            batch_idx,
            batch_size,
            &state.att_shift_states,
            stream,
        )?;
        HipState::write_batch_ffn(
            &mut scratch.ffn_state_gpu,
            batch_idx,
            batch_size,
            &state.ffn_states,
            stream,
        )?;

        // Synchronize to ensure all H2D copies complete
        stream.synchronize()?;

        Ok(())
    }
}

