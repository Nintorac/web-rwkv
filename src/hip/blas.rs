//! rocBLAS GEMV/GEMM wrapper functions.
//!
//! This module provides both low-level and high-level BLAS interfaces:
//!
//! - **Low-level**: `sgemm_f32`, `hgemm_f16` - raw rocBLAS wrappers
//! - **High-level**: `hip_sgemm` - convenience wrapper with automatic memory management
//! - **Context-based**: `HipBlasContext` - reusable handle for efficient batched operations
//!
//! For best performance in forward passes, use `HipBlasContext` to amortize
//! handle creation cost and enable device-to-device operations.

use std::ffi::c_int;

use super::ffi::{
    HipErrorKind, Result, RocblasHandle, ROCBLAS_STATUS_SUCCESS,
    rocblas_handle_create, rocblas_handle_destroy, rocblas_set_stream_wrapper,
    rocblas_to_hip_error,
    launch_hgemm, launch_sgemm, launch_sgemm_ta,
};
use super::device::Stream;
use super::tensor::{TensorShape, TensorHip};

// ============================================================================
// HipBlasContext - Reusable BLAS context for efficient batched operations
// ============================================================================

/// Long-lived BLAS context for efficient GPU compute.
///
/// Holds a rocBLAS handle bound to a single stream. Reuse this context across
/// multiple GEMM calls to amortize handle creation cost and enable kernel warmup.
///
/// # Performance Benefits
///
/// - **Handle reuse**: rocBLAS keeps per-handle temporary device memory
/// - **Stream binding**: All operations execute on the same stream for ordering
/// - **Warmup amortization**: First GEMM may be slower; subsequent calls benefit
///
/// # Usage
///
/// ```rust,ignore
/// // Create once at start of forward pass
/// let ctx = HipBlasContext::new()?;
///
/// // Reuse for all GEMMs
/// ctx.sgemm_into(&weight_r, &input, &mut output_r)?;
/// ctx.sgemm_into(&weight_k, &input, &mut output_k)?;
/// ctx.sgemm_into(&weight_v, &input, &mut output_v)?;
///
/// // Sync before reading results
/// ctx.synchronize()?;
/// ```
pub struct HipBlasContext {
    handle: RocblasHandle,
    stream: Stream,
}

impl HipBlasContext {
    /// Create a new BLAS context with a dedicated stream.
    pub fn new() -> Result<Self> {
        let stream = Stream::new()?;
        let handle = rocblas_create()?;
        rocblas_set_stream(handle, &stream)?;
        Ok(Self { handle, stream })
    }

    /// Create a BLAS context using the null (default) stream.
    ///
    /// Operations on the null stream are implicitly synchronized with
    /// other null-stream operations but may have less overlap potential.
    pub fn with_null_stream() -> Result<Self> {
        let stream = Stream::null();
        let handle = rocblas_create()?;
        rocblas_set_stream(handle, &stream)?;
        Ok(Self { handle, stream })
    }

    /// Get a reference to the underlying stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Synchronize the stream (wait for all enqueued operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Device-to-device SGEMM: output = weight @ input
    ///
    /// All tensors must be GPU-resident. No host copies occur.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix on GPU [M, K] (output_features × input_features)
    /// * `input` - Input matrix on GPU [K, N] (input_features × tokens)
    /// * `output` - Output matrix on GPU [M, N] (output_features × tokens), must be pre-allocated
    ///
    /// # Panics
    /// Panics if dimensions don't match.
    pub fn sgemm_into(
        &self,
        weight: &TensorHip<f32>,
        input: &TensorHip<f32>,
        output: &mut TensorHip<f32>,
    ) -> Result<()> {
        sgemm_f32(self.handle, weight, input, output)
    }

    /// Device-to-device HGEMM: output = weight @ input (FP16)
    ///
    /// All tensors must be GPU-resident. No host copies occur.
    pub fn hgemm_into(
        &self,
        weight: &TensorHip<f32>,  // Actually f16 storage
        input: &TensorHip<f32>,   // Actually f16 storage
        output: &mut TensorHip<f32>, // Actually f16 storage
    ) -> Result<()> {
        hgemm_f16(self.handle, weight, input, output)
    }

    /// Copy host data to a GPU tensor using this context's stream.
    ///
    /// This is useful for staging input data before GEMM operations.
    pub fn upload_to_tensor(&self, data: &[f32], tensor: &mut TensorHip<f32>) -> Result<()> {
        if data.len() != tensor.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Upload size mismatch: data has {} elements, tensor has {}",
                    data.len(),
                    tensor.len()
                ),
            });
        }
        tensor.copy_from_slice(data, &self.stream)
    }

    /// Copy GPU tensor data to host using this context's stream.
    ///
    /// Note: This operation synchronizes the stream before returning.
    pub fn download_from_tensor(&self, tensor: &TensorHip<f32>) -> Result<Vec<f32>> {
        tensor.to_vec(&self.stream)
    }
}

impl Drop for HipBlasContext {
    fn drop(&mut self) {
        // Best-effort cleanup - ignore errors in drop
        let _ = rocblas_destroy(self.handle);
        // Stream is dropped automatically
    }
}

// ============================================================================
// rocBLAS GEMV/GEMM Functions
// ============================================================================

/// Create a rocBLAS handle.
pub fn rocblas_create() -> Result<RocblasHandle> {
    let mut handle: RocblasHandle = std::ptr::null_mut();
    let status = unsafe { rocblas_handle_create(&mut handle) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("Failed to create rocBLAS handle: status {}", status),
        });
    }
    Ok(handle)
}

/// Destroy a rocBLAS handle.
pub fn rocblas_destroy(handle: RocblasHandle) -> Result<()> {
    let status = unsafe { rocblas_handle_destroy(handle) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("Failed to destroy rocBLAS handle: status {}", status),
        });
    }
    Ok(())
}

/// Set the stream for a rocBLAS handle.
pub fn rocblas_set_stream(handle: RocblasHandle, stream: &Stream) -> Result<()> {
    let status = unsafe { rocblas_set_stream_wrapper(handle, stream.handle()) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("Failed to set rocBLAS stream: status {}", status),
        });
    }
    Ok(())
}

/// HGEMM: C = A * B (FP16 matrix multiply)
///
/// Computes C = A * B where:
/// - A is M×K matrix (stored column-major)
/// - B is K×N matrix (stored column-major)
/// - C is M×N matrix (stored column-major)
///
/// For the RWKV use case:
/// - weight is (N, K) = (out_features, in_features) stored as K×N in column-major
/// - input is K×A (in_features × tokens)
/// - output is N×A (out_features × tokens)
///
/// The operation is: output = weight @ input
pub fn hgemm_f16(
    handle: RocblasHandle,
    weight: &TensorHip<f32>,  // Actually f16 stored as f32 for API simplicity
    input: &TensorHip<f32>,   // Actually f16
    output: &mut TensorHip<f32>, // Actually f16
) -> Result<()> {
    // For web-rwkv tensor convention:
    // weight: Shape(N, K) where N is fastest = column-major K×N = N cols, K rows
    //   Actually stored as transpose, so it's M×K where M=N, K=K
    // input: Shape(K, A) = column-major A×K = K rows, A cols -> actually K×A where K rows, A cols
    // output: Shape(N, A) = column-major A×N = N rows, A cols -> N×A
    //
    // We want: output[N,A] = weight[N,K] @ input[K,A]
    // In rocBLAS column-major terms:
    //   C[M,N] = A[M,K] * B[K,N] where M=N_out, N=A_tokens, K=K_in

    let weight_shape = weight.shape();
    let input_shape = input.shape();

    // weight: [N, K, 1, 1] where N is out_features, K is in_features
    // input: [K, A, 1, 1] where K is in_features, A is tokens
    let m = weight_shape[0] as c_int;  // N (output features) - rows of weight
    let k = weight_shape[1] as c_int;  // K (input features) - cols of weight, rows of input
    let n = input_shape[1] as c_int;   // A (tokens) - cols of input

    // Verify dimensions
    if input_shape[0] as c_int != k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "HGEMM dimension mismatch: weight has K={}, input has K={}",
                k, input_shape[0]
            ),
        });
    }

    let status = unsafe {
        launch_hgemm(
            handle,
            m, n, k,
            weight.as_ptr() as *const u16,
            input.as_ptr() as *const u16,
            output.as_mut_ptr() as *mut u16,
        )
    };

    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("rocBLAS HGEMM failed: status {}", status),
        });
    }

    Ok(())
}

/// Low-level SGEMM wrapper that calls rocBLAS with TensorHip buffers.
///
/// Performs: C = alpha * A * B + beta * C (FP32 matrix multiply)
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `weight` - Weight matrix (A) on device
/// * `input` - Input matrix (B) on device
/// * `output` - Output matrix (C) on device
pub fn sgemm_f32(
    handle: RocblasHandle,
    weight: &TensorHip<f32>,
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
) -> Result<()> {
    // For web-rwkv tensor convention:
    // weight: Shape(N, K) where N is fastest = column-major K×N = N cols, K rows
    // input: Shape(K, A) = column-major A×K = K rows, A cols
    // output: Shape(N, A) = column-major A×N = N rows, A cols
    //
    // We want: output[N,A] = weight[N,K] @ input[K,A]
    // In rocBLAS column-major terms:
    //   C[M,N] = A[M,K] * B[K,N] where M=N_out, N=A_tokens, K=K_in

    let weight_shape = weight.shape();
    let input_shape = input.shape();

    // weight: [N, K, 1, 1] where N is out_features, K is in_features
    // input: [K, A, 1, 1] where K is in_features, A is tokens
    let m = weight_shape[0] as c_int;  // N (output features) - rows of weight
    let k = weight_shape[1] as c_int;  // K (input features) - cols of weight, rows of input
    let n = input_shape[1] as c_int;   // A (tokens) - cols of input

    // Verify dimensions
    if input_shape[0] as c_int != k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "SGEMM dimension mismatch: weight has K={}, input has K={}",
                k, input_shape[0]
            ),
        });
    }

    let status = unsafe {
        launch_sgemm(
            handle,
            m, n, k,
            1.0,  // alpha
            weight.as_ptr(),
            input.as_ptr(),
            0.0,  // beta
            output.as_mut_ptr(),
        )
    };

    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("rocBLAS SGEMM failed: status {}", status),
        });
    }

    Ok(())
}

/// High-level SGEMM that manages device memory allocation (column-major weights).
///
/// Performs: output = weight @ input (FP32 matrix multiply)
///
/// NOTE: This expects weights in column-major format. For row-major weights
/// (e.g., from SafeTensors), use `hip_sgemm_ta` instead.
///
/// # Arguments
/// * `weight` - Weight matrix [N, K] where N is output features, K is input features (flattened, column-major)
/// * `input` - Input matrix [K, A] where K is input features, A is tokens (flattened)
/// * `m` - Number of output features (N)
/// * `k` - Number of input features (K)
/// * `n` - Number of tokens (A)
///
/// # Returns
/// * Output vector [N, A] (flattened)
#[allow(dead_code)]
pub fn hip_sgemm(
    weight: &[f32],
    input: &[f32],
    m: usize,   // output features (N)
    k: usize,   // input features (K)
    n: usize,   // tokens (A)
) -> Result<Vec<f32>> {
    if weight.len() != m * k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight size mismatch: expected {}×{}={}, got {}",
                m, k, m * k, weight.len()
            ),
        });
    }
    if input.len() != k * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}×{}={}, got {}",
                k, n, k * n, input.len()
            ),
        });
    }

    let stream = Stream::new()?;

    // Create tensors with appropriate shapes
    // weight: [M, K, 1, 1] where M=output_features, K=input_features
    let weight_shape = TensorShape::new(m, k, 1, 1);
    // input: [K, N, 1, 1] where K=input_features, N=tokens
    let input_shape = TensorShape::new(k, n, 1, 1);
    // output: [M, N, 1, 1] where M=output_features, N=tokens
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight = TensorHip::from_slice(weight, weight_shape, &stream)?;
    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(output_shape)?;

    // Create rocBLAS handle
    let handle = rocblas_create()?;
    rocblas_set_stream(handle, &stream)?;

    // Run GEMM
    sgemm_f32(handle, &d_weight, &d_input, &mut d_output)?;

    // Clean up handle
    rocblas_destroy(handle)?;

    // Copy back result
    d_output.to_vec(&stream)
}

/// High-level SGEMM with transposed A (for row-major weights - DEPRECATED).
///
/// NOTE: This function is deprecated. Use `hip_sgemm` with column-major weights
/// loaded via `load_weight_matrix_f32` instead. Per docs/RWKV7_HIP_BACKEND_PLAN.md:
/// "Use rocBLAS-native column-major storage for GEMM/GEMV...
///  This avoids per-call row/col mapping in rocBLAS"
///
/// Performs: output = weight^T @ input (FP32 matrix multiply)
///
/// # Arguments
/// * `weight` - Weight matrix stored as row-major [M, K] (flattened)
/// * `input` - Input matrix [K, N] where K is input features, N is tokens (flattened)
/// * `m` - Number of output features
/// * `k` - Number of input features
/// * `n` - Number of tokens
///
/// # Returns
/// * Output vector [M, N] (flattened)
#[deprecated(note = "Use hip_sgemm with column-major weights instead")]
pub fn hip_sgemm_ta(
    weight: &[f32],
    input: &[f32],
    m: usize,   // output features
    k: usize,   // input features
    n: usize,   // tokens
) -> Result<Vec<f32>> {
    if weight.len() != m * k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight size mismatch: expected {}×{}={}, got {}",
                m, k, m * k, weight.len()
            ),
        });
    }
    if input.len() != k * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}×{}={}, got {}",
                k, n, k * n, input.len()
            ),
        });
    }

    let stream = Stream::new()?;

    // For transposed A:
    // weight is stored as row-major [M, K] = column-major [K, M]
    // We tell rocBLAS to use shape [K, M] and transpose to get [M, K]
    let weight_shape = TensorShape::new(k, m, 1, 1);  // stored shape
    let input_shape = TensorShape::new(k, n, 1, 1);
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight = TensorHip::from_slice(weight, weight_shape, &stream)?;
    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(output_shape)?;

    let handle = rocblas_create()?;
    rocblas_set_stream(handle, &stream)?;

    let status = unsafe {
        launch_sgemm_ta(
            handle,
            m as c_int, n as c_int, k as c_int,
            1.0,  // alpha
            d_weight.as_ptr(),
            d_input.as_ptr(),
            0.0,  // beta
            d_output.as_mut_ptr(),
        )
    };

    rocblas_destroy(handle)?;

    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { rocblas_to_hip_error(status) },
            message: format!("rocBLAS SGEMM_TA failed: status {}", status),
        });
    }

    d_output.to_vec(&stream)
}

/// High-level HGEMM that manages device memory allocation.
///
/// Performs: output = weight @ input (FP16 matrix multiply)
///
/// # Arguments
/// * `weight` - Weight matrix [N, K] where N is output features, K is input features
/// * `input` - Input matrix [K, A] where K is input features, A is tokens
///
/// # Returns
/// * Output vector [N, A] (flattened)
pub fn hip_hgemm(
    weight: &[f32],
    input: &[f32],
    m: usize,   // output features (N)
    k: usize,   // input features (K)
    n: usize,   // tokens (A)
) -> Result<Vec<f32>> {
    if weight.len() != m * k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight size mismatch: expected {}×{}={}, got {}",
                m, k, m * k, weight.len()
            ),
        });
    }
    if input.len() != k * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}×{}={}, got {}",
                k, n, k * n, input.len()
            ),
        });
    }

    let stream = Stream::new()?;

    // Create tensors with appropriate shapes
    // weight: [M, K, 1, 1] where M=output_features, K=input_features
    let weight_shape = TensorShape::new(m, k, 1, 1);
    // input: [K, N, 1, 1] where K=input_features, N=tokens
    let input_shape = TensorShape::new(k, n, 1, 1);
    // output: [M, N, 1, 1] where M=output_features, N=tokens
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight = TensorHip::from_slice(weight, weight_shape, &stream)?;
    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(output_shape)?;

    // Create rocBLAS handle
    let handle = rocblas_create()?;
    rocblas_set_stream(handle, &stream)?;

    // Run GEMM
    hgemm_f16(handle, &d_weight, &d_input, &mut d_output)?;

    // Clean up handle
    rocblas_destroy(handle)?;

    // Copy back result
    d_output.to_vec(&stream)
}

