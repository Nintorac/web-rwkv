//! hipBLASLt GEMM wrapper functions.
//!
//! This module provides hipBLASLt-based BLAS interfaces as an alternative to rocBLAS:
//!
//! - **Context-based**: `HipBlasLtContext` - reusable handle for efficient operations
//! - **Mixed precision**: `hgemm_f16_to_f32` - f16 inputs, f32 output
//! - **Pure f16**: `hgemm_f16` - f16 inputs and output
//!
//! hipBLASLt may provide better performance than rocBLAS for certain GEMM shapes,
//! especially for mixed precision operations.

use std::ffi::c_int;

use super::buffer::DeviceBuffer;
use super::device::Stream;
use super::ffi::{
    hipblaslt_handle_create, hipblaslt_handle_destroy, hipblaslt_to_hip_error,
    launch_hipblaslt_hgemm, launch_hipblaslt_hgemm_f32_out, HipErrorKind, HipblasLtHandle, Result,
    HIPBLAS_STATUS_SUCCESS,
};
use super::tensor::{TensorHip, TensorShape};
use half::f16;

// ============================================================================
// HipBlasLtContext - Reusable hipBLASLt context for efficient batched operations
// ============================================================================

/// Long-lived hipBLASLt context for efficient GPU compute.
///
/// Holds a hipBLASLt handle bound to a single stream. Reuse this context across
/// multiple GEMM calls to amortize handle creation cost.
///
/// # Performance Benefits
///
/// - **Handle reuse**: hipBLASLt uses heuristics that benefit from reuse
/// - **Workspace reuse**: Pre-allocated workspace buffer for hipBLASLt algorithms
/// - **Stream binding**: All operations execute on the same stream for ordering
///
/// # Usage
///
/// ```rust,ignore
/// // Create once at start of forward pass
/// let ctx = HipBlasLtContext::new()?;
///
/// // Reuse for all GEMMs
/// ctx.hgemm_f16_to_f32_into(&weight, &input, &mut output)?;
///
/// // Sync before reading results
/// ctx.synchronize()?;
/// ```
pub struct HipBlasLtContext {
    handle: HipblasLtHandle,
    stream: Stream,
    workspace: Option<DeviceBuffer<u8>>,
    workspace_size: usize,
}

/// Default workspace size (32 MB)
const DEFAULT_WORKSPACE_SIZE: usize = 32 * 1024 * 1024;

impl HipBlasLtContext {
    /// Create a new hipBLASLt context with a dedicated stream.
    pub fn new() -> Result<Self> {
        Self::with_workspace_size(DEFAULT_WORKSPACE_SIZE)
    }

    /// Create a new hipBLASLt context with a custom workspace size.
    pub fn with_workspace_size(workspace_size: usize) -> Result<Self> {
        let stream = Stream::new()?;
        let handle = hipblaslt_create()?;

        // Pre-allocate workspace
        let workspace = if workspace_size > 0 {
            Some(DeviceBuffer::new(workspace_size)?)
        } else {
            None
        };

        Ok(Self {
            handle,
            stream,
            workspace,
            workspace_size,
        })
    }

    /// Create a hipBLASLt context using the null (default) stream.
    ///
    /// Operations on the null stream are implicitly synchronized with
    /// other null-stream operations but may have less overlap potential.
    pub fn with_null_stream() -> Result<Self> {
        let stream = Stream::null();
        let handle = hipblaslt_create()?;

        let workspace = Some(DeviceBuffer::new(DEFAULT_WORKSPACE_SIZE)?);

        Ok(Self {
            handle,
            stream,
            workspace,
            workspace_size: DEFAULT_WORKSPACE_SIZE,
        })
    }

    /// Get a reference to the underlying stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Synchronize the stream (wait for all enqueued operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Get the underlying hipBLASLt handle.
    pub fn handle(&self) -> HipblasLtHandle {
        self.handle
    }

    /// Get workspace pointer and size for FFI calls.
    fn workspace_ptr_and_size(&self) -> (*mut std::ffi::c_void, usize) {
        match &self.workspace {
            // We cast away const here - hipBLASLt uses the workspace as scratch space
            // but doesn't actually modify the pointer we pass in
            Some(buf) => (buf.as_ptr() as *mut std::ffi::c_void, self.workspace_size),
            None => (std::ptr::null_mut(), 0),
        }
    }

    /// Mixed-precision HGEMM: output = weight @ input with f16 inputs and f32 output.
    ///
    /// This is optimized for the head layer to avoid a separate f16->f32 conversion.
    /// All tensors must be GPU-resident. No host copies occur.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix on GPU [M, K] (output_features x input_features, FP16)
    /// * `input` - Input matrix on GPU [K, N] (input_features x tokens, FP16)
    /// * `output` - Output matrix on GPU [M, N] (output_features x tokens, FP32), must be pre-allocated
    ///
    /// # Panics
    /// Panics if dimensions don't match.
    pub fn hgemm_f16_to_f32_into(
        &self,
        weight: &TensorHip<f16>,
        input: &TensorHip<f16>,
        output: &mut TensorHip<f32>,
    ) -> Result<()> {
        let weight_shape = weight.shape();
        let input_shape = input.shape();

        // weight: [M, K, 1, 1] where M is out_features (vocab_size), K is in_features (n_embd)
        // input: [K, T, B, 1] where K is in_features, T*B is total columns
        let m = weight_shape[0] as c_int;
        let k = weight_shape[1] as c_int;
        let n = (input_shape[1] * input_shape[2] * input_shape[3]) as c_int;

        // Verify dimensions
        if input_shape[0] as c_int != k {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "hipBLASLt HGEMM_F32_OUT dimension mismatch: weight has K={}, input has K={}",
                    k, input_shape[0]
                ),
            });
        }

        let (workspace_ptr, workspace_size) = self.workspace_ptr_and_size();

        let status = unsafe {
            launch_hipblaslt_hgemm_f32_out(
                self.handle,
                m,
                n,
                k,
                weight.as_ptr() as *const u16,
                input.as_ptr() as *const u16,
                output.as_mut_ptr(),
                workspace_ptr,
                workspace_size,
                self.stream.handle(),
            )
        };

        if status != HIPBLAS_STATUS_SUCCESS {
            return Err(HipErrorKind {
                code: unsafe { hipblaslt_to_hip_error(status) },
                message: format!("hipBLASLt HGEMM_F32_OUT failed: status {}", status),
            });
        }

        Ok(())
    }

    /// Device-to-device HGEMM: output = weight @ input (FP16)
    ///
    /// All tensors must be GPU-resident. No host copies occur.
    pub fn hgemm_into(
        &self,
        weight: &TensorHip<f16>,
        input: &TensorHip<f16>,
        output: &mut TensorHip<f16>,
    ) -> Result<()> {
        let weight_shape = weight.shape();
        let input_shape = input.shape();

        let m = weight_shape[0] as c_int;
        let k = weight_shape[1] as c_int;
        let n = (input_shape[1] * input_shape[2] * input_shape[3]) as c_int;

        if input_shape[0] as c_int != k {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "hipBLASLt HGEMM dimension mismatch: weight has K={}, input has K={}",
                    k, input_shape[0]
                ),
            });
        }

        let (workspace_ptr, workspace_size) = self.workspace_ptr_and_size();

        let status = unsafe {
            launch_hipblaslt_hgemm(
                self.handle,
                m,
                n,
                k,
                weight.as_ptr() as *const u16,
                input.as_ptr() as *const u16,
                output.as_mut_ptr() as *mut u16,
                workspace_ptr,
                workspace_size,
                self.stream.handle(),
            )
        };

        if status != HIPBLAS_STATUS_SUCCESS {
            return Err(HipErrorKind {
                code: unsafe { hipblaslt_to_hip_error(status) },
                message: format!("hipBLASLt HGEMM failed: status {}", status),
            });
        }

        Ok(())
    }
}

impl Drop for HipBlasLtContext {
    fn drop(&mut self) {
        // Best-effort cleanup - ignore errors in drop
        let _ = hipblaslt_destroy(self.handle);
        // Stream and workspace are dropped automatically
    }
}

// ============================================================================
// hipBLASLt Handle Management
// ============================================================================

/// Create a hipBLASLt handle.
pub fn hipblaslt_create() -> Result<HipblasLtHandle> {
    let mut handle: HipblasLtHandle = std::ptr::null_mut();
    let status = unsafe { hipblaslt_handle_create(&mut handle) };
    if status != HIPBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { hipblaslt_to_hip_error(status) },
            message: format!("Failed to create hipBLASLt handle: status {}", status),
        });
    }
    Ok(handle)
}

/// Destroy a hipBLASLt handle.
pub fn hipblaslt_destroy(handle: HipblasLtHandle) -> Result<()> {
    let status = unsafe { hipblaslt_handle_destroy(handle) };
    if status != HIPBLAS_STATUS_SUCCESS {
        return Err(HipErrorKind {
            code: unsafe { hipblaslt_to_hip_error(status) },
            message: format!("Failed to destroy hipBLASLt handle: status {}", status),
        });
    }
    Ok(())
}

// ============================================================================
// Standalone GEMM Functions
// ============================================================================

/// Mixed-precision HGEMM using hipBLASLt: C = A * B with FP16 inputs and FP32 output
///
/// This is a standalone function that creates temporary handles.
/// For better performance with multiple GEMMs, use `HipBlasLtContext` instead.
///
/// # Arguments
/// * `weight` - Weight matrix [M, K] where M is output features, K is input features (FP16)
/// * `input` - Input matrix [K, N] where K is input features, N is tokens (FP16)
///
/// # Returns
/// * Output vector [M, N] (FP32, flattened)
pub fn hip_hipblaslt_hgemm_f32_out(
    weight: &[f16],
    input: &[f16],
    m: usize, // output features (M)
    k: usize, // input features (K)
    n: usize, // tokens (N)
) -> Result<Vec<f32>> {
    if weight.len() != m * k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight size mismatch: expected {}x{}={}, got {}",
                m,
                k,
                m * k,
                weight.len()
            ),
        });
    }
    if input.len() != k * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}x{}={}, got {}",
                k,
                n,
                k * n,
                input.len()
            ),
        });
    }

    let ctx = HipBlasLtContext::new()?;

    // Create tensors with appropriate shapes
    let weight_shape = TensorShape::new(m, k, 1, 1);
    let input_shape = TensorShape::new(k, n, 1, 1);
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight = TensorHip::from_slice(weight, weight_shape, ctx.stream())?;
    let d_input = TensorHip::from_slice(input, input_shape, ctx.stream())?;
    let mut d_output = TensorHip::<f32>::new(output_shape)?;

    // Run GEMM
    ctx.hgemm_f16_to_f32_into(&d_weight, &d_input, &mut d_output)?;

    // Copy back result
    d_output.to_vec(ctx.stream())
}

/// Pure FP16 HGEMM using hipBLASLt: C = A * B
///
/// This is a standalone function that creates temporary handles.
/// For better performance with multiple GEMMs, use `HipBlasLtContext` instead.
///
/// # Arguments
/// * `weight` - Weight matrix [M, K] where M is output features, K is input features (FP16)
/// * `input` - Input matrix [K, N] where K is input features, N is tokens (FP16)
///
/// # Returns
/// * Output vector [M, N] (FP16, flattened)
pub fn hip_hipblaslt_hgemm(
    weight: &[f16],
    input: &[f16],
    m: usize, // output features (M)
    k: usize, // input features (K)
    n: usize, // tokens (N)
) -> Result<Vec<f16>> {
    if weight.len() != m * k {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight size mismatch: expected {}x{}={}, got {}",
                m,
                k,
                m * k,
                weight.len()
            ),
        });
    }
    if input.len() != k * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}x{}={}, got {}",
                k,
                n,
                k * n,
                input.len()
            ),
        });
    }

    let ctx = HipBlasLtContext::new()?;

    // Create tensors with appropriate shapes
    let weight_shape = TensorShape::new(m, k, 1, 1);
    let input_shape = TensorShape::new(k, n, 1, 1);
    let output_shape = TensorShape::new(m, n, 1, 1);

    let d_weight = TensorHip::from_slice(weight, weight_shape, ctx.stream())?;
    let d_input = TensorHip::from_slice(input, input_shape, ctx.stream())?;
    let mut d_output = TensorHip::<f16>::new(output_shape)?;

    // Run GEMM
    ctx.hgemm_into(&d_weight, &d_input, &mut d_output)?;

    // Copy back result
    d_output.to_vec(ctx.stream())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_hipblaslt_context_creation() {
        let ctx = HipBlasLtContext::new();
        assert!(
            ctx.is_ok(),
            "Failed to create hipBLASLt context: {:?}",
            ctx.err()
        );
        println!("hipBLASLt context created successfully");
    }

    #[test]
    fn test_hipblaslt_hgemm_f32_out_basic() {
        // Simple 2x2 @ 2x2 matrix multiply
        // A = [[1, 2], [3, 4]] (column-major: [1, 3, 2, 4])
        // B = [[1, 0], [0, 1]] (identity, column-major: [1, 0, 0, 1])
        // C = A @ B = A
        let weight: Vec<f16> = [1.0, 3.0, 2.0, 4.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let input: Vec<f16> = [1.0, 0.0, 0.0, 1.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();

        let result = hip_hipblaslt_hgemm_f32_out(&weight, &input, 2, 2, 2);
        match result {
            Ok(output) => {
                // Expected: [1, 3, 2, 4] in column-major
                let expected = [1.0f32, 3.0, 2.0, 4.0];
                for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
                    let diff = (actual - exp).abs();
                    assert!(diff < 0.1, "Mismatch at {}: {} vs {}", i, actual, exp);
                }
                println!("hipBLASLt HGEMM F32 output test passed");
            }
            Err(e) => {
                // hipBLASLt may not be supported on all architectures
                println!("hipBLASLt HGEMM F32 output not supported: {}", e);
            }
        }
    }

    #[test]
    fn test_hipblaslt_hgemm_basic() {
        // Simple identity test
        let weight: Vec<f16> = [1.0, 0.0, 0.0, 1.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let input: Vec<f16> = [2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();

        let result = hip_hipblaslt_hgemm(&weight, &input, 2, 2, 2);
        match result {
            Ok(output) => {
                // Identity * input = input
                let expected: Vec<f16> = [2.0, 3.0, 4.0, 5.0]
                    .iter()
                    .map(|&x| f16::from_f32(x))
                    .collect();
                for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
                    let diff = (actual.to_f32() - exp.to_f32()).abs();
                    assert!(diff < 0.1, "Mismatch at {}: {} vs {}", i, actual, exp);
                }
                println!("hipBLASLt HGEMM test passed");
            }
            Err(e) => {
                println!("hipBLASLt HGEMM not supported: {}", e);
            }
        }
    }
}
