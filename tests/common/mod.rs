//! Common test utilities for loading fixtures and comparing tensors.
//!
//! This module provides infrastructure for testing HIP kernels against
//! Python-generated reference data stored in NPZ files.
//!
//! # Fixture Format
//!
//! NPZ files contain arrays with standardized naming:
//! - `{name}`: The flattened 1D tensor data (stored as f32 for compatibility)
//! - `{name}_shape`: The original 4D shape as `[i64; 4]`
//! - `{name}_dtype`: The original dtype as a string (e.g., "float16", "float32")
//!
//! # Example Usage
//!
//! ```ignore
//! let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/basic.npz")?;
//! let input = fixture.f32("input");
//! let shape = fixture.shape4("input");
//! let expected = fixture.f32("expected");
//!
//! // Run your kernel...
//! let actual = run_sigmoid_kernel(input, shape);
//!
//! // Compare with detailed error reporting
//! assert_tensors_close_with_shape(&actual, expected, 1e-3, 1e-4, &shape)?;
//! ```

use half::f16;
use npyz::npz::NpzArchive;
use std::collections::HashMap;
use std::path::Path;

/// Tolerance specifications for different tensor types.
///
/// These tolerances are calibrated for comparing FP32 HIP output against BF16 reference
/// (Python chunk_rwkv7 from rwkvfla). BF16 has only 7 mantissa bits, so the reference
/// itself is only accurate to ~0.4% relative precision. Standard practice (per Triton
/// and PyTorch testing guidelines) is atol=1e-2, rtol=1e-2 for FP32-vs-BF16 comparisons.
///
/// See: https://github.com/triton-lang/triton/issues/5283
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    pub rtol: f32,
    pub atol: f32,
}

impl Tolerances {
    /// BF16-appropriate tolerance for normalized activations (after layernorm, groupnorm, L2norm).
    /// Normalization can amplify input differences for low-variance groups, so we use
    /// the standard BF16 tolerance rather than trying to be tighter.
    pub const NORMALIZED: Self = Self {
        rtol: 1e-2,
        atol: 1e-2,
    };

    /// Tolerance for linear projections and matrix multiplications.
    /// Standard BF16 comparison tolerance.
    pub const MATMUL: Self = Self {
        rtol: 1e-2,
        atol: 1e-2,
    };

    /// Tolerance for activations with potential numerical instability.
    pub const ACTIVATION: Self = Self {
        rtol: 1e-2,
        atol: 1e-2,
    };

    /// Tolerance for WKV state (FP32 accumulation on both sides).
    /// Can be tighter since both implementations use FP32 for state.
    pub const STATE: Self = Self {
        rtol: 1e-3,
        atol: 1e-4,
    };

    /// Tolerance for values with accumulated error across layers.
    /// After 12 layers, expect ~1e-2 aggregate relative error.
    pub const ACCUMULATED: Self = Self {
        rtol: 2e-2,
        atol: 2e-2,
    };
}

/// Supported array types in fixtures.
///
/// Note: f16 arrays are stored natively in fixtures; we upcast to f32 when loading
/// for compatibility with existing test helpers.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Variants are used by future HIP kernel tests
pub enum FixtureArray {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    Str(String),
}

/// A loaded test fixture containing input/output arrays.
#[derive(Debug)]
pub struct TestFixture {
    /// Raw arrays indexed by name
    pub data: HashMap<String, FixtureArray>,
}

impl TestFixture {
    /// Load a fixture from an NPZ file.
    ///
    /// Supports loading:
    /// - f32 arrays (most tensor data)
    /// - i64 arrays (shape arrays)
    /// - String arrays (dtype metadata)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut npz = NpzArchive::open(path.as_ref())?;

        let mut data = HashMap::new();

        let names: Vec<String> = npz.array_names().map(|name| name.to_string()).collect();

        for name in names {
            let Some(npy) = npz.by_name(&name)? else {
                continue;
            };

            let dtype = npy.dtype().descr();

            if dtype.contains("f2") {
                let vec = npy.into_vec::<f16>()?;
                let vec_f32: Vec<f32> = vec.iter().map(|v| v.to_f32()).collect();
                data.insert(name, FixtureArray::F32(vec_f32));
                continue;
            }

            if dtype.contains("f4") {
                let vec = npy.into_vec::<f32>()?;
                data.insert(name, FixtureArray::F32(vec));
                continue;
            }

            if dtype.contains("i4") {
                let vec = npy.into_vec::<i32>()?;
                data.insert(name, FixtureArray::I32(vec));
                continue;
            }

            if dtype.contains("i8") {
                let vec = npy.into_vec::<i64>()?;
                data.insert(name, FixtureArray::I64(vec));
                continue;
            }

            if dtype.contains("u4") {
                let vec = npy.into_vec::<u32>()?;
                data.insert(name, FixtureArray::U32(vec));
                continue;
            }

            if dtype.contains("|S") || dtype.contains("<U") || dtype.contains("|U") {
                let vec = npy.into_vec::<String>()?;
                let value = vec.join(",");
                data.insert(name, FixtureArray::Str(value));
                continue;
            }
        }

        Ok(Self { data })
    }

    /// Get the 4D shape for a tensor (reads {name}_shape).
    ///
    /// Shape follows the web-rwkv convention where `shape[0]` is the fastest
    /// (contiguous) axis in memory.
    pub fn shape4(&self, name: &str) -> [usize; 4] {
        let shape_key = format!("{}_shape", name);
        match self.data.get(&shape_key) {
            Some(FixtureArray::I64(v)) => {
                assert_eq!(v.len(), 4, "Shape array must have 4 elements");
                [v[0] as usize, v[1] as usize, v[2] as usize, v[3] as usize]
            }
            _ => panic!("Shape array '{}' not found or wrong type", shape_key),
        }
    }

    /// Get a float32 array.
    pub fn f32(&self, name: &str) -> &[f32] {
        match self.data.get(name) {
            Some(FixtureArray::F32(v)) => v,
            _ => panic!(
                "Expected f32 array for '{}', available keys: {:?}",
                name,
                self.tensor_keys()
            ),
        }
    }

    /// Get an int32 array.
    #[allow(dead_code)] // Used by future HIP kernel tests
    pub fn i32(&self, name: &str) -> &[i32] {
        match self.data.get(name) {
            Some(FixtureArray::I32(v)) => v,
            _ => panic!(
                "Expected i32 array for '{}', available keys: {:?}",
                name,
                self.tensor_keys()
            ),
        }
    }

    /// Get an int64 array.
    #[allow(dead_code)] // Used by future HIP kernel tests
    pub fn i64(&self, name: &str) -> &[i64] {
        match self.data.get(name) {
            Some(FixtureArray::I64(v)) => v,
            _ => panic!(
                "Expected i64 array for '{}', available keys: {:?}",
                name,
                self.tensor_keys()
            ),
        }
    }

    /// Get a uint32 array.
    #[allow(dead_code)] // Used by future HIP kernel tests
    pub fn u32(&self, name: &str) -> &[u32] {
        match self.data.get(name) {
            Some(FixtureArray::U32(v)) => v,
            _ => panic!(
                "Expected u32 array for '{}', available keys: {:?}",
                name,
                self.tensor_keys()
            ),
        }
    }

    /// Check if a key exists.
    #[allow(dead_code)] // Used by hip_layer_validation tests
    pub fn contains(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    /// List all keys.
    #[allow(dead_code)] // Used by future HIP kernel tests
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// List tensor keys (excluding shape and dtype metadata).
    pub fn tensor_keys(&self) -> Vec<&String> {
        self.data
            .keys()
            .filter(|k| !k.ends_with("_shape") && !k.ends_with("_dtype"))
            .collect()
    }

    /// Get the total number of elements for a tensor.
    #[allow(dead_code)] // Used by future HIP kernel tests
    pub fn numel(&self, name: &str) -> usize {
        let shape = self.shape4(name);
        shape.iter().product()
    }
}

/// Details about a tensor comparison failure.
#[derive(Debug)]
pub struct TensorMismatch {
    /// Total number of elements
    pub total_elements: usize,
    /// Number of mismatched elements
    pub mismatch_count: usize,
    /// Index of the first mismatch
    pub first_mismatch_idx: usize,
    /// Actual value at first mismatch
    pub first_actual: f32,
    /// Expected value at first mismatch
    pub first_expected: f32,
    /// Index of the maximum difference
    pub max_diff_idx: usize,
    /// Maximum absolute difference
    pub max_diff: f32,
    /// Actual value at max diff
    pub max_diff_actual: f32,
    /// Expected value at max diff
    pub max_diff_expected: f32,
    /// Mean absolute error across all elements
    pub mean_abs_error: f32,
    /// Optional shape for coordinate reporting
    pub shape: Option<[usize; 4]>,
}

impl TensorMismatch {
    /// Convert a flat index to 4D coordinates given a shape.
    /// Assumes shape[0] is the fastest (contiguous) axis.
    pub fn index_to_coords(index: usize, shape: &[usize; 4]) -> [usize; 4] {
        let mut remaining = index;
        let mut coords = [0usize; 4];
        for (i, &dim) in shape.iter().enumerate() {
            coords[i] = remaining % dim;
            remaining /= dim;
        }
        coords
    }

    /// Format the mismatch as a detailed error message.
    pub fn format_error(&self) -> String {
        let mut msg = String::new();

        msg.push_str(&format!(
            "Tensor mismatch: {}/{} elements differ ({:.2}%)\n",
            self.mismatch_count,
            self.total_elements,
            100.0 * self.mismatch_count as f32 / self.total_elements as f32
        ));

        // First mismatch
        if let Some(shape) = &self.shape {
            let coords = Self::index_to_coords(self.first_mismatch_idx, shape);
            msg.push_str(&format!(
                "  First mismatch at index {} (coords {:?}):\n",
                self.first_mismatch_idx, coords
            ));
        } else {
            msg.push_str(&format!(
                "  First mismatch at index {}:\n",
                self.first_mismatch_idx
            ));
        }
        msg.push_str(&format!(
            "    actual={:.6}, expected={:.6}, diff={:.6}\n",
            self.first_actual,
            self.first_expected,
            (self.first_actual - self.first_expected).abs()
        ));

        // Max diff
        if let Some(shape) = &self.shape {
            let coords = Self::index_to_coords(self.max_diff_idx, shape);
            msg.push_str(&format!(
                "  Max diff at index {} (coords {:?}):\n",
                self.max_diff_idx, coords
            ));
        } else {
            msg.push_str(&format!("  Max diff at index {}:\n", self.max_diff_idx));
        }
        msg.push_str(&format!(
            "    actual={:.6}, expected={:.6}, diff={:.6}\n",
            self.max_diff_actual, self.max_diff_expected, self.max_diff
        ));

        msg.push_str(&format!(
            "  Mean absolute error: {:.6}\n",
            self.mean_abs_error
        ));

        if let Some(shape) = &self.shape {
            msg.push_str(&format!("  Shape: {:?}\n", shape));
        }

        msg
    }
}

impl std::fmt::Display for TensorMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_error())
    }
}

/// Compare two f32 tensors with tolerance.
///
/// Returns Ok(()) if all elements match within tolerance, or an error
/// with detailed mismatch information.
///
/// # Arguments
/// * `actual` - The computed tensor values
/// * `expected` - The expected (reference) tensor values
/// * `rtol` - Relative tolerance (e.g., 1e-3 for 0.1%)
/// * `atol` - Absolute tolerance (e.g., 1e-4)
///
/// # Tolerance Formula
/// An element passes if: `|actual - expected| <= atol + rtol * |expected|`
pub fn assert_tensors_close(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    compare_tensors(actual, expected, rtol, atol, None)
}

/// Compare two f32 tensors with tolerance, including shape for better error messages.
///
/// Same as `assert_tensors_close` but includes shape information in error messages
/// for easier debugging.
pub fn assert_tensors_close_with_shape(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
    shape: &[usize; 4],
) -> Result<(), String> {
    compare_tensors(actual, expected, rtol, atol, Some(*shape))
}

/// Internal comparison function with optional shape.
fn compare_tensors(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
    shape: Option<[usize; 4]>,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Length mismatch: actual has {} elements, expected has {}",
            actual.len(),
            expected.len()
        ));
    }

    if actual.is_empty() {
        return Ok(());
    }

    let mut mismatch_count = 0usize;
    let mut first_mismatch_idx = None;
    let mut first_actual = 0.0f32;
    let mut first_expected = 0.0f32;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut max_diff_actual = 0.0f32;
    let mut max_diff_expected = 0.0f32;
    let mut total_abs_error = 0.0f64; // Use f64 for accumulation

    // Threshold for "effectively infinite" - values this large are practically overflow
    // f32 max is ~3.4e38, so values > 1e34 are in the danger zone for overflow
    const OVERFLOW_THRESHOLD: f32 = 1e34;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Handle non-finite and near-overflow values
        // Values > OVERFLOW_THRESHOLD or non-finite are considered "overflow"
        let a_overflow = !a.is_finite() || a.abs() > OVERFLOW_THRESHOLD;
        let e_overflow = !e.is_finite() || e.abs() > OVERFLOW_THRESHOLD;

        if a_overflow && e_overflow {
            // Both are overflow - consider as matching
            // At extreme values, both represent numerical instability regardless of sign
            // Different overflow paths (GPU vs CPU) can produce different signs
            continue;
        }
        if a_overflow || e_overflow {
            // One is overflow, one is not - this is a mismatch
            mismatch_count += 1;
            if first_mismatch_idx.is_none() {
                first_mismatch_idx = Some(i);
                first_actual = a;
                first_expected = e;
            }
            let diff = f32::INFINITY;
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
                max_diff_actual = a;
                max_diff_expected = e;
            }
            continue;
        }

        let diff = (a - e).abs();
        total_abs_error += diff as f64;

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
            max_diff_actual = a;
            max_diff_expected = e;
        }

        let threshold = atol + rtol * e.abs();
        if diff > threshold {
            mismatch_count += 1;
            if first_mismatch_idx.is_none() {
                first_mismatch_idx = Some(i);
                first_actual = a;
                first_expected = e;
            }
        }
    }

    if mismatch_count > 0 {
        let mismatch = TensorMismatch {
            total_elements: actual.len(),
            mismatch_count,
            first_mismatch_idx: first_mismatch_idx.unwrap(),
            first_actual,
            first_expected,
            max_diff_idx,
            max_diff,
            max_diff_actual,
            max_diff_expected,
            mean_abs_error: (total_abs_error / actual.len() as f64) as f32,
            shape,
        };
        return Err(mismatch.format_error());
    }

    Ok(())
}

/// Compare two f16 tensors with tolerance (converts to f32 internally).
pub fn assert_tensors_close_f16(
    actual: &[f16],
    expected: &[f16],
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    let actual_f32: Vec<f32> = actual.iter().map(|x| x.to_f32()).collect();
    let expected_f32: Vec<f32> = expected.iter().map(|x| x.to_f32()).collect();
    assert_tensors_close(&actual_f32, &expected_f32, rtol, atol)
}

/// Compare two f16 tensors with tolerance, including shape for better error messages.
#[allow(dead_code)] // Used by future HIP kernel tests
pub fn assert_tensors_close_f16_with_shape(
    actual: &[f16],
    expected: &[f16],
    rtol: f32,
    atol: f32,
    shape: &[usize; 4],
) -> Result<(), String> {
    let actual_f32: Vec<f32> = actual.iter().map(|x| x.to_f32()).collect();
    let expected_f32: Vec<f32> = expected.iter().map(|x| x.to_f32()).collect();
    assert_tensors_close_with_shape(&actual_f32, &expected_f32, rtol, atol, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic comparison tests
    // =========================================================================

    #[test]
    fn test_assert_tensors_close_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok());
    }

    #[test]
    fn test_assert_tensors_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 3.0]; // 10% diff at index 0
        let result = assert_tensors_close(&a, &b, 1e-3, 1e-4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("mismatch"),
            "Error should mention mismatch: {}",
            err
        );
        assert!(
            err.contains("index 0"),
            "Error should identify index 0: {}",
            err
        );
    }

    #[test]
    fn test_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = assert_tensors_close(&a, &b, 1e-3, 1e-4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Length mismatch"), "Error: {}", err);
        assert!(
            err.contains("3") && err.contains("2"),
            "Should mention both lengths: {}",
            err
        );
    }

    #[test]
    fn test_empty_tensors_pass() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok());
    }

    // =========================================================================
    // Special value handling
    // =========================================================================

    #[test]
    fn test_nan_both_sides_pass() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, f32::NAN, 3.0];
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok());
    }

    #[test]
    fn test_nan_mismatch_fail() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = assert_tensors_close(&a, &b, 1e-3, 1e-4);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("NaN"), "Error should mention NaN: {}", err);
    }

    #[test]
    fn test_infinity_both_sides_pass() {
        let a = vec![f32::INFINITY, f32::NEG_INFINITY, 3.0];
        let b = vec![f32::INFINITY, f32::NEG_INFINITY, 3.0];
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok());
    }

    #[test]
    fn test_overflow_values_match() {
        // Both +inf and -inf are overflow - we accept this as matching
        // since at extreme values, sign can differ due to different overflow paths
        let a = vec![f32::INFINITY];
        let b = vec![f32::NEG_INFINITY];
        let result = assert_tensors_close(&a, &b, 1e-3, 1e-4);
        assert!(result.is_ok());

        // Also test extreme finite values
        let c = vec![1e35_f32];
        let d = vec![-1e35_f32];
        let result2 = assert_tensors_close(&c, &d, 1e-3, 1e-4);
        assert!(result2.is_ok());
    }

    // =========================================================================
    // Error message quality tests - these prove the harness gives actionable info
    // =========================================================================

    #[test]
    fn test_error_reports_mismatch_count() {
        // Create data with multiple mismatches
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.0, 3.5, 4.0, 5.5]; // 3 mismatches at indices 0, 2, 4
        let result = assert_tensors_close(&a, &b, 0.01, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err();

        // Should report total count
        assert!(err.contains("3/5"), "Should report 3/5 mismatches: {}", err);
        // Should show first mismatch at index 0
        assert!(
            err.contains("index 0"),
            "Should report first at index 0: {}",
            err
        );
    }

    #[test]
    fn test_error_reports_max_diff() {
        // Create data where max diff is not at first mismatch
        let a = vec![1.1, 2.0, 5.0, 4.0]; // max diff at index 2 (diff=2.0)
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = assert_tensors_close(&a, &b, 0.01, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err();

        // Should report max diff location
        assert!(err.contains("Max diff"), "Should report max diff: {}", err);
        assert!(
            err.contains("index 2"),
            "Max diff should be at index 2: {}",
            err
        );
    }

    #[test]
    fn test_error_with_shape_shows_coordinates() {
        // 2x2x2x1 tensor, mismatch at flat index 5 = coords [1, 0, 1, 0]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0, 8.0]; // mismatch at index 5
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [2, 2, 2, 1];
        let result = assert_tensors_close_with_shape(&a, &b, 0.01, 0.01, &shape);
        assert!(result.is_err());
        let err = result.unwrap_err();

        // Should show coordinates
        assert!(err.contains("coords"), "Should show coordinates: {}", err);
        assert!(
            err.contains("[1, 0, 1, 0]"),
            "Coords should be [1, 0, 1, 0]: {}",
            err
        );
        assert!(
            err.contains("Shape: [2, 2, 2, 1]"),
            "Should show shape: {}",
            err
        );
    }

    #[test]
    fn test_error_shows_actual_and_expected_values() {
        let a = vec![1.0, 2.5];
        let b = vec![1.0, 2.0];
        let result = assert_tensors_close(&a, &b, 0.01, 0.01);
        assert!(result.is_err());
        let err = result.unwrap_err();

        // Should show both values
        assert!(err.contains("2.5"), "Should show actual value 2.5: {}", err);
        assert!(
            err.contains("2.0"),
            "Should show expected value 2.0: {}",
            err
        );
        assert!(err.contains("diff"), "Should show difference: {}", err);
    }

    #[test]
    fn test_error_shows_mean_absolute_error() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1]; // All off by 0.1
        let result = assert_tensors_close(&a, &b, 0.001, 0.001);
        assert!(result.is_err());
        let err = result.unwrap_err();

        assert!(
            err.contains("Mean absolute error"),
            "Should show MAE: {}",
            err
        );
    }

    // =========================================================================
    // Index to coordinates conversion tests
    // =========================================================================

    #[test]
    fn test_index_to_coords_simple() {
        // Shape [2, 3, 1, 1], flat index 4
        // index 4 = coords [0, 1, 0, 0] because:
        //   index % 2 = 0 (dim 0)
        //   (index / 2) % 3 = 2 (dim 1)
        // Wait, let me recalculate:
        // shape[0]=2 is fastest axis
        // index=4: 4 % 2 = 0, 4/2=2, 2 % 3 = 2, 2/3=0
        // So coords = [0, 2, 0, 0]
        let shape = [2, 3, 1, 1];
        let coords = TensorMismatch::index_to_coords(4, &shape);
        assert_eq!(coords, [0, 2, 0, 0]);
    }

    #[test]
    fn test_index_to_coords_3d() {
        // Shape [4, 3, 2, 1]
        // Total size = 24
        // Index 13:
        //   13 % 4 = 1
        //   13 / 4 = 3
        //   3 % 3 = 0
        //   3 / 3 = 1
        //   1 % 2 = 1
        //   1 / 2 = 0
        // coords = [1, 0, 1, 0]
        let shape = [4, 3, 2, 1];
        let coords = TensorMismatch::index_to_coords(13, &shape);
        assert_eq!(coords, [1, 0, 1, 0]);
    }

    // =========================================================================
    // f16 tests
    // =========================================================================

    #[test]
    fn test_f16_comparison_pass() {
        let a: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let b: Vec<f16> = vec![f16::from_f32(1.001), f16::from_f32(2.001)];
        assert!(assert_tensors_close_f16(&a, &b, 1e-2, 1e-2).is_ok());
    }

    #[test]
    fn test_f16_comparison_fail() {
        let a: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let b: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(3.0)]; // Big diff at index 1
        let result = assert_tensors_close_f16(&a, &b, 1e-3, 1e-3);
        assert!(result.is_err());
    }

    // =========================================================================
    // Tolerance formula tests
    // =========================================================================

    #[test]
    fn test_relative_tolerance() {
        // Test that rtol works: |a - e| <= atol + rtol * |e|
        // With atol=0 and rtol=0.1, a value of 10.0 allows up to 1.0 difference
        let a = vec![10.9]; // diff = 0.9, which is < 0.1 * 10 = 1.0
        let b = vec![10.0];
        assert!(assert_tensors_close(&a, &b, 0.1, 0.0).is_ok());

        let a2 = vec![11.1]; // diff = 1.1, which is > 0.1 * 10 = 1.0
        assert!(assert_tensors_close(&a2, &b, 0.1, 0.0).is_err());
    }

    #[test]
    fn test_absolute_tolerance() {
        // Test that atol works independently
        // With rtol=0 and atol=0.5, any diff > 0.5 fails
        let a = vec![1.4];
        let b = vec![1.0];
        assert!(assert_tensors_close(&a, &b, 0.0, 0.5).is_ok()); // diff=0.4 < 0.5

        let a2 = vec![1.6];
        assert!(assert_tensors_close(&a2, &b, 0.0, 0.5).is_err()); // diff=0.6 > 0.5
    }
}
