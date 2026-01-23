//! Common test utilities for loading fixtures and comparing tensors.
//!
//! This module provides infrastructure for testing HIP kernels against
//! Python-generated reference data stored in NPZ files.

use half::f16;
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Supported array types in fixtures
#[derive(Debug)]
pub enum FixtureArray {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
}

/// A loaded test fixture containing input/output arrays
pub struct TestFixture {
    /// Raw arrays indexed by name
    pub data: HashMap<String, FixtureArray>,
}

impl TestFixture {
    /// Load a fixture from an NPZ file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path.as_ref())?;
        let mut npz = NpzReader::new(file)?;

        let mut data = HashMap::new();

        for name in npz.names()? {
            // Read array - try different dtypes
            let name_clone = name.clone();

            // First try to read as f32 (most common)
            if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
                let vec: Vec<f32> = arr.into_raw_vec();
                data.insert(name_clone, FixtureArray::F32(vec));
                continue;
            }

            // Try i64 (for shape arrays)
            if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<i64>, ndarray::IxDyn>(&name) {
                let vec: Vec<i64> = arr.into_raw_vec();
                data.insert(name_clone, FixtureArray::I64(vec));
                continue;
            }

            // Try f16 (stored as u16 in npy)
            // Note: NumPy stores f16 which ndarray-npy may read as f32 or fail
            // For now, we assume Python saved f16 as f32 or we handle specially
        }

        Ok(Self { data })
    }

    /// Get the 4D shape for a tensor (reads {name}_shape)
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

    /// Get a float32 array
    pub fn f32(&self, name: &str) -> &[f32] {
        match self.data.get(name) {
            Some(FixtureArray::F32(v)) => v,
            _ => panic!("Expected f32 array for '{}'", name),
        }
    }

    /// Get an int64 array
    pub fn i64(&self, name: &str) -> &[i64] {
        match self.data.get(name) {
            Some(FixtureArray::I64(v)) => v,
            _ => panic!("Expected i64 array for '{}'", name),
        }
    }

    /// Check if a key exists
    pub fn contains(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    /// List all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
}

/// Compare two f32 tensors with tolerance
///
/// Returns Ok(()) if all elements match within tolerance, or an error
/// describing the first mismatch.
pub fn assert_tensors_close(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        ));
    }

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Handle NaN
        if a.is_nan() && e.is_nan() {
            continue;
        }
        if a.is_nan() || e.is_nan() {
            return Err(format!(
                "NaN mismatch at index {}: actual={}, expected={}",
                i, a, e
            ));
        }

        let diff = (a - e).abs();
        let threshold = atol + rtol * e.abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }

        if diff > threshold {
            return Err(format!(
                "Mismatch at index {}: actual={} vs expected={} (diff={}, threshold={})",
                i, a, e, diff, threshold
            ));
        }
    }

    Ok(())
}

/// Compare two f16 tensors with tolerance (converts to f32 internally)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_tensors_close_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_ok());
    }

    #[test]
    fn test_assert_tensors_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 3.0]; // 10% diff
        assert!(assert_tensors_close(&a, &b, 1e-3, 1e-4).is_err());
    }
}
