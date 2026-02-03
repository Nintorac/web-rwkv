//! HIP kernel wrapper functions.

mod elementwise;
pub mod fla;
mod norm;
mod rwkv_ops;
mod wkv;

pub use elementwise::*;
pub use fla::*;
pub use norm::*;
pub use rwkv_ops::*;
pub use wkv::*;

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use crate::hip::device::Stream;
    use crate::hip::tensor::TensorShape;
    use crate::hip::tensor::TensorHip;

    /// Test that copy_f16_to_f32 correctly converts f16 to f32 on GPU
    #[test]
    fn test_copy_f16_to_f32() {
        let stream = Stream::null();

        // Create test data with various values including edge cases
        let data_f16: Vec<f16> = [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001, -0.001, 65504.0, // max f16
            6.1e-5,  // smallest positive normal f16
        ]
        .iter()
        .map(|&x| f16::from_f32(x))
        .collect();

        let n = data_f16.len();
        let shape = TensorShape::new(n, 1, 1, 1);

        // Upload f16 data to GPU
        let input =
            TensorHip::from_slice(&data_f16, shape, &stream).expect("Failed to create f16 tensor");

        // Create output f32 tensor
        let mut output: TensorHip<f32> =
            TensorHip::new(shape).expect("Failed to create f32 tensor");

        // Run conversion kernel
        copy_f16_to_f32(&input, &mut output, &stream).expect("copy_f16_to_f32 failed");

        // Download result
        let result = output.to_vec(&stream).expect("Failed to download result");

        // Verify
        assert_eq!(result.len(), n);
        for (i, (&f16_val, &f32_val)) in data_f16.iter().zip(result.iter()).enumerate() {
            let expected = f16_val.to_f32();
            let diff = (expected - f32_val).abs();
            assert!(
                diff < 1e-6 || diff / expected.abs().max(1e-10) < 1e-6,
                "Mismatch at index {}: f16={} -> expected={}, got={}",
                i,
                f16_val,
                expected,
                f32_val
            );
        }
    }

    /// Test copy_f16_to_f32 with larger data (tests block/grid sizing)
    #[test]
    fn test_copy_f16_to_f32_large() {
        let stream = Stream::null();

        // Test with size that requires multiple blocks (>256 elements)
        let n = 1024 * 4;
        let data_f16: Vec<f16> = (0..n)
            .map(|i| f16::from_f32((i as f32) * 0.1 - (n as f32) * 0.05))
            .collect();

        let shape = TensorShape::new(n, 1, 1, 1);

        let input =
            TensorHip::from_slice(&data_f16, shape, &stream).expect("Failed to create f16 tensor");

        let mut output: TensorHip<f32> =
            TensorHip::new(shape).expect("Failed to create f32 tensor");

        copy_f16_to_f32(&input, &mut output, &stream).expect("copy_f16_to_f32 failed");

        let result = output.to_vec(&stream).expect("Failed to download result");

        assert_eq!(result.len(), n);
        for (i, (&f16_val, &f32_val)) in data_f16.iter().zip(result.iter()).enumerate() {
            let expected = f16_val.to_f32();
            let diff = (expected - f32_val).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: f16={} -> expected={}, got={}",
                i,
                f16_val,
                expected,
                f32_val
            );
        }
    }
}
