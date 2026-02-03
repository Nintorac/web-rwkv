//! HIP kernel wrapper for softmax operation.

use std::ffi::c_int;

use crate::hip::buffer::DeviceBuffer;
use crate::hip::device::Stream;
use crate::hip::ffi::{check, launch_softmax_f32, Result};

/// Launch the softmax kernel on GPU device buffers.
///
/// Per-token softmax over the vocab dimension using a three-pass
/// numerically-stable algorithm (find max, exp+sum, normalize).
///
/// # Arguments
/// * `input` - Input device buffer with V*T elements
/// * `output` - Output device buffer with V*T elements
/// * `vocab_size` - Number of elements per token (V)
/// * `num_tokens` - Number of tokens (T)
/// * `stream` - HIP stream for asynchronous execution
pub fn softmax_f32(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    vocab_size: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        check(launch_softmax_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            vocab_size as c_int,
            num_tokens as c_int,
            stream.handle(),
        ))
    }
}
