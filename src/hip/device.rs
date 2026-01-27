//! HIP device context and stream management.

use std::ptr;

use super::ffi::{
    HipStream, HipErrorKind, Result, check,
    get_device_count, get_device_name, get_gcn_arch_name, set_device,
    get_device_total_memory, get_device_mp_count, get_device_warp_size,
    get_device_compute_capability, is_device_integrated, device_supports_cooperative_launch,
    hip_stream_create, hip_stream_destroy, hip_stream_synchronize,
    hip_device_synchronize,
};

/// HIP context that manages device selection and provides a default stream.
///
/// The context ensures that GPU operations use the correct device and provides
/// a convenient stream for asynchronous operations.
///
/// Note: Device properties are queried on-demand using safe FFI functions that
/// don't require struct layout matching between ROCm versions.
pub struct HipContext {
    device_id: i32,
    stream: Stream,
}

impl HipContext {
    /// Create a new HIP context using device 0 (default).
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create a new HIP context for a specific device.
    pub fn with_device(device_id: i32) -> Result<Self> {
        // Verify device exists
        let count = get_device_count()?;
        if device_id < 0 || device_id >= count {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Invalid device ID {}: only {} device(s) available",
                    device_id, count
                ),
            });
        }

        // Set the device
        set_device(device_id)?;

        // Use null stream as workaround for ROCm stream creation issues
        // TODO: Switch to Stream::new() when ROCm stream creation is fixed
        let stream = Stream::null();

        Ok(Self { device_id, stream })
    }

    /// Get the device ID for this context.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get the device name.
    pub fn device_name(&self) -> Result<String> {
        get_device_name(self.device_id)
    }

    /// Get the GCN architecture name (e.g., "gfx1151").
    pub fn gcn_arch_name(&self) -> Result<String> {
        get_gcn_arch_name(self.device_id)
    }

    /// Get the total global memory in bytes.
    pub fn total_memory(&self) -> Result<usize> {
        get_device_total_memory(self.device_id)
    }

    /// Get the number of multiprocessors (compute units).
    pub fn multiprocessor_count(&self) -> Result<i32> {
        get_device_mp_count(self.device_id)
    }

    /// Get the warp (wavefront) size.
    pub fn warp_size(&self) -> Result<i32> {
        get_device_warp_size(self.device_id)
    }

    /// Get the compute capability (major.minor).
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        get_device_compute_capability(self.device_id)
    }

    /// Check if this is an integrated (APU) device.
    pub fn is_integrated(&self) -> Result<bool> {
        is_device_integrated(self.device_id)
    }

    /// Check if cooperative launch is supported.
    pub fn supports_cooperative_launch(&self) -> Result<bool> {
        device_supports_cooperative_launch(self.device_id)
    }

    /// Get a reference to the default stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Synchronize the context's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Create a new stream associated with this context's device.
    ///
    /// Note: On some ROCm versions, this may crash. Use the default stream
    /// via `stream()` as a workaround.
    pub fn create_stream(&self) -> Result<Stream> {
        set_device(self.device_id)?;
        Stream::new()
    }

    /// Check if this device matches the expected architecture (e.g., "gfx1151").
    pub fn is_arch(&self, expected: &str) -> Result<bool> {
        Ok(self.gcn_arch_name()?.contains(expected))
    }
}

/// A HIP stream for asynchronous operations.
///
/// Supports both explicit streams (created with `new()`) and the default null stream
/// (created with `null()`). The null stream is the default HIP stream and doesn't
/// require explicit creation - use it as a workaround when hipStreamCreate fails.
pub struct Stream {
    handle: HipStream,
    /// Whether this is an owned stream that needs to be destroyed
    owned: bool,
}

impl Stream {
    /// Create a new HIP stream.
    ///
    /// Note: On some ROCm versions, hipStreamCreate may crash. Use `null()` as a
    /// workaround if you encounter this issue.
    pub fn new() -> Result<Self> {
        let mut handle: HipStream = ptr::null_mut();
        unsafe { check(hip_stream_create(&mut handle))? };
        Ok(Self {
            handle,
            owned: true,
        })
    }

    /// Get a wrapper around the default (null) stream.
    ///
    /// The null stream is the default HIP stream used when no explicit stream is
    /// specified. Operations on the null stream are serialized with all other
    /// streams on the same device.
    ///
    /// This is useful as a workaround for hipStreamCreate crashes on some ROCm
    /// versions.
    pub fn null() -> Self {
        Self {
            handle: ptr::null_mut(),
            owned: false,
        }
    }

    /// Synchronize the stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        if self.handle.is_null() {
            // For null stream, use device synchronize
            device_synchronize()
        } else {
            unsafe { check(hip_stream_synchronize(self.handle)) }
        }
    }

    /// Get the raw stream handle
    pub fn handle(&self) -> HipStream {
        self.handle
    }

    /// Check if this is the null (default) stream
    pub fn is_null(&self) -> bool {
        self.handle.is_null()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe {
                hip_stream_destroy(self.handle);
            }
        }
    }
}

/// Synchronize the default stream / device
pub fn device_synchronize() -> Result<()> {
    unsafe { check(hip_device_synchronize()) }
}

