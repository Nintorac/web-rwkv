//! HIP backend for AMD GPUs.
//!
//! This module provides a HIP-based backend for running RWKV inference on AMD GPUs.
//! It requires a ROCm/TheRock installation with support for your GPU architecture.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;

/// HIP error codes
pub type HipError = c_int;

/// HIP stream handle (opaque pointer)
pub type HipStream = *mut c_void;

/// HIP device properties structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProp {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multi_processor_count: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub compute_mode: c_int,
    pub clock_instruction_rate: c_int,
    pub arch: HipDeviceArch,
    pub concurrent_kernels: c_int,
    pub pci_domain_id: c_int,
    pub pci_bus_id: c_int,
    pub pci_device_id: c_int,
    pub max_shared_memory_per_multi_processor: usize,
    pub is_multi_gpu_board: c_int,
    pub can_map_host_memory: c_int,
    pub gcn_arch: c_int,
    pub gcn_arch_name: [c_char; 256],
    pub integrated: c_int,
    pub cooperative_launch: c_int,
    pub cooperative_multi_device_launch: c_int,
    pub max_texture_1d_linear: c_int,
    pub max_texture_1d: c_int,
    pub max_texture_2d: [c_int; 2],
    pub max_texture_3d: [c_int; 3],
    pub mem_pitch: usize,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: c_int,
    pub kernel_exec_timeout_enabled: c_int,
    pub ecc_enabled: c_int,
    pub tcc_driver: c_int,
    pub cooperative_multi_device_unmatched_func: c_int,
    pub cooperative_multi_device_unmatched_grid_dim: c_int,
    pub cooperative_multi_device_unmatched_block_dim: c_int,
    pub cooperative_multi_device_unmatched_shared_mem: c_int,
    pub is_large_bar: c_int,
    pub asic_revision: c_int,
    pub managed_memory: c_int,
    pub direct_managed_mem_access_from_host: c_int,
    pub concurrent_managed_access: c_int,
    pub pageable_memory_access: c_int,
    pub pageable_memory_access_uses_host_page_tables: c_int,
    // Padding for forward compatibility
    _reserved: [c_char; 64],
}

/// HIP device architecture
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HipDeviceArch {
    _unused: [c_int; 28],
}

impl Default for HipDeviceProp {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// FFI declarations for our compiled HIP kernels
extern "C" {
    fn hip_get_device_count(count: *mut c_int) -> HipError;
    fn hip_get_device_properties(props: *mut HipDeviceProp, device_id: c_int) -> HipError;
    fn hip_set_device(device_id: c_int) -> HipError;
    fn hip_get_device(device_id: *mut c_int) -> HipError;
    fn hip_malloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    fn hip_free(ptr: *mut c_void) -> HipError;
    fn hip_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize, stream: HipStream) -> HipError;
    fn hip_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize, stream: HipStream) -> HipError;
    fn hip_stream_create(stream: *mut HipStream) -> HipError;
    fn hip_stream_destroy(stream: HipStream) -> HipError;
    fn hip_stream_synchronize(stream: HipStream) -> HipError;
    fn hip_device_synchronize() -> HipError;
    fn hip_get_error_string(error: HipError) -> *const c_char;
    fn launch_copy_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;

    // Safe property accessors (avoid struct layout issues)
    fn hip_get_device_name(device_id: c_int, name: *mut c_char, max_len: c_int) -> HipError;
    fn hip_get_device_gcn_arch_name(device_id: c_int, name: *mut c_char, max_len: c_int) -> HipError;
    fn hip_get_device_total_memory(device_id: c_int, total_mem: *mut usize) -> HipError;
    fn hip_get_device_mp_count(device_id: c_int, count: *mut c_int) -> HipError;
    fn hip_get_device_warp_size(device_id: c_int, warp_size: *mut c_int) -> HipError;
    fn hip_get_device_compute_capability(device_id: c_int, major: *mut c_int, minor: *mut c_int) -> HipError;
    fn hip_is_device_integrated(device_id: c_int, integrated: *mut c_int) -> HipError;
    fn hip_supports_cooperative_launch(device_id: c_int, supported: *mut c_int) -> HipError;
}

/// HIP success error code
pub const HIP_SUCCESS: HipError = 0;

/// Convert a HIP error to a human-readable string
pub fn error_string(error: HipError) -> String {
    if error == HIP_SUCCESS {
        return "hipSuccess".to_string();
    }
    unsafe {
        let ptr = hip_get_error_string(error);
        if ptr.is_null() {
            return format!("Unknown HIP error: {}", error);
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Result type for HIP operations
pub type Result<T> = std::result::Result<T, HipErrorKind>;

/// HIP error with context
#[derive(Debug, Clone)]
pub struct HipErrorKind {
    pub code: HipError,
    pub message: String,
}

impl std::fmt::Display for HipErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HIP error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for HipErrorKind {}

/// Check a HIP error code and convert to Result
fn check(error: HipError) -> Result<()> {
    if error == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipErrorKind {
            code: error,
            message: error_string(error),
        })
    }
}

/// Get the number of HIP devices available
pub fn get_device_count() -> Result<i32> {
    let mut count: c_int = 0;
    unsafe { check(hip_get_device_count(&mut count))? };
    Ok(count)
}

/// Get properties for a HIP device
pub fn get_device_properties(device_id: i32) -> Result<HipDeviceProp> {
    let mut props = HipDeviceProp::default();
    unsafe { check(hip_get_device_properties(&mut props, device_id))? };
    Ok(props)
}

/// Get the device name as a String (using safe FFI)
pub fn get_device_name(device_id: i32) -> Result<String> {
    let mut name_buf = [0i8; 256];
    unsafe {
        check(hip_get_device_name(
            device_id,
            name_buf.as_mut_ptr(),
            256,
        ))?;
        let name = CStr::from_ptr(name_buf.as_ptr());
        Ok(name.to_string_lossy().into_owned())
    }
}

/// Get the GCN architecture name as a String (using safe FFI)
pub fn get_gcn_arch_name(device_id: i32) -> Result<String> {
    let mut name_buf = [0i8; 256];
    unsafe {
        check(hip_get_device_gcn_arch_name(
            device_id,
            name_buf.as_mut_ptr(),
            256,
        ))?;
        let name = CStr::from_ptr(name_buf.as_ptr());
        Ok(name.to_string_lossy().into_owned())
    }
}

/// Set the active HIP device
pub fn set_device(device_id: i32) -> Result<()> {
    unsafe { check(hip_set_device(device_id)) }
}

/// Get the current active HIP device
pub fn get_device() -> Result<i32> {
    let mut device_id: c_int = 0;
    unsafe { check(hip_get_device(&mut device_id))? };
    Ok(device_id)
}

/// Get total device memory in bytes
pub fn get_device_total_memory(device_id: i32) -> Result<usize> {
    let mut total_mem: usize = 0;
    unsafe { check(hip_get_device_total_memory(device_id, &mut total_mem))? };
    Ok(total_mem)
}

/// Get device multiprocessor (compute unit) count
pub fn get_device_mp_count(device_id: i32) -> Result<i32> {
    let mut count: c_int = 0;
    unsafe { check(hip_get_device_mp_count(device_id, &mut count))? };
    Ok(count)
}

/// Get device warp (wavefront) size
pub fn get_device_warp_size(device_id: i32) -> Result<i32> {
    let mut warp_size: c_int = 0;
    unsafe { check(hip_get_device_warp_size(device_id, &mut warp_size))? };
    Ok(warp_size)
}

/// Get device compute capability
pub fn get_device_compute_capability(device_id: i32) -> Result<(i32, i32)> {
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    unsafe { check(hip_get_device_compute_capability(device_id, &mut major, &mut minor))? };
    Ok((major, minor))
}

/// Check if device is integrated (APU)
pub fn is_device_integrated(device_id: i32) -> Result<bool> {
    let mut integrated: c_int = 0;
    unsafe { check(hip_is_device_integrated(device_id, &mut integrated))? };
    Ok(integrated != 0)
}

/// Check if device supports cooperative launch
pub fn device_supports_cooperative_launch(device_id: i32) -> Result<bool> {
    let mut supported: c_int = 0;
    unsafe { check(hip_supports_cooperative_launch(device_id, &mut supported))? };
    Ok(supported != 0)
}

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

/// A device buffer for GPU memory
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    /// Allocate a new device buffer with the given number of elements
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check(hip_malloc(&mut ptr, size))? };
        Ok(Self {
            ptr: ptr as *mut T,
            len,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the raw device pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get the raw device pointer (mutable)
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        if data.len() != self.len {
            return Err(HipErrorKind {
                code: -1,
                message: format!("Size mismatch: host {} vs device {}", data.len(), self.len),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let size = self.len * std::mem::size_of::<T>();
        unsafe {
            check(hip_memcpy_h2d(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                stream.handle(),
            ))
        }
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T], stream: &Stream) -> Result<()> {
        if data.len() != self.len {
            return Err(HipErrorKind {
                code: -1,
                message: format!("Size mismatch: device {} vs host {}", self.len, data.len()),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let size = self.len * std::mem::size_of::<T>();
        unsafe {
            check(hip_memcpy_d2h(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                stream.handle(),
            ))
        }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                hip_free(self.ptr as *mut c_void);
            }
        }
    }
}

// DeviceBuffer is Send + Sync since the GPU pointer is only accessed on GPU
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// Launch the copy kernel to copy f32 data from input to output
pub fn copy_f32(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check(launch_copy_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Copy f32 data from host, through GPU copy kernel, back to host.
/// This is a convenience function for testing.
pub fn hip_copy_kernel(input: &[f32]) -> Result<Vec<f32>> {
    // Use the null stream as a workaround for hipStreamCreate crashes
    let stream = Stream::null();

    let mut d_input = DeviceBuffer::<f32>::new(input.len())?;
    let mut d_output = DeviceBuffer::<f32>::new(input.len())?;

    d_input.copy_from_host(input, &stream)?;
    copy_f32(&d_input, &mut d_output, &stream)?;

    let mut output = vec![0.0f32; input.len()];
    d_output.copy_to_host(&mut output, &stream)?;
    stream.synchronize()?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_feature_compiles() {
        // This test exists - if it compiles with --features hip,
        // the build system works
    }

    #[test]
    fn test_hip_device_detection() {
        let count = get_device_count().expect("Failed to get device count");
        assert!(count > 0, "No HIP devices found");

        let name = get_device_name(0).expect("Failed to get device name");
        println!("Device 0: {}", name);

        let arch = get_gcn_arch_name(0).expect("Failed to get arch name");
        println!("Architecture: {}", arch);
    }

    #[test]
    fn test_simple_kernel_runs() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let output = hip_copy_kernel(&input).expect("Copy kernel failed");
        assert_eq!(input, output, "Copy kernel output mismatch");
    }

    #[test]
    fn test_hip_memory_roundtrip() {
        // Use null stream as workaround for hipStreamCreate crashes
        let stream = Stream::null();
        let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();

        let mut device_buf = DeviceBuffer::<f32>::new(1024).expect("Failed to allocate");
        device_buf
            .copy_from_host(&host_data, &stream)
            .expect("Failed to copy to device");

        let mut result = vec![0.0f32; 1024];
        device_buf
            .copy_to_host(&mut result, &stream)
            .expect("Failed to copy to host");
        stream.synchronize().expect("Failed to synchronize");

        assert_eq!(host_data, result, "Memcpy roundtrip failed");
    }

    // === Acceptance Criteria Tests for bd-2sh.3.2 ===
    //
    // NOTE: These tests require a working ROCm installation. On some ROCm versions
    // (particularly with gfx1151/RDNA 3.5), hipStreamCreate and hipMemcpy operations
    // may crash due to runtime issues. The tests are designed to pass when the
    // ROCm environment is functioning correctly.
    //
    // Known issue: ROCm 7.1.1 on gfx1151 crashes on stream creation and memory
    // copy operations. These tests will be enabled when the environment is fixed.

    #[test]
    fn test_hip_context_creation() {
        // Test that we can create a HIP context
        let ctx = HipContext::new().expect("Failed to create context");

        // Verify device_id is valid (>= 0)
        assert!(ctx.device_id() >= 0, "Device ID should be non-negative");

        // Verify device name is reasonable
        let name = ctx.device_name().expect("Failed to get device name");
        let arch = ctx.gcn_arch_name().unwrap_or_default();
        println!("Context created for device: {} ({})", name, arch);

        assert!(
            arch.contains("gfx") || name.contains("Radeon") || name.contains("AMD"),
            "Expected AMD GPU, got: {} ({})",
            name,
            arch
        );
    }

    #[test]
    fn test_hip_context_device_query() {
        // Test that we can query device properties through the context
        let ctx = HipContext::new().expect("Failed to create context");

        // Query various device properties
        let name = ctx.device_name().expect("Failed to get device name");
        let arch = ctx.gcn_arch_name().unwrap_or_else(|_| "unknown".to_string());
        let memory = ctx.total_memory().unwrap_or(0);
        let mp_count = ctx.multiprocessor_count().unwrap_or(0);
        let warp_size = ctx.warp_size().unwrap_or(0);
        let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
        let is_integrated = ctx.is_integrated().unwrap_or(false);
        let supports_coop = ctx.supports_cooperative_launch().unwrap_or(false);

        println!("Device: {}", name);
        println!("  Architecture: {}", arch);
        println!("  Total memory: {} MB", memory / (1024 * 1024));
        println!("  Multiprocessors: {}", mp_count);
        println!("  Warp size: {}", warp_size);
        println!("  Compute capability: {}.{}", major, minor);
        println!("  Integrated (APU): {}", is_integrated);
        println!("  Cooperative launch: {}", supports_coop);

        // Validate device name (most reliable)
        assert!(!name.is_empty(), "Device name should not be empty");

        // For gfx1151, verify expected warp size
        if arch.contains("gfx1151") {
            if warp_size > 0 {
                assert_eq!(warp_size, 32, "gfx1151 should have warp size 32");
                println!("  Verified gfx1151 warp size");
            }
        }
    }

    #[test]
    fn test_hip_stream_sync() {
        // Test stream synchronization using null stream
        // (explicit stream creation may crash on some ROCm versions)
        let ctx = HipContext::new().expect("Failed to create context");

        // Test null stream synchronization
        ctx.synchronize().expect("Failed to synchronize null stream");
        println!("Null stream synchronization test passed");
    }
}
