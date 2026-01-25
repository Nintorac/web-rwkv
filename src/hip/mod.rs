//! HIP backend for AMD GPUs.
//!
//! This module provides a HIP-based backend for running RWKV inference on AMD GPUs.
//! It requires a ROCm/TheRock installation with support for your GPU architecture.

mod runtime;

pub use runtime::HipRuntime;

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
    fn hip_malloc_managed(ptr: *mut *mut c_void, size: usize) -> HipError;
    fn hip_free(ptr: *mut c_void) -> HipError;
    fn hip_memset(ptr: *mut c_void, value: c_int, size: usize) -> HipError;
    fn hip_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize, stream: HipStream) -> HipError;
    fn hip_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize, stream: HipStream) -> HipError;
    fn hip_stream_create(stream: *mut HipStream) -> HipError;
    fn hip_stream_destroy(stream: HipStream) -> HipError;
    fn hip_stream_synchronize(stream: HipStream) -> HipError;
    fn hip_device_synchronize() -> HipError;
    fn hip_get_error_string(error: HipError) -> *const c_char;
    fn launch_copy_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_decay_exp_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_lerp_f32(a: *const f32, b: *const f32, t: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_sigmoid_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_squared_relu_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_softplus_decay_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_layer_norm_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        eps: f32,
        stream: HipStream
    ) -> HipError;
    fn launch_group_norm_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        g: c_int,
        eps: f32,
        stream: HipStream
    ) -> HipError;
    fn launch_l2_norm_f32(
        input: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        head_size: c_int,
        eps: f32,
        stream: HipStream
    ) -> HipError;
    fn launch_tanh_f32(input: *const f32, output: *mut f32, n: c_int, stream: HipStream) -> HipError;
    fn launch_token_shift_f32(
        x: *const f32,
        state_in: *const f32,
        mix: *const f32,
        output: *mut f32,
        state_out: *mut f32,
        c: c_int,
        t: c_int,
        stream: HipStream
    ) -> HipError;
    fn launch_channel_mix_state_f32(
        x: *const f32,
        state_in: *const f32,
        x_k: *const f32,
        output: *mut f32,
        state_out: *mut f32,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream
    ) -> HipError;
    fn launch_wkv_bonus_f32(
        r: *const f32,
        k: *const f32,
        v: *const f32,
        r_k: *const f32,
        output: *mut f32,
        n: c_int,    // head_size
        h: c_int,    // n_heads
        t: c_int,    // tokens
        b: c_int,    // batch
        stream: HipStream
    ) -> HipError;
    fn launch_control_k_f32(
        k_a: *const f32,   // per-channel weight [C, 1, 1, 1]
        a: *const f32,     // attention [C, T, B, 1]
        k: *const f32,     // key [C, T, B, 1]
        output: *mut f32,  // output [C, T, B, 1]
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream
    ) -> HipError;
    fn launch_wkv7_f32(
        w_decay: *const f32,   // [N, H, T, B] - pre-computed decay
        q: *const f32,         // [N, H, T, B]
        k: *const f32,         // [N, H, T, B]
        v: *const f32,         // [N, H, T, B]
        a: *const f32,         // [N, H, T, B]
        b: *const f32,         // [N, H, T, B]
        state_in: *const f32,  // [N, N, H, B]
        output: *mut f32,      // [N, H, T, B]
        state_out: *mut f32,   // [N, N, H, B]
        n: c_int,    // head_size
        h: c_int,    // n_heads
        t: c_int,    // tokens
        b: c_int,    // batch
        stream: HipStream
    ) -> HipError;
    fn launch_wkv7_f32_masked(
        w_decay: *const f32,   // [N, H, T, B] - pre-computed decay
        q: *const f32,         // [N, H, T, B]
        k: *const f32,         // [N, H, T, B]
        v: *const f32,         // [N, H, T, B]
        a: *const f32,         // [N, H, T, B]
        b: *const f32,         // [N, H, T, B]
        state_in: *const f32,  // [N, N, H, B]
        output: *mut f32,      // [N, H, T, B]
        state_out: *mut f32,   // [N, N, H, B]
        lengths: *const c_int, // [B] - real sequence length per batch
        n: c_int,    // head_size
        h: c_int,    // n_heads
        t: c_int,    // tokens (padded)
        b: c_int,    // batch
        stream: HipStream
    ) -> HipError;

    // Safe property accessors (avoid struct layout issues)
    fn hip_get_device_name(device_id: c_int, name: *mut c_char, max_len: c_int) -> HipError;
    fn hip_get_device_gcn_arch_name(device_id: c_int, name: *mut c_char, max_len: c_int) -> HipError;
    fn hip_get_device_total_memory(device_id: c_int, total_mem: *mut usize) -> HipError;
    fn hip_get_device_mp_count(device_id: c_int, count: *mut c_int) -> HipError;
    fn hip_get_device_warp_size(device_id: c_int, warp_size: *mut c_int) -> HipError;
    fn hip_get_device_compute_capability(device_id: c_int, major: *mut c_int, minor: *mut c_int) -> HipError;
    fn hip_is_device_integrated(device_id: c_int, integrated: *mut c_int) -> HipError;
    fn hip_supports_cooperative_launch(device_id: c_int, supported: *mut c_int) -> HipError;

    // rocBLAS functions
    fn rocblas_handle_create(handle: *mut RocblasHandle) -> RocblasStatus;
    fn rocblas_handle_destroy(handle: RocblasHandle) -> RocblasStatus;
    fn rocblas_set_stream_wrapper(handle: RocblasHandle, stream: HipStream) -> RocblasStatus;
    fn launch_hgemm(
        handle: RocblasHandle,
        m: c_int,       // rows of A and C
        n: c_int,       // cols of B and C
        k: c_int,       // cols of A, rows of B
        a: *const u16,  // M×K matrix (FP16 as u16)
        b: *const u16,  // K×N matrix (FP16 as u16)
        c: *mut u16     // M×N matrix (FP16 as u16)
    ) -> RocblasStatus;
    fn launch_sgemm(
        handle: RocblasHandle,
        m: c_int,       // rows of A and C
        n: c_int,       // cols of B and C
        k: c_int,       // cols of A, rows of B
        alpha: f32,
        a: *const f32,  // M×K matrix
        b: *const f32,  // K×N matrix
        beta: f32,
        c: *mut f32     // M×N matrix
    ) -> RocblasStatus;
    fn launch_sgemm_ta(
        handle: RocblasHandle,
        m: c_int,       // rows of output C (output features)
        n: c_int,       // cols of B and C (tokens)
        k: c_int,       // input features
        alpha: f32,
        a: *const f32,  // stored as K×M (row-major [M, K])
        b: *const f32,  // K×N matrix
        beta: f32,
        c: *mut f32     // M×N matrix
    ) -> RocblasStatus;
    fn rocblas_to_hip_error(status: RocblasStatus) -> HipError;
}

/// rocBLAS handle type (opaque pointer)
pub type RocblasHandle = *mut c_void;

/// rocBLAS status code
pub type RocblasStatus = c_int;

/// rocBLAS success status
pub const ROCBLAS_STATUS_SUCCESS: RocblasStatus = 0;

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

/// Memory allocation type for HIP tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Device-only memory (hipMalloc)
    Device,
    /// Managed/unified memory (hipMallocManaged) - accessible from CPU and GPU
    Managed,
}

/// A 4D tensor shape following web-rwkv conventions.
///
/// The shape is stored as `[x, y, z, w]` where `x` (shape[0]) is the fastest-moving
/// axis (contiguous in memory). This is opposite to PyTorch's convention.
///
/// Linear index formula: `offset(x, y, z, w) = x + y*X + z*X*Y + w*X*Y*Z`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    dims: [usize; 4],
}

impl TensorShape {
    /// Create a new shape.
    pub fn new(x: usize, y: usize, z: usize, w: usize) -> Self {
        Self { dims: [x, y, z, w] }
    }

    /// Create from a slice, padding with 1s for missing dimensions.
    pub fn from_slice(slice: &[usize]) -> Self {
        let mut dims = [1, 1, 1, 1];
        for (i, &d) in slice.iter().take(4).enumerate() {
            dims[i] = d;
        }
        Self { dims }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shape is empty.
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }

    /// Get dimension at index.
    pub fn dim(&self, index: usize) -> usize {
        self.dims[index]
    }

    /// Get all dimensions as a slice.
    pub fn dims(&self) -> &[usize; 4] {
        &self.dims
    }

    /// Compute strides for this shape.
    /// Stride[i] = product of dims[0..i]
    pub fn strides(&self) -> [usize; 4] {
        [
            1,
            self.dims[0],
            self.dims[0] * self.dims[1],
            self.dims[0] * self.dims[1] * self.dims[2],
        ]
    }

    /// Convert shaped indices to linear index.
    pub fn linear_index(&self, x: usize, y: usize, z: usize, w: usize) -> usize {
        let strides = self.strides();
        x * strides[0] + y * strides[1] + z * strides[2] + w * strides[3]
    }
}

impl std::ops::Index<usize> for TensorShape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {})",
            self.dims[0], self.dims[1], self.dims[2], self.dims[3]
        )
    }
}

/// A view into a tensor, defining a sub-region with custom strides.
#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    /// Shape of the view
    pub shape: TensorShape,
    /// Strides (in elements) for each dimension
    pub strides: [usize; 4],
    /// Offset (in elements) from the start of the underlying buffer
    pub offset: usize,
}

impl TensorView {
    /// Create a contiguous view for the full tensor.
    pub fn contiguous(shape: TensorShape) -> Self {
        Self {
            shape,
            strides: shape.strides(),
            offset: 0,
        }
    }

    /// Create a view with custom strides and offset.
    pub fn new(shape: TensorShape, strides: [usize; 4], offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Check if this view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.strides() && self.offset == 0
    }

    /// Convert shaped indices to linear index within this view.
    pub fn linear_index(&self, x: usize, y: usize, z: usize, w: usize) -> usize {
        self.offset + x * self.strides[0] + y * self.strides[1] + z * self.strides[2] + w * self.strides[3]
    }

    /// Create a sub-view (slice) of this view.
    ///
    /// Each range is `(start, end)` for the corresponding dimension.
    /// The new view shares the same underlying strides but has a new offset and shape.
    pub fn slice(
        &self,
        x_range: (usize, usize),
        y_range: (usize, usize),
        z_range: (usize, usize),
        w_range: (usize, usize),
    ) -> Result<Self> {
        // Validate ranges
        if x_range.1 > self.shape[0] || y_range.1 > self.shape[1]
            || z_range.1 > self.shape[2] || w_range.1 > self.shape[3]
        {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Slice out of bounds: ({:?}, {:?}, {:?}, {:?}) for shape {}",
                    x_range, y_range, z_range, w_range, self.shape
                ),
            });
        }

        let new_shape = TensorShape::new(
            x_range.1 - x_range.0,
            y_range.1 - y_range.0,
            z_range.1 - z_range.0,
            w_range.1 - w_range.0,
        );

        let new_offset = self.offset
            + x_range.0 * self.strides[0]
            + y_range.0 * self.strides[1]
            + z_range.0 * self.strides[2]
            + w_range.0 * self.strides[3];

        Ok(Self {
            shape: new_shape,
            strides: self.strides,
            offset: new_offset,
        })
    }
}

/// A HIP tensor with shape, strides, and GPU memory.
///
/// Follows web-rwkv conventions where shape[0] is the fastest-moving axis.
/// Supports both device-only and managed (unified) memory.
pub struct TensorHip<T> {
    ptr: *mut T,
    /// Total allocated elements (may be larger than view.shape.len() for non-contiguous views)
    allocated_len: usize,
    /// View into the tensor data
    view: TensorView,
    /// Memory type (device or managed)
    memory_type: MemoryType,
    /// Whether this tensor owns its memory (false for views)
    owned: bool,
}

impl<T> std::fmt::Debug for TensorHip<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorHip")
            .field("shape", &self.view.shape)
            .field("memory_type", &self.memory_type)
            .field("len", &self.allocated_len)
            .field("owned", &self.owned)
            .finish()
    }
}

impl<T: Copy> TensorHip<T> {
    /// Allocate a new tensor with the given shape using device memory.
    pub fn new(shape: TensorShape) -> Result<Self> {
        Self::with_memory_type(shape, MemoryType::Device)
    }

    /// Allocate a new tensor with the given shape using managed (unified) memory.
    pub fn managed(shape: TensorShape) -> Result<Self> {
        Self::with_memory_type(shape, MemoryType::Managed)
    }

    /// Allocate a new tensor with the specified memory type.
    pub fn with_memory_type(shape: TensorShape, memory_type: MemoryType) -> Result<Self> {
        let len = shape.len();
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                allocated_len: 0,
                view: TensorView::contiguous(shape),
                memory_type,
                owned: true,
            });
        }

        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            match memory_type {
                MemoryType::Device => check(hip_malloc(&mut ptr, size))?,
                MemoryType::Managed => check(hip_malloc_managed(&mut ptr, size))?,
            }
        }

        Ok(Self {
            ptr: ptr as *mut T,
            allocated_len: len,
            view: TensorView::contiguous(shape),
            memory_type,
            owned: true,
        })
    }

    /// Allocate a new tensor initialized to zeros.
    pub fn zeros(shape: TensorShape) -> Result<Self> {
        let tensor = Self::new(shape)?;
        if tensor.allocated_len > 0 {
            let size = tensor.allocated_len * std::mem::size_of::<T>();
            unsafe {
                check(hip_memset(tensor.ptr as *mut c_void, 0, size))?;
            }
        }
        Ok(tensor)
    }

    /// Allocate a managed tensor initialized to zeros.
    pub fn zeros_managed(shape: TensorShape) -> Result<Self> {
        let tensor = Self::managed(shape)?;
        if tensor.allocated_len > 0 {
            let size = tensor.allocated_len * std::mem::size_of::<T>();
            unsafe {
                check(hip_memset(tensor.ptr as *mut c_void, 0, size))?;
            }
        }
        Ok(tensor)
    }

    /// Create a tensor from host data.
    pub fn from_slice(data: &[T], shape: TensorShape, stream: &Stream) -> Result<Self> {
        if data.len() != shape.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Data length {} doesn't match shape {} (len={})",
                    data.len(),
                    shape,
                    shape.len()
                ),
            });
        }

        let tensor = Self::new(shape)?;
        if tensor.allocated_len > 0 {
            let size = tensor.allocated_len * std::mem::size_of::<T>();
            unsafe {
                check(hip_memcpy_h2d(
                    tensor.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size,
                    stream.handle(),
                ))?;
            }
        }
        Ok(tensor)
    }

    /// Create a tensor from host data using managed (unified) memory.
    ///
    /// Managed memory can be accessed by both CPU and GPU, which is useful for
    /// model weights on APUs where zero-copy access avoids transfer overhead.
    pub fn from_slice_managed(data: &[T], shape: TensorShape, stream: &Stream) -> Result<Self> {
        if data.len() != shape.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Data length {} doesn't match shape {} (len={})",
                    data.len(),
                    shape,
                    shape.len()
                ),
            });
        }

        let tensor = Self::managed(shape)?;
        if tensor.allocated_len > 0 {
            let size = tensor.allocated_len * std::mem::size_of::<T>();
            unsafe {
                check(hip_memcpy_h2d(
                    tensor.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size,
                    stream.handle(),
                ))?;
            }
        }
        Ok(tensor)
    }

    /// Get the shape of this tensor.
    pub fn shape(&self) -> TensorShape {
        self.view.shape
    }

    /// Get the strides of this tensor.
    pub fn strides(&self) -> [usize; 4] {
        self.view.strides
    }

    /// Get the view offset.
    pub fn offset(&self) -> usize {
        self.view.offset
    }

    /// Get the total number of elements in the view.
    pub fn len(&self) -> usize {
        self.view.shape.len()
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.view.shape.is_empty()
    }

    /// Check if this tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }

    /// Get the memory type.
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    /// Get the raw device pointer (at the view offset).
    pub fn as_ptr(&self) -> *const T {
        unsafe { self.ptr.add(self.view.offset) }
    }

    /// Get the raw device pointer (mutable, at the view offset).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.ptr.add(self.view.offset) }
    }

    /// Copy data from device to host.
    /// The tensor must be contiguous.
    pub fn to_vec(&self, stream: &Stream) -> Result<Vec<T>> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "Cannot copy non-contiguous tensor to host directly".to_string(),
            });
        }

        let len = self.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let mut result = vec![unsafe { std::mem::zeroed() }; len];
        let size = len * std::mem::size_of::<T>();
        unsafe {
            check(hip_memcpy_d2h(
                result.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                stream.handle(),
            ))?;
        }
        stream.synchronize()?;
        Ok(result)
    }

    /// Copy data from host to this tensor.
    /// The tensor must be contiguous.
    pub fn copy_from_slice(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "Cannot copy to non-contiguous tensor directly".to_string(),
            });
        }

        if data.len() != self.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!("Size mismatch: host {} vs tensor {}", data.len(), self.len()),
            });
        }

        if self.len() == 0 {
            return Ok(());
        }

        let size = self.len() * std::mem::size_of::<T>();
        unsafe {
            check(hip_memcpy_h2d(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                stream.handle(),
            ))
        }
    }

    /// Create a view (slice) of this tensor.
    ///
    /// The view shares the same underlying memory but has different shape/offset.
    /// Note: The returned view does not own memory and will not free it on drop.
    pub fn view(
        &self,
        x_range: (usize, usize),
        y_range: (usize, usize),
        z_range: (usize, usize),
        w_range: (usize, usize),
    ) -> Result<TensorHip<T>> {
        let new_view = self.view.slice(x_range, y_range, z_range, w_range)?;

        Ok(TensorHip {
            ptr: self.ptr,
            allocated_len: self.allocated_len,
            view: new_view,
            memory_type: self.memory_type,
            owned: false, // View doesn't own memory
        })
    }

    /// Get the TensorView for this tensor.
    pub fn tensor_view(&self) -> TensorView {
        self.view
    }
}

impl<T> Drop for TensorHip<T> {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                hip_free(self.ptr as *mut c_void);
            }
        }
    }
}

// TensorHip is Send + Sync since the GPU pointer is only accessed on GPU
unsafe impl<T: Send> Send for TensorHip<T> {}
unsafe impl<T: Sync> Sync for TensorHip<T> {}

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

/// Launch the decay exponential kernel: out = exp(-exp(x))
///
/// This is the time decay transformation used in RWKV7.
/// Numerically stable for all finite inputs.
pub fn decay_exp_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
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
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "decay_exp_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_decay_exp_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute decay exponential on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_decay_exp(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    decay_exp_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the lerp kernel: out = a + t * (b - a) (linear interpolation)
///
/// This is used for mixing operations in RWKV7.
/// All tensors must have the same length and be contiguous.
pub fn lerp_f32(
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    t: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let n = a.len();
    if b.len() != n || t.len() != n || output.len() != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, t={}, output={}",
                n, b.len(), t.len(), output.len()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() || !t.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "lerp_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_lerp_f32(
            a.as_ptr(),
            b.as_ptr(),
            t.as_ptr(),
            output.as_mut_ptr(),
            n as c_int,
            stream.handle(),
        ))
    }
}

/// Compute linear interpolation on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_lerp(a: &[f32], b: &[f32], t: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() || a.len() != t.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Size mismatch: a={}, b={}, t={}", a.len(), b.len(), t.len()),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(a.len(), 1, 1, 1);

    let d_a = TensorHip::from_slice(a, shape, &stream)?;
    let d_b = TensorHip::from_slice(b, shape, &stream)?;
    let d_t = TensorHip::from_slice(t, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    lerp_f32(&d_a, &d_b, &d_t, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the sigmoid kernel: out = 1 / (1 + exp(-x))
///
/// Standard sigmoid activation function.
pub fn sigmoid_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
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
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "sigmoid_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_sigmoid_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute sigmoid on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_sigmoid(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    sigmoid_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the squared ReLU kernel: out = max(0, x)^2
///
/// Used in RWKV7 channel mixing.
pub fn squared_relu_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
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
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "squared_relu_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_squared_relu_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute squared ReLU on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_squared_relu(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    squared_relu_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the softplus decay kernel: out = log(sigmoid(x)) - 0.5
///
/// Used for RWKV7 time decay computation.
/// Numerically stable for all finite inputs.
pub fn softplus_decay_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
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
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "softplus_decay_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_softplus_decay_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute softplus decay on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_softplus_decay(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    softplus_decay_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the layer normalization kernel.
///
/// Layer normalization normalizes each vector of length C (channel dimension)
/// independently. Used in RWKV7 for normalizing activations.
///
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1] where C is the channel dimension
/// * `weight` - Per-channel weight of shape [C, 1, 1, 1]
/// * `bias` - Per-channel bias of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `eps` - Epsilon for numerical stability (typically 1e-5)
/// * `stream` - HIP stream for asynchronous execution
pub fn layer_norm_f32(
    input: &TensorHip<f32>,
    weight: &TensorHip<f32>,
    bias: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [C, N, 1, 1]
    let c = input.shape()[0];  // Channel dimension (normalize over this)
    let n = input.shape()[1];  // Number of vectors

    // Validate shapes
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
            ),
        });
    }
    if weight.shape()[0] != c || bias.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias shape mismatch: expected [{}, 1, 1, 1], got weight={}, bias={}",
                c, weight.shape(), bias.shape()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "layer_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_layer_norm_f32(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute layer normalization on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `input` - Input data of shape [C, N, 1, 1] flattened (C*N elements)
/// * `weight` - Weight data of shape [C, 1, 1, 1] (C elements)
/// * `bias` - Bias data of shape [C, 1, 1, 1] (C elements)
/// * `c` - Channel dimension (length of each vector to normalize)
/// * `n` - Number of vectors to normalize
/// * `eps` - Epsilon for numerical stability
pub fn hip_layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c: usize,
    n: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    // Validate sizes
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {} ({}*{}), got {}",
                c * n, c, n, input.len()
            ),
        });
    }
    if weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Weight/bias size mismatch: expected {}, got weight={}, bias={}",
                c, weight.len(), bias.len()
            ),
        });
    }

    let stream = Stream::null();

    // Create tensors with proper shapes
    let input_shape = TensorShape::new(c, n, 1, 1);
    let param_shape = TensorShape::new(c, 1, 1, 1);

    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let d_weight = TensorHip::from_slice(weight, param_shape, &stream)?;
    let d_bias = TensorHip::from_slice(bias, param_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;

    layer_norm_f32(&d_input, &d_weight, &d_bias, &mut d_output, eps, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the group normalization kernel.
///
/// Group normalization divides channels into groups and normalizes within each group.
/// Used in RWKV7 with 12 groups (H = 12 heads) and eps = 64e-5.
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1]
/// * `weight` - Per-channel weight of shape [C, 1, 1, 1]
/// * `bias` - Per-channel bias of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `num_groups` - Number of groups (must divide C evenly)
/// * `eps` - Epsilon for numerical stability
/// * `stream` - HIP stream
pub fn group_norm_f32(
    input: &TensorHip<f32>,
    weight: &TensorHip<f32>,
    bias: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    num_groups: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    let n = input.shape()[1];

    if c % num_groups != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by num_groups {}",
                c, num_groups
            ),
        });
    }
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "group_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_group_norm_f32(
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            num_groups as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute group normalization on host data.
pub fn hip_group_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c: usize,
    n: usize,
    num_groups: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Input size mismatch: expected {}, got {}", c * n, input.len()),
        });
    }
    if weight.len() != c || bias.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Weight/bias size mismatch: expected {}", c),
        });
    }

    let stream = Stream::null();
    let input_shape = TensorShape::new(c, n, 1, 1);
    let param_shape = TensorShape::new(c, 1, 1, 1);

    let d_input = TensorHip::from_slice(input, input_shape, &stream)?;
    let d_weight = TensorHip::from_slice(weight, param_shape, &stream)?;
    let d_bias = TensorHip::from_slice(bias, param_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;

    group_norm_f32(&d_input, &d_weight, &d_bias, &mut d_output, num_groups, eps, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the L2 normalization kernel.
///
/// L2 normalization normalizes each head to unit L2 norm.
/// Used for key normalization in RWKV7.
///
/// # Arguments
/// * `input` - Input tensor of shape [C, N, 1, 1] where C = H * head_size
/// * `output` - Output tensor of shape [C, N, 1, 1]
/// * `head_size` - Size of each head (normalize over this dimension)
/// * `eps` - Epsilon for numerical stability
/// * `stream` - HIP stream
pub fn l2_norm_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    head_size: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    let c = input.shape()[0];
    let n = input.shape()[1];

    if c % head_size != 0 {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Channel count {} must be divisible by head_size {}",
                c, head_size
            ),
        });
    }
    if output.shape()[0] != c || output.shape()[1] != n {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, n, output.shape()
            ),
        });
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "l2_norm_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_l2_norm_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            n as c_int,
            head_size as c_int,
            eps,
            stream.handle(),
        ))
    }
}

/// Compute L2 normalization on host data.
pub fn hip_l2_norm(
    input: &[f32],
    c: usize,
    n: usize,
    head_size: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if input.len() != c * n {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Input size mismatch: expected {}, got {}", c * n, input.len()),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(c, n, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    l2_norm_f32(&d_input, &mut d_output, head_size, eps, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the tanh kernel: out = tanh(x)
pub fn tanh_f32(
    input: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
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
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "tanh_f32 requires contiguous tensors".to_string(),
        });
    }
    unsafe {
        check(launch_tanh_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

/// Compute tanh on host data.
pub fn hip_tanh(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    tanh_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the token shift kernel.
///
/// Token shift implements RWKV's time-mixing operation:
/// - output[t] = x[t] + mix * (prev[t] - x[t])
/// - Where prev[0] = state_in, prev[t>0] = x[t-1]
/// - state_out = x[T-1] (last token becomes state for next batch)
///
/// # Arguments
/// * `x` - Input tensor of shape [C, T, 1, 1]
/// * `state_in` - Previous state of shape [C, 1, 1, 1]
/// * `mix` - Per-channel mixing factor of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, T, 1, 1]
/// * `state_out` - New state of shape [C, 1, 1, 1]
/// * `stream` - HIP stream
pub fn token_shift_f32(
    x: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    mix: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = x.shape()[0];
    let t = x.shape()[1];

    if output.shape()[0] != c || output.shape()[1] != t {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, t, output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_out.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, 1, 1, 1]",
                c
            ),
        });
    }
    if mix.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Mix shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, mix.shape()
            ),
        });
    }

    unsafe {
        check(launch_token_shift_f32(
            x.as_ptr(),
            state_in.as_ptr(),
            mix.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            c as c_int,
            t as c_int,
            stream.handle(),
        ))
    }
}

/// Compute token shift on host data, returning (output, state_out).
pub fn hip_token_shift(
    x: &[f32],
    state_in: &[f32],
    mix: &[f32],
    c: usize,
    t: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if x.len() != c * t {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x size mismatch: expected {}, got {}", c * t, x.len()),
        });
    }
    if state_in.len() != c || mix.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state/mix size mismatch: expected {}", c),
        });
    }

    let stream = Stream::null();

    let x_shape = TensorShape::new(c, t, 1, 1);
    let state_shape = TensorShape::new(c, 1, 1, 1);

    let d_x = TensorHip::from_slice(x, x_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_mix = TensorHip::from_slice(mix, state_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(x_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    token_shift_f32(&d_x, &d_state_in, &d_mix, &mut d_output, &mut d_state_out, &stream)?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// Launch the channel-mix state kernel.
///
/// Same as token shift but with batch dimension.
/// Input: [C, T, B, 1], state: [C, B, 1, 1], x_k: [C, 1, 1, 1]
///
/// # Arguments
/// * `x` - Input tensor of shape [C, T, B, 1]
/// * `state_in` - Previous state per batch of shape [C, B, 1, 1]
/// * `x_k` - Per-channel mixing factor of shape [C, 1, 1, 1]
/// * `output` - Output tensor of shape [C, T, B, 1]
/// * `state_out` - New state per batch of shape [C, B, 1, 1]
/// * `stream` - HIP stream
pub fn channel_mix_state_f32(
    x: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    x_k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = x.shape()[0];
    let t = x.shape()[1];
    let b = x.shape()[2];

    if output.shape() != x.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                x.shape(), output.shape()
            ),
        });
    }
    if state_in.shape()[0] != c || state_in.shape()[1] != b {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "State shape mismatch: expected [{}, {}, 1, 1], got {}",
                c, b, state_in.shape()
            ),
        });
    }
    if x_k.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "x_k shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, x_k.shape()
            ),
        });
    }

    unsafe {
        check(launch_channel_mix_state_f32(
            x.as_ptr(),
            state_in.as_ptr(),
            x_k.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            c as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute channel-mix state on host data, returning (output, state_out).
pub fn hip_channel_mix_state(
    x: &[f32],
    state_in: &[f32],
    x_k: &[f32],
    c: usize,
    t: usize,
    b: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if x.len() != c * t * b {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x size mismatch: expected {}, got {}", c * t * b, x.len()),
        });
    }
    if state_in.len() != c * b {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", c * b, state_in.len()),
        });
    }
    if x_k.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("x_k size mismatch: expected {}, got {}", c, x_k.len()),
        });
    }

    let stream = Stream::null();

    let x_shape = TensorShape::new(c, t, b, 1);
    let state_shape = TensorShape::new(c, b, 1, 1);
    let xk_shape = TensorShape::new(c, 1, 1, 1);

    let d_x = TensorHip::from_slice(x, x_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_x_k = TensorHip::from_slice(x_k, xk_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(x_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    channel_mix_state_f32(&d_x, &d_state_in, &d_x_k, &mut d_output, &mut d_state_out, &stream)?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// Launch the WKV bonus kernel (time_first).
///
/// Computes: output = (r * k * r_k).sum(dim=head_size) * v
/// This is the "time_first" bonus attention on the current token.
///
/// # Arguments
/// * `r` - Receptance tensor of shape [N, H, T, B] where N=head_size, H=n_heads
/// * `k` - Key tensor of shape [N, H, T, B]
/// * `v` - Value tensor of shape [N, H, T, B]
/// * `r_k` - Per-head bonus weight of shape [N, H, 1, 1]
/// * `output` - Output tensor of shape [N, H, T, B]
/// * `stream` - HIP stream
pub fn wkv_bonus_f32(
    r: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    r_k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    // Shape: [N, H, T, B] where N=head_size
    let n = r.shape()[0];  // head_size
    let h = r.shape()[1];  // n_heads
    let t = r.shape()[2];  // tokens
    let b = r.shape()[3];  // batch

    // Validate shapes
    if k.shape() != r.shape() || v.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Shape mismatch: r={}, k={}, v={}",
                r.shape(), k.shape(), v.shape()
            ),
        });
    }
    if output.shape() != r.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                r.shape(), output.shape()
            ),
        });
    }
    if r_k.shape()[0] != n || r_k.shape()[1] != h {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "r_k shape mismatch: expected [{}, {}, 1, 1], got {}",
                n, h, r_k.shape()
            ),
        });
    }
    if !r.is_contiguous() || !k.is_contiguous() || !v.is_contiguous()
        || !r_k.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv_bonus_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv_bonus_f32(
            r.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            r_k.as_ptr(),
            output.as_mut_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV bonus on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `r` - Receptance data of shape [N, H, T, B] flattened
/// * `k` - Key data of shape [N, H, T, B] flattened
/// * `v` - Value data of shape [N, H, T, B] flattened
/// * `r_k` - Per-head bonus weight of shape [N, H, 1, 1] flattened
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens
/// * `b` - batch
pub fn hip_wkv_bonus(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    r_k: &[f32],
    n: usize,  // head_size
    h: usize,  // n_heads
    t: usize,  // tokens
    b: usize,  // batch
) -> Result<Vec<f32>> {
    let expected_len = n * h * t * b;
    let rk_len = n * h;

    if r.len() != expected_len || k.len() != expected_len || v.len() != expected_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got r={}, k={}, v={}",
                expected_len, r.len(), k.len(), v.len()
            ),
        });
    }
    if r_k.len() != rk_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("r_k size mismatch: expected {}, got {}", rk_len, r_k.len()),
        });
    }

    let stream = Stream::null();

    let data_shape = TensorShape::new(n, h, t, b);
    let rk_shape = TensorShape::new(n, h, 1, 1);

    let d_r = TensorHip::from_slice(r, data_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, data_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, data_shape, &stream)?;
    let d_r_k = TensorHip::from_slice(r_k, rk_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(data_shape)?;

    wkv_bonus_f32(&d_r, &d_k, &d_v, &d_r_k, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the control-K kernel (replacement key).
///
/// Computes: output = k * (1 + (a - 1) * k_a)
/// This creates the replacement key for RWKV7 time mixing.
///
/// # Arguments
/// * `k_a` - Per-channel control weight of shape [C, 1, 1, 1]
/// * `a` - Attention tensor of shape [C, T, B, 1]
/// * `k` - Key tensor of shape [C, T, B, 1]
/// * `output` - Output tensor of shape [C, T, B, 1]
/// * `stream` - HIP stream
pub fn control_k_f32(
    k_a: &TensorHip<f32>,
    a: &TensorHip<f32>,
    k: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    let c = k.shape()[0];
    let t = k.shape()[1];
    let b = k.shape()[2];

    if output.shape() != k.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                k.shape(), output.shape()
            ),
        });
    }
    if a.shape() != k.shape() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "a shape mismatch: expected {}, got {}",
                k.shape(), a.shape()
            ),
        });
    }
    if k_a.shape()[0] != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "k_a shape mismatch: expected [{}, 1, 1, 1], got {}",
                c, k_a.shape()
            ),
        });
    }
    if !k_a.is_contiguous() || !a.is_contiguous() || !k.is_contiguous() || !output.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: "control_k_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_control_k_f32(
            k_a.as_ptr(),
            a.as_ptr(),
            k.as_ptr(),
            output.as_mut_ptr(),
            c as c_int,
            t as c_int,
            b as c_int,
            stream.handle(),
        ))
    }
}

/// Compute control-K on host data, returning results.
/// This is a convenience function for testing.
///
/// # Arguments
/// * `k_a` - Per-channel control weight of size C
/// * `a` - Attention data of shape [C, T, B, 1] flattened
/// * `k` - Key data of shape [C, T, B, 1] flattened
/// * `c` - Channel dimension
/// * `t` - Token dimension
/// * `b` - Batch dimension
pub fn hip_control_k(
    k_a: &[f32],
    a: &[f32],
    k: &[f32],
    c: usize,
    t: usize,
    b: usize,
) -> Result<Vec<f32>> {
    let expected_len = c * t * b;

    if k.len() != expected_len || a.len() != expected_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got k={}, a={}",
                expected_len, k.len(), a.len()
            ),
        });
    }
    if k_a.len() != c {
        return Err(HipErrorKind {
            code: -1,
            message: format!("k_a size mismatch: expected {}, got {}", c, k_a.len()),
        });
    }

    let stream = Stream::null();

    let data_shape = TensorShape::new(c, t, b, 1);
    let ka_shape = TensorShape::new(c, 1, 1, 1);

    let d_k_a = TensorHip::from_slice(k_a, ka_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, data_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, data_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(data_shape)?;

    control_k_f32(&d_k_a, &d_a, &d_k, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

/// Launch the WKV7 core kernel.
///
/// Implements RWKV7's time mixing attention mechanism:
///   1. sa = sum(a[j] * state[i,j])  - attention over state
///   2. state[i,j] = state[i,j] * w[j] + sa * b[j] + k[j] * v[i]  - update
///   3. y[i] = sum(state[i,j] * q[j])  - output
///
/// # Arguments
/// * `w_decay` - Pre-computed decay tensor [N, H, T, B] where decay = exp(-exp(w_raw))
/// * `q` - Query tensor [N, H, T, B]
/// * `k` - Key tensor [N, H, T, B]
/// * `v` - Value tensor [N, H, T, B]
/// * `a` - Attention component tensor [N, H, T, B]
/// * `b` - Attention component tensor [N, H, T, B]
/// * `state_in` - Input state tensor [N, N, H, B]
/// * `output` - Output tensor [N, H, T, B]
/// * `state_out` - Output state tensor [N, N, H, B]
/// * `stream` - HIP stream
pub fn wkv7_f32(
    w_decay: &TensorHip<f32>,
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [N, H, T, B]
    let n = w_decay.shape()[0];  // head_size
    let h = w_decay.shape()[1];  // n_heads
    let t = w_decay.shape()[2];  // tokens
    let b_size = w_decay.shape()[3];  // batch

    // Validate input shapes
    let input_shape = w_decay.shape();
    if q.shape() != input_shape || k.shape() != input_shape || v.shape() != input_shape
        || a.shape() != input_shape || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input shape mismatch: w_decay={}, q={}, k={}, v={}, a={}, b={}",
                w_decay.shape(), q.shape(), k.shape(), v.shape(), a.shape(), b.shape()
            ),
        });
    }

    // Validate output shape
    if output.shape() != input_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                input_shape, output.shape()
            ),
        });
    }

    // Validate state shapes: [N, N, H, B]
    let state_shape = TensorShape::new(n, n, h, b_size);
    if state_in.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_in shape mismatch: expected {}, got {}",
                state_shape, state_in.shape()
            ),
        });
    }
    if state_out.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_out shape mismatch: expected {}, got {}",
                state_shape, state_out.shape()
            ),
        });
    }

    // Check contiguity
    if !w_decay.is_contiguous() || !q.is_contiguous() || !k.is_contiguous()
        || !v.is_contiguous() || !a.is_contiguous() || !b.is_contiguous()
        || !state_in.is_contiguous() || !output.is_contiguous() || !state_out.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_f32 requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_f32(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state_in.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV7 on host data, returning (output, state_out).
/// This is a convenience function for testing.
///
/// # Arguments
/// * `w_decay` - Pre-computed decay data [N, H, T, B] flattened
/// * `q`, `k`, `v`, `a`, `b` - Input data [N, H, T, B] flattened
/// * `state_in` - Input state [N, N, H, B] flattened
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens
/// * `b` - batch
pub fn hip_wkv7(
    w_decay: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    state_in: &[f32],
    n: usize,
    h: usize,
    t: usize,
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let input_len = n * h * t * batch;
    let state_len = n * n * h * batch;

    if w_decay.len() != input_len || q.len() != input_len || k.len() != input_len
        || v.len() != input_len || a.len() != input_len || b.len() != input_len
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got w={}, q={}, k={}, v={}, a={}, b={}",
                input_len, w_decay.len(), q.len(), k.len(), v.len(), a.len(), b.len()
            ),
        });
    }
    if state_in.len() != state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", state_len, state_in.len()),
        });
    }

    let stream = Stream::null();

    let input_shape = TensorShape::new(n, h, t, batch);
    let state_shape = TensorShape::new(n, n, h, batch);

    let d_w_decay = TensorHip::from_slice(w_decay, input_shape, &stream)?;
    let d_q = TensorHip::from_slice(q, input_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, input_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, input_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, input_shape, &stream)?;
    let d_b = TensorHip::from_slice(b, input_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    wkv7_f32(
        &d_w_decay, &d_q, &d_k, &d_v, &d_a, &d_b,
        &d_state_in, &mut d_output, &mut d_state_out, &stream
    )?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
}

/// WKV7 kernel with length masking for variable-length batched sequences.
/// Skips state updates for padding positions, preserving: state(seq + padding) == state(seq).
///
/// # Arguments
/// * `w_decay` - Pre-computed decay tensor [N, H, T, B]
/// * `q`, `k`, `v`, `a`, `b` - Input tensors [N, H, T, B]
/// * `state_in` - Input state tensor [N, N, H, B]
/// * `output` - Output tensor [N, H, T, B]
/// * `state_out` - Output state tensor [N, N, H, B]
/// * `lengths` - Real sequence length per batch [B] (as i32)
/// * `stream` - HIP stream
pub fn wkv7_f32_masked(
    w_decay: &TensorHip<f32>,
    q: &TensorHip<f32>,
    k: &TensorHip<f32>,
    v: &TensorHip<f32>,
    a: &TensorHip<f32>,
    b: &TensorHip<f32>,
    state_in: &TensorHip<f32>,
    output: &mut TensorHip<f32>,
    state_out: &mut TensorHip<f32>,
    lengths: &TensorHip<i32>,
    stream: &Stream,
) -> Result<()> {
    // Input shape: [N, H, T, B]
    let n = w_decay.shape()[0];  // head_size
    let h = w_decay.shape()[1];  // n_heads
    let t = w_decay.shape()[2];  // tokens (padded)
    let b_size = w_decay.shape()[3];  // batch

    // Validate input shapes
    let input_shape = w_decay.shape();
    if q.shape() != input_shape || k.shape() != input_shape || v.shape() != input_shape
        || a.shape() != input_shape || b.shape() != input_shape
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input shape mismatch: w_decay={}, q={}, k={}, v={}, a={}, b={}",
                w_decay.shape(), q.shape(), k.shape(), v.shape(), a.shape(), b.shape()
            ),
        });
    }

    // Validate output shape
    if output.shape() != input_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Output shape mismatch: expected {}, got {}",
                input_shape, output.shape()
            ),
        });
    }

    // Validate state shapes: [N, N, H, B]
    let state_shape = TensorShape::new(n, n, h, b_size);
    if state_in.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_in shape mismatch: expected {}, got {}",
                state_shape, state_in.shape()
            ),
        });
    }
    if state_out.shape() != state_shape {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "state_out shape mismatch: expected {}, got {}",
                state_shape, state_out.shape()
            ),
        });
    }

    // Validate lengths shape: should have b_size elements
    let lengths_len = lengths.shape().len();
    if lengths_len != b_size {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "lengths size mismatch: expected {}, got {}",
                b_size, lengths_len
            ),
        });
    }

    // Check contiguity
    if !w_decay.is_contiguous() || !q.is_contiguous() || !k.is_contiguous()
        || !v.is_contiguous() || !a.is_contiguous() || !b.is_contiguous()
        || !state_in.is_contiguous() || !output.is_contiguous() || !state_out.is_contiguous()
        || !lengths.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: "wkv7_f32_masked requires contiguous tensors".to_string(),
        });
    }

    unsafe {
        check(launch_wkv7_f32_masked(
            w_decay.as_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            state_in.as_ptr(),
            output.as_mut_ptr(),
            state_out.as_mut_ptr(),
            lengths.as_ptr(),
            n as c_int,
            h as c_int,
            t as c_int,
            b_size as c_int,
            stream.handle(),
        ))
    }
}

/// Compute WKV7 with length masking on host data, returning (output, state_out).
/// This is a convenience function for testing.
///
/// # Arguments
/// * `w_decay` - Pre-computed decay data [N, H, T, B] flattened
/// * `q`, `k`, `v`, `a`, `b` - Input data [N, H, T, B] flattened
/// * `state_in` - Input state [N, N, H, B] flattened
/// * `lengths` - Real sequence length per batch [B]
/// * `n` - head_size
/// * `h` - n_heads
/// * `t` - tokens (padded)
/// * `batch` - batch size
pub fn hip_wkv7_masked(
    w_decay: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    state_in: &[f32],
    lengths: &[i32],
    n: usize,
    h: usize,
    t: usize,
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let input_len = n * h * t * batch;
    let state_len = n * n * h * batch;

    if w_decay.len() != input_len || q.len() != input_len || k.len() != input_len
        || v.len() != input_len || a.len() != input_len || b.len() != input_len
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Input size mismatch: expected {}, got w={}, q={}, k={}, v={}, a={}, b={}",
                input_len, w_decay.len(), q.len(), k.len(), v.len(), a.len(), b.len()
            ),
        });
    }
    if state_in.len() != state_len {
        return Err(HipErrorKind {
            code: -1,
            message: format!("state_in size mismatch: expected {}, got {}", state_len, state_in.len()),
        });
    }
    if lengths.len() != batch {
        return Err(HipErrorKind {
            code: -1,
            message: format!("lengths size mismatch: expected {}, got {}", batch, lengths.len()),
        });
    }

    let stream = Stream::null();

    let input_shape = TensorShape::new(n, h, t, batch);
    let state_shape = TensorShape::new(n, n, h, batch);
    let lengths_shape = TensorShape::new(batch, 1, 1, 1);

    let d_w_decay = TensorHip::from_slice(w_decay, input_shape, &stream)?;
    let d_q = TensorHip::from_slice(q, input_shape, &stream)?;
    let d_k = TensorHip::from_slice(k, input_shape, &stream)?;
    let d_v = TensorHip::from_slice(v, input_shape, &stream)?;
    let d_a = TensorHip::from_slice(a, input_shape, &stream)?;
    let d_b = TensorHip::from_slice(b, input_shape, &stream)?;
    let d_state_in = TensorHip::from_slice(state_in, state_shape, &stream)?;
    let d_lengths = TensorHip::from_slice(lengths, lengths_shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(input_shape)?;
    let mut d_state_out = TensorHip::<f32>::new(state_shape)?;

    wkv7_f32_masked(
        &d_w_decay, &d_q, &d_k, &d_v, &d_a, &d_b,
        &d_state_in, &mut d_output, &mut d_state_out,
        &d_lengths, &stream
    )?;

    let output = d_output.to_vec(&stream)?;
    let state_out = d_state_out.to_vec(&stream)?;

    Ok((output, state_out))
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

// ============================================================================
// Model Loading for RWKV7 HIP Backend
// ============================================================================

use half::f16;
use std::path::Path;

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

/// A single layer's layer normalization weights.
#[derive(Debug)]
pub struct LayerNormHip {
    pub weight: TensorHip<f32>,
    pub bias: TensorHip<f32>,
}

/// Attention weights for a single layer.
#[derive(Debug)]
pub struct AttentionHip {
    // Token shift mix weights
    pub x_r: TensorHip<f32>,
    pub x_w: TensorHip<f32>,
    pub x_k: TensorHip<f32>,
    pub x_v: TensorHip<f32>,
    pub x_a: TensorHip<f32>,
    pub x_g: TensorHip<f32>,

    // Decay LoRA
    pub w0: TensorHip<f32>,
    pub w1: TensorHip<f32>,
    pub w2: TensorHip<f32>,

    // Learning rate LoRA
    pub a0: TensorHip<f32>,
    pub a1: TensorHip<f32>,
    pub a2: TensorHip<f32>,

    // Gate LoRA
    pub g1: TensorHip<f32>,
    pub g2: TensorHip<f32>,

    // Value residual LoRA (layers > 0)
    pub v0: Option<TensorHip<f32>>,
    pub v1: Option<TensorHip<f32>>,
    pub v2: Option<TensorHip<f32>>,

    // Key normalization weights
    pub r_k: TensorHip<f32>,
    pub k_k: TensorHip<f32>,
    pub k_a: TensorHip<f32>,

    // Projection matrices (column-major for rocBLAS)
    pub w_r: TensorHip<f32>,  // Receptance: [n_embd, n_embd]
    pub w_k: TensorHip<f32>,  // Key: [n_embd, n_embd]
    pub w_v: TensorHip<f32>,  // Value: [n_embd, n_embd]
    pub w_o: TensorHip<f32>,  // Output: [n_embd, n_embd]

    // Group normalization
    pub gn: LayerNormHip,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnHip {
    // Token shift mix weight
    pub x_k: TensorHip<f32>,

    // Projection matrices
    pub w_k: TensorHip<f32>,  // Key (expand): [n_hidden, n_embd]
    pub w_v: TensorHip<f32>,  // Value (contract): [n_embd, n_hidden]
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
    pub w: TensorHip<f32>,  // [n_vocab, n_embd]
}

/// Output head weights.
#[derive(Debug)]
pub struct HeadHip {
    pub ln: LayerNormHip,
    pub w: TensorHip<f32>,  // [n_vocab, n_embd]
}

/// RWKV7 model loaded into HIP memory.
///
/// Weights are stored in managed memory for zero-copy APU access.
/// All tensors use FP32 internally (converted from FP16 at load time).
#[derive(Debug)]
pub struct Rwkv7Hip {
    pub info: Rwkv7ModelInfo,
    pub embed: EmbedHip,
    pub head: HeadHip,
    pub layers: Vec<LayerHip>,
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

/// State for HIP RWKV7 batched inference.
///
/// Holds the recurrent state needed to continue inference from a previous position.
/// Supports batch sizes > 1 for high-throughput inference.
///
/// # State Layout
/// All state tensors include a batch dimension:
/// - `att_states`: WKV recurrent state `[n_layer][head_size * head_size * n_head * batch]`
/// - `att_shift_states`: Attention token shift state `[n_layer][n_embd * batch]`
/// - `ffn_states`: FFN token shift state `[n_layer][n_embd * batch]`
///
/// # Usage
/// ```ignore
/// // Batched inference (B=4 sequences)
/// let mut state = HipState::new(&model.info, 4);
/// let tokens: Vec<&[u32]> = vec![&seq1, &seq2, &seq3, &seq4];
/// let logits = model.forward_with_state(&tokens, &mut state)?;
///
/// // Streaming with batching: process one token per sequence
/// let next_tokens: Vec<&[u32]> = vec![&[t1], &[t2], &[t3], &[t4]];
/// let logits = model.forward_with_state(&next_tokens, &mut state)?;
///
/// // Single sequence (B=1) for simple use cases
/// let mut state = HipState::new(&model.info, 1);
/// let logits = model.forward_with_state(&[&tokens], &mut state)?;
/// ```
#[derive(Debug, Clone)]
pub struct HipState {
    /// Batch size this state was created for
    pub batch_size: usize,
    /// WKV state per layer: [head_size * head_size * n_head * batch] per layer
    pub att_states: Vec<Vec<f32>>,
    /// Attention token shift state per layer: [n_embd * batch] per layer
    pub att_shift_states: Vec<Vec<f32>>,
    /// FFN token shift state per layer: [n_embd * batch] per layer
    pub ffn_states: Vec<Vec<f32>>,
    /// Value residual from first layer for RWKV7, persisted across chunks: [n_embd * batch]
    pub v_first: Option<Vec<f32>>,
}

impl HipState {
    /// Create a fresh state initialized to zeros.
    ///
    /// # Arguments
    /// * `info` - Model info containing dimensions
    /// * `batch_size` - Number of sequences to process in parallel
    pub fn new(info: &Rwkv7ModelInfo, batch_size: usize) -> Self {
        let n_layer = info.n_layer;
        let n_embd = info.n_embd;
        let head_size = info.head_size;
        let n_head = info.n_head;

        HipState {
            batch_size,
            att_states: (0..n_layer)
                .map(|_| vec![0.0f32; head_size * head_size * n_head * batch_size])
                .collect(),
            att_shift_states: (0..n_layer)
                .map(|_| vec![0.0f32; n_embd * batch_size])
                .collect(),
            ffn_states: (0..n_layer)
                .map(|_| vec![0.0f32; n_embd * batch_size])
                .collect(),
            v_first: None,
        }
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        for state in &mut self.att_states {
            state.fill(0.0);
        }
        for state in &mut self.att_shift_states {
            state.fill(0.0);
        }
        for state in &mut self.ffn_states {
            state.fill(0.0);
        }
        self.v_first = None;
    }
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
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            // row-major index: i * cols + j
            // column-major index: j * rows + i
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

/// Load a tensor from SafeTensors, converting f16 to f32 and loading into managed HIP memory.
fn load_tensor_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f32>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    // Convert f16 bytes to f32 vec
    let f32_data: Vec<f32> = match dtype {
        safetensors::Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        safetensors::Dtype::F32 => {
            bytemuck::cast_slice(data).to_vec()
        }
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    // Convert shape to TensorShape (web-rwkv convention: shape[0] is fastest axis)
    // SafeTensors stores shape as [slow, ..., fast], so we need to reverse
    let hip_shape = match shape_st.len() {
        1 => TensorShape::new(shape_st[0], 1, 1, 1),
        2 => TensorShape::new(shape_st[1], shape_st[0], 1, 1),
        3 => TensorShape::new(shape_st[2], shape_st[1], shape_st[0], 1),
        4 => TensorShape::new(shape_st[3], shape_st[2], shape_st[1], shape_st[0]),
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported shape {:?} for tensor {}", shape_st, name
        ))),
    };

    TensorHip::from_slice_managed(&f32_data, hip_shape, stream).map_err(ModelLoadError::from)
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
fn load_weight_matrix_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    stream: &Stream,
) -> std::result::Result<TensorHip<f32>, ModelLoadError> {
    let tensor = st.tensor(name).map_err(|e| ModelLoadError::SafeTensor(format!("{}: {}", name, e)))?;

    let shape_st = tensor.shape();
    let dtype = tensor.dtype();
    let data = tensor.data();

    if shape_st.len() != 2 {
        return Err(ModelLoadError::InvalidModel(format!(
            "Weight matrix {} must be 2D, got {:?}", name, shape_st
        )));
    }

    let rows = shape_st[0];  // out_features (M)
    let cols = shape_st[1];  // in_features (K)

    // Convert f16 bytes to f32 vec
    let f32_data: Vec<f32> = match dtype {
        safetensors::Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        safetensors::Dtype::F32 => {
            bytemuck::cast_slice(data).to_vec()
        }
        safetensors::Dtype::BF16 => {
            let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        _ => return Err(ModelLoadError::InvalidModel(format!(
            "Unsupported dtype {:?} for tensor {}", dtype, name
        ))),
    };

    // Transpose from row-major to column-major
    let transposed = transpose_2d(&f32_data, rows, cols);

    // Shape is [M, K] where M=rows (out_features), K=cols (in_features)
    // This is the natural column-major representation for rocBLAS
    // Note: This differs from load_tensor_f32 which reverses dimensions
    let hip_shape = TensorShape::new(rows, cols, 1, 1);

    TensorHip::from_slice_managed(&transposed, hip_shape, stream).map_err(ModelLoadError::from)
}

/// Load layer normalization weights.
fn load_layer_norm(
    st: &safetensors::SafeTensors,
    prefix: &str,
    stream: &Stream,
) -> std::result::Result<LayerNormHip, ModelLoadError> {
    let weight = load_tensor_f32(st, &format!("{}.weight", prefix), stream)?;
    let bias = load_tensor_f32(st, &format!("{}.bias", prefix), stream)?;
    Ok(LayerNormHip { weight, bias })
}

impl Rwkv7Hip {
    /// Load an RWKV7 model from a SafeTensors file.
    ///
    /// Weights are loaded into managed (unified) memory for efficient APU access.
    /// All weights are converted to FP32 for computation.
    ///
    /// # Arguments
    /// * `path` - Path to the .st (SafeTensors) file
    ///
    /// # Returns
    /// The loaded model with all weights in HIP memory, or an error.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, ModelLoadError> {
        let data = std::fs::read(path.as_ref())?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| ModelLoadError::SafeTensor(format!("Failed to parse SafeTensors: {}", e)))?;

        let stream = Stream::null();

        // Detect model dimensions from tensor shapes
        let embed_tensor = st.tensor("emb.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("emb.weight: {}", e)))?;
        let embed_shape = embed_tensor.shape();  // [n_vocab, n_embd]
        let n_vocab = embed_shape[0];
        let n_embd = embed_shape[1];

        // Get n_head from r_k tensor (RWKV7-specific)
        let r_k_tensor = st.tensor("blocks.0.att.r_k")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.att.r_k: {}", e)))?;
        let n_head = r_k_tensor.shape()[0];
        let head_size = n_embd / n_head;

        // Get n_hidden from FFN key weight
        let ffn_k_tensor = st.tensor("blocks.0.ffn.key.weight")
            .map_err(|e| ModelLoadError::SafeTensor(format!("blocks.0.ffn.key.weight: {}", e)))?;
        let n_hidden = ffn_k_tensor.shape()[0];

        // Count layers
        let n_layer = st.names().iter()
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

        log::info!("Loading RWKV7 model: {} layers, {} embd, {} heads, {} vocab",
            n_layer, n_embd, n_head, n_vocab);

        // Load embedding (no transpose - it's a lookup table, not GEMM)
        let embed = EmbedHip {
            ln: load_layer_norm(&st, "blocks.0.ln0", &stream)?,
            w: load_tensor_f32(&st, "emb.weight", &stream)?,
        };

        // Load output head (transpose to column-major for GEMM)
        let head = HeadHip {
            ln: load_layer_norm(&st, "ln_out", &stream)?,
            w: load_weight_matrix_f32(&st, "head.weight", &stream)?,
        };

        // Load layers
        let mut layers = Vec::with_capacity(n_layer);
        for layer_idx in 0..n_layer {
            let prefix = format!("blocks.{}", layer_idx);

            // Attention layer norm
            let att_ln = load_layer_norm(&st, &format!("{}.ln1", prefix), &stream)?;

            // FFN layer norm
            let ffn_ln = load_layer_norm(&st, &format!("{}.ln2", prefix), &stream)?;

            // Attention weights
            let att = AttentionHip {
                x_r: load_tensor_f32(&st, &format!("{}.att.x_r", prefix), &stream)?,
                x_w: load_tensor_f32(&st, &format!("{}.att.x_w", prefix), &stream)?,
                x_k: load_tensor_f32(&st, &format!("{}.att.x_k", prefix), &stream)?,
                x_v: load_tensor_f32(&st, &format!("{}.att.x_v", prefix), &stream)?,
                x_a: load_tensor_f32(&st, &format!("{}.att.x_a", prefix), &stream)?,
                x_g: load_tensor_f32(&st, &format!("{}.att.x_g", prefix), &stream)?,

                w0: load_tensor_f32(&st, &format!("{}.att.w0", prefix), &stream)?,
                w1: load_weight_matrix_f32(&st, &format!("{}.att.w1", prefix), &stream)?,
                w2: load_weight_matrix_f32(&st, &format!("{}.att.w2", prefix), &stream)?,

                a0: load_tensor_f32(&st, &format!("{}.att.a0", prefix), &stream)?,
                a1: load_weight_matrix_f32(&st, &format!("{}.att.a1", prefix), &stream)?,
                a2: load_weight_matrix_f32(&st, &format!("{}.att.a2", prefix), &stream)?,

                g1: load_weight_matrix_f32(&st, &format!("{}.att.g1", prefix), &stream)?,
                g2: load_weight_matrix_f32(&st, &format!("{}.att.g2", prefix), &stream)?,

                // Value residual LoRA (only for layers > 0)
                v0: if layer_idx > 0 {
                    Some(load_tensor_f32(&st, &format!("{}.att.v0", prefix), &stream)?)
                } else {
                    None
                },
                v1: if layer_idx > 0 {
                    Some(load_weight_matrix_f32(&st, &format!("{}.att.v1", prefix), &stream)?)
                } else {
                    None
                },
                v2: if layer_idx > 0 {
                    Some(load_weight_matrix_f32(&st, &format!("{}.att.v2", prefix), &stream)?)
                } else {
                    None
                },

                r_k: load_tensor_f32(&st, &format!("{}.att.r_k", prefix), &stream)?,
                k_k: load_tensor_f32(&st, &format!("{}.att.k_k", prefix), &stream)?,
                k_a: load_tensor_f32(&st, &format!("{}.att.k_a", prefix), &stream)?,

                w_r: load_weight_matrix_f32(&st, &format!("{}.att.receptance.weight", prefix), &stream)?,
                w_k: load_weight_matrix_f32(&st, &format!("{}.att.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f32(&st, &format!("{}.att.value.weight", prefix), &stream)?,
                w_o: load_weight_matrix_f32(&st, &format!("{}.att.output.weight", prefix), &stream)?,

                gn: load_layer_norm(&st, &format!("{}.att.ln_x", prefix), &stream)?,
            };

            // FFN weights (transpose weight matrices to column-major for GEMM)
            let ffn = FfnHip {
                x_k: load_tensor_f32(&st, &format!("{}.ffn.x_k", prefix), &stream)?,
                w_k: load_weight_matrix_f32(&st, &format!("{}.ffn.key.weight", prefix), &stream)?,
                w_v: load_weight_matrix_f32(&st, &format!("{}.ffn.value.weight", prefix), &stream)?,
            };

            layers.push(LayerHip { att_ln, ffn_ln, att, ffn });
        }

        // Synchronize to ensure all transfers are complete
        stream.synchronize()?;

        log::info!("RWKV7 model loaded successfully");

        Ok(Self { info, embed, head, layers })
    }

    /// Get a reference to a specific weight tensor by name (for spot-checking).
    ///
    /// Name format examples:
    /// - "emb.weight" - embedding weights
    /// - "blocks.0.att.receptance.weight" - layer 0 attention receptance
    /// - "blocks.5.ffn.key.weight" - layer 5 FFN key weights
    /// - "head.weight" - output head weights
    pub fn get_weight(&self, name: &str) -> Option<&TensorHip<f32>> {
        if name == "emb.weight" {
            return Some(&self.embed.w);
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
    pub fn read_weight_head(&self, name: &str, n: usize) -> std::result::Result<Vec<f32>, ModelLoadError> {
        let tensor = self.get_weight(name)
            .ok_or_else(|| ModelLoadError::InvalidModel(format!("Weight not found: {}", name)))?;

        let stream = Stream::null();
        let all_data = tensor.to_vec(&stream)?;
        let n = n.min(all_data.len());
        Ok(all_data[..n].to_vec())
    }

    /// Run a full forward pass on input tokens (single sequence, fresh state).
    ///
    /// Convenience wrapper for `forward_with_state` with batch_size=1 and fresh state.
    ///
    /// # Arguments
    /// * `tokens` - Input token IDs for a single sequence
    ///
    /// # Returns
    /// Logits tensor of shape [vocab_size * T] for each input token.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut state = HipState::new(&self.info, 1);
        self.forward_with_state(&[tokens], &mut state)
    }

    /// Run a batched forward pass with explicit state management.
    ///
    /// This is the core inference API supporting:
    /// - **Batched inference**: Process multiple sequences in parallel (B > 1)
    /// - **Streaming**: Token-by-token generation with state carryover
    /// - **Chunked processing**: Long context in pieces with state carryover
    ///
    /// # Arguments
    /// * `tokens` - Batch of sequences. Each `&[u32]` is one sequence.
    ///              All sequences must have the same length T.
    /// * `state` - Mutable reference to inference state. Must have `batch_size == tokens.len()`.
    ///
    /// # Returns
    /// Logits tensor of shape [vocab_size * T * B] in column-major layout [V, T, B].
    ///
    /// # Example
    /// ```ignore
    /// // Batched prefill: 4 sequences of 128 tokens each
    /// let mut state = HipState::new(&model.info, 4);
    /// let sequences: Vec<&[u32]> = vec![&seq1, &seq2, &seq3, &seq4];
    /// let logits = model.forward_with_state(&sequences, &mut state)?;
    ///
    /// // Batched decode: generate next token for all 4 sequences
    /// let next_tokens: Vec<&[u32]> = vec![&[t1], &[t2], &[t3], &[t4]];
    /// let logits = model.forward_with_state(&next_tokens, &mut state)?;
    /// ```
    pub fn forward_with_state(&self, tokens: &[&[u32]], state: &mut HipState) -> Result<Vec<f32>> {
        let b = tokens.len();
        if b == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty batch".to_string(),
            });
        }
        if b != state.batch_size {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Batch size mismatch: tokens has {} sequences but state has batch_size={}",
                    b, state.batch_size
                ),
            });
        }

        let t = tokens[0].len();
        if t == 0 {
            return Err(HipErrorKind {
                code: -1,
                message: "Empty sequence".to_string(),
            });
        }
        // Verify all sequences have the same length
        for (i, seq) in tokens.iter().enumerate() {
            if seq.len() != t {
                return Err(HipErrorKind {
                    code: -1,
                    message: format!(
                        "Sequence length mismatch: sequence 0 has {} tokens but sequence {} has {}",
                        t, i, seq.len()
                    ),
                });
            }
        }

        let stream = Stream::null();
        let n_embd = self.info.n_embd;
        let n_head = self.info.n_head;
        let head_size = self.info.head_size;
        let n_vocab = self.info.n_vocab;
        let n_layer = self.info.n_layer;
        let n_hidden = self.info.n_hidden;

        // Embedding lookup: tokens[b][t] -> x[c, t, b]
        // Layout: x[c, t, b] = x[b * T * C + t * C + c]
        // embed.w is row-major [n_vocab, n_embd], so embed[token, c] = emb_data[token * n_embd + c]
        let emb_data = self.embed.w.to_vec(&stream)?;
        let mut x = vec![0.0f32; n_embd * t * b];
        for batch_idx in 0..b {
            for time_idx in 0..t {
                let token = tokens[batch_idx][time_idx] as usize;
                for c in 0..n_embd {
                    let idx = batch_idx * t * n_embd + time_idx * n_embd + c;
                    x[idx] = emb_data[token * n_embd + c];
                }
            }
        }

        // v_first is computed fresh each forward call (not part of persistent state)
        let mut v_first: Option<Vec<f32>> = None;

        // Process each layer
        for layer_idx in 0..n_layer {
            let layer = &self.layers[layer_idx];

            // Apply ln0 for layer 0
            if layer_idx == 0 {
                let ln0_w = self.embed.ln.weight.to_vec(&stream)?;
                let ln0_b = self.embed.ln.bias.to_vec(&stream)?;
                x = hip_layer_norm(&x, &ln0_w, &ln0_b, n_embd, t * b, 1e-5)?;
            }

            // ==== Time-Mix (Attention) ====
            let ln1_w = layer.att_ln.weight.to_vec(&stream)?;
            let ln1_b = layer.att_ln.bias.to_vec(&stream)?;
            let x_ln1 = hip_layer_norm(&x, &ln1_w, &ln1_b, n_embd, t * b, 1e-5)?;

            // Token shift for attention - all shifts use the same old state
            let att_shift_state = &state.att_shift_states[layer_idx];
            let x_r = layer.att.x_r.to_vec(&stream)?;
            let x_w = layer.att.x_w.to_vec(&stream)?;
            let x_k = layer.att.x_k.to_vec(&stream)?;
            let x_v = layer.att.x_v.to_vec(&stream)?;
            let x_a = layer.att.x_a.to_vec(&stream)?;
            let x_g = layer.att.x_g.to_vec(&stream)?;

            let (xr, new_att_shift) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_r, n_embd, t, b)?;
            let (xw, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_w, n_embd, t, b)?;
            let (xk, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_k, n_embd, t, b)?;
            let (xv, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_v, n_embd, t, b)?;
            let (xa, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_a, n_embd, t, b)?;
            let (xg, _) = hip_channel_mix_state(&x_ln1, att_shift_state, &x_g, n_embd, t, b)?;

            // Update shift state after all shifts are computed
            state.att_shift_states[layer_idx] = new_att_shift;

            // Linear projections: r, k, v
            let w_r = layer.att.w_r.to_vec(&stream)?;
            let w_k = layer.att.w_k.to_vec(&stream)?;
            let w_v = layer.att.w_v.to_vec(&stream)?;
            let r = hip_sgemm(&w_r, &xr, n_embd, n_embd, t * b)?;
            let k = hip_sgemm(&w_k, &xk, n_embd, n_embd, t * b)?;
            let mut v = hip_sgemm(&w_v, &xv, n_embd, n_embd, t * b)?;

            // w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
            let w0 = layer.att.w0.to_vec(&stream)?;
            let w1 = layer.att.w1.to_vec(&stream)?;
            let w2 = layer.att.w2.to_vec(&stream)?;
            // Column-major weights: dim(0) is output features (LoRA dim for w1)
            let w1_dim = layer.att.w1.shape().dim(0);
            let w_lora1 = hip_sgemm(&w1, &xw, w1_dim, n_embd, t * b)?;
            let w_lora1_tanh = hip_tanh(&w_lora1)?;
            let w_lora2 = hip_sgemm(&w2, &w_lora1_tanh, n_embd, w1_dim, t * b)?;
            let mut w: Vec<f32> = w0.iter().cycle().take(w_lora2.len())
                .zip(w_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            w = hip_softplus_decay(&w)?;

            // a = sigmoid(a0 + (xa @ a1) @ a2)
            let a0 = layer.att.a0.to_vec(&stream)?;
            let a1 = layer.att.a1.to_vec(&stream)?;
            let a2 = layer.att.a2.to_vec(&stream)?;
            let a1_dim = layer.att.a1.shape().dim(0);
            let a_lora1 = hip_sgemm(&a1, &xa, a1_dim, n_embd, t * b)?;
            let a_lora2 = hip_sgemm(&a2, &a_lora1, n_embd, a1_dim, t * b)?;
            let a_biased: Vec<f32> = a0.iter().cycle().take(a_lora2.len())
                .zip(a_lora2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let a = hip_sigmoid(&a_biased)?;

            // g = sigmoid(xg @ g1) @ g2
            let g1 = layer.att.g1.to_vec(&stream)?;
            let g2 = layer.att.g2.to_vec(&stream)?;
            let g1_dim = layer.att.g1.shape().dim(0);
            let g_lora1 = hip_sgemm(&g1, &xg, g1_dim, n_embd, t * b)?;
            let g_lora1_sigmoid = hip_sigmoid(&g_lora1)?;
            let g = hip_sgemm(&g2, &g_lora1_sigmoid, n_embd, g1_dim, t * b)?;

            // Value residual (layers > 0)
            if layer_idx > 0 {
                if let (Some(v0), Some(v1), Some(v2), Some(ref vf)) =
                    (&layer.att.v0, &layer.att.v1, &layer.att.v2, &v_first) {
                    let v0_data = v0.to_vec(&stream)?;
                    let v1_data = v1.to_vec(&stream)?;
                    let v2_data = v2.to_vec(&stream)?;
                    let v1_dim = v1.shape().dim(0);
                    let v_lora1 = hip_sgemm(&v1_data, &v, v1_dim, n_embd, t * b)?;
                    let v_lora2 = hip_sgemm(&v2_data, &v_lora1, n_embd, v1_dim, t * b)?;
                    let v_biased: Vec<f32> = v0_data.iter().cycle().take(v_lora2.len())
                        .zip(v_lora2.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    let v_residual = hip_sigmoid(&v_biased)?;
                    // v = v + (v_first - v) * v_residual
                    v = v.iter().zip(vf.iter()).zip(v_residual.iter())
                        .map(|((&vi, &vfi), &vri)| vi + (vfi - vi) * vri)
                        .collect();
                }
            } else {
                v_first = Some(v.clone());
            }

            // L2 normalize k
            let k_k = layer.att.k_k.to_vec(&stream)?;
            let k_scaled: Vec<f32> = k.iter().zip(k_k.iter().cycle())
                .map(|(&ki, &kki)| ki * kki)
                .collect();
            let kk = hip_l2_norm(&k_scaled, n_embd, t * b, head_size, 1e-12)?;

            // Control K: k = k * (1 + (a - 1) * k_a)
            let k_a = layer.att.k_a.to_vec(&stream)?;
            let k_ctrl = hip_control_k(&k_a, &a, &k, n_embd, t, b)?;

            // WKV inputs
            let wkv_a: Vec<f32> = kk.iter().map(|&x| -x).collect();
            let wkv_b: Vec<f32> = kk.iter().zip(a.iter()).map(|(&kki, &ai)| kki * ai).collect();
            let w_decay: Vec<f32> = w.iter().map(|&wi| wi.exp()).collect();

            // Run WKV7
            let (wkv_output, new_att_state) = hip_wkv7(
                &w_decay, &r, &k_ctrl, &v, &wkv_a, &wkv_b,
                &state.att_states[layer_idx], head_size, n_head, t, b
            )?;
            state.att_states[layer_idx] = new_att_state;

            // WKV bonus
            let r_k = layer.att.r_k.to_vec(&stream)?;
            let wkv_bonus = hip_wkv_bonus(&r, &k_ctrl, &v, &r_k, head_size, n_head, t, b)?;

            // Combine WKV output and bonus
            let x_att: Vec<f32> = wkv_output.iter().zip(wkv_bonus.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // Group norm
            let gn_w = layer.att.gn.weight.to_vec(&stream)?;
            let gn_b = layer.att.gn.bias.to_vec(&stream)?;
            let x_att_gn = hip_group_norm(&x_att, &gn_w, &gn_b, n_embd, t * b, n_head, 64e-5)?;

            // Gate and output projection
            let x_att_gated: Vec<f32> = x_att_gn.iter().zip(g.iter())
                .map(|(&xi, &gi)| xi * gi)
                .collect();
            let w_o = layer.att.w_o.to_vec(&stream)?;
            let x_att_out = hip_sgemm(&w_o, &x_att_gated, n_embd, n_embd, t * b)?;

            // Residual
            x = x.iter().zip(x_att_out.iter()).map(|(&a, &b)| a + b).collect();

            // ==== Channel-Mix (FFN) ====
            let ln2_w = layer.ffn_ln.weight.to_vec(&stream)?;
            let ln2_b = layer.ffn_ln.bias.to_vec(&stream)?;
            let x_ln2 = hip_layer_norm(&x, &ln2_w, &ln2_b, n_embd, t * b, 1e-5)?;

            // Token shift for FFN
            let ffn_x_k = layer.ffn.x_k.to_vec(&stream)?;
            let (xk_ffn, new_ffn_state) = hip_channel_mix_state(&x_ln2, &state.ffn_states[layer_idx], &ffn_x_k, n_embd, t, b)?;
            state.ffn_states[layer_idx] = new_ffn_state;

            // Key projection + squared ReLU
            let ffn_w_k = layer.ffn.w_k.to_vec(&stream)?;
            let k_ffn = hip_sgemm(&ffn_w_k, &xk_ffn, n_hidden, n_embd, t * b)?;
            let k_sq = hip_squared_relu(&k_ffn)?;

            // Value projection
            let ffn_w_v = layer.ffn.w_v.to_vec(&stream)?;
            let x_ffn_out = hip_sgemm(&ffn_w_v, &k_sq, n_embd, n_hidden, t * b)?;

            // Residual
            x = x.iter().zip(x_ffn_out.iter()).map(|(&a, &b)| a + b).collect();
        }

        // ==== Output Head ====
        let ln_out_w = self.head.ln.weight.to_vec(&stream)?;
        let ln_out_b = self.head.ln.bias.to_vec(&stream)?;
        let x_ln_out = hip_layer_norm(&x, &ln_out_w, &ln_out_b, n_embd, t * b, 1e-5)?;

        // Head projection
        let head_w = self.head.w.to_vec(&stream)?;
        let logits = hip_sgemm(&head_w, &x_ln_out, n_vocab, n_embd, t * b)?;

        Ok(logits)
    }
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

    // === Acceptance Criteria Tests for bd-2sh.3.3 (Tensor Abstraction) ===

    #[test]
    fn test_tensor_alloc() {
        // Test tensor allocation with various shapes and memory types
        let stream = Stream::null();

        // Test device memory allocation
        let shape = TensorShape::new(2, 3, 4, 1);
        let tensor = TensorHip::<f32>::new(shape).expect("Failed to allocate device tensor");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.len(), 24);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.memory_type(), MemoryType::Device);
        println!("Device tensor allocated: shape={}, len={}", tensor.shape(), tensor.len());

        // Test managed memory allocation
        let managed_tensor = TensorHip::<f32>::managed(shape).expect("Failed to allocate managed tensor");
        assert_eq!(managed_tensor.shape(), shape);
        assert_eq!(managed_tensor.memory_type(), MemoryType::Managed);
        println!("Managed tensor allocated: shape={}", managed_tensor.shape());

        // Test zeros initialization
        let zeros_tensor = TensorHip::<f32>::zeros(shape).expect("Failed to allocate zeros tensor");
        let data = zeros_tensor.to_vec(&stream).expect("Failed to read zeros");
        assert!(data.iter().all(|&x| x == 0.0), "Zeros tensor should be all zeros");
        println!("Zeros tensor verified: all {} elements are zero", data.len());

        // Test empty tensor
        let empty_shape = TensorShape::new(0, 1, 1, 1);
        let empty_tensor = TensorHip::<f32>::new(empty_shape).expect("Failed to allocate empty tensor");
        assert!(empty_tensor.is_empty());
        println!("Empty tensor allocated successfully");
    }

    #[test]
    fn test_tensor_copy_roundtrip() {
        // Test copying data to device and back
        let stream = Stream::null();

        // Create test data
        let shape = TensorShape::new(4, 3, 2, 1);
        let host_data: Vec<f32> = (0..24).map(|i| i as f32).collect();

        // Upload to device
        let tensor = TensorHip::from_slice(&host_data, shape, &stream)
            .expect("Failed to create tensor from slice");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.len(), 24);

        // Download back to host
        let result = tensor.to_vec(&stream).expect("Failed to copy tensor to host");
        assert_eq!(host_data, result, "Roundtrip data should match");
        println!("Tensor roundtrip verified: {} elements match", result.len());

        // Test with copy_from_slice
        let mut tensor2 = TensorHip::<f32>::new(shape).expect("Failed to allocate tensor");
        tensor2.copy_from_slice(&host_data, &stream).expect("Failed to copy from slice");
        stream.synchronize().expect("Failed to sync");

        let result2 = tensor2.to_vec(&stream).expect("Failed to read tensor2");
        assert_eq!(host_data, result2, "copy_from_slice data should match");
        println!("copy_from_slice roundtrip verified");
    }

    #[test]
    fn test_tensor_view_strides() {
        // Test tensor views and stride calculations
        let stream = Stream::null();

        // Create a 4x3x2 tensor with known data
        // shape[0]=4 is fastest (contiguous)
        let shape = TensorShape::new(4, 3, 2, 1);
        let host_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = TensorHip::from_slice(&host_data, shape, &stream)
            .expect("Failed to create tensor");

        // Verify strides: [1, 4, 12, 24]
        let strides = tensor.strides();
        assert_eq!(strides, [1, 4, 12, 24], "Strides should be [1, 4, 12, 24]");
        println!("Verified strides: {:?}", strides);

        // Test linear index calculation
        // Element at (x=1, y=2, z=1, w=0) should be at index: 1 + 2*4 + 1*12 = 21
        let idx = shape.linear_index(1, 2, 1, 0);
        assert_eq!(idx, 21, "Linear index (1,2,1,0) should be 21");
        println!("Linear index (1,2,1,0) = {}", idx);

        // Create a view: first row of each "page" (x=0..4, y=0..1, z=0..2)
        let view = tensor.view((0, 4), (0, 1), (0, 2), (0, 1))
            .expect("Failed to create view");
        assert_eq!(view.shape(), TensorShape::new(4, 1, 2, 1));
        assert_eq!(view.len(), 8);
        assert!(!view.is_contiguous(), "View with gap in y should not be contiguous");
        println!("View shape: {}, contiguous: {}", view.shape(), view.is_contiguous());

        // Test view strides (should inherit parent strides)
        let view_strides = view.strides();
        assert_eq!(view_strides, strides, "View should inherit parent strides");
        println!("View strides: {:?}", view_strides);

        // Test TensorView offset calculation
        let tv = tensor.tensor_view();
        assert_eq!(tv.offset, 0, "Original tensor offset should be 0");

        // Create view starting at (2, 1, 0, 0)
        let view2 = tensor.view((2, 4), (1, 3), (0, 2), (0, 1))
            .expect("Failed to create view2");
        let expected_offset = 2 * 1 + 1 * 4 + 0 * 12; // = 6
        assert_eq!(view2.offset(), expected_offset, "View offset should be 6");
        println!("View2 offset: {} (expected {})", view2.offset(), expected_offset);

        // Verify view shape
        assert_eq!(view2.shape(), TensorShape::new(2, 2, 2, 1));
        println!("View2 shape: {}", view2.shape());

        // Test contiguous view (full slice should be contiguous)
        let full_view = tensor.view((0, 4), (0, 3), (0, 2), (0, 1))
            .expect("Failed to create full view");
        assert!(full_view.is_contiguous(), "Full view should be contiguous");
        println!("Full view is contiguous: {}", full_view.is_contiguous());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.3 (Decay Exponential Kernel) ===

    #[test]
    fn test_decay_exp() {
        // Test basic decay_exp functionality: out = exp(-exp(x))
        // Also serves as the primary acceptance test when fixtures are loaded

        // Test known values
        let input = vec![
            0.0,   // exp(-exp(0)) = exp(-1) ≈ 0.3679
            -1.0,  // exp(-exp(-1)) = exp(-0.3679) ≈ 0.6922
            1.0,   // exp(-exp(1)) = exp(-2.718) ≈ 0.0660
            -5.0,  // exp(-exp(-5)) ≈ exp(-0.0067) ≈ 0.9933
            5.0,   // exp(-exp(5)) ≈ exp(-148.4) ≈ 0
        ];

        let output = hip_decay_exp(&input).expect("decay_exp kernel failed");

        // Expected values (computed with Python: np.exp(-np.exp(x)))
        let expected = vec![
            0.36787944,  // exp(-1)
            0.69220066,  // exp(-exp(-1))
            0.06598804,  // exp(-exp(1))
            0.99330715,  // exp(-exp(-5))
            0.0,         // exp(-exp(5)) ≈ 0 (underflow)
        ];

        // Check each value with tolerance
        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-3 + 1e-3 * exp.abs(); // rtol=1e-3, atol=1e-3
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic decay_exp test passed: {} values verified", output.len());
    }

    #[test]
    fn test_decay_exp_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -50.0,  // Very negative: exp(-exp(-50)) ≈ 1
            -10.0,  // Negative: exp(-exp(-10)) ≈ 1
            -5.0,   // Moderate negative
            -1.0,   // Small negative
            0.0,    // Zero
            1.0,    // Small positive
            5.0,    // Moderate positive: exp(-exp(5)) ≈ 0
            10.0,   // exp(-exp(10)) ≈ 0 (extreme underflow)
            80.0,   // At clamping boundary
            100.0,  // Beyond clamping: should be 0, not NaN/Inf
        ];

        let output = hip_decay_exp(&input).expect("decay_exp stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i, val, input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(output[0] > 0.999, "exp(-exp(-50)) should be ≈1, got {}", output[0]);
        assert!(output[1] > 0.999, "exp(-exp(-10)) should be ≈1, got {}", output[1]);
        assert!(output[7] < 0.001, "exp(-exp(10)) should be ≈0, got {}", output[7]);
        assert!(output[8] < 0.001, "exp(-exp(80)) should be ≈0, got {}", output[8]);
        assert_eq!(output[9], 0.0, "exp(-exp(100)) should be exactly 0");

        println!("Numerical stability test passed: all {} values are finite and in [0,1]", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.4 (Lerp Kernel) ===

    #[test]
    fn test_lerp() {
        // Test basic lerp functionality: out = a + t * (b - a)
        let a = vec![0.0, 1.0, 2.0, 10.0, -5.0];
        let b = vec![10.0, 5.0, 2.0, 0.0, 5.0];
        let t = vec![0.0, 0.5, 1.0, 0.25, 0.5];

        let output = hip_lerp(&a, &b, &t).expect("lerp kernel failed");

        // Expected: lerp(a, b, t) = a + t * (b - a)
        // [0] lerp(0, 10, 0) = 0
        // [1] lerp(1, 5, 0.5) = 1 + 0.5 * 4 = 3
        // [2] lerp(2, 2, 1) = 2
        // [3] lerp(10, 0, 0.25) = 10 + 0.25 * (-10) = 7.5
        // [4] lerp(-5, 5, 0.5) = -5 + 0.5 * 10 = 0
        let expected = vec![0.0, 3.0, 2.0, 7.5, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic lerp test passed: {} values verified", output.len());
    }

    #[test]
    fn test_lerp_edge_cases() {
        // Test edge cases: t outside [0, 1] (extrapolation)
        let a = vec![0.0, 0.0, 100.0];
        let b = vec![10.0, 10.0, 0.0];
        let t = vec![-0.5, 1.5, 2.0];

        let output = hip_lerp(&a, &b, &t).expect("lerp edge case test failed");

        // Expected with extrapolation:
        // [0] lerp(0, 10, -0.5) = 0 + (-0.5) * 10 = -5
        // [1] lerp(0, 10, 1.5) = 0 + 1.5 * 10 = 15
        // [2] lerp(100, 0, 2.0) = 100 + 2.0 * (-100) = -100
        let expected = vec![-5.0, 15.0, -100.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Lerp edge case test passed: extrapolation works correctly");
    }

    // === Acceptance Criteria Tests for bd-2sh.4.1 (Sigmoid Kernel) ===

    #[test]
    fn test_sigmoid() {
        // Test basic sigmoid functionality: out = 1 / (1 + exp(-x))
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_sigmoid(&input).expect("sigmoid kernel failed");

        // Expected: sigmoid(x) = 1 / (1 + exp(-x))
        // sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.7311
        // sigmoid(-1) ≈ 0.2689
        // sigmoid(2) ≈ 0.8808
        // sigmoid(-2) ≈ 0.1192
        let expected = vec![0.5, 0.7310586, 0.26894143, 0.880797, 0.11920292];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Basic sigmoid test passed: {} values verified", output.len());
    }

    #[test]
    fn test_sigmoid_edge_cases() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0,  // Very negative: sigmoid → 0
            -50.0,   // Large negative
            -10.0,   // Moderate negative
            0.0,     // Zero: sigmoid = 0.5
            10.0,    // Moderate positive
            50.0,    // Large positive
            100.0,   // Very positive: sigmoid → 1
        ];

        let output = hip_sigmoid(&input).expect("sigmoid edge case test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i, val, input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(output[0] < 1e-10, "sigmoid(-100) should be ≈0, got {}", output[0]);
        assert!(output[1] < 1e-10, "sigmoid(-50) should be ≈0, got {}", output[1]);
        assert!((output[3] - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {}", output[3]);
        assert!(output[5] >= 1.0 - 1e-10, "sigmoid(50) should be ≈1, got {}", output[5]);
        assert!(output[6] >= 1.0 - 1e-10, "sigmoid(100) should be ≈1, got {}", output[6]);

        println!("Sigmoid edge case test passed: all {} values are finite and in [0,1]", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.2 (Squared ReLU Kernel) ===

    #[test]
    fn test_squared_relu() {
        // Test basic squared ReLU: out = max(0, x)^2
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

        let output = hip_squared_relu(&input).expect("squared_relu kernel failed");

        // Expected: max(0, x)^2
        // [-2] -> 0^2 = 0
        // [-1] -> 0^2 = 0
        // [0]  -> 0^2 = 0
        // [1]  -> 1^2 = 1
        // [2]  -> 2^2 = 4
        // [3]  -> 3^2 = 9
        let expected = vec![0.0, 0.0, 0.0, 1.0, 4.0, 9.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Squared ReLU test passed: {} values verified", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.14 (Softplus Decay Kernel) ===

    #[test]
    fn test_softplus_decay() {
        // Test softplus decay: out = log(sigmoid(x)) - 0.5
        let input = vec![0.0, 1.0, -1.0, 5.0, -5.0];

        let output = hip_softplus_decay(&input).expect("softplus_decay kernel failed");

        // Expected: log(sigmoid(x)) - 0.5
        // log(sigmoid(0)) - 0.5 = log(0.5) - 0.5 ≈ -0.693 - 0.5 = -1.193
        // log(sigmoid(1)) - 0.5 ≈ -0.313 - 0.5 = -0.813
        // log(sigmoid(-1)) - 0.5 ≈ -1.313 - 0.5 = -1.813
        // log(sigmoid(5)) - 0.5 ≈ -0.0067 - 0.5 ≈ -0.507
        // log(sigmoid(-5)) - 0.5 ≈ -5.0067 - 0.5 ≈ -5.507
        let expected: Vec<f32> = input.iter().map(|&x| {
            let log_sigmoid = -(1.0f32 + (-x).exp()).ln();
            log_sigmoid - 0.5
        }).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, actual, exp, diff
            );
        }
        println!("Softplus decay test passed: {} values verified", output.len());
    }

    #[test]
    fn test_softplus_decay_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0,  // Very negative: result ≈ x - 0.5 = -100.5
            -50.0,   // Large negative
            -20.0,   // At clamping boundary
            0.0,     // Zero
            20.0,    // At clamping boundary
            50.0,    // Large positive
            100.0,   // Very positive: result ≈ -0.5
        ];

        let output = hip_softplus_decay(&input).expect("softplus_decay stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (input={})", i, input[i]
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})", i, input[i]
            );
        }

        // Verify expected behavior at extremes
        // For large negative x: result ≈ x - 0.5
        assert!((output[0] - (-100.5)).abs() < 0.1, "softplus_decay(-100) should be ≈-100.5, got {}", output[0]);
        // For large positive x: result ≈ -0.5
        assert!((output[6] - (-0.5)).abs() < 0.01, "softplus_decay(100) should be ≈-0.5, got {}", output[6]);

        println!("Softplus decay stability test passed: all {} values are finite", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.5 (Layer Normalization Kernel) ===

    #[test]
    fn test_layer_norm_basic() {
        // Test basic layer normalization with a simple 2-vector case
        // Input: 2 vectors of length 4
        // Shape: [4, 2, 1, 1] where 4 is the channel dimension (fastest axis)

        // Vector 0: [1.0, 2.0, 3.0, 4.0] -> mean=2.5, var=1.25
        // Vector 1: [0.0, 4.0, 2.0, 6.0] -> mean=3.0, var=5.0
        let input = vec![
            1.0, 2.0, 3.0, 4.0,  // vector 0 (elements at offsets 0-3)
            0.0, 4.0, 2.0, 6.0,  // vector 1 (elements at offsets 4-7)
        ];

        // Weight = 1.0 (no scaling)
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        // Bias = 0.0 (no offset)
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let c = 4; // Channel dimension
        let n = 2; // Number of vectors
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm kernel failed");

        // Expected for vector 0: (x - 2.5) / sqrt(1.25 + eps)
        // std0 = sqrt(1.25) ≈ 1.118034
        // normalized: [-1.342, -0.447, 0.447, 1.342]
        let mean0 = 2.5f32;
        let std0 = (1.25f32 + eps).sqrt();

        // Expected for vector 1: (x - 3.0) / sqrt(5.0 + eps)
        // std1 = sqrt(5.0) ≈ 2.236
        // normalized: [-1.342, 0.447, -0.447, 1.342]
        let mean1 = 3.0f32;
        let std1 = (5.0f32 + eps).sqrt();

        let expected = vec![
            (1.0 - mean0) / std0, (2.0 - mean0) / std0, (3.0 - mean0) / std0, (4.0 - mean0) / std0,
            (0.0 - mean1) / std1, (4.0 - mean1) / std1, (2.0 - mean1) / std1, (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic layer_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_layer_norm_with_affine() {
        // Test layer normalization with weight and bias
        // Input: 1 vector of length 4
        // Shape: [4, 1, 1, 1]

        // Vector: [0.0, 2.0, 4.0, 6.0] -> mean=3.0, var=5.0
        let input = vec![0.0, 2.0, 4.0, 6.0];

        // Weight = [2.0, 1.0, 0.5, 0.0]
        let weight = vec![2.0, 1.0, 0.5, 0.0];
        // Bias = [1.0, 0.0, -1.0, 5.0]
        let bias = vec![1.0, 0.0, -1.0, 5.0];

        let c = 4;
        let n = 1;
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm with affine failed");

        // normalized = (x - 3.0) / sqrt(5.0 + eps)
        // then: output = normalized * weight + bias
        let mean = 3.0f32;
        let std = (5.0f32 + eps).sqrt();

        let expected: Vec<f32> = (0..4).map(|i| {
            let x = input[i];
            let normalized = (x - mean) / std;
            normalized * weight[i] + bias[i]
        }).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Layer norm with affine test passed: {} values verified", output.len());
    }

    #[test]
    fn test_layer_norm_numerical_stability() {
        // Test with values that could cause numerical issues
        // Very small variance (constant input + small noise)
        let c = 128;
        let n = 4;

        let mut input = vec![0.0f32; c * n];
        // Fill with near-constant values
        for i in 0..n {
            for j in 0..c {
                // Small variation around 1000.0
                input[i * c + j] = 1000.0 + (j as f32) * 0.001;
            }
        }

        let weight = vec![1.0f32; c];
        let bias = vec![0.0f32; c];
        let eps = 1e-5;

        let output = hip_layer_norm(&input, &weight, &bias, c, n, eps)
            .expect("layer_norm stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(
                !val.is_nan(),
                "NaN at index {} (vec={}, ch={})", i, i / c, i % c
            );
            assert!(
                !val.is_infinite(),
                "Inf at index {} (vec={}, ch={})", i, i / c, i % c
            );
        }

        // Normalized values should have mean ≈ 0 and std ≈ 1 for each vector
        for vec_idx in 0..n {
            let start = vec_idx * c;
            let end = start + c;
            let vec_output = &output[start..end];

            let sum: f32 = vec_output.iter().sum();
            let mean = sum / (c as f32);

            // Mean should be close to 0 (within tolerance due to FP32 accumulation)
            // With 128 elements and FP32 arithmetic, numerical errors can accumulate
            assert!(
                mean.abs() < 1e-3,
                "Vector {} mean not close to 0: {}", vec_idx, mean
            );
        }

        println!("Layer norm stability test passed: {} values are finite with correct mean", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.6 (Group Normalization Kernel) ===

    #[test]
    fn test_group_norm_basic() {
        // Test group normalization with 2 groups on 1 vector
        // Input: 1 vector of length 8, divided into 2 groups of 4
        // Shape: [8, 1, 1, 1]
        let input = vec![
            // Group 0: [1, 2, 3, 4] -> mean=2.5, var=1.25
            1.0, 2.0, 3.0, 4.0,
            // Group 1: [0, 2, 4, 6] -> mean=3.0, var=5.0
            0.0, 2.0, 4.0, 6.0,
        ];

        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];

        let c = 8;
        let n = 1;
        let num_groups = 2;
        let eps = 1e-5;

        let output = hip_group_norm(&input, &weight, &bias, c, n, num_groups, eps)
            .expect("group_norm kernel failed");

        // Expected: normalize each group independently
        let mean0 = 2.5f32;
        let std0 = (1.25f32 + eps).sqrt();
        let mean1 = 3.0f32;
        let std1 = (5.0f32 + eps).sqrt();

        let expected = vec![
            (1.0 - mean0) / std0, (2.0 - mean0) / std0, (3.0 - mean0) / std0, (4.0 - mean0) / std0,
            (0.0 - mean1) / std1, (2.0 - mean1) / std1, (4.0 - mean1) / std1, (6.0 - mean1) / std1,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic group_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_group_norm_multiple_vectors() {
        // Test group normalization on multiple vectors
        // 2 vectors of length 4, 2 groups of 2 channels each
        let input = vec![
            // Vector 0: groups [1, 3], [2, 4]
            1.0, 3.0, 2.0, 4.0,
            // Vector 1: groups [0, 2], [1, 3]
            0.0, 2.0, 1.0, 3.0,
        ];

        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];

        let c = 4;
        let n = 2;
        let num_groups = 2;
        let eps = 1e-5;

        let output = hip_group_norm(&input, &weight, &bias, c, n, num_groups, eps)
            .expect("group_norm multiple vectors failed");

        // Each group should have mean ≈ 0 after normalization
        assert_eq!(output.len(), 8);
        println!("Group norm multiple vectors test passed: {} values", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.7 (L2 Normalization Kernel) ===

    #[test]
    fn test_l2_norm_basic() {
        // Test L2 normalization on a simple case
        // 1 vector of length 4, with head_size=4 (1 head)
        let input = vec![3.0, 0.0, 4.0, 0.0];  // L2 norm = 5

        let c = 4;
        let n = 1;
        let head_size = 4;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps)
            .expect("l2_norm kernel failed");

        // Expected: input / 5 = [0.6, 0.0, 0.8, 0.0]
        let expected = vec![0.6, 0.0, 0.8, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic l2_norm test passed: {} values verified", output.len());
    }

    #[test]
    fn test_l2_norm_per_head() {
        // Test L2 normalization with multiple heads
        // 1 vector of length 4, with head_size=2 (2 heads)
        let input = vec![
            3.0, 4.0,  // Head 0: norm = 5
            5.0, 12.0, // Head 1: norm = 13
        ];

        let c = 4;
        let n = 1;
        let head_size = 2;
        let eps = 1e-12;

        let output = hip_l2_norm(&input, c, n, head_size, eps)
            .expect("l2_norm per head failed");

        // Expected: normalize each head independently
        let expected = vec![
            3.0/5.0, 4.0/5.0,
            5.0/13.0, 12.0/13.0,
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("L2 norm per head test passed: {} values verified", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.13 (Tanh Kernel) ===

    #[test]
    fn test_tanh_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_tanh(&input).expect("tanh kernel failed");

        // Expected: tanh(x)
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i, actual, exp, diff
            );
        }
        println!("Basic tanh test passed: {} values verified", output.len());
    }

    #[test]
    fn test_tanh_edge_cases() {
        let input = vec![
            -100.0, // Very negative: tanh → -1
            -10.0,
            0.0,    // tanh(0) = 0
            10.0,
            100.0,  // Very positive: tanh → 1
        ];

        let output = hip_tanh(&input).expect("tanh edge cases failed");

        // Verify no NaN or Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {}", i);
            assert!(!val.is_infinite(), "Inf at index {}", i);
            assert!(val >= -1.0 && val <= 1.0, "Value out of [-1,1] at index {}: {}", i, val);
        }

        // Check extremes
        assert!((output[0] - (-1.0)).abs() < 1e-6, "tanh(-100) should be ≈-1");
        assert!(output[2].abs() < 1e-6, "tanh(0) should be ≈0");
        assert!((output[4] - 1.0).abs() < 1e-6, "tanh(100) should be ≈1");

        println!("Tanh edge cases test passed");
    }

    // === Acceptance Criteria Tests for bd-2sh.4.9 (Token Shift Kernel) ===

    #[test]
    fn test_token_shift_basic() {
        // Test token shift with 2 channels and 3 tokens
        // x shape: [2, 3, 1, 1]
        let x = vec![
            // Token 0: [1.0, 2.0]
            1.0, 2.0,
            // Token 1: [3.0, 4.0]
            3.0, 4.0,
            // Token 2: [5.0, 6.0]
            5.0, 6.0,
        ];

        // Initial state: [0.0, 0.0]
        let state_in = vec![0.0, 0.0];

        // Mix factor: 0.5 (blend 50% of previous into current)
        let mix = vec![0.5, 0.5];

        let c = 2;
        let t = 3;

        let (output, state_out) = hip_token_shift(&x, &state_in, &mix, c, t)
            .expect("token_shift kernel failed");

        // Formula: output[t] = x[t] + mix * (prev - x[t])
        // Token 0: x[0] + 0.5*(state - x[0]) = 1 + 0.5*(0-1) = [0.5, 1.0]
        // Token 1: x[1] + 0.5*(x[0] - x[1]) = 3 + 0.5*(1-3) = [2.0, 3.0]
        // Token 2: x[2] + 0.5*(x[1] - x[2]) = 5 + 0.5*(3-5) = [4.0, 5.0]
        let expected = vec![
            0.5, 1.0,  // Token 0
            2.0, 3.0,  // Token 1
            4.0, 5.0,  // Token 2
        ];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Output mismatch at index {}: actual={:.6}, expected={:.6}",
                i, actual, exp
            );
        }

        // State out should be x[last] = [5.0, 6.0]
        assert!((state_out[0] - 5.0).abs() < 1e-5, "state_out[0] should be 5.0");
        assert!((state_out[1] - 6.0).abs() < 1e-5, "state_out[1] should be 6.0");

        println!("Token shift basic test passed");
    }

    #[test]
    fn test_token_shift_no_mix() {
        // Test with mix=0 (pass through current, no blending)
        let x = vec![1.0, 2.0, 3.0, 4.0];  // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![0.0, 0.0];

        let (output, _) = hip_token_shift(&x, &state_in, &mix, 2, 2)
            .expect("token_shift no mix failed");

        // With mix=0: output = x + 0*(prev - x) = x
        // So output equals x directly
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}", i, actual, exp
            );
        }
        println!("Token shift no mix test passed");
    }

    #[test]
    fn test_token_shift_full_mix() {
        // Test with mix=1 (full blending with previous)
        let x = vec![1.0, 2.0, 3.0, 4.0];  // [2, 2]
        let state_in = vec![10.0, 20.0];
        let mix = vec![1.0, 1.0];

        let (output, _) = hip_token_shift(&x, &state_in, &mix, 2, 2)
            .expect("token_shift full mix failed");

        // With mix=1: output = x + 1*(prev - x) = prev
        // Token 0: prev = state_in = [10.0, 20.0]
        // Token 1: prev = x[0] = [1.0, 2.0]
        let expected = vec![10.0, 20.0, 1.0, 2.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Mismatch at {}: {} vs {}", i, actual, exp
            );
        }
        println!("Token shift full mix test passed");
    }

    // === State I/O and Batched Inference Tests (bd-2sh.5.6) ===

    /// Test that HipState is correctly sized for batched inference.
    #[test]
    fn test_hip_state_batched_sizing() {
        let info = Rwkv7ModelInfo {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            head_size: 64,
            n_vocab: 65536,
            n_hidden: 2048,
        };

        let batch_size = 4;
        let state = HipState::new(&info, batch_size);

        assert_eq!(state.batch_size, 4);
        assert_eq!(state.att_states.len(), 12);
        assert_eq!(state.att_shift_states.len(), 12);
        assert_eq!(state.ffn_states.len(), 12);

        // Check per-layer sizes include batch dimension
        let expected_att_state_size = 64 * 64 * 12 * 4; // head_size² * n_head * batch
        let expected_shift_state_size = 768 * 4; // n_embd * batch

        assert_eq!(state.att_states[0].len(), expected_att_state_size);
        assert_eq!(state.att_shift_states[0].len(), expected_shift_state_size);
        assert_eq!(state.ffn_states[0].len(), expected_shift_state_size);

        println!("HipState batched sizing test passed (B=4)");
    }

    /// Test that forward() convenience wrapper works.
    #[test]
    fn test_forward_convenience_wrapper() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
        let tokens = vec![1u32, 2, 3, 4, 5];

        let logits = model.forward(&tokens).expect("forward() failed");

        // Should return vocab_size * T logits
        let expected_len = model.info.n_vocab * tokens.len();
        assert_eq!(logits.len(), expected_len,
            "Expected {} logits, got {}", expected_len, logits.len());

        println!("forward() convenience wrapper test passed");
    }

    /// Test batched inference with B=2 produces same results as sequential B=1.
    #[test]
    fn test_batched_matches_sequential() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // Two sequences
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];

        // Run sequentially with B=1
        let logits1 = model.forward(&seq1).expect("seq1 forward failed");
        let logits2 = model.forward(&seq2).expect("seq2 forward failed");

        // Run batched with B=2
        let mut state = HipState::new(&model.info, 2);
        let batched_logits = model.forward_with_state(&[&seq1, &seq2], &mut state)
            .expect("batched forward failed");

        // Batched output should have shape [vocab_size, T, B]
        let vocab = model.info.n_vocab;
        let t = 3;
        let b = 2;
        assert_eq!(batched_logits.len(), vocab * t * b);

        // Compare: batched logits should match sequential
        // Layout: batched[v, t, b] = batched[b * T * V + t * V + v]
        let mut max_diff = 0.0f32;
        for ti in 0..t {
            for vi in 0..vocab {
                // Sequential: logits1[v + t*V], logits2[v + t*V]
                let seq1_val = logits1[vi + ti * vocab];
                let seq2_val = logits2[vi + ti * vocab];

                // Batched: logits[v, t, 0] and logits[v, t, 1]
                let batch1_val = batched_logits[0 * t * vocab + ti * vocab + vi];
                let batch2_val = batched_logits[1 * t * vocab + ti * vocab + vi];

                let diff1 = (seq1_val - batch1_val).abs();
                let diff2 = (seq2_val - batch2_val).abs();
                max_diff = max_diff.max(diff1).max(diff2);

                // Allow small tolerance for numerical differences
                assert!(diff1 < 1e-4, "seq1 mismatch at t={}, v={}: {} vs {}", ti, vi, seq1_val, batch1_val);
                assert!(diff2 < 1e-4, "seq2 mismatch at t={}, v={}: {} vs {}", ti, vi, seq2_val, batch2_val);
            }
        }

        println!("Batched matches sequential test PASSED (max_diff={})", max_diff);
    }

    /// Test streaming equivalence with batched state.
    #[test]
    fn test_batched_streaming_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        let tokens: Vec<u32> = vec![1, 2, 3];

        // Single-sequence batch forward
        let mut state1 = HipState::new(&model.info, 1);
        let logits_batch = model.forward_with_state(&[&tokens], &mut state1)
            .expect("batch forward failed");

        // Single-sequence streaming (token by token)
        let mut state2 = HipState::new(&model.info, 1);
        let mut logits_stream = Vec::new();
        for &tok in &tokens {
            let logits = model.forward_with_state(&[&[tok]], &mut state2)
                .expect("streaming forward failed");
            logits_stream.extend(logits);
        }

        assert_eq!(logits_batch.len(), logits_stream.len());

        let mut max_diff = 0.0f32;
        for (i, (batch, stream)) in logits_batch.iter().zip(logits_stream.iter()).enumerate() {
            let diff = (batch - stream).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 1e-3, "Streaming mismatch at {}: {} vs {}", i, batch, stream);
        }

        println!("Batched streaming equivalence test PASSED (max_diff={})", max_diff);
    }

    /// Test chunked processing with batched state.
    #[test]
    fn test_batched_chunked_equivalence() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // 10 tokens, chunk into [4, 4, 2]
        let tokens: Vec<u32> = (1..=10).collect();
        let chunk_size = 4;

        // Full forward
        let mut state1 = HipState::new(&model.info, 1);
        let logits_full = model.forward_with_state(&[&tokens], &mut state1)
            .expect("full forward failed");

        // Chunked forward
        let mut state2 = HipState::new(&model.info, 1);
        let mut logits_chunked = Vec::new();
        for chunk in tokens.chunks(chunk_size) {
            let logits = model.forward_with_state(&[chunk], &mut state2)
                .expect("chunked forward failed");
            logits_chunked.extend(logits);
        }

        assert_eq!(logits_full.len(), logits_chunked.len());

        let mut max_diff = 0.0f32;
        for (i, (full, chunked)) in logits_full.iter().zip(logits_chunked.iter()).enumerate() {
            let diff = (full - chunked).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 1e-3, "Chunked mismatch at {}: {} vs {}", i, full, chunked);
        }

        println!("Batched chunked equivalence test PASSED (max_diff={})", max_diff);
    }

    /// Test state evolution in batched mode.
    #[test]
    fn test_batched_state_evolution() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6];

        let mut state = HipState::new(&model.info, 2);

        // State starts at zero
        let att_sum_before: f32 = state.att_states.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .sum();
        assert_eq!(att_sum_before, 0.0);

        // Run forward
        let _ = model.forward_with_state(&[&seq1, &seq2], &mut state)
            .expect("forward failed");

        // State should have evolved
        let att_sum_after: f32 = state.att_states.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .sum();
        assert!(att_sum_after > 0.0, "att_states should be non-zero after forward");

        println!("Batched state evolution test PASSED");
    }

    /// Test batch size mismatch error.
    #[test]
    fn test_batch_size_mismatch_error() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // State with batch_size=2, but provide 3 sequences
        let mut state = HipState::new(&model.info, 2);
        let result = model.forward_with_state(
            &[&[1u32], &[2u32], &[3u32]],
            &mut state
        );

        assert!(result.is_err(), "Should error on batch size mismatch");
        println!("Batch size mismatch error test PASSED");
    }

    /// Test sequence length mismatch error.
    #[test]
    fn test_sequence_length_mismatch_error() {
        use std::path::Path;

        let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
        if !Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let model = Rwkv7Hip::load(model_path).expect("Failed to load model");

        // Two sequences with different lengths
        let mut state = HipState::new(&model.info, 2);
        let result = model.forward_with_state(
            &[&[1u32, 2, 3], &[4u32, 5]],  // length 3 vs length 2
            &mut state
        );

        assert!(result.is_err(), "Should error on sequence length mismatch");
        println!("Sequence length mismatch error test PASSED");
    }

    /// Test HipState reset.
    #[test]
    fn test_hip_state_reset() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        let mut state = HipState::new(&info, 2);

        // Fill with values
        for s in &mut state.att_states { s.fill(1.0); }
        for s in &mut state.att_shift_states { s.fill(2.0); }
        for s in &mut state.ffn_states { s.fill(3.0); }

        state.reset();

        // All should be zero
        assert!(state.att_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.att_shift_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.ffn_states.iter().all(|s| s.iter().all(|&x| x == 0.0)));
        assert!(state.v_first.is_none());

        println!("HipState reset test PASSED");
    }

    /// Test HipState v_first persistence across operations.
    #[test]
    fn test_hip_state_v_first_persistence() {
        let info = Rwkv7ModelInfo {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            head_size: 32,
            n_vocab: 100,
            n_hidden: 128,
        };

        // Fresh state should have v_first = None
        let mut state = HipState::new(&info, 1);
        assert!(state.v_first.is_none(), "New state should have v_first = None");

        // Set v_first and verify it persists
        state.v_first = Some(vec![1.0f32; 64]);
        assert!(state.v_first.is_some(), "v_first should persist after assignment");
        assert_eq!(state.v_first.as_ref().unwrap().len(), 64);

        // Reset should clear v_first
        state.reset();
        assert!(state.v_first.is_none(), "Reset should clear v_first to None");

        println!("HipState v_first persistence test PASSED");
    }
}
