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
}
