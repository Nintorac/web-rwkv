//! HIP tensor types with shape and memory management.

use std::ffi::c_void;
use std::ptr;

use super::ffi::{
    HipErrorKind, Result, check,
    hip_malloc, hip_malloc_managed, hip_free, hip_memset,
    hip_memcpy_h2d, hip_memcpy_d2h,
};
use super::device::Stream;
use super::buffer::MemoryType;


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

    /// Copy tensor data to a pre-allocated host buffer asynchronously.
    ///
    /// Unlike `to_vec()`, this method does NOT synchronize the stream before returning.
    /// The caller must ensure synchronization before reading the data, typically by
    /// recording an event after this call and waiting on it:
    ///
    /// ```ignore
    /// tensor.copy_to_slice_async(&mut buffer, stream)?;
    /// let event = Event::new()?;
    /// event.record(stream)?;
    /// // ... do other work ...
    /// event.synchronize()?;  // Now buffer is safe to read
    /// ```
    ///
    /// The tensor must be contiguous and the buffer must have exactly `self.len()` elements.
    pub fn copy_to_slice_async(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "Cannot copy non-contiguous tensor directly".to_string(),
            });
        }

        if dst.len() != self.len() {
            return Err(HipErrorKind {
                code: -1,
                message: format!("Size mismatch: dst {} vs tensor {}", dst.len(), self.len()),
            });
        }

        let len = self.len();
        if len == 0 {
            return Ok(());
        }

        let size = len * std::mem::size_of::<T>();
        unsafe {
            check(hip_memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                stream.handle(),
            ))
        }
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

    /// Copy data from host to this tensor at a specific element offset.
    ///
    /// This allows writing a partial slice to a larger pre-allocated buffer.
    /// The tensor must be contiguous.
    ///
    /// # Arguments
    /// * `data` - Host data to copy
    /// * `offset` - Element offset into the tensor (not bytes)
    /// * `stream` - HIP stream for async copy
    pub fn copy_from_slice_at(&mut self, data: &[T], offset: usize, stream: &Stream) -> Result<()> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "Cannot copy to non-contiguous tensor directly".to_string(),
            });
        }

        if offset + data.len() > self.allocated_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Copy would exceed buffer: offset {} + len {} > allocated {}",
                    offset, data.len(), self.allocated_len
                ),
            });
        }

        if data.is_empty() {
            return Ok(());
        }

        let size = data.len() * std::mem::size_of::<T>();
        let dst = unsafe { self.ptr.add(offset) };
        unsafe {
            check(hip_memcpy_h2d(
                dst as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                stream.handle(),
            ))
        }
    }

    /// Fill the tensor with zeros.
    ///
    /// This sets all bytes of the tensor to zero, which is equivalent to
    /// setting all numeric elements to 0.
    pub fn fill_zero(&mut self) -> Result<()> {
        if self.allocated_len == 0 {
            return Ok(());
        }

        let size = self.allocated_len * std::mem::size_of::<T>();
        unsafe {
            check(hip_memset(self.ptr as *mut c_void, 0, size))
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

    /// Create a contiguous view with a smaller shape.
    ///
    /// This allows reusing pre-allocated buffers with different sequence lengths.
    /// The new shape must fit within the allocated memory.
    ///
    /// # Arguments
    /// * `new_shape` - The new shape, which must have total elements <= allocated_len
    ///
    /// # Returns
    /// A new TensorHip that shares the same memory but has the new shape.
    /// The returned tensor does not own the memory and will not free it on drop.
    pub fn resized_view(&self, new_shape: TensorShape) -> Result<TensorHip<T>> {
        let new_len = new_shape.len();
        if new_len > self.allocated_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "New shape {} (len={}) exceeds allocated size {}",
                    new_shape, new_len, self.allocated_len
                ),
            });
        }

        Ok(TensorHip {
            ptr: self.ptr,
            allocated_len: self.allocated_len,
            view: TensorView::contiguous(new_shape),
            memory_type: self.memory_type,
            owned: false, // View doesn't own memory
        })
    }

    /// Create a mutable contiguous view with a smaller shape.
    ///
    /// Same as `resized_view` but for mutable access.
    pub fn resized_view_mut(&mut self, new_shape: TensorShape) -> Result<TensorHip<T>> {
        let new_len = new_shape.len();
        if new_len > self.allocated_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "New shape {} (len={}) exceeds allocated size {}",
                    new_shape, new_len, self.allocated_len
                ),
            });
        }

        Ok(TensorHip {
            ptr: self.ptr,
            allocated_len: self.allocated_len,
            view: TensorView::contiguous(new_shape),
            memory_type: self.memory_type,
            owned: false, // View doesn't own memory
        })
    }

    /// Reinterpret the tensor with a different shape without copying data.
    ///
    /// This is a zero-copy reshape operation. The new shape must have the same
    /// total number of elements as the current shape. The tensor must be contiguous.
    ///
    /// # Example
    /// ```ignore
    /// // Reshape [n_embd, t, b] to [head_size, n_head, t, b]
    /// // where n_embd = head_size * n_head
    /// let reshaped = tensor.reshape_view(new_shape)?;
    /// ```
    pub fn reshape_view(&self, new_shape: TensorShape) -> Result<TensorHip<T>> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "reshape_view requires contiguous tensor".to_string(),
            });
        }

        let old_len = self.len();
        let new_len = new_shape.len();
        if old_len != new_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Cannot reshape: old shape {} (len={}) != new shape {} (len={})",
                    self.shape(), old_len, new_shape, new_len
                ),
            });
        }

        Ok(TensorHip {
            ptr: self.ptr,
            allocated_len: self.allocated_len,
            view: TensorView::contiguous(new_shape),
            memory_type: self.memory_type,
            owned: false, // View doesn't own memory
        })
    }

    /// Reinterpret the tensor with a different shape without copying data (mutable).
    ///
    /// Same as `reshape_view` but returns a mutable view.
    pub fn reshape_view_mut(&mut self, new_shape: TensorShape) -> Result<TensorHip<T>> {
        if !self.is_contiguous() {
            return Err(HipErrorKind {
                code: -1,
                message: "reshape_view_mut requires contiguous tensor".to_string(),
            });
        }

        let old_len = self.len();
        let new_len = new_shape.len();
        if old_len != new_len {
            return Err(HipErrorKind {
                code: -1,
                message: format!(
                    "Cannot reshape: old shape {} (len={}) != new shape {} (len={})",
                    self.shape(), old_len, new_shape, new_len
                ),
            });
        }

        Ok(TensorHip {
            ptr: self.ptr,
            allocated_len: self.allocated_len,
            view: TensorView::contiguous(new_shape),
            memory_type: self.memory_type,
            owned: false, // View doesn't own memory
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::device::{Stream, Event};

    #[test]
    fn test_tensor_new_empty() {
        let tensor: TensorHip<f32> = TensorHip::new(TensorShape::new(0, 1, 1, 1))
            .expect("Failed to create empty tensor");
        assert_eq!(tensor.len(), 0);
        assert!(tensor.is_empty());
    }

    #[test]
    fn test_tensor_new_1d() {
        let tensor: TensorHip<f32> = TensorHip::new(TensorShape::from_slice(&[100]))
            .expect("Failed to create 1D tensor");
        assert_eq!(tensor.shape().dim(0), 100);
        assert_eq!(tensor.len(), 100);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_new_3d() {
        let tensor: TensorHip<f32> = TensorHip::new(TensorShape::from_slice(&[4, 8, 16]))
            .expect("Failed to create 3D tensor");
        assert_eq!(tensor.shape().dim(0), 4);
        assert_eq!(tensor.shape().dim(1), 8);
        assert_eq!(tensor.shape().dim(2), 16);
        assert_eq!(tensor.len(), 4 * 8 * 16);
    }

    #[test]
    fn test_tensor_from_slice() {
        let stream = Stream::null();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let tensor = TensorHip::from_slice(&data, TensorShape::new(10, 10, 1, 1), &stream)
            .expect("Failed to create tensor from slice");

        assert_eq!(tensor.shape().dim(0), 10);
        assert_eq!(tensor.shape().dim(1), 10);
        assert_eq!(tensor.len(), 100);
    }

    #[test]
    fn test_tensor_to_vec_round_trip() {
        let stream = Stream::null();
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();

        let tensor = TensorHip::from_slice(&data, TensorShape::new(8, 8, 1, 1), &stream)
            .expect("Failed to create tensor");

        let result = tensor.to_vec(&stream).expect("Failed to read tensor");

        assert_eq!(result.len(), data.len());
        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            assert!((expected - actual).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_tensor_copy_to_slice_async() {
        let stream = Stream::null();
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.25).collect();

        // Create tensor and upload data
        let tensor = TensorHip::from_slice(&data, TensorShape::new(16, 16, 1, 1), &stream)
            .expect("Failed to create tensor");

        // Create destination buffer
        let mut dst = vec![0.0f32; 256];

        // Async copy
        tensor.copy_to_slice_async(&mut dst, &stream)
            .expect("copy_to_slice_async failed");

        // Record event and sync
        let event = Event::new().expect("Failed to create event");
        event.record(&stream).expect("Failed to record event");
        event.synchronize().expect("Failed to sync event");

        // Verify data
        for (i, (&expected, &actual)) in data.iter().zip(dst.iter()).enumerate() {
            assert!((expected - actual).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_tensor_copy_to_slice_async_size_mismatch() {
        let stream = Stream::null();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let tensor = TensorHip::from_slice(&data, TensorShape::new(10, 10, 1, 1), &stream)
            .expect("Failed to create tensor");

        // Wrong size buffer
        let mut dst = vec![0.0f32; 50];

        let result = tensor.copy_to_slice_async(&mut dst, &stream);
        assert!(result.is_err(), "Should fail with size mismatch");
    }

    #[test]
    fn test_tensor_copy_to_slice_async_empty() {
        let stream = Stream::null();
        let tensor: TensorHip<f32> = TensorHip::new(TensorShape::new(0, 1, 1, 1))
            .expect("Failed to create empty tensor");

        let mut dst: Vec<f32> = vec![];

        // Should succeed (no-op for empty)
        tensor.copy_to_slice_async(&mut dst, &stream)
            .expect("Empty copy should succeed");
    }

    #[test]
    fn test_tensor_is_contiguous() {
        let tensor: TensorHip<f32> = TensorHip::new(TensorShape::from_slice(&[10, 20, 30]))
            .expect("Failed to create tensor");

        // Fresh tensor should be contiguous
        assert!(tensor.is_contiguous(), "New tensor should be contiguous");
    }

    #[test]
    fn test_tensor_fill_zero() {
        let stream = Stream::null();
        let mut tensor: TensorHip<f32> = TensorHip::new(TensorShape::from_slice(&[100]))
            .expect("Failed to create tensor");

        // Fill with zeros
        tensor.fill_zero().expect("Failed to fill tensor");

        let result = tensor.to_vec(&stream).expect("Failed to read tensor");
        for &v in &result {
            assert_eq!(v, 0.0, "Value should be 0.0 after fill_zero");
        }
    }
}
