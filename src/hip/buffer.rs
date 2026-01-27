//! HIP device buffer management.

use std::ffi::c_void;
use std::ptr;

use super::ffi::{
    HipErrorKind, Result, check,
    hip_malloc, hip_free,
    hip_memcpy_h2d, hip_memcpy_d2h,
};
use super::device::Stream;

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
