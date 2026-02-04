//! Pinned (page-locked) host memory for faster DMA transfers.
//!
//! Pinned memory is allocated using `hipHostMalloc` which page-locks the memory,
//! enabling faster host-to-device and device-to-host transfers via DMA.

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr;

use super::ffi::{
    check, hip_host_free, hip_host_malloc, hip_memcpy_d2h, hip_memcpy_h2d, HipStream, Result,
    HIP_HOST_MALLOC_DEFAULT,
};

/// Pinned (page-locked) host memory buffer.
///
/// This buffer is allocated using `hipHostMalloc` which page-locks the memory,
/// enabling faster DMA transfers between host and device. Use this for data
/// that needs to be frequently transferred to/from the GPU.
///
/// Note: Pinned memory is a limited resource. Only use it for hot-path transfers.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> PinnedBuffer<T> {
    /// Allocate a new pinned buffer with the given length.
    ///
    /// # Panics
    /// Panics if the allocation size overflows or if T is a zero-sized type.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }

        let layout = Layout::array::<T>(len).expect("allocation size overflow");
        assert!(layout.size() > 0, "cannot allocate ZST buffer");

        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check(hip_host_malloc(
                &mut ptr,
                layout.size(),
                HIP_HOST_MALLOC_DEFAULT,
            ))?;
        }

        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Get the length of the buffer in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice view of the buffer.
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Get a mutable slice view of the buffer.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    /// Get a raw pointer to the buffer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get a mutable raw pointer to the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from host to device asynchronously.
    ///
    /// The destination pointer must have enough space for `self.len()` elements.
    /// The copy is asynchronous with respect to the host; call `stream.synchronize()`
    /// to wait for completion if needed.
    ///
    /// # Safety
    /// The destination pointer must be valid for `self.len()` elements.
    pub unsafe fn copy_to_device_async(&self, dst: *mut T, stream: HipStream) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        let size = self.len * std::mem::size_of::<T>();
        check(hip_memcpy_h2d(
            dst as *mut std::ffi::c_void,
            self.ptr as *const std::ffi::c_void,
            size,
            stream,
        ))
    }

    /// Copy data from device to this pinned buffer asynchronously.
    ///
    /// The source pointer must have at least `self.len()` elements.
    /// The copy is asynchronous with respect to the host; call `stream.synchronize()`
    /// to wait for completion if needed.
    ///
    /// # Safety
    /// The source pointer must be valid for `self.len()` elements.
    pub unsafe fn copy_from_device_async(
        &mut self,
        src: *const T,
        stream: HipStream,
    ) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        let size = self.len * std::mem::size_of::<T>();
        check(hip_memcpy_d2h(
            self.ptr as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            size,
            stream,
        ))
    }

    /// Copy from a slice into this pinned buffer.
    ///
    /// # Panics
    /// Panics if the slice length doesn't match the buffer length.
    pub fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(
            src.len(),
            self.len,
            "source slice length doesn't match buffer length"
        );
        if self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len);
            }
        }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                // Ignore errors during drop
                let _ = hip_host_free(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

// Safety: The buffer contains only Copy types and the pointer is exclusively owned
unsafe impl<T: Copy + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuffer<T> {}

impl<T> std::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl<T: Copy> Clone for PinnedBuffer<T> {
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self {
                ptr: ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            };
        }

        // Allocate a new pinned buffer and copy data
        let mut new_buf = Self::new(self.len).expect("Failed to allocate pinned buffer for clone");
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_buf.ptr, self.len);
        }
        new_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::device::Stream;
    use crate::hip::ffi::{check as ffi_check, hip_free, hip_malloc};

    #[test]
    fn test_pinned_buffer_empty() {
        let buf: PinnedBuffer<f32> = PinnedBuffer::new(0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pinned_buffer_allocation() {
        // Test various sizes
        let sizes = [1, 16, 256, 1024, 4096];
        for &size in &sizes {
            let buf: PinnedBuffer<f32> = PinnedBuffer::new(size).expect(&format!(
                "Failed to allocate pinned buffer of size {}",
                size
            ));
            assert_eq!(buf.len(), size);
            assert!(!buf.is_empty());
            assert!(!buf.as_ptr().is_null());
        }
    }

    #[test]
    fn test_pinned_buffer_slice_access() {
        let mut buf: PinnedBuffer<f32> = PinnedBuffer::new(100).unwrap();

        // Write via mutable slice
        let slice = buf.as_slice_mut();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32 * 2.0;
        }

        // Read back via immutable slice
        let slice = buf.as_slice();
        for (i, &v) in slice.iter().enumerate() {
            assert_eq!(v, i as f32 * 2.0);
        }
    }

    #[test]
    fn test_pinned_buffer_copy_from_slice() {
        let mut buf: PinnedBuffer<f32> = PinnedBuffer::new(10).unwrap();
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();

        buf.copy_from_slice(&data);

        assert_eq!(buf.as_slice(), &data[..]);
    }

    #[test]
    #[should_panic(expected = "source slice length doesn't match buffer length")]
    fn test_pinned_buffer_copy_from_slice_size_mismatch() {
        let mut buf: PinnedBuffer<f32> = PinnedBuffer::new(10).unwrap();
        let data: Vec<f32> = (0..5).map(|i| i as f32).collect();

        buf.copy_from_slice(&data); // Should panic
    }

    #[test]
    fn test_pinned_buffer_async_round_trip() {
        // This test verifies the full async copy path:
        // 1. Host → pinned buffer
        // 2. Pinned buffer → device (async)
        // 3. Device → pinned buffer (async)
        // 4. Verify data integrity

        let size = 1024;
        let stream = Stream::null();

        // Allocate device memory
        let mut device_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let byte_size = size * std::mem::size_of::<f32>();
        unsafe {
            ffi_check(hip_malloc(&mut device_ptr, byte_size))
                .expect("Failed to allocate device memory");
        }

        // Create source data
        let src_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();

        // Create pinned buffers
        let mut src_buf: PinnedBuffer<f32> = PinnedBuffer::new(size).unwrap();
        let mut dst_buf: PinnedBuffer<f32> = PinnedBuffer::new(size).unwrap();

        // Copy source data to pinned buffer
        src_buf.copy_from_slice(&src_data);

        // Async copy: pinned → device → pinned
        unsafe {
            src_buf
                .copy_to_device_async(device_ptr as *mut f32, stream.handle())
                .expect("H→D async copy failed");

            dst_buf
                .copy_from_device_async(device_ptr as *const f32, stream.handle())
                .expect("D→H async copy failed");
        }

        // Synchronize to ensure copies complete
        stream.synchronize().expect("Stream sync failed");

        // Verify data
        assert_eq!(
            dst_buf.as_slice(),
            src_data.as_slice(),
            "Data mismatch after async round-trip"
        );

        // Clean up device memory
        unsafe {
            let _ = hip_free(device_ptr);
        }
    }

    #[test]
    fn test_pinned_buffer_async_empty() {
        // Async copies on empty buffers should be no-ops
        let buf: PinnedBuffer<f32> = PinnedBuffer::new(0).unwrap();
        let stream = Stream::null();

        // These should succeed (no-op)
        unsafe {
            buf.copy_to_device_async(std::ptr::null_mut(), stream.handle())
                .expect("Empty H→D should succeed");
        }

        let mut buf2: PinnedBuffer<f32> = PinnedBuffer::new(0).unwrap();
        unsafe {
            buf2.copy_from_device_async(std::ptr::null(), stream.handle())
                .expect("Empty D→H should succeed");
        }
    }

    #[test]
    fn test_pinned_buffer_different_types() {
        // Test with different types
        let buf_u32: PinnedBuffer<u32> = PinnedBuffer::new(100).unwrap();
        assert_eq!(buf_u32.len(), 100);

        let buf_i64: PinnedBuffer<i64> = PinnedBuffer::new(50).unwrap();
        assert_eq!(buf_i64.len(), 50);

        let buf_f64: PinnedBuffer<f64> = PinnedBuffer::new(25).unwrap();
        assert_eq!(buf_f64.len(), 25);
    }
}
