//! HIP device context and stream management.

use std::ptr;

use super::ffi::{
    check, device_supports_cooperative_launch, get_device_compute_capability, get_device_count,
    get_device_mp_count, get_device_name, get_device_total_memory, get_device_warp_size,
    get_gcn_arch_name, hip_device_synchronize, hip_event_create, hip_event_destroy,
    hip_event_query, hip_event_record, hip_event_synchronize, hip_stream_create,
    hip_stream_destroy, hip_stream_synchronize, hip_stream_wait_event, is_device_integrated,
    set_device, HipErrorKind, HipEvent, HipStream, Result, HIP_ERROR_NOT_READY, HIP_SUCCESS,
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

    /// Make this stream wait for an event to complete before continuing.
    ///
    /// All operations enqueued on this stream after this call will wait until
    /// the specified event has completed. This enables stream-to-stream dependencies.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        unsafe { check(hip_stream_wait_event(self.handle, event.handle(), 0)) }
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

/// A HIP event for fine-grained asynchronous synchronization.
///
/// Events can be recorded on a stream and then waited on, either from the host
/// (via `synchronize()`) or from another stream (via `Stream::wait_event()`).
/// This enables overlapping compute and data transfers.
pub struct Event {
    handle: HipEvent,
}

impl Event {
    /// Create a new HIP event.
    pub fn new() -> Result<Self> {
        let mut handle: HipEvent = ptr::null_mut();
        unsafe { check(hip_event_create(&mut handle))? };
        Ok(Self { handle })
    }

    /// Record this event on a stream.
    ///
    /// The event will be marked as complete when all operations submitted to the
    /// stream before this call have finished.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { check(hip_event_record(self.handle, stream.handle())) }
    }

    /// Block the host until this event completes.
    ///
    /// This is a blocking operation that waits for all operations recorded before
    /// this event to finish.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(hip_event_synchronize(self.handle)) }
    }

    /// Query whether this event has completed (non-blocking).
    ///
    /// Returns `Ok(true)` if the event has completed, `Ok(false)` if it's still
    /// pending, or an error if something went wrong.
    pub fn query(&self) -> Result<bool> {
        let result = unsafe { hip_event_query(self.handle) };
        if result == HIP_SUCCESS {
            Ok(true)
        } else if result == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            check(result)?;
            unreachable!()
        }
    }

    /// Get the raw event handle.
    pub fn handle(&self) -> HipEvent {
        self.handle
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                hip_event_destroy(self.handle);
            }
        }
    }
}

// Safety: Event handles are thread-safe to send between threads
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

/// Synchronize the default stream / device
pub fn device_synchronize() -> Result<()> {
    unsafe { check(hip_device_synchronize()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_new() {
        // Event::new() should succeed on initialized HIP
        let event = Event::new();
        assert!(event.is_ok(), "Event::new() failed: {:?}", event.err());
    }

    #[test]
    fn test_event_record_and_synchronize() {
        // Create event and stream
        let event = Event::new().expect("Failed to create event");
        let stream = Stream::null();

        // Record event on stream
        let result = event.record(&stream);
        assert!(result.is_ok(), "Event::record() failed: {:?}", result.err());

        // Synchronize should succeed (event is already complete on null stream)
        let result = event.synchronize();
        assert!(
            result.is_ok(),
            "Event::synchronize() failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_event_query_complete() {
        // Create event and stream
        let event = Event::new().expect("Failed to create event");
        let stream = Stream::null();

        // Record event
        event.record(&stream).expect("Failed to record event");

        // Synchronize to ensure completion
        stream.synchronize().expect("Failed to sync stream");

        // Query should return true (complete)
        let is_complete = event.query().expect("Event::query() failed");
        assert!(is_complete, "Event should be complete after stream sync");
    }

    #[test]
    fn test_stream_null() {
        // Null stream should work without explicit creation
        let stream = Stream::null();
        assert!(stream.is_null(), "Null stream should report is_null()");
        assert!(
            stream.handle().is_null(),
            "Null stream handle should be null"
        );

        // Synchronize should work
        let result = stream.synchronize();
        assert!(
            result.is_ok(),
            "Null stream sync failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_stream_wait_event() {
        // Create two streams and an event
        let stream1 = Stream::null();
        let stream2 = Stream::null(); // Both null for compatibility
        let event = Event::new().expect("Failed to create event");

        // Record event on stream1
        event.record(&stream1).expect("Failed to record event");

        // Make stream2 wait on the event
        let result = stream2.wait_event(&event);
        assert!(
            result.is_ok(),
            "Stream::wait_event() failed: {:?}",
            result.err()
        );

        // Both should sync without issue
        stream1.synchronize().expect("stream1 sync failed");
        stream2.synchronize().expect("stream2 sync failed");
    }

    #[test]
    fn test_event_multiple_records() {
        // An event can be recorded multiple times (replaces previous record point)
        let event = Event::new().expect("Failed to create event");
        let stream = Stream::null();

        // Record multiple times
        event.record(&stream).expect("First record failed");
        event.record(&stream).expect("Second record failed");
        event.record(&stream).expect("Third record failed");

        // Should sync fine
        event
            .synchronize()
            .expect("Sync after multiple records failed");
    }

    #[test]
    fn test_hip_context_new() {
        // HipContext::new() should succeed if HIP is available
        let ctx = HipContext::new();
        assert!(ctx.is_ok(), "HipContext::new() failed: {:?}", ctx.err());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.device_id(), 0);
    }

    #[test]
    fn test_hip_context_device_properties() {
        let ctx = HipContext::new().expect("Failed to create HipContext");

        // These should all succeed on a valid HIP device
        let name = ctx.device_name();
        assert!(name.is_ok(), "device_name() failed: {:?}", name.err());

        let arch = ctx.gcn_arch_name();
        assert!(arch.is_ok(), "gcn_arch_name() failed: {:?}", arch.err());

        let mem = ctx.total_memory();
        assert!(mem.is_ok(), "total_memory() failed: {:?}", mem.err());
        assert!(mem.unwrap() > 0, "Total memory should be > 0");

        let mps = ctx.multiprocessor_count();
        assert!(
            mps.is_ok(),
            "multiprocessor_count() failed: {:?}",
            mps.err()
        );
        assert!(mps.unwrap() > 0, "MP count should be > 0");

        let warp = ctx.warp_size();
        assert!(warp.is_ok(), "warp_size() failed: {:?}", warp.err());
        // AMD GPUs typically have warp size 64
        assert!(warp.unwrap() > 0, "Warp size should be > 0");
    }

    #[test]
    fn test_hip_context_creation() {
        let ctx = HipContext::new().expect("Failed to create context");
        assert!(ctx.device_id() >= 0, "Device ID should be non-negative");

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
        let ctx = HipContext::new().expect("Failed to create context");

        let name = ctx.device_name().expect("Failed to get device name");
        let arch = ctx
            .gcn_arch_name()
            .unwrap_or_else(|_| "unknown".to_string());
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

        assert!(!name.is_empty(), "Device name should not be empty");

        if arch.contains("gfx1151") {
            if warp_size > 0 {
                assert_eq!(warp_size, 32, "gfx1151 should have warp size 32");
                println!("  Verified gfx1151 warp size");
            }
        }
    }

    #[test]
    fn test_hip_stream_sync() {
        let ctx = HipContext::new().expect("Failed to create context");
        ctx.synchronize()
            .expect("Failed to synchronize null stream");
        println!("Null stream synchronization test passed");
    }
}
