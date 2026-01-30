//! WGPU profiling via GPU timestamp queries.
//!
//! Enable with the `wgpu-prof` feature flag. Control at runtime via `WEB_RWKV_WGPU_PROF` env var.

#[cfg(feature = "wgpu-prof")]
mod enabled {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    use wgpu::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, QuerySet,
        QuerySetDescriptor, QueryType, Queue,
    };

    /// Maximum number of timestamp pairs (start + end) per profiling session.
    /// Note: QuerySet max is 4096, so MAX_TIMESTAMPS * 2 must not exceed that.
    const MAX_TIMESTAMPS: u32 = 2048;

    /// WGPU profiler using GPU timestamp queries.
    pub struct WgpuProf {
        label: &'static str,
        enabled: bool,
        query_set: Arc<QuerySet>,
        resolve_buffer: Arc<Buffer>,
        readback_buffer: Arc<Buffer>,
        timestamp_period: f32,
        labels: Vec<&'static str>,
        query_index: u32,
        totals: BTreeMap<&'static str, Duration>,
    }

    impl WgpuProf {
        /// Create a new profiler. Checks `WEB_RWKV_WGPU_PROF` env var.
        pub fn new(label: &'static str, device: &Device, queue: &Queue) -> Self {
            let enabled = std::env::var("WEB_RWKV_WGPU_PROF")
                .map(|v| v != "0")
                .unwrap_or(true);

            let timestamp_period = queue.get_timestamp_period();

            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("wgpu_prof_query_set"),
                ty: QueryType::Timestamp,
                count: MAX_TIMESTAMPS * 2,
            });

            // Buffer to resolve timestamps into
            let resolve_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("wgpu_prof_resolve"),
                size: (MAX_TIMESTAMPS * 2 * 8) as u64, // u64 per timestamp
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Buffer to read timestamps back to CPU
            let readback_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("wgpu_prof_readback"),
                size: (MAX_TIMESTAMPS * 2 * 8) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Self {
                label,
                enabled,
                query_set: Arc::new(query_set),
                resolve_buffer: Arc::new(resolve_buffer),
                readback_buffer: Arc::new(readback_buffer),
                timestamp_period,
                labels: Vec::with_capacity(MAX_TIMESTAMPS as usize),
                query_index: 0,
                totals: BTreeMap::new(),
            }
        }

        /// Check if profiling is enabled.
        #[inline]
        pub fn enabled(&self) -> bool {
            self.enabled
        }

        /// Get the query set for use in compute passes.
        #[inline]
        pub fn query_set(&self) -> &QuerySet {
            &self.query_set
        }

        /// Record a start timestamp for a labeled operation.
        /// Returns the query index to use with `write_timestamp`.
        #[inline]
        pub fn start(&mut self, label: &'static str) -> Option<u32> {
            if !self.enabled || self.query_index >= MAX_TIMESTAMPS * 2 - 1 {
                return None;
            }
            self.labels.push(label);
            let idx = self.query_index;
            self.query_index += 1;
            Some(idx)
        }

        /// Record an end timestamp.
        /// Returns the query index to use with `write_timestamp`.
        #[inline]
        pub fn end(&mut self) -> Option<u32> {
            if !self.enabled || self.query_index >= MAX_TIMESTAMPS * 2 {
                return None;
            }
            let idx = self.query_index;
            self.query_index += 1;
            Some(idx)
        }

        /// Resolve timestamps to buffer. Call after all profiled work is submitted.
        pub fn resolve(&self, encoder: &mut CommandEncoder) {
            if !self.enabled || self.query_index == 0 {
                return;
            }
            encoder.resolve_query_set(
                &self.query_set,
                0..self.query_index,
                &self.resolve_buffer,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buffer,
                0,
                &self.readback_buffer,
                0,
                (self.query_index as u64) * 8,
            );
        }

        /// Read timestamps and accumulate into totals. Call after GPU work completes.
        pub fn accumulate(&mut self, device: &Device) {
            if !self.enabled || self.query_index == 0 {
                return;
            }

            // Map the readback buffer
            let slice = self.readback_buffer.slice(..);
            let (sender, receiver) = flume::bounded(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            if receiver.recv().ok().and_then(|r| r.ok()).is_none() {
                log::warn!("[wgpu-prof] failed to map readback buffer");
                return;
            }

            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            // Process timestamp pairs
            for (i, label) in self.labels.iter().enumerate() {
                let start_idx = i * 2;
                let end_idx = start_idx + 1;
                if end_idx < timestamps.len() {
                    let start_ts = timestamps[start_idx];
                    let end_ts = timestamps[end_idx];
                    if end_ts >= start_ts {
                        let delta_ns = (end_ts - start_ts) as f64 * self.timestamp_period as f64;
                        let delta = Duration::from_nanos(delta_ns as u64);
                        *self.totals.entry(label).or_insert(Duration::ZERO) += delta;
                    }
                }
            }

            drop(data);
            self.readback_buffer.unmap();

            // Reset for next frame
            self.labels.clear();
            self.query_index = 0;
        }

        /// Print accumulated timings to stderr.
        pub fn print(&self, context: &str) {
            if !self.enabled || self.totals.is_empty() {
                return;
            }
            eprintln!("[wgpu-prof] {} {}", self.label, context);
            for (label, dur) in &self.totals {
                eprintln!(
                    "[wgpu-prof]   {:<16} {:>8.3} ms",
                    label,
                    dur.as_secs_f64() * 1000.0
                );
            }
        }

        /// Clear accumulated totals.
        pub fn clear(&mut self) {
            self.totals.clear();
        }
    }
}

#[cfg(feature = "wgpu-prof")]
pub use enabled::WgpuProf;

// No-op implementation when feature is disabled
#[cfg(not(feature = "wgpu-prof"))]
mod disabled {
    use wgpu::{CommandEncoder, Device, QuerySet, Queue};

    /// No-op profiler when `wgpu-prof` feature is disabled.
    #[derive(Default)]
    pub struct WgpuProf;

    impl WgpuProf {
        #[inline]
        pub fn new(_label: &'static str, _device: &Device, _queue: &Queue) -> Self {
            Self
        }

        #[inline]
        pub fn enabled(&self) -> bool {
            false
        }

        #[inline]
        pub fn query_set(&self) -> Option<&QuerySet> {
            None
        }

        #[inline]
        pub fn start(&mut self, _label: &'static str) -> Option<u32> {
            None
        }

        #[inline]
        pub fn end(&mut self) -> Option<u32> {
            None
        }

        #[inline]
        pub fn resolve(&self, _encoder: &mut CommandEncoder) {}

        #[inline]
        pub fn accumulate(&mut self, _device: &Device) {}

        #[inline]
        pub fn print(&self, _context: &str) {}

        #[inline]
        pub fn clear(&mut self) {}
    }
}

#[cfg(not(feature = "wgpu-prof"))]
pub use disabled::WgpuProf;
