//! Runtime profiling utilities for wgpu backend.
//!
//! This module provides profiling support for measuring GPU execution times.
//! Enable with the `wgpu-prof` feature flag.
//!
//! # Example
//!
//! ```ignore
//! use web_rwkv::runtime::profiling::ProfiledRuntime;
//!
//! // Wrap your runtime
//! let profiled = ProfiledRuntime::new(runtime);
//!
//! // Run inference
//! let output = profiled.infer(input).await?;
//!
//! // Print profiling results
//! profiled.print_profile("batch=256 tokens=1");
//! ```

#[cfg(feature = "wgpu-prof")]
mod enabled {
    use std::sync::{Arc, Mutex};

    use crate::context::Context;
    use crate::tensor::prof::WgpuProf;

    /// A profiled runtime wrapper that collects GPU timing information.
    pub struct ProfiledContext {
        pub context: Context,
        pub profiler: Arc<Mutex<WgpuProf>>,
    }

    impl ProfiledContext {
        /// Create a new profiled context wrapper.
        pub fn new(context: Context) -> Self {
            let profiler = WgpuProf::new("forward", &context.device, &context.queue);
            Self {
                context,
                profiler: Arc::new(Mutex::new(profiler)),
            }
        }

        /// Get the underlying context.
        pub fn context(&self) -> &Context {
            &self.context
        }

        /// Get the profiler for manual timing operations.
        pub fn profiler(&self) -> Arc<Mutex<WgpuProf>> {
            self.profiler.clone()
        }

        /// Print collected profiling results.
        pub fn print_profile(&self, context_str: &str) {
            if let Ok(prof) = self.profiler.lock() {
                prof.print(context_str);
            }
        }

        /// Clear collected profiling results.
        pub fn clear_profile(&self) {
            if let Ok(mut prof) = self.profiler.lock() {
                prof.clear();
            }
        }

        /// Resolve timestamps after GPU work submission.
        /// Call this after submitting commands but before reading results.
        pub fn resolve_timestamps(&self) {
            let mut encoder = self.context.device.create_command_encoder(&Default::default());
            if let Ok(prof) = self.profiler.lock() {
                prof.resolve(&mut encoder);
            }
            self.context.queue.submit(Some(encoder.finish()));
        }

        /// Accumulate timestamp results after GPU work completes.
        /// Call this after the GPU has finished processing.
        pub fn accumulate_timestamps(&self) {
            if let Ok(mut prof) = self.profiler.lock() {
                prof.accumulate(&self.context.device);
            }
        }
    }

    /// Helper to run a profiled inference step.
    ///
    /// This function:
    /// 1. Encodes operations with profiling timestamps
    /// 2. Submits commands to the queue
    /// 3. Resolves timestamps
    /// 4. Waits for GPU completion
    /// 5. Accumulates timing results
    pub fn run_profiled<F>(
        context: &Context,
        profiler: &mut WgpuProf,
        ops: &crate::tensor::ops::TensorOp,
    ) -> Vec<wgpu::CommandBuffer> {
        context.encode_profiled(ops, profiler)
    }
}

#[cfg(feature = "wgpu-prof")]
pub use enabled::*;

#[cfg(not(feature = "wgpu-prof"))]
mod disabled {
    use crate::context::Context;

    /// No-op profiled context when feature is disabled.
    pub struct ProfiledContext {
        pub context: Context,
    }

    impl ProfiledContext {
        pub fn new(context: Context) -> Self {
            Self { context }
        }

        pub fn context(&self) -> &Context {
            &self.context
        }

        pub fn print_profile(&self, _context_str: &str) {}

        pub fn clear_profile(&self) {}

        pub fn resolve_timestamps(&self) {}

        pub fn accumulate_timestamps(&self) {}
    }
}

#[cfg(not(feature = "wgpu-prof"))]
pub use disabled::*;
