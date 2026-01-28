//! # web-rwkv-bench
//!
//! Benchmarking utilities for the web-rwkv project.
//!
//! This crate provides:
//! - Config parsing for benchmark settings (skip conditions, limits)
//! - Skip condition evaluation to filter benchmark cases
//! - Limits tracking for controlling benchmark execution
//! - Environment/build metadata collection for benchmark run headers
//! - Sweep execution engine for cartesian expansion and case lifecycle management
//!
//! # Sweep Execution
//!
//! The sweep engine performs cartesian product expansion across benchmark dimensions:
//! - models
//! - backends (including wgpu variants)
//! - batch_sizes
//! - token_chunk_sizes
//! - scenarios (with scenario-specific parameters like seq_lens, decode_steps)
//!
//! Skip conditions are applied during expansion, and limits are tracked to control execution.
//!
//! ```rust,ignore
//! use web_rwkv_bench::sweep::{SweepEngine, SweepConfig, Hooks};
//! use web_rwkv_bench::config::{SkipConditions, Limits};
//!
//! let config = SweepConfig { /* ... */ };
//! let mut engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
//!
//! let hooks = Hooks::new()
//!     .on_case_setup(|params| { /* setup */ Ok(()) })
//!     .on_case_teardown(|params| { /* teardown */ });
//!
//! let mut executor = engine.execute_with_hooks(&hooks);
//! while let Some(case) = executor.next_case() {
//!     case.setup(&hooks)?;
//!     // run benchmark
//!     executor.record_executed();
//!     case.teardown(&hooks);
//! }
//! let summary = executor.finish();
//! ```

pub mod config;
pub mod metadata;
pub mod skip;
pub mod sweep;

pub use config::{CustomRule, Limits, SkipConditions};
pub use metadata::{collect_run_metadata, BuildInfo, GitInfo, GpuInfo, HostInfo, RunMetadata};
pub use skip::{LimitsTracker, SkipReason};
pub use sweep::{CaseParams, ExpandedCase, Hooks, SweepConfig, SweepEngine, SweepSummary};
