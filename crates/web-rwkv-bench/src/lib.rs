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
//! - JSONL writer for benchmark output with append-only semantics
//! - Prefill-uniform scenario with length sets and TTFT tracking
//! - Prefill-mixed named case patterns for mixed-batch benchmarks
//! - Error classification for continue-on-error policy
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
//!
//! # Prefill-Uniform Scenario
//!
//! The `prefill_uniform` module provides utilities for measuring prompt processing performance:
//!
//! ```rust
//! use web_rwkv_bench::prefill_uniform::{canonical_length_list, target_mode_lengths};
//!
//! // Get canonical lengths for chunk size 128
//! let lengths = canonical_length_list(128);
//! // Returns [1, 32, 64, 127, 128, 129, 256, 512, 1024]
//!
//! // Get lengths for total-token target mode
//! let lengths = target_mode_lengths(128, 4); // chunk=128, batch=4
//! ```
//!
//! # Error Handling
//!
//! The benchmark runner uses a continue-on-error policy by default:
//!
//! ```rust,ignore
//! use web_rwkv_bench::error::{BenchError, classify_error, ErrorContext};
//! use web_rwkv_bench::jsonl::{ErrorKind, Status};
//!
//! // When a benchmark case fails, classify the error
//! let error = BenchError::OutOfMemory { message: "allocation failed".to_string() };
//! let (kind, message) = classify_error(&error);
//!
//! // Write a status=error record to JSONL (see jsonl module for full example)
//! ```
//!
//! # Prefill-Mixed Patterns
//!
//! The `prefill_mixed` module provides deterministic length vector generation for
//! mixed-batch prefill benchmarks:
//!
//! ```rust,ignore
//! use web_rwkv_bench::prefill_mixed::{generate_lengths, MixedCaseId};
//!
//! // Generate lengths for staircase_8 pattern with B=8, C=256
//! let lengths = generate_lengths(MixedCaseId::Staircase8, 8, 256).unwrap();
//! // lengths = [16, 32, 64, 128, 192, 256, 512, 1024]
//! ```

pub mod config;
pub mod error;
pub mod jsonl;
pub mod metadata;
pub mod prefill_mixed;
pub mod prefill_uniform;
pub mod skip;
pub mod sweep;

pub use config::{CustomRule, Limits, SkipConditions};
pub use error::{classify_error, classify_error_message, BenchError, BenchResult, ErrorContext};
pub use jsonl::{
    generate_case_id, generate_run_id, generate_timestamp_utc, round_chunk_size,
    CaseIdParams, CaseIdentity, DecodeMetrics, ErrorKind, GpuInfo as JsonlGpuInfo,
    HostInfo as JsonlHostInfo, JsonlError, JsonlResult, JsonlWriter, MeasureRecord, Metrics,
    PrefillMetrics, RunHeader, Scenario, ScenarioParams, Status, SCHEMA_VERSION,
};
pub use metadata::{collect_run_metadata, BuildInfo, GitInfo, GpuInfo, HostInfo, RunMetadata};
pub use prefill_mixed::{
    all_mixed_case_ids, generate_lengths, generate_lengths_from_str, total_tokens, MixedCaseError,
    MixedCaseId, MixedCaseResult,
};
pub use prefill_uniform::{
    all_prefill_lengths, canonical_length_list, seq_len_from_total_tokens, target_mode_lengths,
    total_token_targets, LengthMetadata, LengthMode, PrefillResult, PrefillUniformConfig,
    TokenGenerator, TtftTracker,
};
pub use skip::{LimitsTracker, SkipReason};
pub use sweep::{CaseParams, ExpandedCase, Hooks, SweepConfig, SweepEngine, SweepSummary};
