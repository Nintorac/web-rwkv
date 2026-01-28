//! # web-rwkv-bench
//!
//! Benchmarking utilities for the web-rwkv project.
//!
//! This crate provides:
//! - Config parsing for benchmark settings (skip conditions, limits)
//! - Skip condition evaluation to filter benchmark cases
//! - Limits tracking for controlling benchmark execution

pub mod config;
pub mod skip;

pub use config::{CustomRule, Limits, SkipConditions};
pub use skip::{LimitsTracker, SkipReason};
