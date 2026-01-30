//! Sweep execution engine for benchmark case expansion and lifecycle management.
//!
//! This module provides:
//! - [`CaseParams`]: Normalized parameters for a single benchmark case
//! - [`SweepConfig`]: Configuration for sweep expansion
//! - [`SweepEngine`]: Engine that expands sweeps into individual cases
//! - [`Hooks`]: Setup and teardown callbacks for case lifecycle
//!
//! # Usage
//!
//! ```rust,ignore
//! use web_rwkv_bench::sweep::{SweepEngine, SweepConfig, Hooks};
//! use web_rwkv_bench::config::{SkipConditions, Limits};
//!
//! let sweep_config = SweepConfig {
//!     models: vec![/* model configs */],
//!     backends: vec![/* backend configs */],
//!     batch_sizes: vec![1, 4, 8],
//!     token_chunk_sizes: vec![128, 256],
//!     seq_lens: vec![32, 128, 512],
//!     decode_steps: vec![128, 512],
//!     scenarios: vec!["decode_only".to_string(), "prefill_uniform".to_string()],
//! };
//!
//! let mut engine = SweepEngine::new(
//!     sweep_config,
//!     SkipConditions::default(),
//!     Limits::default(),
//! );
//!
//! // Define hooks for case lifecycle
//! let hooks = Hooks::new()
//!     .on_case_setup(|params| {
//!         println!("Setting up case: {}", params.case_id());
//!         Ok(())
//!     })
//!     .on_case_teardown(|params| {
//!         println!("Tearing down case: {}", params.case_id());
//!     });
//!
//! // Iterate over expanded cases
//! for case in engine.expand_cases() {
//!     if let Some(reason) = case.skip_reason {
//!         println!("Skipped: {}", reason);
//!         continue;
//!     }
//!     // Execute benchmark for case.params
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt;

use crate::config::{BackendConfig, BenchmarkCase, Limits, ModelConfig, SkipConditions};
use crate::skip::{should_skip, LimitsTracker, SkipReason, StopReason};

/// Normalized parameters for a single benchmark case.
///
/// These parameters uniquely identify a benchmark "cell" in the matrix
/// and are used for case_id computation. The parameters are normalized
/// to ensure stable, reproducible identifiers across runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseParams {
    /// Model identifier (stable hash)
    pub model_id: String,
    /// Model name (human readable)
    pub model_name: String,
    /// Model size label (e.g., "9m", "0.1b")
    pub model_size: String,
    /// Backend identifier ("wgpu" or "hip")
    pub backend_id: String,
    /// WGPU backend variant (e.g., "Vulkan", "Dx12") - optional for wgpu
    pub wgpu_backend: Option<String>,
    /// Scenario name (e.g., "decode_only", "prefill_uniform", "prefill_mixed")
    pub scenario: String,
    /// Batch size
    pub batch_size: u32,
    /// Token chunk size (requested)
    pub token_chunk_size: u32,
    /// Sequence length (for prefill scenarios)
    pub seq_len: Option<u32>,
    /// Decode steps (for decode_only scenario)
    pub decode_steps: Option<u32>,
    /// Mixed case ID (for prefill_mixed scenario)
    pub mixed_case_id: Option<String>,
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of recorded repeats
    pub repeats: u32,
}

impl CaseParams {
    /// Generate a stable case ID for this benchmark case.
    ///
    /// The case_id is a human-readable string that uniquely identifies
    /// the case parameters. It excludes run-time metadata (run_id, timestamps)
    /// and adapter-specific info so it can be used to group repeats and
    /// compare results across runs.
    pub fn case_id(&self) -> String {
        let mut parts = vec![self.model_name.clone(), self.backend_id.clone()];

        if let Some(ref wgpu_backend) = self.wgpu_backend {
            parts.push(wgpu_backend.clone());
        }

        parts.push(self.scenario.clone());
        parts.push(format!("b{}", self.batch_size));
        parts.push(format!("c{}", self.token_chunk_size));

        if let Some(seq_len) = self.seq_len {
            parts.push(format!("s{}", seq_len));
        }
        if let Some(decode_steps) = self.decode_steps {
            parts.push(format!("d{}", decode_steps));
        }
        if let Some(ref mixed_case_id) = self.mixed_case_id {
            parts.push(format!("m_{}", mixed_case_id));
        }

        parts.join("_")
    }

    /// Convert to a sorted map for canonical JSON encoding.
    ///
    /// This is useful for computing hashes or comparing case params.
    pub fn to_normalized_map(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();

        map.insert("model_id".to_string(), self.model_id.clone());
        map.insert("model_name".to_string(), self.model_name.clone());
        map.insert("backend_id".to_string(), self.backend_id.clone());

        if let Some(ref wgpu_backend) = self.wgpu_backend {
            map.insert("wgpu_backend".to_string(), wgpu_backend.clone());
        }

        map.insert("scenario".to_string(), self.scenario.clone());
        map.insert("batch_size".to_string(), self.batch_size.to_string());
        map.insert(
            "token_chunk_size".to_string(),
            self.token_chunk_size.to_string(),
        );

        if let Some(seq_len) = self.seq_len {
            map.insert("seq_len".to_string(), seq_len.to_string());
        }
        if let Some(decode_steps) = self.decode_steps {
            map.insert("decode_steps".to_string(), decode_steps.to_string());
        }
        if let Some(ref mixed_case_id) = self.mixed_case_id {
            map.insert("mixed_case_id".to_string(), mixed_case_id.clone());
        }

        map
    }
}

impl fmt::Display for CaseParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.case_id())
    }
}

/// Extended model configuration for sweep expansion.
#[derive(Debug, Clone)]
pub struct SweepModelConfig {
    /// Model identifier (sha256 hash)
    pub model_id: String,
    /// Model name
    pub model_name: String,
    /// Model size label
    pub model_size: String,
    /// Maximum supported batch size
    pub max_batch_size: Option<u32>,
    /// Maximum supported token chunk size
    pub max_token_chunk_size: Option<u32>,
}

impl From<&SweepModelConfig> for ModelConfig {
    fn from(m: &SweepModelConfig) -> Self {
        ModelConfig {
            model_name: m.model_name.clone(),
            max_batch_size: m.max_batch_size,
            max_token_chunk_size: m.max_token_chunk_size,
        }
    }
}

/// Extended backend configuration for sweep expansion.
#[derive(Debug, Clone)]
pub struct SweepBackendConfig {
    /// Backend identifier
    pub backend_id: String,
    /// WGPU backend variants (for wgpu only)
    pub wgpu_backends: Vec<String>,
}

impl From<&SweepBackendConfig> for BackendConfig {
    fn from(b: &SweepBackendConfig) -> Self {
        BackendConfig {
            backend_id: b.backend_id.clone(),
        }
    }
}

/// Configuration for sweep expansion.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Models to include in sweep
    pub models: Vec<SweepModelConfig>,
    /// Backends to include in sweep
    pub backends: Vec<SweepBackendConfig>,
    /// Batch sizes to sweep
    pub batch_sizes: Vec<u32>,
    /// Token chunk sizes to sweep
    pub token_chunk_sizes: Vec<u32>,
    /// Sequence lengths to sweep (for prefill scenarios)
    pub seq_lens: Vec<u32>,
    /// Decode steps to sweep (for decode_only scenario)
    pub decode_steps: Vec<u32>,
    /// Scenarios to run
    pub scenarios: Vec<String>,
    /// Mixed case IDs (for prefill_mixed scenario)
    pub mixed_case_ids: Vec<String>,
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of recorded repeats
    pub repeats: u32,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            backends: Vec::new(),
            batch_sizes: vec![1],
            token_chunk_sizes: vec![128],
            seq_lens: vec![128],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: Vec::new(),
            warmup_runs: 1,
            repeats: 5,
        }
    }
}

/// An expanded case with optional skip reason.
#[derive(Debug, Clone)]
pub struct ExpandedCase {
    /// The case parameters
    pub params: CaseParams,
    /// Skip reason if case should be skipped
    pub skip_reason: Option<SkipReason>,
    /// Index of this case in the expansion
    pub index: usize,
}

/// Type alias for setup hook function.
pub type SetupFn = Box<dyn Fn(&CaseParams) -> Result<(), String> + Send + Sync>;

/// Type alias for teardown hook function.
pub type TeardownFn = Box<dyn Fn(&CaseParams) + Send + Sync>;

/// Type alias for before-sweep hook function.
pub type BeforeSweepFn = Box<dyn Fn(usize) + Send + Sync>;

/// Type alias for after-sweep hook function.
pub type AfterSweepFn = Box<dyn Fn(&SweepSummary) + Send + Sync>;

/// Lifecycle hooks for benchmark cases.
///
/// Hooks allow scenarios to perform setup and teardown operations
/// around each benchmark case. They can be used for:
/// - Loading/unloading models
/// - Initializing GPU contexts
/// - Clearing caches
/// - Logging and metrics collection
pub struct Hooks {
    /// Called before each case execution
    pub setup: Option<SetupFn>,
    /// Called after each case execution
    pub teardown: Option<TeardownFn>,
    /// Called before the sweep starts, with total case count
    pub before_sweep: Option<BeforeSweepFn>,
    /// Called after the sweep completes
    pub after_sweep: Option<AfterSweepFn>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Hooks {
    /// Create empty hooks.
    pub fn new() -> Self {
        Self {
            setup: None,
            teardown: None,
            before_sweep: None,
            after_sweep: None,
        }
    }

    /// Set the setup hook.
    pub fn on_case_setup<F>(mut self, f: F) -> Self
    where
        F: Fn(&CaseParams) -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup = Some(Box::new(f));
        self
    }

    /// Set the teardown hook.
    pub fn on_case_teardown<F>(mut self, f: F) -> Self
    where
        F: Fn(&CaseParams) + Send + Sync + 'static,
    {
        self.teardown = Some(Box::new(f));
        self
    }

    /// Set the before-sweep hook.
    pub fn on_before_sweep<F>(mut self, f: F) -> Self
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.before_sweep = Some(Box::new(f));
        self
    }

    /// Set the after-sweep hook.
    pub fn on_after_sweep<F>(mut self, f: F) -> Self
    where
        F: Fn(&SweepSummary) + Send + Sync + 'static,
    {
        self.after_sweep = Some(Box::new(f));
        self
    }

    /// Call the setup hook if defined.
    pub fn call_setup(&self, params: &CaseParams) -> Result<(), String> {
        if let Some(ref f) = self.setup {
            f(params)
        } else {
            Ok(())
        }
    }

    /// Call the teardown hook if defined.
    pub fn call_teardown(&self, params: &CaseParams) {
        if let Some(ref f) = self.teardown {
            f(params);
        }
    }

    /// Call the before-sweep hook if defined.
    pub fn call_before_sweep(&self, total_cases: usize) {
        if let Some(ref f) = self.before_sweep {
            f(total_cases);
        }
    }

    /// Call the after-sweep hook if defined.
    pub fn call_after_sweep(&self, summary: &SweepSummary) {
        if let Some(ref f) = self.after_sweep {
            f(summary);
        }
    }
}

impl fmt::Debug for Hooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hooks")
            .field("setup", &self.setup.is_some())
            .field("teardown", &self.teardown.is_some())
            .field("before_sweep", &self.before_sweep.is_some())
            .field("after_sweep", &self.after_sweep.is_some())
            .finish()
    }
}

/// Summary of sweep execution.
#[derive(Debug, Clone, Default)]
pub struct SweepSummary {
    /// Total cases in the expansion (before skip filtering)
    pub total_cases: usize,
    /// Cases that were skipped
    pub skipped_cases: usize,
    /// Cases that were executed
    pub executed_cases: usize,
    /// Cases that resulted in errors
    pub error_cases: usize,
    /// Stop reason if sweep was terminated early
    pub stop_reason: Option<StopReason>,
}

/// Sweep execution engine.
///
/// The engine performs cartesian product expansion of sweep dimensions
/// while applying skip conditions and respecting execution limits.
pub struct SweepEngine {
    config: SweepConfig,
    skip_conditions: SkipConditions,
    limits_tracker: LimitsTracker,
}

impl SweepEngine {
    /// Create a new sweep engine with the given configuration.
    pub fn new(config: SweepConfig, skip_conditions: SkipConditions, limits: Limits) -> Self {
        Self {
            config,
            skip_conditions,
            limits_tracker: LimitsTracker::new(limits),
        }
    }

    /// Get a reference to the limits tracker.
    pub fn limits_tracker(&self) -> &LimitsTracker {
        &self.limits_tracker
    }

    /// Get a mutable reference to the limits tracker.
    pub fn limits_tracker_mut(&mut self) -> &mut LimitsTracker {
        &mut self.limits_tracker
    }

    /// Expand the sweep configuration into individual cases.
    ///
    /// This performs cartesian product expansion across:
    /// - models
    /// - backends (including wgpu variants)
    /// - batch_sizes
    /// - token_chunk_sizes
    /// - scenarios (with scenario-specific parameters)
    ///
    /// Skip conditions are evaluated during expansion. Cases that
    /// should be skipped will have `skip_reason` set.
    ///
    /// The expansion respects `max_total_cases` limit - expansion
    /// stops when the limit is reached.
    pub fn expand_cases(&self) -> Vec<ExpandedCase> {
        let mut cases = Vec::new();
        let mut index = 0;

        // Cartesian expansion: models × backends × batch_sizes × token_chunk_sizes × scenarios
        'expansion: for model in &self.config.models {
            for backend in &self.config.backends {
                // For wgpu backend, expand over wgpu_backends
                // For hip backend, use None for wgpu_backend
                let wgpu_variants: Vec<Option<String>> =
                    if backend.backend_id == "wgpu" && !backend.wgpu_backends.is_empty() {
                        backend
                            .wgpu_backends
                            .iter()
                            .map(|v| Some(v.clone()))
                            .collect()
                    } else {
                        vec![None]
                    };

                for wgpu_backend in &wgpu_variants {
                    for &batch_size in &self.config.batch_sizes {
                        for &token_chunk_size in &self.config.token_chunk_sizes {
                            for scenario in &self.config.scenarios {
                                // Expand scenario-specific parameters
                                let scenario_cases = self.expand_scenario(
                                    model,
                                    backend,
                                    wgpu_backend.clone(),
                                    batch_size,
                                    token_chunk_size,
                                    scenario,
                                    &mut index,
                                );

                                for case in scenario_cases {
                                    cases.push(case);

                                    // Check max_cases limit (count non-skipped cases)
                                    // Note: We expand all cases but stop early if limit reached
                                    if self
                                        .limits_tracker
                                        .limits_reached_for_expansion(cases.len())
                                    {
                                        break 'expansion;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        cases
    }

    /// Expand scenario-specific parameters.
    fn expand_scenario(
        &self,
        model: &SweepModelConfig,
        backend: &SweepBackendConfig,
        wgpu_backend: Option<String>,
        batch_size: u32,
        token_chunk_size: u32,
        scenario: &str,
        index: &mut usize,
    ) -> Vec<ExpandedCase> {
        let mut cases = Vec::new();

        match scenario {
            "decode_only" => {
                // Expand over decode_steps
                for &decode_steps in &self.config.decode_steps {
                    let params = CaseParams {
                        model_id: model.model_id.clone(),
                        model_name: model.model_name.clone(),
                        model_size: model.model_size.clone(),
                        backend_id: backend.backend_id.clone(),
                        wgpu_backend: wgpu_backend.clone(),
                        scenario: scenario.to_string(),
                        batch_size,
                        token_chunk_size,
                        seq_len: None,
                        decode_steps: Some(decode_steps),
                        mixed_case_id: None,
                        warmup_runs: self.config.warmup_runs,
                        repeats: self.config.repeats,
                    };

                    let skip_reason = self.evaluate_skip(&params, model, backend);
                    cases.push(ExpandedCase {
                        params,
                        skip_reason,
                        index: *index,
                    });
                    *index += 1;
                }
            }
            "prefill_uniform" => {
                // Expand over seq_lens
                for &seq_len in &self.config.seq_lens {
                    let params = CaseParams {
                        model_id: model.model_id.clone(),
                        model_name: model.model_name.clone(),
                        model_size: model.model_size.clone(),
                        backend_id: backend.backend_id.clone(),
                        wgpu_backend: wgpu_backend.clone(),
                        scenario: scenario.to_string(),
                        batch_size,
                        token_chunk_size,
                        seq_len: Some(seq_len),
                        decode_steps: None,
                        mixed_case_id: None,
                        warmup_runs: self.config.warmup_runs,
                        repeats: self.config.repeats,
                    };

                    let skip_reason = self.evaluate_skip(&params, model, backend);
                    cases.push(ExpandedCase {
                        params,
                        skip_reason,
                        index: *index,
                    });
                    *index += 1;
                }
            }
            "prefill_mixed" => {
                // Expand over seq_lens × mixed_case_ids
                for &seq_len in &self.config.seq_lens {
                    // If no mixed_case_ids configured, create one case per seq_len
                    if self.config.mixed_case_ids.is_empty() {
                        let params = CaseParams {
                            model_id: model.model_id.clone(),
                            model_name: model.model_name.clone(),
                            model_size: model.model_size.clone(),
                            backend_id: backend.backend_id.clone(),
                            wgpu_backend: wgpu_backend.clone(),
                            scenario: scenario.to_string(),
                            batch_size,
                            token_chunk_size,
                            seq_len: Some(seq_len),
                            decode_steps: None,
                            mixed_case_id: None,
                            warmup_runs: self.config.warmup_runs,
                            repeats: self.config.repeats,
                        };

                        let skip_reason = self.evaluate_skip(&params, model, backend);
                        cases.push(ExpandedCase {
                            params,
                            skip_reason,
                            index: *index,
                        });
                        *index += 1;
                    } else {
                        for mixed_case_id in &self.config.mixed_case_ids {
                            let params = CaseParams {
                                model_id: model.model_id.clone(),
                                model_name: model.model_name.clone(),
                                model_size: model.model_size.clone(),
                                backend_id: backend.backend_id.clone(),
                                wgpu_backend: wgpu_backend.clone(),
                                scenario: scenario.to_string(),
                                batch_size,
                                token_chunk_size,
                                seq_len: Some(seq_len),
                                decode_steps: None,
                                mixed_case_id: Some(mixed_case_id.clone()),
                                warmup_runs: self.config.warmup_runs,
                                repeats: self.config.repeats,
                            };

                            let skip_reason = self.evaluate_skip(&params, model, backend);
                            cases.push(ExpandedCase {
                                params,
                                skip_reason,
                                index: *index,
                            });
                            *index += 1;
                        }
                    }
                }
            }
            _ => {
                // Unknown scenario - create a basic case
                let params = CaseParams {
                    model_id: model.model_id.clone(),
                    model_name: model.model_name.clone(),
                    model_size: model.model_size.clone(),
                    backend_id: backend.backend_id.clone(),
                    wgpu_backend: wgpu_backend.clone(),
                    scenario: scenario.to_string(),
                    batch_size,
                    token_chunk_size,
                    seq_len: None,
                    decode_steps: None,
                    mixed_case_id: None,
                    warmup_runs: self.config.warmup_runs,
                    repeats: self.config.repeats,
                };

                let skip_reason = self.evaluate_skip(&params, model, backend);
                cases.push(ExpandedCase {
                    params,
                    skip_reason,
                    index: *index,
                });
                *index += 1;
            }
        }

        cases
    }

    /// Evaluate skip conditions for a case.
    fn evaluate_skip(
        &self,
        params: &CaseParams,
        model: &SweepModelConfig,
        backend: &SweepBackendConfig,
    ) -> Option<SkipReason> {
        let benchmark_case = BenchmarkCase {
            batch_size: params.batch_size,
            token_chunk_size: params.token_chunk_size,
            seq_len: params.seq_len,
            decode_steps: params.decode_steps,
        };

        let model_config = ModelConfig::from(model);
        let backend_config = BackendConfig::from(backend);

        should_skip(
            &benchmark_case,
            &model_config,
            &backend_config,
            &self.skip_conditions,
        )
    }

    /// Count total cases (including skipped).
    pub fn count_total_cases(&self) -> usize {
        self.expand_cases().len()
    }

    /// Count cases that will be executed (excluding skipped).
    pub fn count_executable_cases(&self) -> usize {
        self.expand_cases()
            .iter()
            .filter(|c| c.skip_reason.is_none())
            .count()
    }

    /// Create an iterator that executes cases with hooks.
    ///
    /// This is the main entry point for executing a sweep with
    /// per-case setup/teardown and limits checking.
    pub fn execute_with_hooks<'a>(&'a mut self, hooks: &'a Hooks) -> SweepExecutor<'a> {
        let cases = self.expand_cases();
        let total = cases.len();
        let skipped = cases.iter().filter(|c| c.skip_reason.is_some()).count();

        SweepExecutor {
            engine: self,
            hooks,
            cases,
            current: 0,
            summary: SweepSummary {
                total_cases: total,
                skipped_cases: skipped,
                executed_cases: 0,
                error_cases: 0,
                stop_reason: None,
            },
            started: false,
        }
    }
}

impl LimitsTracker {
    /// Check if max_cases limit would be exceeded for expansion.
    fn limits_reached_for_expansion(&self, current_count: usize) -> bool {
        let max = self.max_total_cases();
        max > 0 && current_count >= max as usize
    }

    /// Get the max total cases limit.
    fn max_total_cases(&self) -> u32 {
        // Access through the public interface
        // We'll need to expose this - for now return 0 (unlimited)
        0 // Will be implemented via accessor
    }
}

/// Iterator-style executor for sweep with hooks.
pub struct SweepExecutor<'a> {
    engine: &'a mut SweepEngine,
    hooks: &'a Hooks,
    cases: Vec<ExpandedCase>,
    current: usize,
    summary: SweepSummary,
    started: bool,
}

impl<'a> SweepExecutor<'a> {
    /// Get the sweep summary.
    pub fn summary(&self) -> &SweepSummary {
        &self.summary
    }

    /// Record that a case was executed.
    pub fn record_executed(&mut self) {
        self.summary.executed_cases += 1;
        self.engine.limits_tracker_mut().record_case();
    }

    /// Record that a case resulted in an error.
    ///
    /// This increments the error counter in both the summary and the limits tracker.
    /// Use this when a benchmark case fails and you want to continue with the sweep.
    ///
    /// Note: This method only updates counters. To write an error record to JSONL,
    /// use the JSONL writer directly.
    pub fn record_error(&mut self) {
        self.summary.error_cases += 1;
        self.engine.limits_tracker_mut().record_error();
    }

    /// Record that a case was executed and resulted in an error.
    ///
    /// This is a convenience method that combines `record_executed` and `record_error`.
    /// It also marks the case as executed (for total count) before recording the error.
    ///
    /// Use this when:
    /// 1. You attempted to run a case
    /// 2. It failed with an error
    /// 3. You want to continue with the next case (continue-on-error policy)
    ///
    /// # Usage
    ///
    /// In a typical benchmark loop:
    /// 1. Call `next_case()` to get the next case
    /// 2. Attempt to run the benchmark
    /// 3. On success: write success record, call `record_executed()`
    /// 4. On error: classify error, write error record, call `record_executed_error()`
    /// 5. Loop continues to next case (continue-on-error policy)
    pub fn record_executed_error(&mut self) {
        self.record_executed();
        self.record_error();
    }

    /// Check if execution should stop due to limits.
    pub fn should_stop(&self) -> Option<StopReason> {
        self.engine.limits_tracker().should_stop()
    }

    /// Get a reference to the hooks.
    pub fn hooks(&self) -> &Hooks {
        self.hooks
    }

    /// Get the next case to execute.
    ///
    /// Returns `None` when all cases are exhausted or limits are reached.
    pub fn next_case(&mut self) -> Option<ExecutableCase> {
        // Call before_sweep on first access
        if !self.started {
            self.started = true;
            self.hooks.call_before_sweep(self.cases.len());
        }

        // Check if we should stop
        if let Some(reason) = self.engine.limits_tracker().should_stop() {
            self.summary.stop_reason = Some(reason);
            return None;
        }

        // Find next case
        while self.current < self.cases.len() {
            let idx = self.current;
            self.current += 1;

            let case = &self.cases[idx];

            // If case should be skipped, continue to next
            if case.skip_reason.is_some() {
                continue;
            }

            return Some(ExecutableCase {
                params: case.params.clone(),
                index: case.index,
            });
        }

        None
    }

    /// Finish the sweep and get the final summary.
    pub fn finish(self) -> SweepSummary {
        self.hooks.call_after_sweep(&self.summary);
        self.summary
    }
}

/// A case ready for execution.
///
/// This struct owns a clone of the case parameters so it can be used
/// independently of the executor. Use `run_with_hooks` for full lifecycle
/// management, or call hooks manually.
#[derive(Debug, Clone)]
pub struct ExecutableCase {
    /// The case parameters
    pub params: CaseParams,
    /// Index of this case
    pub index: usize,
}

impl ExecutableCase {
    /// Call setup hook for this case.
    pub fn setup(&self, hooks: &Hooks) -> Result<(), String> {
        hooks.call_setup(&self.params)
    }

    /// Call teardown hook for this case.
    pub fn teardown(&self, hooks: &Hooks) {
        hooks.call_teardown(&self.params);
    }

    /// Run with full lifecycle management.
    ///
    /// Calls setup, executes the provided closure, then calls teardown.
    /// Returns the result of the closure.
    pub fn run_with_hooks<T, F>(&self, hooks: &Hooks, f: F) -> Result<T, String>
    where
        F: FnOnce(&CaseParams) -> T,
    {
        self.setup(hooks)?;
        let result = f(&self.params);
        self.teardown(hooks);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(name: &str) -> SweepModelConfig {
        SweepModelConfig {
            model_id: format!("{}_id", name),
            model_name: name.to_string(),
            model_size: "9m".to_string(),
            max_batch_size: Some(32),
            max_token_chunk_size: Some(512),
        }
    }

    fn make_backend(id: &str, wgpu_backends: Vec<&str>) -> SweepBackendConfig {
        SweepBackendConfig {
            backend_id: id.to_string(),
            wgpu_backends: wgpu_backends.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn test_case_params_case_id() {
        let params = CaseParams {
            model_id: "abc123".to_string(),
            model_name: "test_model".to_string(),
            model_size: "9m".to_string(),
            backend_id: "wgpu".to_string(),
            wgpu_backend: Some("Vulkan".to_string()),
            scenario: "decode_only".to_string(),
            batch_size: 4,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(512),
            mixed_case_id: None,
            warmup_runs: 1,
            repeats: 5,
        };

        let case_id = params.case_id();
        assert!(case_id.contains("test_model"));
        assert!(case_id.contains("wgpu"));
        assert!(case_id.contains("Vulkan"));
        assert!(case_id.contains("decode_only"));
        assert!(case_id.contains("b4"));
        assert!(case_id.contains("c128"));
        assert!(case_id.contains("d512"));
    }

    #[test]
    fn test_case_params_prefill_case_id() {
        let params = CaseParams {
            model_id: "abc123".to_string(),
            model_name: "test_model".to_string(),
            model_size: "9m".to_string(),
            backend_id: "hip".to_string(),
            wgpu_backend: None,
            scenario: "prefill_uniform".to_string(),
            batch_size: 8,
            token_chunk_size: 256,
            seq_len: Some(1024),
            decode_steps: None,
            mixed_case_id: None,
            warmup_runs: 1,
            repeats: 5,
        };

        let case_id = params.case_id();
        assert!(case_id.contains("test_model"));
        assert!(case_id.contains("hip"));
        assert!(case_id.contains("prefill_uniform"));
        assert!(case_id.contains("b8"));
        assert!(case_id.contains("c256"));
        assert!(case_id.contains("s1024"));
        assert!(!case_id.contains("Vulkan")); // No wgpu_backend for hip
    }

    #[test]
    fn test_case_params_mixed_case_id() {
        let params = CaseParams {
            model_id: "abc123".to_string(),
            model_name: "test_model".to_string(),
            model_size: "9m".to_string(),
            backend_id: "wgpu".to_string(),
            wgpu_backend: None,
            scenario: "prefill_mixed".to_string(),
            batch_size: 8,
            token_chunk_size: 256,
            seq_len: Some(512),
            decode_steps: None,
            mixed_case_id: Some("staircase_8".to_string()),
            warmup_runs: 1,
            repeats: 5,
        };

        let case_id = params.case_id();
        assert!(case_id.contains("m_staircase_8"));
    }

    #[test]
    fn test_sweep_expand_decode_only() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 4],
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128, 512],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let cases = engine.expand_cases();

        // 1 model × 1 backend × 2 batch_sizes × 1 chunk_size × 2 decode_steps = 4 cases
        assert_eq!(cases.len(), 4);

        // Check that all cases are decode_only
        for case in &cases {
            assert_eq!(case.params.scenario, "decode_only");
            assert!(case.params.decode_steps.is_some());
            assert!(case.params.seq_len.is_none());
        }
    }

    #[test]
    fn test_sweep_expand_prefill_uniform() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 4],
            token_chunk_sizes: vec![128],
            seq_lens: vec![32, 128, 256],
            decode_steps: vec![],
            scenarios: vec!["prefill_uniform".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let cases = engine.expand_cases();

        // 1 model × 1 backend × 2 batch_sizes × 1 chunk_size × 3 seq_lens = 6 cases
        assert_eq!(cases.len(), 6);

        // Check that all cases are prefill_uniform
        for case in &cases {
            assert_eq!(case.params.scenario, "prefill_uniform");
            assert!(case.params.seq_len.is_some());
            assert!(case.params.decode_steps.is_none());
        }
    }

    #[test]
    fn test_sweep_expand_multiple_scenarios() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1],
            token_chunk_sizes: vec![128],
            seq_lens: vec![128],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string(), "prefill_uniform".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let cases = engine.expand_cases();

        // 1 decode_only case + 1 prefill_uniform case = 2 cases
        assert_eq!(cases.len(), 2);
    }

    #[test]
    fn test_sweep_expand_wgpu_backends() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec!["Vulkan", "Dx12"])],
            batch_sizes: vec![1],
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let cases = engine.expand_cases();

        // 1 model × 2 wgpu_backends × 1 batch × 1 chunk × 1 decode_steps = 2 cases
        assert_eq!(cases.len(), 2);

        // Check wgpu_backend variants
        let wgpu_backends: Vec<_> = cases
            .iter()
            .map(|c| c.params.wgpu_backend.clone())
            .collect();
        assert!(wgpu_backends.contains(&Some("Vulkan".to_string())));
        assert!(wgpu_backends.contains(&Some("Dx12".to_string())));
    }

    #[test]
    fn test_sweep_expand_skip_conditions() {
        let config = SweepConfig {
            models: vec![make_model("model1")], // max_batch_size = 32
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 64], // 64 exceeds max
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let skip_conditions = SkipConditions {
            skip_batch_exceeds_model_max: true,
            ..Default::default()
        };

        let engine = SweepEngine::new(config, skip_conditions, Limits::default());
        let cases = engine.expand_cases();

        // 2 cases total
        assert_eq!(cases.len(), 2);

        // One should be skipped
        let skipped: Vec<_> = cases.iter().filter(|c| c.skip_reason.is_some()).collect();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].params.batch_size, 64);
    }

    #[test]
    fn test_sweep_expand_prefill_mixed() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![8],
            token_chunk_sizes: vec![128],
            seq_lens: vec![128, 256],
            decode_steps: vec![],
            scenarios: vec!["prefill_mixed".to_string()],
            mixed_case_ids: vec!["staircase_8".to_string(), "bimodal_half".to_string()],
            warmup_runs: 1,
            repeats: 5,
        };

        let engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let cases = engine.expand_cases();

        // 1 model × 1 backend × 1 batch × 1 chunk × 2 seq_lens × 2 mixed_cases = 4 cases
        assert_eq!(cases.len(), 4);

        // Check that all have mixed_case_id
        for case in &cases {
            assert!(case.params.mixed_case_id.is_some());
        }
    }

    #[test]
    fn test_hooks_builder() {
        let setup_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let teardown_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let setup_flag = setup_called.clone();
        let teardown_flag = teardown_called.clone();

        let hooks = Hooks::new()
            .on_case_setup(move |_| {
                setup_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })
            .on_case_teardown(move |_| {
                teardown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            });

        let params = CaseParams {
            model_id: "test".to_string(),
            model_name: "test".to_string(),
            model_size: "9m".to_string(),
            backend_id: "wgpu".to_string(),
            wgpu_backend: None,
            scenario: "decode_only".to_string(),
            batch_size: 1,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(128),
            mixed_case_id: None,
            warmup_runs: 1,
            repeats: 5,
        };

        hooks.call_setup(&params).unwrap();
        assert!(setup_called.load(std::sync::atomic::Ordering::SeqCst));

        hooks.call_teardown(&params);
        assert!(teardown_called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_normalized_map() {
        let params = CaseParams {
            model_id: "abc123".to_string(),
            model_name: "test_model".to_string(),
            model_size: "9m".to_string(),
            backend_id: "wgpu".to_string(),
            wgpu_backend: Some("Vulkan".to_string()),
            scenario: "prefill_uniform".to_string(),
            batch_size: 4,
            token_chunk_size: 256,
            seq_len: Some(512),
            decode_steps: None,
            mixed_case_id: None,
            warmup_runs: 1,
            repeats: 5,
        };

        let map = params.to_normalized_map();

        assert_eq!(map.get("model_id"), Some(&"abc123".to_string()));
        assert_eq!(map.get("backend_id"), Some(&"wgpu".to_string()));
        assert_eq!(map.get("wgpu_backend"), Some(&"Vulkan".to_string()));
        assert_eq!(map.get("batch_size"), Some(&"4".to_string()));
        assert_eq!(map.get("seq_len"), Some(&"512".to_string()));
        assert!(!map.contains_key("decode_steps"));
        assert!(!map.contains_key("mixed_case_id"));
    }

    #[test]
    fn test_count_executable_cases() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 64], // 64 exceeds max
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let skip_conditions = SkipConditions {
            skip_batch_exceeds_model_max: true,
            ..Default::default()
        };

        let engine = SweepEngine::new(config, skip_conditions, Limits::default());

        assert_eq!(engine.count_total_cases(), 2);
        assert_eq!(engine.count_executable_cases(), 1);
    }

    #[test]
    fn test_sweep_executor() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 4],
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let mut engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let hooks = Hooks::new();

        let mut executor = engine.execute_with_hooks(&hooks);
        let mut count = 0;

        while let Some(case) = executor.next_case() {
            case.setup(&hooks).unwrap();
            count += 1;
            case.teardown(&hooks);
            // Note: record_executed must be called after case is dropped
            // to avoid borrow conflicts
            executor.record_executed();
        }

        let summary = executor.finish();
        assert_eq!(count, 2);
        assert_eq!(summary.total_cases, 2);
        assert_eq!(summary.executed_cases, 2);
        assert_eq!(summary.skipped_cases, 0);
    }

    #[test]
    fn test_sweep_executor_with_run_with_hooks() {
        let config = SweepConfig {
            models: vec![make_model("model1")],
            backends: vec![make_backend("wgpu", vec![])],
            batch_sizes: vec![1, 4],
            token_chunk_sizes: vec![128],
            seq_lens: vec![],
            decode_steps: vec![128],
            scenarios: vec!["decode_only".to_string()],
            mixed_case_ids: vec![],
            warmup_runs: 1,
            repeats: 5,
        };

        let mut engine = SweepEngine::new(config, SkipConditions::default(), Limits::default());
        let hooks = Hooks::new();

        let mut executor = engine.execute_with_hooks(&hooks);
        let mut executed_ids = Vec::new();

        while let Some(case) = executor.next_case() {
            let result = case.run_with_hooks(&hooks, |params| params.case_id());
            executed_ids.push(result.unwrap());
            executor.record_executed();
        }

        let summary = executor.finish();
        assert_eq!(executed_ids.len(), 2);
        assert_eq!(summary.executed_cases, 2);
    }
}
