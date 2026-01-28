//! Integration test runner harness for web-rwkv benchmarks.
//!
//! This module provides a config-driven benchmark runner that executes performance
//! sweeps across model, backend, batch size, and sequence length dimensions.
//!
//! # Invocation
//!
//! The benchmark tests are marked `#[ignore]` so they only run on demand:
//!
//! ```bash
//! # Run with default config (benchmarks/config.yaml) and smoke profile
//! cargo test --release --test benchmarks -- --ignored --nocapture
//!
//! # Run with custom config and profile
//! WEB_RWKV_BENCH_CONFIG=benchmarks/config.yaml WEB_RWKV_BENCH_PROFILE=dev \
//!     cargo test --release --test benchmarks -- --ignored --nocapture
//!
//! # Run with HIP backend support
//! WEB_RWKV_BENCH_CONFIG=benchmarks/config.yaml WEB_RWKV_BENCH_PROFILE=full \
//!     cargo test --release --features hip --test benchmarks -- --ignored --nocapture
//! ```
//!
//! # Environment Variables
//!
//! - `WEB_RWKV_BENCH_CONFIG`: Path to YAML config file (default: `benchmarks/config.yaml`)
//! - `WEB_RWKV_BENCH_PROFILE`: Profile name to use (default: `smoke`)
//!
//! # Output
//!
//! Results are written to JSONL files in the directory specified by `output.directory`
//! in the config file. Each run creates a new file with the pattern specified by
//! `output.filename_pattern`.

use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

/// Default config file path
const DEFAULT_CONFIG_PATH: &str = "benchmarks/config.yaml";

/// Default profile name
const DEFAULT_PROFILE: &str = "smoke";

// =============================================================================
// CONFIG STRUCTS
// =============================================================================

/// Top-level benchmark configuration
#[derive(Debug, Deserialize)]
pub struct BenchConfig {
    /// Schema version for compatibility checking
    pub schema_version: u32,

    /// Named profiles for different benchmark scopes
    pub profiles: HashMap<String, Profile>,

    /// Model definitions
    pub models: Vec<ModelEntry>,

    /// Backend definitions
    pub backends: Vec<BackendEntry>,

    /// Scenario definitions
    pub scenarios: HashMap<String, ScenarioConfig>,

    /// Mixed case definitions for prefill_mixed scenario
    #[serde(default)]
    pub mixed_cases: HashMap<String, MixedCase>,

    /// Sweep definitions
    pub sweeps: HashMap<String, SweepConfig>,

    /// Output configuration
    pub output: OutputConfig,

    /// Skip conditions
    #[serde(default)]
    pub skip_conditions: SkipConditions,

    /// Global limits
    #[serde(default)]
    pub limits: Limits,

    /// Shared controls (defaults for all scenarios)
    #[serde(default)]
    pub shared_controls: SharedControls,
}

/// A named profile that specifies which models, backends, and scenarios to run
#[derive(Debug, Deserialize)]
pub struct Profile {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// List of model names to include
    pub models: Vec<String>,

    /// List of backend IDs to include
    pub backends: Vec<String>,

    /// List of scenario names to run
    pub scenarios: Vec<String>,

    /// Batch sizes to sweep
    pub batch_sizes: Vec<u32>,

    /// Token chunk sizes to sweep
    pub token_chunk_sizes: Vec<u32>,

    /// Decode steps (for decode_only scenario)
    #[serde(default)]
    pub decode_steps: Vec<u32>,

    /// Sequence lengths (for prefill scenarios)
    #[serde(default)]
    pub seq_lens: Vec<u32>,

    /// Number of warmup runs
    #[serde(default = "default_warmup_runs")]
    pub warmup_runs: u32,

    /// Number of recorded repeats
    #[serde(default = "default_repeats")]
    pub repeats: u32,
}

fn default_warmup_runs() -> u32 {
    1
}

fn default_repeats() -> u32 {
    5
}

/// Model entry defining a model to benchmark
#[derive(Debug, Deserialize, Clone)]
pub struct ModelEntry {
    /// SHA256 hash of the model file (stable identity)
    pub model_id: String,

    /// Human-readable model name
    pub model_name: String,

    /// Size label for grouping (e.g., "9m", "0.1b", "2.9b")
    pub model_size: String,

    /// Path to the safetensors file
    pub path: String,

    /// Tags for filtering and metadata
    #[serde(default)]
    pub tags: HashMap<String, serde_yaml::Value>,

    /// Maximum batch size this model supports
    #[serde(default)]
    pub max_batch_size: Option<u32>,

    /// Maximum token chunk size this model supports
    #[serde(default)]
    pub max_token_chunk_size: Option<u32>,

    /// Whether to skip this model
    #[serde(default)]
    pub skip: bool,
}

/// Backend entry defining a backend to benchmark
#[derive(Debug, Deserialize, Clone)]
pub struct BackendEntry {
    /// Backend identifier ("wgpu" or "hip")
    pub backend_id: String,

    /// WGPU backend variants (for wgpu only)
    #[serde(default)]
    pub wgpu_backends: Vec<String>,

    /// Adapter selection strategy
    #[serde(default)]
    pub adapter_selection: Option<String>,

    /// Whether to skip this backend
    #[serde(default)]
    pub skip: bool,
}

/// Scenario configuration
#[derive(Debug, Deserialize)]
pub struct ScenarioConfig {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Default parameters for this scenario
    #[serde(default)]
    pub defaults: HashMap<String, serde_yaml::Value>,
}

/// Mixed case definition for prefill_mixed scenario
#[derive(Debug, Deserialize)]
pub struct MixedCase {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Base pattern divisors (relative to C)
    #[serde(default)]
    pub base_pattern_divisors: Vec<f64>,

    /// Short divisor for bimodal/one_long patterns
    #[serde(default)]
    pub short_divisor: Option<f64>,

    /// Long divisor for bimodal/one_long patterns
    #[serde(default)]
    pub long_divisor: Option<f64>,

    /// Distribution type
    #[serde(default)]
    pub distribution: Option<String>,

    /// Number of long prompts
    #[serde(default)]
    pub long_count: Option<u32>,

    /// Adapt rule
    #[serde(default)]
    pub adapt_rule: Option<String>,

    /// Base lengths for realistic patterns
    #[serde(default)]
    pub base_lengths: Vec<u32>,

    /// Base chunk size for scaling
    #[serde(default)]
    pub base_chunk_size: Option<u32>,

    /// Scale rule
    #[serde(default)]
    pub scale_rule: Option<String>,
}

/// Sweep configuration defining the parameter matrix
#[derive(Debug, Deserialize)]
pub struct SweepConfig {
    /// Models to include
    pub models: Vec<String>,

    /// Backends to include
    pub backends: Vec<String>,

    /// Batch sizes to sweep
    pub batch_sizes: Vec<u32>,

    /// Token chunk sizes to sweep
    pub token_chunk_sizes: Vec<u32>,

    /// Sequence lengths to sweep
    #[serde(default)]
    pub seq_lens: Vec<u32>,

    /// Decode steps to sweep
    #[serde(default)]
    pub decode_steps: Vec<u32>,
}

/// Output configuration
#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    /// Directory for benchmark results
    pub directory: String,

    /// File naming pattern
    pub filename_pattern: String,

    /// Output mode ("create_new" or "append")
    #[serde(default = "default_output_mode")]
    pub mode: String,

    /// Whether to compress output
    #[serde(default)]
    pub compress: bool,

    /// Whether to include run header record
    #[serde(default = "default_true")]
    pub include_run_header: bool,

    /// Whether to pretty-print JSON
    #[serde(default)]
    pub pretty_print: bool,
}

fn default_output_mode() -> String {
    "create_new".to_string()
}

fn default_true() -> bool {
    true
}

/// Skip conditions configuration
#[derive(Debug, Deserialize, Default)]
pub struct SkipConditions {
    /// Skip if batch_size exceeds model's max
    #[serde(default = "default_true")]
    pub skip_batch_exceeds_model_max: bool,

    /// Skip if token_chunk_size exceeds model's max
    #[serde(default = "default_true")]
    pub skip_chunk_exceeds_model_max: bool,

    /// Skip known failure combinations
    #[serde(default = "default_true")]
    pub skip_known_failures: bool,

    /// Skip if OOM is predicted
    #[serde(default)]
    pub skip_oom_predicted: bool,

    /// Custom skip rules
    #[serde(default)]
    pub custom_rules: Vec<CustomSkipRule>,
}

/// Custom skip rule
#[derive(Debug, Deserialize)]
pub struct CustomSkipRule {
    /// Rule name
    pub name: String,

    /// Rule description
    #[serde(default)]
    pub description: String,

    /// Condition expression
    pub condition: String,
}

/// Global limits configuration
#[derive(Debug, Deserialize, Default)]
pub struct Limits {
    /// Maximum total cases to run
    #[serde(default = "default_max_cases")]
    pub max_total_cases: u32,

    /// Maximum runtime in seconds (0 = unlimited)
    #[serde(default)]
    pub max_runtime_seconds: u32,

    /// Stop on first error
    #[serde(default)]
    pub fail_fast: bool,

    /// Maximum errors before aborting
    #[serde(default)]
    pub max_errors: u32,

    /// Per-case timeout in seconds
    #[serde(default = "default_case_timeout")]
    pub case_timeout_seconds: u32,

    /// Maximum memory usage percentage
    #[serde(default = "default_max_memory_percent")]
    pub max_memory_percent: u32,
}

fn default_max_cases() -> u32 {
    10000
}

fn default_case_timeout() -> u32 {
    300
}

fn default_max_memory_percent() -> u32 {
    90
}

/// Shared controls (defaults for all scenarios)
#[derive(Debug, Deserialize, Default)]
pub struct SharedControls {
    /// PRNG seed for reproducibility
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Warmup configuration
    #[serde(default)]
    pub warmup: WarmupConfig,

    /// Number of recorded repeats
    #[serde(default = "default_repeats_u32")]
    pub repeats: u32,

    /// Timing configuration
    #[serde(default)]
    pub timing: TimingConfig,

    /// Error handling policy
    #[serde(default)]
    pub error_policy: ErrorPolicy,
}

fn default_seed() -> u64 {
    42
}

fn default_repeats_u32() -> u32 {
    5
}

/// Warmup configuration
#[derive(Debug, Deserialize, Default)]
pub struct WarmupConfig {
    /// Number of warmup runs (not recorded)
    #[serde(default = "default_warmup_runs")]
    pub warmup_runs: u32,

    /// Warmup steps per run
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u32,
}

fn default_warmup_steps() -> u32 {
    64
}

/// Timing configuration
#[derive(Debug, Deserialize, Default)]
pub struct TimingConfig {
    /// Exclude model load time from measurements
    #[serde(default = "default_true")]
    pub exclude_model_load: bool,

    /// Settle delay between repeats (milliseconds)
    #[serde(default = "default_settle_delay")]
    pub settle_delay_ms: u32,
}

fn default_settle_delay() -> u32 {
    100
}

/// Error handling policy
#[derive(Debug, Deserialize, Default)]
pub struct ErrorPolicy {
    /// Continue on error (write status=error records)
    #[serde(default = "default_true")]
    pub continue_on_error: bool,

    /// Classify errors in JSONL records
    #[serde(default = "default_true")]
    pub classify_errors: bool,
}

// =============================================================================
// CONFIG LOADING
// =============================================================================

/// Error type for config loading
#[derive(Debug)]
pub enum ConfigError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Profile not found
    ProfileNotFound(String),
    /// Model not found
    ModelNotFound(String),
    /// Backend not found
    BackendNotFound(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::FileNotFound(path) => write!(f, "Config file not found: {}", path),
            ConfigError::ParseError(msg) => write!(f, "Config parse error: {}", msg),
            ConfigError::ProfileNotFound(name) => write!(f, "Profile not found: {}", name),
            ConfigError::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            ConfigError::BackendNotFound(name) => write!(f, "Backend not found: {}", name),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load benchmark configuration from the specified path
pub fn load_config(path: &str) -> Result<BenchConfig, ConfigError> {
    let path = Path::new(path);
    if !path.exists() {
        return Err(ConfigError::FileNotFound(path.display().to_string()));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| ConfigError::ParseError(format!("Failed to read config: {}", e)))?;

    let config: BenchConfig = serde_yaml::from_str(&content)
        .map_err(|e| ConfigError::ParseError(format!("Failed to parse YAML: {}", e)))?;

    Ok(config)
}

/// Load config from environment variable or default path
pub fn load_config_from_env() -> Result<BenchConfig, ConfigError> {
    let config_path = env::var("WEB_RWKV_BENCH_CONFIG")
        .unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());

    println!("[bench] Loading config from: {}", config_path);
    load_config(&config_path)
}

/// Get the selected profile name from environment variable or default
pub fn get_profile_name() -> String {
    env::var("WEB_RWKV_BENCH_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.to_string())
}

/// Get a profile from the config by name
pub fn get_profile<'a>(config: &'a BenchConfig, name: &str) -> Result<&'a Profile, ConfigError> {
    config
        .profiles
        .get(name)
        .ok_or_else(|| ConfigError::ProfileNotFound(name.to_string()))
}

/// Resolve model entries from a list of model names
pub fn resolve_models(config: &BenchConfig, names: &[String]) -> Result<Vec<ModelEntry>, ConfigError> {
    let mut models = Vec::new();
    for name in names {
        let model = config
            .models
            .iter()
            .find(|m| &m.model_name == name)
            .ok_or_else(|| ConfigError::ModelNotFound(name.clone()))?;
        if !model.skip {
            models.push(model.clone());
        }
    }
    Ok(models)
}

/// Resolve backend entries from a list of backend IDs
pub fn resolve_backends(config: &BenchConfig, ids: &[String]) -> Result<Vec<BackendEntry>, ConfigError> {
    let mut backends = Vec::new();
    for id in ids {
        let backend = config
            .backends
            .iter()
            .find(|b| &b.backend_id == id)
            .ok_or_else(|| ConfigError::BackendNotFound(id.clone()))?;
        if !backend.skip {
            backends.push(backend.clone());
        }
    }
    Ok(backends)
}

// =============================================================================
// BENCHMARK CASE
// =============================================================================

/// A single benchmark case to execute
#[derive(Debug, Clone)]
pub struct BenchCase {
    /// Model to benchmark
    pub model: ModelEntry,
    /// Backend to use
    pub backend: BackendEntry,
    /// Scenario name
    pub scenario: String,
    /// Batch size
    pub batch_size: u32,
    /// Token chunk size
    pub token_chunk_size: u32,
    /// Sequence length (for prefill scenarios)
    pub seq_len: Option<u32>,
    /// Decode steps (for decode_only scenario)
    pub decode_steps: Option<u32>,
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of recorded repeats
    pub repeats: u32,
}

impl BenchCase {
    /// Generate a stable case ID for this benchmark case
    pub fn case_id(&self) -> String {
        let mut parts = vec![
            self.model.model_name.clone(),
            self.backend.backend_id.clone(),
            self.scenario.clone(),
            format!("b{}", self.batch_size),
            format!("c{}", self.token_chunk_size),
        ];

        if let Some(seq_len) = self.seq_len {
            parts.push(format!("s{}", seq_len));
        }
        if let Some(decode_steps) = self.decode_steps {
            parts.push(format!("d{}", decode_steps));
        }

        parts.join("_")
    }
}

/// Expand a profile into a list of benchmark cases
pub fn expand_profile(config: &BenchConfig, profile: &Profile) -> Result<Vec<BenchCase>, ConfigError> {
    let models = resolve_models(config, &profile.models)?;
    let backends = resolve_backends(config, &profile.backends)?;

    let mut cases = Vec::new();

    for model in &models {
        for backend in &backends {
            for &batch_size in &profile.batch_sizes {
                // Check batch size limit
                if let Some(max) = model.max_batch_size {
                    if batch_size > max {
                        println!(
                            "[bench] Skipping batch_size={} for {} (max={})",
                            batch_size, model.model_name, max
                        );
                        continue;
                    }
                }

                for &token_chunk_size in &profile.token_chunk_sizes {
                    // Check chunk size limit
                    if let Some(max) = model.max_token_chunk_size {
                        if token_chunk_size > max {
                            println!(
                                "[bench] Skipping token_chunk_size={} for {} (max={})",
                                token_chunk_size, model.model_name, max
                            );
                            continue;
                        }
                    }

                    for scenario in &profile.scenarios {
                        match scenario.as_str() {
                            "decode_only" => {
                                for &decode_steps in &profile.decode_steps {
                                    cases.push(BenchCase {
                                        model: model.clone(),
                                        backend: backend.clone(),
                                        scenario: scenario.clone(),
                                        batch_size,
                                        token_chunk_size,
                                        seq_len: None,
                                        decode_steps: Some(decode_steps),
                                        warmup_runs: profile.warmup_runs,
                                        repeats: profile.repeats,
                                    });
                                }
                            }
                            "prefill_uniform" | "prefill_mixed" => {
                                for &seq_len in &profile.seq_lens {
                                    cases.push(BenchCase {
                                        model: model.clone(),
                                        backend: backend.clone(),
                                        scenario: scenario.clone(),
                                        batch_size,
                                        token_chunk_size,
                                        seq_len: Some(seq_len),
                                        decode_steps: None,
                                        warmup_runs: profile.warmup_runs,
                                        repeats: profile.repeats,
                                    });
                                }
                            }
                            _ => {
                                println!("[bench] Unknown scenario: {}", scenario);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(cases)
}

// =============================================================================
// TESTS
// =============================================================================

/// Smoke test: load config, select profile, expand cases
#[test]
#[ignore]
fn bench_smoke() {
    println!("\n=== web-rwkv Benchmark Runner ===\n");

    // Load configuration
    let config = match load_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Failed to load config: {}", e);
            eprintln!("[bench] Make sure {} exists or set WEB_RWKV_BENCH_CONFIG", DEFAULT_CONFIG_PATH);
            return;
        }
    };

    println!("[bench] Config schema version: {}", config.schema_version);
    println!("[bench] Available profiles: {:?}", config.profiles.keys().collect::<Vec<_>>());
    println!("[bench] Available models: {:?}", config.models.iter().map(|m| &m.model_name).collect::<Vec<_>>());
    println!("[bench] Available backends: {:?}", config.backends.iter().map(|b| &b.backend_id).collect::<Vec<_>>());

    // Get selected profile
    let profile_name = get_profile_name();
    println!("\n[bench] Selected profile: {}", profile_name);

    let profile = match get_profile(&config, &profile_name) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[bench] {}", e);
            return;
        }
    };

    println!("[bench] Profile description: {}", profile.description);
    println!("[bench] Models: {:?}", profile.models);
    println!("[bench] Backends: {:?}", profile.backends);
    println!("[bench] Scenarios: {:?}", profile.scenarios);
    println!("[bench] Batch sizes: {:?}", profile.batch_sizes);
    println!("[bench] Token chunk sizes: {:?}", profile.token_chunk_sizes);

    // Expand to benchmark cases
    let cases = match expand_profile(&config, profile) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Failed to expand profile: {}", e);
            return;
        }
    };

    println!("\n[bench] Expanded {} benchmark cases:", cases.len());
    for (i, case) in cases.iter().enumerate() {
        println!(
            "[bench]   {}. {} (batch={}, chunk={}, seq={:?}, steps={:?})",
            i + 1,
            case.case_id(),
            case.batch_size,
            case.token_chunk_size,
            case.seq_len,
            case.decode_steps,
        );
    }

    // Output configuration
    println!("\n[bench] Output configuration:");
    println!("[bench]   Directory: {}", config.output.directory);
    println!("[bench]   Pattern: {}", config.output.filename_pattern);
    println!("[bench]   Mode: {}", config.output.mode);

    // Limits
    println!("\n[bench] Limits:");
    println!("[bench]   Max total cases: {}", config.limits.max_total_cases);
    println!("[bench]   Max runtime: {}s", config.limits.max_runtime_seconds);
    println!("[bench]   Fail fast: {}", config.limits.fail_fast);
    println!("[bench]   Case timeout: {}s", config.limits.case_timeout_seconds);

    println!("\n[bench] Harness ready. Sweep execution will be implemented in subsequent tickets.");
    println!("[bench] (bd-2ey.7: Sweep execution engine)");
    println!("[bench] (bd-2ey.8: JSONL writer)");
}

/// Test that validates config parsing
#[test]
#[ignore]
fn bench_validate_config() {
    println!("\n=== Config Validation ===\n");

    let config = match load_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Config validation failed: {}", e);
            panic!("Config validation failed");
        }
    };

    // Validate schema version
    assert!(config.schema_version >= 1, "Schema version must be >= 1");

    // Validate at least one profile exists
    assert!(!config.profiles.is_empty(), "At least one profile required");

    // Validate at least one model exists
    assert!(!config.models.is_empty(), "At least one model required");

    // Validate at least one backend exists
    assert!(!config.backends.is_empty(), "At least one backend required");

    // Validate model paths exist (warning only)
    for model in &config.models {
        let path = Path::new(&model.path);
        if !path.exists() {
            println!(
                "[bench] WARNING: Model file not found: {} ({})",
                model.model_name, model.path
            );
        } else {
            println!("[bench] OK: {} -> {}", model.model_name, model.path);
        }
    }

    // Validate profile references
    for (name, profile) in &config.profiles {
        for model_name in &profile.models {
            let found = config.models.iter().any(|m| &m.model_name == model_name);
            assert!(found, "Profile '{}' references unknown model: {}", name, model_name);
        }
        for backend_id in &profile.backends {
            let found = config.backends.iter().any(|b| &b.backend_id == backend_id);
            assert!(found, "Profile '{}' references unknown backend: {}", name, backend_id);
        }
    }

    println!("\n[bench] Config validation passed!");
}
