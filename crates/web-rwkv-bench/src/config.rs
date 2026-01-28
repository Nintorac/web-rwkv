//! Configuration types for benchmark skip conditions and limits.
//!
//! These types are designed to be deserialized from the benchmark config.yaml file.

use serde::{Deserialize, Serialize};

/// Configuration for skip conditions that determine when benchmark cases should be skipped.
///
/// Skip conditions help avoid running cases that would fail or are not meaningful,
/// such as cases where batch_size exceeds the model's maximum supported batch size.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkipConditions {
    /// Skip if batch_size exceeds model's max_batch_size
    #[serde(default)]
    pub skip_batch_exceeds_model_max: bool,

    /// Skip if token_chunk_size exceeds model's max_token_chunk_size
    #[serde(default)]
    pub skip_chunk_exceeds_model_max: bool,

    /// Skip combinations known to fail (e.g., hip + certain models)
    #[serde(default)]
    pub skip_known_failures: bool,

    /// Skip if estimated memory exceeds available GPU memory
    #[serde(default)]
    pub skip_oom_predicted: bool,

    /// Custom skip rules (evaluated in order)
    #[serde(default)]
    pub custom_rules: Vec<CustomRule>,
}

/// A custom skip rule with a name, description, and condition expression.
///
/// Condition expressions support simple comparisons and logical operators:
/// - Variables: `batch_size`, `seq_len`, `token_chunk_size`, `model_name`, `backend_id`
/// - Comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
/// - Logical operators: `and`, `or`, `not`
/// - String matching: `==` for exact match
///
/// # Examples
///
/// ```yaml
/// - name: "skip_large_batch_large_seq"
///   description: "Skip batch_size > 16 when seq_len > 1024"
///   condition: "batch_size > 16 and seq_len > 1024"
///
/// - name: "skip_puzzle15_hip"
///   description: "Skip puzzle15 model on hip backend"
///   condition: "model_name == 'rwkv_puzzle15' and backend_id == 'hip'"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRule {
    /// Unique name for the rule (used for logging and skip reasons)
    pub name: String,

    /// Human-readable description of what this rule does
    #[serde(default)]
    pub description: String,

    /// Condition expression to evaluate
    pub condition: String,
}

/// Global limits for benchmark execution.
///
/// These limits help control resource usage and provide fail-fast behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    /// Maximum total number of benchmark cases to run (0 = unlimited)
    #[serde(default = "default_max_total_cases")]
    pub max_total_cases: u32,

    /// Maximum total runtime in seconds (0 = unlimited)
    #[serde(default)]
    pub max_runtime_seconds: u64,

    /// Stop on first error vs continue and log errors
    #[serde(default)]
    pub fail_fast: bool,

    /// Maximum errors before aborting (0 = unlimited)
    #[serde(default)]
    pub max_errors: u32,

    /// Per-case timeout in seconds
    #[serde(default = "default_case_timeout")]
    pub case_timeout_seconds: u64,

    /// Maximum memory usage threshold (percentage of available GPU memory)
    #[serde(default = "default_max_memory_percent")]
    pub max_memory_percent: u32,
}

fn default_max_total_cases() -> u32 {
    10000
}

fn default_case_timeout() -> u64 {
    300
}

fn default_max_memory_percent() -> u32 {
    90
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_total_cases: default_max_total_cases(),
            max_runtime_seconds: 0,
            fail_fast: false,
            max_errors: 0,
            case_timeout_seconds: default_case_timeout(),
            max_memory_percent: default_max_memory_percent(),
        }
    }
}

/// Model configuration data needed for skip condition evaluation.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model identifier/name
    pub model_name: String,
    /// Maximum supported batch size (None = unlimited)
    pub max_batch_size: Option<u32>,
    /// Maximum supported token chunk size (None = unlimited)
    pub max_token_chunk_size: Option<u32>,
}

/// Backend configuration data needed for skip condition evaluation.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend identifier (e.g., "wgpu", "hip")
    pub backend_id: String,
}

/// Benchmark case parameters needed for skip condition evaluation.
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    /// Batch size for this case
    pub batch_size: u32,
    /// Token chunk size for this case
    pub token_chunk_size: u32,
    /// Sequence length (for prefill scenarios)
    pub seq_len: Option<u32>,
    /// Decode steps (for decode scenarios)
    pub decode_steps: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_conditions_default() {
        let skip = SkipConditions::default();
        assert!(!skip.skip_batch_exceeds_model_max);
        assert!(!skip.skip_chunk_exceeds_model_max);
        assert!(!skip.skip_known_failures);
        assert!(!skip.skip_oom_predicted);
        assert!(skip.custom_rules.is_empty());
    }

    #[test]
    fn test_limits_default() {
        let limits = Limits::default();
        assert_eq!(limits.max_total_cases, 10000);
        assert_eq!(limits.max_runtime_seconds, 0);
        assert!(!limits.fail_fast);
        assert_eq!(limits.max_errors, 0);
        assert_eq!(limits.case_timeout_seconds, 300);
        assert_eq!(limits.max_memory_percent, 90);
    }

    #[test]
    fn test_skip_conditions_deserialize() {
        let yaml = r#"
skip_batch_exceeds_model_max: true
skip_chunk_exceeds_model_max: true
skip_known_failures: false
skip_oom_predicted: true
custom_rules:
  - name: "test_rule"
    description: "A test rule"
    condition: "batch_size > 8"
"#;
        let skip: SkipConditions = serde_yaml::from_str(yaml).unwrap();
        assert!(skip.skip_batch_exceeds_model_max);
        assert!(skip.skip_chunk_exceeds_model_max);
        assert!(!skip.skip_known_failures);
        assert!(skip.skip_oom_predicted);
        assert_eq!(skip.custom_rules.len(), 1);
        assert_eq!(skip.custom_rules[0].name, "test_rule");
    }

    #[test]
    fn test_limits_deserialize() {
        let yaml = r#"
max_total_cases: 500
max_runtime_seconds: 1800
fail_fast: true
max_errors: 10
case_timeout_seconds: 120
max_memory_percent: 80
"#;
        let limits: Limits = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(limits.max_total_cases, 500);
        assert_eq!(limits.max_runtime_seconds, 1800);
        assert!(limits.fail_fast);
        assert_eq!(limits.max_errors, 10);
        assert_eq!(limits.case_timeout_seconds, 120);
        assert_eq!(limits.max_memory_percent, 80);
    }
}
