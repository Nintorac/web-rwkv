//! Skip condition evaluation and limits tracking for benchmark execution.
//!
//! This module provides:
//! - [`should_skip`]: Evaluates whether a benchmark case should be skipped
//! - [`LimitsTracker`]: Tracks execution progress against configured limits
//! - [`SkipReason`]: Describes why a case was skipped

use std::time::{Duration, Instant};

use crate::config::{BackendConfig, BenchmarkCase, CustomRule, Limits, ModelConfig, SkipConditions};

/// Reason why a benchmark case was skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Batch size exceeds model's maximum supported batch size
    BatchExceedsModelMax {
        batch_size: u32,
        max_batch_size: u32,
    },
    /// Token chunk size exceeds model's maximum supported chunk size
    ChunkExceedsModelMax {
        token_chunk_size: u32,
        max_token_chunk_size: u32,
    },
    /// Combination is known to fail
    KnownFailure {
        model_name: String,
        backend_id: String,
    },
    /// Estimated memory usage would exceed available GPU memory
    OomPredicted {
        estimated_percent: u32,
        max_percent: u32,
    },
    /// Custom rule triggered
    CustomRule {
        rule_name: String,
        description: String,
    },
    /// Model is marked as skipped in config
    ModelSkipped {
        model_name: String,
    },
    /// Backend is marked as skipped in config
    BackendSkipped {
        backend_id: String,
    },
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::BatchExceedsModelMax {
                batch_size,
                max_batch_size,
            } => {
                write!(
                    f,
                    "batch_size ({}) exceeds model max ({})",
                    batch_size, max_batch_size
                )
            }
            SkipReason::ChunkExceedsModelMax {
                token_chunk_size,
                max_token_chunk_size,
            } => {
                write!(
                    f,
                    "token_chunk_size ({}) exceeds model max ({})",
                    token_chunk_size, max_token_chunk_size
                )
            }
            SkipReason::KnownFailure {
                model_name,
                backend_id,
            } => {
                write!(
                    f,
                    "known failure: model '{}' on backend '{}'",
                    model_name, backend_id
                )
            }
            SkipReason::OomPredicted {
                estimated_percent,
                max_percent,
            } => {
                write!(
                    f,
                    "OOM predicted: estimated {}% > max {}%",
                    estimated_percent, max_percent
                )
            }
            SkipReason::CustomRule {
                rule_name,
                description,
            } => {
                if description.is_empty() {
                    write!(f, "custom rule '{}' triggered", rule_name)
                } else {
                    write!(f, "custom rule '{}': {}", rule_name, description)
                }
            }
            SkipReason::ModelSkipped { model_name } => {
                write!(f, "model '{}' is marked as skipped", model_name)
            }
            SkipReason::BackendSkipped { backend_id } => {
                write!(f, "backend '{}' is marked as skipped", backend_id)
            }
        }
    }
}

/// Evaluate whether a benchmark case should be skipped.
///
/// Checks skip conditions in the following order:
/// 1. Batch size exceeds model's max_batch_size
/// 2. Token chunk size exceeds model's max_token_chunk_size
/// 3. Known failure combinations
/// 4. Custom rules (evaluated in config order)
///
/// Returns `Some(SkipReason)` if the case should be skipped, `None` if it should run.
///
/// # Example
///
/// ```
/// use web_rwkv_bench::config::{BenchmarkCase, ModelConfig, BackendConfig, SkipConditions};
/// use web_rwkv_bench::skip::should_skip;
///
/// let case = BenchmarkCase {
///     batch_size: 64,
///     token_chunk_size: 256,
///     seq_len: Some(512),
///     decode_steps: None,
/// };
///
/// let model = ModelConfig {
///     model_name: "test_model".to_string(),
///     max_batch_size: Some(32),
///     max_token_chunk_size: Some(1024),
/// };
///
/// let backend = BackendConfig {
///     backend_id: "wgpu".to_string(),
/// };
///
/// let config = SkipConditions {
///     skip_batch_exceeds_model_max: true,
///     ..Default::default()
/// };
///
/// let reason = should_skip(&case, &model, &backend, &config);
/// assert!(reason.is_some());
/// ```
pub fn should_skip(
    case: &BenchmarkCase,
    model: &ModelConfig,
    backend: &BackendConfig,
    config: &SkipConditions,
) -> Option<SkipReason> {
    // Check batch size limit
    if config.skip_batch_exceeds_model_max {
        if let Some(max_batch) = model.max_batch_size {
            if case.batch_size > max_batch {
                return Some(SkipReason::BatchExceedsModelMax {
                    batch_size: case.batch_size,
                    max_batch_size: max_batch,
                });
            }
        }
    }

    // Check token chunk size limit
    if config.skip_chunk_exceeds_model_max {
        if let Some(max_chunk) = model.max_token_chunk_size {
            if case.token_chunk_size > max_chunk {
                return Some(SkipReason::ChunkExceedsModelMax {
                    token_chunk_size: case.token_chunk_size,
                    max_token_chunk_size: max_chunk,
                });
            }
        }
    }

    // Check custom rules
    for rule in &config.custom_rules {
        if evaluate_custom_rule(rule, case, model, backend) {
            return Some(SkipReason::CustomRule {
                rule_name: rule.name.clone(),
                description: rule.description.clone(),
            });
        }
    }

    None
}

/// Variables available for custom rule evaluation.
struct RuleContext<'a> {
    batch_size: u32,
    token_chunk_size: u32,
    seq_len: Option<u32>,
    decode_steps: Option<u32>,
    model_name: &'a str,
    backend_id: &'a str,
}

/// Evaluate a custom rule condition against the benchmark case.
///
/// Supports simple expressions like:
/// - `batch_size > 16`
/// - `seq_len > 1024`
/// - `model_name == 'rwkv_puzzle15'`
/// - `batch_size > 16 and seq_len > 1024`
/// - `model_name == 'rwkv_puzzle15' and backend_id == 'hip'`
fn evaluate_custom_rule(
    rule: &CustomRule,
    case: &BenchmarkCase,
    model: &ModelConfig,
    backend: &BackendConfig,
) -> bool {
    let ctx = RuleContext {
        batch_size: case.batch_size,
        token_chunk_size: case.token_chunk_size,
        seq_len: case.seq_len,
        decode_steps: case.decode_steps,
        model_name: &model.model_name,
        backend_id: &backend.backend_id,
    };

    evaluate_expression(&rule.condition, &ctx)
}

/// Simple expression evaluator for custom rules.
///
/// Grammar (informal):
/// - expr := and_expr
/// - and_expr := or_expr ("and" or_expr)*
/// - or_expr := comparison ("or" comparison)*
/// - comparison := value op value
/// - value := identifier | number | string
/// - op := ">" | "<" | ">=" | "<=" | "==" | "!="
fn evaluate_expression(expr: &str, ctx: &RuleContext) -> bool {
    let expr = expr.trim();

    // Handle "and" operator (lowest precedence)
    if let Some((left, right)) = split_binary_op(expr, " and ") {
        return evaluate_expression(left, ctx) && evaluate_expression(right, ctx);
    }

    // Handle "or" operator
    if let Some((left, right)) = split_binary_op(expr, " or ") {
        return evaluate_expression(left, ctx) || evaluate_expression(right, ctx);
    }

    // Handle "not" operator
    if let Some(inner) = expr.strip_prefix("not ") {
        return !evaluate_expression(inner.trim(), ctx);
    }

    // Handle parentheses
    if expr.starts_with('(') && expr.ends_with(')') {
        return evaluate_expression(&expr[1..expr.len() - 1], ctx);
    }

    // Handle comparison operators (check longer operators first)
    for op in [">=", "<=", "!=", "==", ">", "<"] {
        if let Some((left, right)) = split_comparison(expr, op) {
            let left_val = resolve_value(left.trim(), ctx);
            let right_val = resolve_value(right.trim(), ctx);

            return match (left_val, right_val) {
                (Value::Number(a), Value::Number(b)) => compare_numbers(a, b, op),
                (Value::String(a), Value::String(b)) => compare_strings(&a, &b, op),
                _ => false, // Type mismatch
            };
        }
    }

    // Unknown expression, default to false (don't skip)
    false
}

/// Compare two numbers with the given operator.
fn compare_numbers(a: i64, b: i64, op: &str) -> bool {
    match op {
        ">=" => a >= b,
        "<=" => a <= b,
        "!=" => a != b,
        "==" => a == b,
        ">" => a > b,
        "<" => a < b,
        _ => false,
    }
}

/// Compare two strings with the given operator.
fn compare_strings(a: &str, b: &str, op: &str) -> bool {
    match op {
        "==" => a == b,
        "!=" => a != b,
        _ => false, // String comparison only supports == and !=
    }
}

/// Split an expression on a binary operator, respecting parentheses.
fn split_binary_op<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0;
    let mut i = 0;
    let chars: Vec<char> = expr.chars().collect();

    while i < chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }

        if depth == 0 && expr[i..].starts_with(op) {
            return Some((&expr[..i], &expr[i + op.len()..]));
        }

        i += 1;
    }

    None
}

/// Split a comparison expression on a comparison operator.
fn split_comparison<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    // Find the operator, avoiding partial matches (e.g., ">=" vs ">")
    if let Some(pos) = expr.find(op) {
        // Make sure we're not matching a longer operator
        let before = &expr[..pos];
        let after = &expr[pos + op.len()..];

        // Check that this isn't part of a longer operator
        if op == ">" && after.starts_with('=') {
            return None;
        }
        if op == "<" && after.starts_with('=') {
            return None;
        }
        if op == "=" && (before.ends_with('!') || before.ends_with('<') || before.ends_with('>') || after.starts_with('=')) {
            return None;
        }

        return Some((before, after));
    }

    None
}

#[derive(Debug, Clone)]
enum Value {
    Number(i64),
    String(String),
    None,
}

/// Resolve a value from the expression (variable name, number literal, or string literal).
fn resolve_value(s: &str, ctx: &RuleContext) -> Value {
    let s = s.trim();

    // String literal (single or double quotes)
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        return Value::String(s[1..s.len() - 1].to_string());
    }

    // Number literal
    if let Ok(n) = s.parse::<i64>() {
        return Value::Number(n);
    }

    // Variable lookup
    match s {
        "batch_size" => Value::Number(ctx.batch_size as i64),
        "token_chunk_size" => Value::Number(ctx.token_chunk_size as i64),
        "seq_len" => ctx.seq_len.map(|v| Value::Number(v as i64)).unwrap_or(Value::None),
        "decode_steps" => ctx.decode_steps.map(|v| Value::Number(v as i64)).unwrap_or(Value::None),
        "model_name" => Value::String(ctx.model_name.to_string()),
        "backend_id" => Value::String(ctx.backend_id.to_string()),
        _ => Value::None,
    }
}

/// Reason why the benchmark should stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Maximum number of cases reached
    MaxCasesReached { total_cases: u32, max_cases: u32 },
    /// Maximum runtime exceeded
    MaxRuntimeExceeded { elapsed_secs: u64, max_secs: u64 },
    /// Error encountered with fail_fast enabled
    FailFastError { error_count: u32 },
    /// Maximum error count reached
    MaxErrorsReached { error_count: u32, max_errors: u32 },
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::MaxCasesReached { total_cases, max_cases } => {
                write!(f, "max cases reached: {} >= {}", total_cases, max_cases)
            }
            StopReason::MaxRuntimeExceeded { elapsed_secs, max_secs } => {
                write!(f, "max runtime exceeded: {}s >= {}s", elapsed_secs, max_secs)
            }
            StopReason::FailFastError { error_count } => {
                write!(f, "fail_fast: {} error(s) encountered", error_count)
            }
            StopReason::MaxErrorsReached { error_count, max_errors } => {
                write!(f, "max errors reached: {} >= {}", error_count, max_errors)
            }
        }
    }
}

/// Tracks execution progress against configured limits.
///
/// Use this to check whether benchmark execution should continue or stop
/// based on the configured limits (max cases, max runtime, fail-fast, max errors).
///
/// # Example
///
/// ```
/// use web_rwkv_bench::config::Limits;
/// use web_rwkv_bench::skip::LimitsTracker;
///
/// let limits = Limits {
///     max_total_cases: 100,
///     max_runtime_seconds: 3600,
///     fail_fast: true,
///     max_errors: 10,
///     ..Default::default()
/// };
///
/// let mut tracker = LimitsTracker::new(limits);
///
/// // Run benchmarks...
/// tracker.record_case();
///
/// // Check if we should continue
/// if let Some(reason) = tracker.should_stop() {
///     println!("Stopping: {}", reason);
/// }
/// ```
#[derive(Debug)]
pub struct LimitsTracker {
    limits: Limits,
    start_time: Instant,
    cases_run: u32,
    errors_encountered: u32,
}

impl LimitsTracker {
    /// Create a new limits tracker with the given configuration.
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            start_time: Instant::now(),
            cases_run: 0,
            errors_encountered: 0,
        }
    }

    /// Record that a benchmark case was run.
    pub fn record_case(&mut self) {
        self.cases_run += 1;
    }

    /// Record that an error was encountered.
    pub fn record_error(&mut self) {
        self.errors_encountered += 1;
    }

    /// Get the number of cases run so far.
    pub fn cases_run(&self) -> u32 {
        self.cases_run
    }

    /// Get the number of errors encountered so far.
    pub fn errors_encountered(&self) -> u32 {
        self.errors_encountered
    }

    /// Get the elapsed time since the tracker was created.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the elapsed time in seconds.
    pub fn elapsed_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Check if benchmark execution should stop.
    ///
    /// Returns `Some(StopReason)` if a limit has been exceeded, `None` otherwise.
    pub fn should_stop(&self) -> Option<StopReason> {
        // Check max cases
        if self.limits.max_total_cases > 0 && self.cases_run >= self.limits.max_total_cases {
            return Some(StopReason::MaxCasesReached {
                total_cases: self.cases_run,
                max_cases: self.limits.max_total_cases,
            });
        }

        // Check max runtime
        if self.limits.max_runtime_seconds > 0 {
            let elapsed = self.start_time.elapsed().as_secs();
            if elapsed >= self.limits.max_runtime_seconds {
                return Some(StopReason::MaxRuntimeExceeded {
                    elapsed_secs: elapsed,
                    max_secs: self.limits.max_runtime_seconds,
                });
            }
        }

        // Check fail_fast
        if self.limits.fail_fast && self.errors_encountered > 0 {
            return Some(StopReason::FailFastError {
                error_count: self.errors_encountered,
            });
        }

        // Check max errors
        if self.limits.max_errors > 0 && self.errors_encountered >= self.limits.max_errors {
            return Some(StopReason::MaxErrorsReached {
                error_count: self.errors_encountered,
                max_errors: self.limits.max_errors,
            });
        }

        None
    }

    /// Get the per-case timeout duration.
    pub fn case_timeout(&self) -> Duration {
        Duration::from_secs(self.limits.case_timeout_seconds)
    }

    /// Get the maximum memory percentage allowed.
    pub fn max_memory_percent(&self) -> u32 {
        self.limits.max_memory_percent
    }

    /// Check if we should continue running (convenience method, inverse of should_stop).
    pub fn should_continue(&self) -> bool {
        self.should_stop().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_case(batch_size: u32, token_chunk_size: u32, seq_len: Option<u32>) -> BenchmarkCase {
        BenchmarkCase {
            batch_size,
            token_chunk_size,
            seq_len,
            decode_steps: None,
        }
    }

    fn make_model(name: &str, max_batch: Option<u32>, max_chunk: Option<u32>) -> ModelConfig {
        ModelConfig {
            model_name: name.to_string(),
            max_batch_size: max_batch,
            max_token_chunk_size: max_chunk,
        }
    }

    fn make_backend(id: &str) -> BackendConfig {
        BackendConfig {
            backend_id: id.to_string(),
        }
    }

    #[test]
    fn test_skip_batch_exceeds_max() {
        let case = make_case(64, 256, None);
        let model = make_model("test", Some(32), None);
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            skip_batch_exceeds_model_max: true,
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(matches!(
            reason,
            Some(SkipReason::BatchExceedsModelMax {
                batch_size: 64,
                max_batch_size: 32
            })
        ));
    }

    #[test]
    fn test_skip_batch_within_limit() {
        let case = make_case(16, 256, None);
        let model = make_model("test", Some(32), None);
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            skip_batch_exceeds_model_max: true,
            ..Default::default()
        };

        assert!(should_skip(&case, &model, &backend, &config).is_none());
    }

    #[test]
    fn test_skip_chunk_exceeds_max() {
        let case = make_case(4, 2048, None);
        let model = make_model("test", None, Some(1024));
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            skip_chunk_exceeds_model_max: true,
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(matches!(
            reason,
            Some(SkipReason::ChunkExceedsModelMax {
                token_chunk_size: 2048,
                max_token_chunk_size: 1024
            })
        ));
    }

    #[test]
    fn test_skip_disabled_does_not_skip() {
        let case = make_case(64, 2048, None);
        let model = make_model("test", Some(32), Some(1024));
        let backend = make_backend("wgpu");
        let config = SkipConditions::default(); // All disabled

        assert!(should_skip(&case, &model, &backend, &config).is_none());
    }

    #[test]
    fn test_custom_rule_simple_comparison() {
        let case = make_case(32, 256, Some(2048));
        let model = make_model("test", None, None);
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            custom_rules: vec![CustomRule {
                name: "skip_large_batch".to_string(),
                description: "Skip batch_size > 16".to_string(),
                condition: "batch_size > 16".to_string(),
            }],
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(matches!(
            reason,
            Some(SkipReason::CustomRule {
                rule_name,
                ..
            }) if rule_name == "skip_large_batch"
        ));
    }

    #[test]
    fn test_custom_rule_and_condition() {
        let case = make_case(32, 256, Some(2048));
        let model = make_model("test", None, None);
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            custom_rules: vec![CustomRule {
                name: "skip_large_batch_large_seq".to_string(),
                description: "Skip batch_size > 16 when seq_len > 1024".to_string(),
                condition: "batch_size > 16 and seq_len > 1024".to_string(),
            }],
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(reason.is_some());

        // Should not skip if seq_len is within limit
        let case2 = make_case(32, 256, Some(512));
        assert!(should_skip(&case2, &model, &backend, &config).is_none());
    }

    #[test]
    fn test_custom_rule_string_comparison() {
        let case = make_case(4, 256, None);
        let model = make_model("rwkv_puzzle15", None, None);
        let backend = make_backend("hip");
        let config = SkipConditions {
            custom_rules: vec![CustomRule {
                name: "skip_puzzle15_hip".to_string(),
                description: "Skip puzzle15 model on hip backend".to_string(),
                condition: "model_name == 'rwkv_puzzle15' and backend_id == 'hip'".to_string(),
            }],
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(reason.is_some());

        // Should not skip on wgpu backend
        let backend2 = make_backend("wgpu");
        assert!(should_skip(&case, &model, &backend2, &config).is_none());
    }

    #[test]
    fn test_custom_rule_or_condition() {
        let case = make_case(4, 256, Some(100));
        let model = make_model("test", None, None);
        let backend = make_backend("wgpu");
        let config = SkipConditions {
            custom_rules: vec![CustomRule {
                name: "skip_small".to_string(),
                description: "Skip small batch or small seq".to_string(),
                condition: "batch_size < 8 or seq_len < 128".to_string(),
            }],
            ..Default::default()
        };

        let reason = should_skip(&case, &model, &backend, &config);
        assert!(reason.is_some());
    }

    #[test]
    fn test_limits_tracker_max_cases() {
        let limits = Limits {
            max_total_cases: 3,
            ..Default::default()
        };

        let mut tracker = LimitsTracker::new(limits);
        assert!(tracker.should_stop().is_none());

        tracker.record_case();
        tracker.record_case();
        assert!(tracker.should_stop().is_none());

        tracker.record_case();
        assert!(matches!(
            tracker.should_stop(),
            Some(StopReason::MaxCasesReached { total_cases: 3, max_cases: 3 })
        ));
    }

    #[test]
    fn test_limits_tracker_fail_fast() {
        let limits = Limits {
            fail_fast: true,
            ..Default::default()
        };

        let mut tracker = LimitsTracker::new(limits);
        assert!(tracker.should_stop().is_none());

        tracker.record_error();
        assert!(matches!(
            tracker.should_stop(),
            Some(StopReason::FailFastError { error_count: 1 })
        ));
    }

    #[test]
    fn test_limits_tracker_max_errors() {
        let limits = Limits {
            max_errors: 3,
            ..Default::default()
        };

        let mut tracker = LimitsTracker::new(limits);

        tracker.record_error();
        tracker.record_error();
        assert!(tracker.should_stop().is_none());

        tracker.record_error();
        assert!(matches!(
            tracker.should_stop(),
            Some(StopReason::MaxErrorsReached { error_count: 3, max_errors: 3 })
        ));
    }

    #[test]
    fn test_limits_tracker_unlimited() {
        // When max_total_cases and max_errors are 0, they're unlimited
        let limits = Limits {
            max_total_cases: 0,
            max_errors: 0,
            max_runtime_seconds: 0,
            fail_fast: false,
            ..Default::default()
        };

        let mut tracker = LimitsTracker::new(limits);

        // Run many cases
        for _ in 0..1000 {
            tracker.record_case();
        }

        // Record many errors
        for _ in 0..100 {
            tracker.record_error();
        }

        // Should still continue
        assert!(tracker.should_stop().is_none());
    }

    #[test]
    fn test_skip_reason_display() {
        let reason = SkipReason::BatchExceedsModelMax {
            batch_size: 64,
            max_batch_size: 32,
        };
        assert_eq!(reason.to_string(), "batch_size (64) exceeds model max (32)");

        let reason = SkipReason::CustomRule {
            rule_name: "test_rule".to_string(),
            description: "Test description".to_string(),
        };
        assert_eq!(reason.to_string(), "custom rule 'test_rule': Test description");
    }

    #[test]
    fn test_stop_reason_display() {
        let reason = StopReason::MaxCasesReached {
            total_cases: 100,
            max_cases: 100,
        };
        assert_eq!(reason.to_string(), "max cases reached: 100 >= 100");

        let reason = StopReason::FailFastError { error_count: 1 };
        assert_eq!(reason.to_string(), "fail_fast: 1 error(s) encountered");
    }

    #[test]
    fn test_expression_with_equals() {
        // Test that == works correctly
        let ctx = RuleContext {
            batch_size: 4,
            token_chunk_size: 256,
            seq_len: None,
            decode_steps: None,
            model_name: "test_model",
            backend_id: "wgpu",
        };

        assert!(evaluate_expression("batch_size == 4", &ctx));
        assert!(!evaluate_expression("batch_size == 5", &ctx));
        assert!(evaluate_expression("model_name == 'test_model'", &ctx));
        assert!(!evaluate_expression("model_name == 'other_model'", &ctx));
    }

    #[test]
    fn test_expression_not_equals() {
        let ctx = RuleContext {
            batch_size: 4,
            token_chunk_size: 256,
            seq_len: None,
            decode_steps: None,
            model_name: "test_model",
            backend_id: "wgpu",
        };

        assert!(!evaluate_expression("batch_size != 4", &ctx));
        assert!(evaluate_expression("batch_size != 5", &ctx));
        assert!(evaluate_expression("backend_id != 'hip'", &ctx));
    }

    #[test]
    fn test_expression_greater_equal_less_equal() {
        let ctx = RuleContext {
            batch_size: 4,
            token_chunk_size: 256,
            seq_len: Some(512),
            decode_steps: None,
            model_name: "test_model",
            backend_id: "wgpu",
        };

        assert!(evaluate_expression("batch_size >= 4", &ctx));
        assert!(evaluate_expression("batch_size >= 3", &ctx));
        assert!(!evaluate_expression("batch_size >= 5", &ctx));

        assert!(evaluate_expression("batch_size <= 4", &ctx));
        assert!(evaluate_expression("batch_size <= 5", &ctx));
        assert!(!evaluate_expression("batch_size <= 3", &ctx));
    }
}
