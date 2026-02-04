//! Error classification for benchmark execution.
//!
//! This module provides:
//! - [`BenchError`]: A unified error type for benchmark operations
//! - [`classify_error`]: Function to classify errors into `ErrorKind` categories
//!
//! # Error Classification
//!
//! Errors are classified into the following categories:
//! - `OutOfMemory`: GPU memory allocation failures
//! - `DeviceLost`: GPU device lost or disconnected
//! - `UnsupportedBackend`: Backend not available on this system
//! - `ModelLoadFailed`: Model file not found or failed to load
//! - `InferenceError`: Error during inference execution
//! - `Timeout`: Operation exceeded time limit
//! - `Other`: Unclassified errors
//!
//! # Continue-on-Error Policy
//!
//! The benchmark runner uses a continue-on-error policy by default:
//! - Errors are caught and classified
//! - A `status=error` record is written to JSONL with error details
//! - The sweep continues to the next case
//! - The `fail_fast` option can override this behavior
//!
//! # Example
//!
//! ```
//! use web_rwkv_bench::error::{BenchError, classify_error};
//! use web_rwkv_bench::jsonl::ErrorKind;
//!
//! // Classify an error
//! let error = BenchError::ModelLoad {
//!     path: "model.st".to_string(),
//!     message: "file not found".to_string(),
//! };
//!
//! let (kind, message) = classify_error(&error);
//! assert_eq!(kind, ErrorKind::ModelLoadFailed);
//! ```

use std::fmt;
use std::time::Duration;

use crate::jsonl::ErrorKind;

/// Unified error type for benchmark operations.
///
/// This enum captures all error conditions that can occur during benchmark
/// execution, from model loading through inference and measurement.
#[derive(Debug)]
pub enum BenchError {
    /// Out of memory error (GPU allocation failed)
    OutOfMemory {
        /// Descriptive message about the allocation that failed
        message: String,
    },

    /// Device lost error (GPU device disconnected or reset)
    DeviceLost {
        /// Backend that lost the device
        backend: String,
        /// Additional context about the error
        message: String,
    },

    /// Backend not supported on this system
    UnsupportedBackend {
        /// The backend that was requested
        backend: String,
        /// Reason why it's not supported
        reason: String,
    },

    /// Model file not found or failed to load
    ModelLoad {
        /// Path to the model file
        path: String,
        /// Error message from the loader
        message: String,
    },

    /// Inference error during benchmark execution
    Inference {
        /// The operation that failed
        operation: String,
        /// Error message
        message: String,
    },

    /// Operation timed out
    Timeout {
        /// The operation that timed out
        operation: String,
        /// The timeout duration
        timeout: Duration,
    },

    /// IO error
    Io {
        /// The operation that failed
        operation: String,
        /// Error message
        message: String,
    },

    /// Configuration error
    Config {
        /// What configuration was invalid
        what: String,
        /// Error message
        message: String,
    },

    /// Generic/other error
    Other {
        /// Error message
        message: String,
    },
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::OutOfMemory { message } => {
                write!(f, "out of memory: {}", message)
            }
            BenchError::DeviceLost { backend, message } => {
                write!(f, "device lost ({}): {}", backend, message)
            }
            BenchError::UnsupportedBackend { backend, reason } => {
                write!(f, "unsupported backend '{}': {}", backend, reason)
            }
            BenchError::ModelLoad { path, message } => {
                write!(f, "failed to load model '{}': {}", path, message)
            }
            BenchError::Inference { operation, message } => {
                write!(f, "inference error in {}: {}", operation, message)
            }
            BenchError::Timeout { operation, timeout } => {
                write!(f, "{} timed out after {:?}", operation, timeout)
            }
            BenchError::Io { operation, message } => {
                write!(f, "IO error in {}: {}", operation, message)
            }
            BenchError::Config { what, message } => {
                write!(f, "config error in {}: {}", what, message)
            }
            BenchError::Other { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for BenchError {}

/// Classify an error into an ErrorKind category.
///
/// Returns a tuple of (ErrorKind, human-readable message) that can be
/// written to the JSONL output.
///
/// # Classification Rules
///
/// The function inspects the error variant and maps it to the appropriate
/// `ErrorKind`:
///
/// - `BenchError::OutOfMemory` -> `ErrorKind::OutOfMemory`
/// - `BenchError::DeviceLost` -> `ErrorKind::DeviceLost`
/// - `BenchError::UnsupportedBackend` -> `ErrorKind::UnsupportedBackend`
/// - `BenchError::ModelLoad` -> `ErrorKind::ModelLoadFailed`
/// - `BenchError::Inference` -> `ErrorKind::InferenceError`
/// - `BenchError::Timeout` -> `ErrorKind::Timeout`
/// - Others -> `ErrorKind::Other`
///
/// # Example
///
/// ```
/// use web_rwkv_bench::error::{BenchError, classify_error};
/// use web_rwkv_bench::jsonl::ErrorKind;
///
/// let error = BenchError::OutOfMemory {
///     message: "failed to allocate 4GB tensor".to_string(),
/// };
///
/// let (kind, message) = classify_error(&error);
/// assert_eq!(kind, ErrorKind::OutOfMemory);
/// assert!(message.contains("4GB"));
/// ```
pub fn classify_error(error: &BenchError) -> (ErrorKind, String) {
    match error {
        BenchError::OutOfMemory { message } => {
            (ErrorKind::OutOfMemory, format!("OOM: {}", message))
        }
        BenchError::DeviceLost { backend, message } => (
            ErrorKind::DeviceLost,
            format!("Device lost ({}): {}", backend, message),
        ),
        BenchError::UnsupportedBackend { backend, reason } => (
            ErrorKind::UnsupportedBackend,
            format!("Backend '{}' not supported: {}", backend, reason),
        ),
        BenchError::ModelLoad { path, message } => (
            ErrorKind::ModelLoadFailed,
            format!("Model load failed '{}': {}", path, message),
        ),
        BenchError::Inference { operation, message } => (
            ErrorKind::InferenceError,
            format!("Inference error in {}: {}", operation, message),
        ),
        BenchError::Timeout { operation, timeout } => (
            ErrorKind::Timeout,
            format!("{} timed out after {:?}", operation, timeout),
        ),
        BenchError::Io { operation, message } => (
            ErrorKind::Other,
            format!("IO error in {}: {}", operation, message),
        ),
        BenchError::Config { what, message } => (
            ErrorKind::Other,
            format!("Config error in {}: {}", what, message),
        ),
        BenchError::Other { message } => (ErrorKind::Other, message.clone()),
    }
}

/// Classify an error from a string message by pattern matching.
///
/// This function attempts to classify errors based on common patterns
/// in error messages. It's useful when working with generic error types
/// that don't provide structured error information.
///
/// # Pattern Matching
///
/// The function looks for these patterns in the error message:
/// - "out of memory", "oom", "allocation failed" -> `OutOfMemory`
/// - "device lost", "device removed" -> `DeviceLost`
/// - "not supported", "unsupported", "unavailable" -> `UnsupportedBackend`
/// - "file not found", "no such file" -> `ModelLoadFailed`
/// - "timeout", "timed out" -> `Timeout`
///
/// # Example
///
/// ```
/// use web_rwkv_bench::error::classify_error_message;
/// use web_rwkv_bench::jsonl::ErrorKind;
///
/// let (kind, _) = classify_error_message("GPU allocation failed: out of memory");
/// assert_eq!(kind, ErrorKind::OutOfMemory);
/// ```
pub fn classify_error_message(message: &str) -> (ErrorKind, String) {
    let lower = message.to_lowercase();

    // Check for OOM patterns
    if lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("allocation failed")
        || lower.contains("allocate failed")
        || lower.contains("memory allocation")
        || lower.contains("insufficient memory")
    {
        return (ErrorKind::OutOfMemory, message.to_string());
    }

    // Check for device lost patterns
    if lower.contains("device_lost")
        || lower.contains("device lost")
        || (lower.contains("device") && lower.contains("removed"))
        || lower.contains("device reset")
        || lower.contains("gpu hung")
        || lower.contains("command buffer")
    {
        return (ErrorKind::DeviceLost, message.to_string());
    }

    // Check for unsupported backend patterns
    if lower.contains("not supported")
        || lower.contains("unsupported")
        || lower.contains("unavailable")
        || lower.contains("no adapter")
        || lower.contains("no device")
        || lower.contains("backend not found")
    {
        return (ErrorKind::UnsupportedBackend, message.to_string());
    }

    // Check for file not found patterns
    if lower.contains("file not found")
        || lower.contains("no such file")
        || lower.contains("cannot find")
        || lower.contains("does not exist")
        || lower.contains("failed to open")
        || lower.contains("failed to load")
    {
        return (ErrorKind::ModelLoadFailed, message.to_string());
    }

    // Check for timeout patterns
    if lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("deadline exceeded")
    {
        return (ErrorKind::Timeout, message.to_string());
    }

    // Check for inference-specific patterns
    if lower.contains("inference")
        || lower.contains("infer")
        || lower.contains("forward pass")
        || lower.contains("tensor")
        || lower.contains("shape mismatch")
    {
        return (ErrorKind::InferenceError, message.to_string());
    }

    // Default to Other
    (ErrorKind::Other, message.to_string())
}

/// Result type for benchmark operations.
pub type BenchResult<T> = Result<T, BenchError>;

/// A wrapper that captures an error along with context for JSONL recording.
///
/// This struct is used to pass error information through the benchmark
/// pipeline so it can be recorded with full context.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The classified error kind
    pub kind: ErrorKind,
    /// Human-readable error message
    pub message: String,
    /// Additional context (e.g., what case was running)
    pub context: Option<String>,
}

impl ErrorContext {
    /// Create a new error context from a BenchError.
    pub fn from_bench_error(error: &BenchError, context: Option<String>) -> Self {
        let (kind, message) = classify_error(error);
        Self {
            kind,
            message,
            context,
        }
    }

    /// Create a new error context from a string message.
    pub fn from_message(message: &str, context: Option<String>) -> Self {
        let (kind, msg) = classify_error_message(message);
        Self {
            kind,
            message: msg,
            context,
        }
    }

    /// Create an error context for an unknown/other error.
    pub fn other(message: String) -> Self {
        Self {
            kind: ErrorKind::Other,
            message,
            context: None,
        }
    }

    /// Get the full error message including context.
    pub fn full_message(&self) -> String {
        match &self.context {
            Some(ctx) => format!("{} ({})", self.message, ctx),
            None => self.message.clone(),
        }
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_bench_error_oom() {
        let error = BenchError::OutOfMemory {
            message: "failed to allocate 4GB tensor".to_string(),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::OutOfMemory);
        assert!(message.contains("4GB"));
    }

    #[test]
    fn test_classify_bench_error_device_lost() {
        let error = BenchError::DeviceLost {
            backend: "Vulkan".to_string(),
            message: "device was reset".to_string(),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::DeviceLost);
        assert!(message.contains("Vulkan"));
    }

    #[test]
    fn test_classify_bench_error_unsupported() {
        let error = BenchError::UnsupportedBackend {
            backend: "hip".to_string(),
            reason: "no AMD GPU found".to_string(),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::UnsupportedBackend);
        assert!(message.contains("hip"));
    }

    #[test]
    fn test_classify_bench_error_model_load() {
        let error = BenchError::ModelLoad {
            path: "/path/to/model.st".to_string(),
            message: "file not found".to_string(),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::ModelLoadFailed);
        assert!(message.contains("/path/to/model.st"));
    }

    #[test]
    fn test_classify_bench_error_inference() {
        let error = BenchError::Inference {
            operation: "forward pass".to_string(),
            message: "tensor shape mismatch".to_string(),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::InferenceError);
        assert!(message.contains("forward pass"));
    }

    #[test]
    fn test_classify_bench_error_timeout() {
        let error = BenchError::Timeout {
            operation: "inference".to_string(),
            timeout: Duration::from_secs(300),
        };
        let (kind, message) = classify_error(&error);
        assert_eq!(kind, ErrorKind::Timeout);
        assert!(message.contains("inference"));
    }

    #[test]
    fn test_classify_error_message_oom() {
        let (kind, _) = classify_error_message("GPU allocation failed: out of memory");
        assert_eq!(kind, ErrorKind::OutOfMemory);

        let (kind, _) = classify_error_message("OOM error during tensor creation");
        assert_eq!(kind, ErrorKind::OutOfMemory);

        let (kind, _) = classify_error_message("Insufficient memory for operation");
        assert_eq!(kind, ErrorKind::OutOfMemory);
    }

    #[test]
    fn test_classify_error_message_device_lost() {
        let (kind, _) = classify_error_message("VK_ERROR_DEVICE_LOST");
        assert_eq!(kind, ErrorKind::DeviceLost);

        let (kind, _) = classify_error_message("Device was removed");
        assert_eq!(kind, ErrorKind::DeviceLost);
    }

    #[test]
    fn test_classify_error_message_unsupported() {
        let (kind, _) = classify_error_message("Vulkan backend not supported on this system");
        assert_eq!(kind, ErrorKind::UnsupportedBackend);

        let (kind, _) = classify_error_message("No adapter found for requested backend");
        assert_eq!(kind, ErrorKind::UnsupportedBackend);
    }

    #[test]
    fn test_classify_error_message_file_not_found() {
        let (kind, _) = classify_error_message("Model file not found: /path/to/model.st");
        assert_eq!(kind, ErrorKind::ModelLoadFailed);

        let (kind, _) = classify_error_message("Failed to load model weights");
        assert_eq!(kind, ErrorKind::ModelLoadFailed);
    }

    #[test]
    fn test_classify_error_message_timeout() {
        let (kind, _) = classify_error_message("Operation timed out after 300s");
        assert_eq!(kind, ErrorKind::Timeout);
    }

    #[test]
    fn test_classify_error_message_inference() {
        let (kind, _) = classify_error_message("Tensor shape mismatch in forward pass");
        assert_eq!(kind, ErrorKind::InferenceError);
    }

    #[test]
    fn test_classify_error_message_other() {
        let (kind, _) = classify_error_message("Unknown error occurred");
        assert_eq!(kind, ErrorKind::Other);
    }

    #[test]
    fn test_bench_error_display() {
        let error = BenchError::OutOfMemory {
            message: "test".to_string(),
        };
        assert_eq!(error.to_string(), "out of memory: test");

        let error = BenchError::ModelLoad {
            path: "model.st".to_string(),
            message: "not found".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "failed to load model 'model.st': not found"
        );
    }

    #[test]
    fn test_error_context_from_bench_error() {
        let error = BenchError::OutOfMemory {
            message: "allocation failed".to_string(),
        };
        let ctx = ErrorContext::from_bench_error(&error, Some("case_123".to_string()));

        assert_eq!(ctx.kind, ErrorKind::OutOfMemory);
        assert!(ctx.message.contains("allocation failed"));
        assert!(ctx.full_message().contains("case_123"));
    }

    #[test]
    fn test_error_context_from_message() {
        let ctx = ErrorContext::from_message("out of memory error", None);
        assert_eq!(ctx.kind, ErrorKind::OutOfMemory);
    }

    #[test]
    fn test_error_context_other() {
        let ctx = ErrorContext::other("something went wrong".to_string());
        assert_eq!(ctx.kind, ErrorKind::Other);
        assert_eq!(ctx.message, "something went wrong");
    }
}
