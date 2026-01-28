//! JSONL output writer for benchmark results.
//!
//! This module provides append-only JSONL writing for benchmark output, following
//! the schema defined in `benchmarks/schema/jsonl_schema.md`.
//!
//! # Output Format
//!
//! Each benchmark output file contains:
//! 1. One **run header record** (`type: "run"`) at the start with environment metadata
//! 2. Multiple **measurement records** (`type: "measure"`), one per (case x repeat)
//!
//! # Safety
//!
//! The writer uses append-only file operations with explicit flushing after each
//! record to ensure that partial results are preserved if the process crashes.
//!
//! # Example
//!
//! ```no_run
//! use web_rwkv_bench::jsonl::{JsonlWriter, RunHeader, MeasureRecord, Scenario, Status};
//! use web_rwkv_bench::jsonl::{HostInfo, GpuInfo, DecodeMetrics};
//! use std::path::Path;
//!
//! // Create writer and write run header
//! let mut writer = JsonlWriter::create(Path::new("results.jsonl")).unwrap();
//!
//! let header = RunHeader {
//!     run_id: "20260128T083000Z_a1b2c3".to_string(),
//!     started_at_utc: "2026-01-28T08:30:00Z".to_string(),
//!     git_sha: "d7327a6".to_string(),
//!     git_dirty: false,
//!     crate_version: "0.10.0".to_string(),
//!     rustc_version: "1.82.0".to_string(),
//!     host: HostInfo {
//!         os: "linux".to_string(),
//!         cpu: "AMD Ryzen 9".to_string(),
//!         ram_gb: 64.0,
//!     },
//!     gpu: GpuInfo {
//!         adapter_name: "AMD Radeon".to_string(),
//!         backend_api: "Vulkan".to_string(),
//!         driver_version: None,
//!         driver_info: None,
//!     },
//!     uname: None,
//! };
//!
//! writer.write_run_header(&header).unwrap();
//!
//! // Write measurement records...
//! ```

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::prefill_uniform::PrefillResult;

/// Current schema version for JSONL output.
pub const SCHEMA_VERSION: u32 = 1;

/// Error type for JSONL writer operations.
#[derive(Debug)]
pub enum JsonlError {
    /// IO error during file operations
    Io(io::Error),
    /// JSON serialization error
    Serialization(String),
    /// Writer is in an invalid state
    InvalidState(String),
}

impl std::fmt::Display for JsonlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonlError::Io(e) => write!(f, "IO error: {}", e),
            JsonlError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            JsonlError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for JsonlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JsonlError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for JsonlError {
    fn from(e: io::Error) -> Self {
        JsonlError::Io(e)
    }
}

impl From<serde_json::Error> for JsonlError {
    fn from(e: serde_json::Error) -> Self {
        JsonlError::Serialization(e.to_string())
    }
}

/// Result type for JSONL writer operations.
pub type JsonlResult<T> = Result<T, JsonlError>;

// =============================================================================
// RUN HEADER RECORD
// =============================================================================

/// Host system information for the run header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Operating system (e.g., "linux", "windows", "macos")
    pub os: String,
    /// CPU model name
    pub cpu: String,
    /// Total system RAM in gigabytes
    pub ram_gb: f64,
}

/// GPU/adapter information for the run header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU adapter name as reported by the backend
    pub adapter_name: String,
    /// Graphics API backend (e.g., "Vulkan", "Dx12", "Metal", "hip")
    pub backend_api: String,
    /// Driver version if obtainable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
    /// Additional driver information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_info: Option<String>,
}

/// Run header record capturing benchmark execution environment metadata.
///
/// This record is written once at the start of each benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunHeader {
    /// Unique run identifier (timestamp + random suffix)
    pub run_id: String,
    /// ISO 8601 timestamp when the run started
    pub started_at_utc: String,
    /// Git commit SHA of the codebase
    pub git_sha: String,
    /// Whether the working directory had uncommitted changes
    pub git_dirty: bool,
    /// Version of the web-rwkv crate
    pub crate_version: String,
    /// Rust compiler version used for the build
    pub rustc_version: String,
    /// Host system information
    pub host: HostInfo,
    /// GPU/adapter information
    pub gpu: GpuInfo,
    /// Full uname output (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uname: Option<String>,
}

/// Internal wrapper for the run header record with type and version fields.
#[derive(Debug, Serialize)]
struct RunHeaderRecord<'a> {
    #[serde(rename = "type")]
    record_type: &'static str,
    schema_version: u32,
    #[serde(flatten)]
    header: &'a RunHeader,
}

// =============================================================================
// MEASUREMENT RECORD
// =============================================================================

/// Benchmark scenario type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scenario {
    /// Decode-only benchmark (seq_len=1 per call, many steps)
    DecodeOnly,
    /// Prefill with uniform prompt lengths across batches
    PrefillUniform,
    /// Prefill with mixed prompt lengths per batch element
    PrefillMixed,
}

impl Scenario {
    /// Convert scenario to string for case_id generation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Scenario::DecodeOnly => "decode_only",
            Scenario::PrefillUniform => "prefill_uniform",
            Scenario::PrefillMixed => "prefill_mixed",
        }
    }
}

/// Execution status for a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Benchmark completed successfully
    Ok,
    /// Benchmark was skipped
    Skipped,
    /// Benchmark failed with an error
    Error,
}

/// Error classification for failed or skipped benchmarks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Backend not supported on this system
    UnsupportedBackend,
    /// Out of memory
    OutOfMemory,
    /// Operation timed out
    Timeout,
    /// Model file not found or failed to load
    ModelLoadFailed,
    /// Inference error during benchmark
    InferenceError,
    /// Device lost during operation
    DeviceLost,
    /// Other error
    Other,
}

/// Decode-specific metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodeMetrics {
    /// Total time for all decode steps in milliseconds
    pub decode_total_ms: f64,
    /// Number of decode steps executed
    pub decode_steps: u32,
    /// Total tokens generated (batch_size * decode_steps)
    pub decode_tokens: u32,
    /// Decode throughput in tokens per second
    pub decode_tok_per_s: f64,
    /// 50th percentile step latency (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_step_ms_p50: Option<f64>,
    /// 95th percentile step latency (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_step_ms_p95: Option<f64>,
}

/// Prefill-specific metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefillMetrics {
    /// Total time for prefill in milliseconds
    pub prefill_total_ms: f64,
    /// Total tokens processed across all sequences
    pub total_prompt_tokens: u32,
    /// Prefill throughput in tokens per second
    pub prefill_tok_per_s: f64,
    /// Number of infer() calls made during prefill
    pub num_infer_calls: u32,
    /// Time to first token per sequence in milliseconds
    pub ttft_ms_local: Vec<f64>,
    /// Minimum TTFT across sequences
    pub ttft_min_ms: f64,
    /// Median TTFT across sequences
    pub ttft_p50_ms: f64,
    /// Maximum TTFT across sequences
    pub ttft_max_ms: f64,
}

impl From<PrefillResult> for PrefillMetrics {
    fn from(result: PrefillResult) -> Self {
        Self {
            prefill_total_ms: result.prefill_total_ms,
            total_prompt_tokens: result.total_prompt_tokens,
            prefill_tok_per_s: result.prefill_tok_per_s,
            num_infer_calls: result.num_infer_calls,
            ttft_ms_local: result.ttft_ms_local,
            ttft_min_ms: result.ttft_min_ms,
            ttft_p50_ms: result.ttft_p50_ms,
            ttft_max_ms: result.ttft_max_ms,
        }
    }
}

/// Case identity fields for a measurement record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseIdentity {
    /// SHA256 hash of the model weights file
    pub model_id: String,
    /// Human-readable model name
    pub model_name: String,
    /// Path to the model file
    pub model_path: String,
    /// Human-readable size label (e.g., "0.1b", "2.9b")
    pub model_size: String,
    /// RWKV architecture version
    pub rwkv_version: String,
    /// Backend identifier ("wgpu" or "hip")
    pub backend_id: String,
    /// Graphics API variant
    pub wgpu_backend: String,
    /// Batch size for this case
    pub batch_size: u32,
    /// Requested token chunk size from config
    pub token_chunk_size_requested: u32,
    /// Effective token chunk size (rounded to multiple of 32)
    pub token_chunk_size_effective: u32,
}

/// Scenario-specific parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioParams {
    /// Decode-only scenario parameters
    Decode {
        /// Number of decode steps
        decode_steps: u32,
    },
    /// Prefill-uniform scenario parameters
    PrefillUniform {
        /// Uniform sequence length
        seq_len: u32,
    },
    /// Prefill-mixed scenario parameters
    PrefillMixed {
        /// Array of sequence lengths
        seq_lens: Vec<u32>,
        /// Mixed case identifier
        mixed_case_id: String,
    },
}

/// Metrics union for different scenario types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Metrics {
    /// Decode metrics
    Decode(DecodeMetrics),
    /// Prefill metrics
    Prefill(PrefillMetrics),
    /// No metrics (for skipped/error cases)
    None,
}

/// Measurement record for a single benchmark case x repeat.
#[derive(Debug, Clone)]
pub struct MeasureRecord {
    /// Run identifier (matches the run header)
    pub run_id: String,
    /// Stable case identifier
    pub case_id: String,
    /// Zero-based repeat index within this case
    pub repeat_index: u32,
    /// Benchmark scenario
    pub scenario: Scenario,
    /// Execution status
    pub status: Status,
    /// Error classification (when status is not Ok)
    pub error_kind: Option<ErrorKind>,
    /// Error details (when status is not Ok)
    pub error_message: Option<String>,
    /// Case identity fields
    pub case_identity: CaseIdentity,
    /// Scenario-specific parameters
    pub scenario_params: ScenarioParams,
    /// Metrics (when status is Ok)
    pub metrics: Option<Metrics>,
}

/// Internal wrapper for the measurement record with flattened structure.
#[derive(Debug, Serialize)]
struct MeasureRecordSerialized<'a> {
    #[serde(rename = "type")]
    record_type: &'static str,
    schema_version: u32,
    run_id: &'a str,
    case_id: &'a str,
    repeat_index: u32,
    scenario: Scenario,
    status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_kind: Option<&'a ErrorKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_message: Option<&'a str>,
    // Flattened case identity
    model_id: &'a str,
    model_name: &'a str,
    model_path: &'a str,
    model_size: &'a str,
    rwkv_version: &'a str,
    backend_id: &'a str,
    wgpu_backend: &'a str,
    batch_size: u32,
    token_chunk_size_requested: u32,
    token_chunk_size_effective: u32,
    // Scenario-specific params (conditionally included)
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seq_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seq_lens: Option<&'a Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mixed_case_id: Option<&'a str>,
    // Decode metrics (conditionally included)
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_total_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tok_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_step_ms_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_step_ms_p95: Option<f64>,
    // Prefill metrics (conditionally included)
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_total_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_tok_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_infer_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms_local: Option<&'a Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_min_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_max_ms: Option<f64>,
}

impl MeasureRecord {
    /// Convert to serializable form.
    fn to_serialized(&self) -> MeasureRecordSerialized<'_> {
        // Extract scenario params
        let (decode_steps, seq_len, seq_lens, mixed_case_id) = match &self.scenario_params {
            ScenarioParams::Decode { decode_steps } => (Some(*decode_steps), None, None, None),
            ScenarioParams::PrefillUniform { seq_len } => (None, Some(*seq_len), None, None),
            ScenarioParams::PrefillMixed { seq_lens, mixed_case_id } => {
                (None, None, Some(seq_lens), Some(mixed_case_id.as_str()))
            }
        };

        // Extract metrics
        let (
            decode_total_ms, decode_tokens, decode_tok_per_s, decode_step_ms_p50, decode_step_ms_p95,
            prefill_total_ms, total_prompt_tokens, prefill_tok_per_s, num_infer_calls,
            ttft_ms_local, ttft_min_ms, ttft_p50_ms, ttft_max_ms,
        ) = match &self.metrics {
            Some(Metrics::Decode(m)) => (
                Some(m.decode_total_ms),
                Some(m.decode_tokens),
                Some(m.decode_tok_per_s),
                m.decode_step_ms_p50,
                m.decode_step_ms_p95,
                None, None, None, None, None, None, None, None,
            ),
            Some(Metrics::Prefill(m)) => (
                None, None, None, None, None,
                Some(m.prefill_total_ms),
                Some(m.total_prompt_tokens),
                Some(m.prefill_tok_per_s),
                Some(m.num_infer_calls),
                Some(&m.ttft_ms_local),
                Some(m.ttft_min_ms),
                Some(m.ttft_p50_ms),
                Some(m.ttft_max_ms),
            ),
            Some(Metrics::None) | None => (
                None, None, None, None, None,
                None, None, None, None, None, None, None, None,
            ),
        };

        MeasureRecordSerialized {
            record_type: "measure",
            schema_version: SCHEMA_VERSION,
            run_id: &self.run_id,
            case_id: &self.case_id,
            repeat_index: self.repeat_index,
            scenario: self.scenario,
            status: self.status,
            error_kind: self.error_kind.as_ref(),
            error_message: self.error_message.as_deref(),
            model_id: &self.case_identity.model_id,
            model_name: &self.case_identity.model_name,
            model_path: &self.case_identity.model_path,
            model_size: &self.case_identity.model_size,
            rwkv_version: &self.case_identity.rwkv_version,
            backend_id: &self.case_identity.backend_id,
            wgpu_backend: &self.case_identity.wgpu_backend,
            batch_size: self.case_identity.batch_size,
            token_chunk_size_requested: self.case_identity.token_chunk_size_requested,
            token_chunk_size_effective: self.case_identity.token_chunk_size_effective,
            decode_steps,
            seq_len,
            seq_lens,
            mixed_case_id,
            decode_total_ms,
            decode_tokens,
            decode_tok_per_s,
            decode_step_ms_p50,
            decode_step_ms_p95,
            prefill_total_ms,
            total_prompt_tokens,
            prefill_tok_per_s,
            num_infer_calls,
            ttft_ms_local,
            ttft_min_ms,
            ttft_p50_ms,
            ttft_max_ms,
        }
    }
}

// =============================================================================
// CASE ID GENERATION
// =============================================================================

/// Parameters for case_id generation.
///
/// The case_id is computed from normalized parameters that identify a benchmark
/// "cell" in the matrix, excluding run-specific info (run_id, timestamps, adapter name).
#[derive(Debug, Clone)]
pub struct CaseIdParams<'a> {
    /// Benchmark scenario
    pub scenario: Scenario,
    /// Model identifier or name
    pub model_id: &'a str,
    /// Backend identifier
    pub backend_id: &'a str,
    /// WGPU backend variant
    pub wgpu_backend: &'a str,
    /// Batch size
    pub batch_size: u32,
    /// Effective token chunk size
    pub token_chunk_size_effective: u32,
    /// Decode steps (for decode_only)
    pub decode_steps: Option<u32>,
    /// Sequence length (for prefill_uniform)
    pub seq_len: Option<u32>,
    /// Mixed case ID (for prefill_mixed)
    pub mixed_case_id: Option<&'a str>,
}

/// Generate a human-readable case_id from normalized parameters.
///
/// Format: `{scenario}:{model_short}:{backend_id}:{wgpu_backend}:bs{batch}:c{chunk}:{scenario_params}`
///
/// # Example
///
/// ```
/// use web_rwkv_bench::jsonl::{generate_case_id, CaseIdParams, Scenario};
///
/// let params = CaseIdParams {
///     scenario: Scenario::DecodeOnly,
///     model_id: "rwkv7_0.1b",
///     backend_id: "wgpu",
///     wgpu_backend: "Vulkan",
///     batch_size: 4,
///     token_chunk_size_effective: 2048,
///     decode_steps: Some(100),
///     seq_len: None,
///     mixed_case_id: None,
/// };
///
/// let case_id = generate_case_id(&params);
/// assert_eq!(case_id, "decode_only:rwkv7_0.1b:wgpu:Vulkan:bs4:c2048:steps100");
/// ```
pub fn generate_case_id(params: &CaseIdParams) -> String {
    let mut parts = vec![
        params.scenario.as_str().to_string(),
        params.model_id.to_string(),
        params.backend_id.to_string(),
        params.wgpu_backend.to_string(),
        format!("bs{}", params.batch_size),
        format!("c{}", params.token_chunk_size_effective),
    ];

    // Add scenario-specific suffix
    match params.scenario {
        Scenario::DecodeOnly => {
            if let Some(steps) = params.decode_steps {
                parts.push(format!("steps{}", steps));
            }
        }
        Scenario::PrefillUniform => {
            if let Some(len) = params.seq_len {
                parts.push(format!("len{}", len));
            }
        }
        Scenario::PrefillMixed => {
            if let Some(case_id) = params.mixed_case_id {
                parts.push(case_id.to_string());
            }
        }
    }

    parts.join(":")
}

// =============================================================================
// JSONL WRITER
// =============================================================================

/// Append-only JSONL writer for benchmark results.
///
/// The writer ensures that each record is flushed to disk immediately after writing,
/// providing crash safety. If the process crashes mid-benchmark, all previously
/// written records will be preserved.
pub struct JsonlWriter {
    writer: BufWriter<File>,
    run_id: Option<String>,
    records_written: u64,
}

impl JsonlWriter {
    /// Create a new JSONL writer, creating a new file.
    ///
    /// Returns an error if the file already exists.
    pub fn create(path: &Path) -> JsonlResult<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            run_id: None,
            records_written: 0,
        })
    }

    /// Create a new JSONL writer, appending to an existing file or creating a new one.
    ///
    /// Use this for append mode where multiple runs may be written to the same file.
    pub fn open_append(path: &Path) -> JsonlResult<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            run_id: None,
            records_written: 0,
        })
    }

    /// Write the run header record.
    ///
    /// This should be called once at the start of a benchmark run. The run_id
    /// from the header will be stored and used to validate subsequent measure records.
    pub fn write_run_header(&mut self, header: &RunHeader) -> JsonlResult<()> {
        let record = RunHeaderRecord {
            record_type: "run",
            schema_version: SCHEMA_VERSION,
            header,
        };

        let json = serde_json::to_string(&record)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()?;

        self.run_id = Some(header.run_id.clone());
        self.records_written += 1;

        Ok(())
    }

    /// Write a measurement record.
    ///
    /// The run_id in the record should match the run header's run_id.
    pub fn write_measure(&mut self, record: &MeasureRecord) -> JsonlResult<()> {
        // Validate run_id matches if we have one
        if let Some(ref expected_run_id) = self.run_id {
            if &record.run_id != expected_run_id {
                return Err(JsonlError::InvalidState(format!(
                    "Record run_id '{}' does not match header run_id '{}'",
                    record.run_id, expected_run_id
                )));
            }
        }

        let serialized = record.to_serialized();
        let json = serde_json::to_string(&serialized)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()?;

        self.records_written += 1;

        Ok(())
    }

    /// Write an error record for a failed benchmark case.
    ///
    /// This is a convenience method that creates a MeasureRecord with status=error
    /// and the appropriate error_kind and error_message fields.
    ///
    /// # Arguments
    ///
    /// * `run_id` - The run identifier (should match the run header)
    /// * `case_id` - The case identifier
    /// * `repeat_index` - Which repeat attempt this is (0-indexed)
    /// * `scenario` - The benchmark scenario type
    /// * `case_identity` - Model/backend configuration details
    /// * `scenario_params` - Scenario-specific parameters
    /// * `error_kind` - The classified error type
    /// * `error_message` - Human-readable error description
    pub fn write_error(
        &mut self,
        run_id: &str,
        case_id: &str,
        repeat_index: u32,
        scenario: Scenario,
        case_identity: CaseIdentity,
        scenario_params: ScenarioParams,
        error_kind: ErrorKind,
        error_message: String,
    ) -> JsonlResult<()> {
        let record = MeasureRecord {
            run_id: run_id.to_string(),
            case_id: case_id.to_string(),
            repeat_index,
            scenario,
            status: Status::Error,
            error_kind: Some(error_kind),
            error_message: Some(error_message),
            case_identity,
            scenario_params,
            metrics: None,
        };

        self.write_measure(&record)
    }

    /// Get the number of records written so far.
    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Get the run_id if a run header has been written.
    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    /// Explicitly flush all buffered data to disk.
    ///
    /// Note: Each write operation already flushes, so this is only needed
    /// if you want to ensure OS-level sync.
    pub fn flush(&mut self) -> JsonlResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Sync all data to the underlying storage device.
    ///
    /// This is more expensive than flush() but ensures data durability
    /// even in the event of a power failure.
    pub fn sync(&mut self) -> JsonlResult<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a run_id from the current timestamp and a random suffix.
///
/// Format: `{timestamp}_{random}` where timestamp is `YYYYMMDDTHHMMSSz`
/// and random is a 6-character hex string.
pub fn generate_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Convert to simple timestamp format
    let secs = now.as_secs();
    let days_since_epoch = secs / 86400;
    let secs_today = secs % 86400;

    // Simple date calculation (approximate, good enough for IDs)
    let years = days_since_epoch / 365;
    let year = 1970 + years;
    let day_of_year = days_since_epoch % 365;
    let month = (day_of_year / 30).min(11) + 1;
    let day = (day_of_year % 30) + 1;

    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    // Random suffix using system time nanos
    let nanos = now.subsec_nanos();
    let random_suffix = format!("{:06x}", nanos & 0xFFFFFF);

    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z_{}",
        year, month, day, hours, minutes, seconds, random_suffix
    )
}

/// Generate an ISO 8601 UTC timestamp for the current time.
pub fn generate_timestamp_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();
    let days_since_epoch = secs / 86400;
    let secs_today = secs % 86400;

    let years = days_since_epoch / 365;
    let year = 1970 + years;
    let day_of_year = days_since_epoch % 365;
    let month = (day_of_year / 30).min(11) + 1;
    let day = (day_of_year % 30) + 1;

    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Round a token chunk size to the nearest multiple of 32 (as required by web-rwkv).
pub fn round_chunk_size(requested: u32) -> u32 {
    ((requested + 31) / 32) * 32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_file() -> std::path::PathBuf {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "test_jsonl_{}_{}.jsonl",
            std::process::id(),
            counter
        ));
        // Clean up any existing file
        let _ = fs::remove_file(&path);
        path
    }

    #[test]
    fn test_generate_case_id_decode() {
        let params = CaseIdParams {
            scenario: Scenario::DecodeOnly,
            model_id: "rwkv7_0.1b",
            backend_id: "wgpu",
            wgpu_backend: "Vulkan",
            batch_size: 4,
            token_chunk_size_effective: 2048,
            decode_steps: Some(100),
            seq_len: None,
            mixed_case_id: None,
        };

        let case_id = generate_case_id(&params);
        assert_eq!(case_id, "decode_only:rwkv7_0.1b:wgpu:Vulkan:bs4:c2048:steps100");
    }

    #[test]
    fn test_generate_case_id_prefill_uniform() {
        let params = CaseIdParams {
            scenario: Scenario::PrefillUniform,
            model_id: "rwkv7_0.1b",
            backend_id: "wgpu",
            wgpu_backend: "Vulkan",
            batch_size: 4,
            token_chunk_size_effective: 2048,
            decode_steps: None,
            seq_len: Some(512),
            mixed_case_id: None,
        };

        let case_id = generate_case_id(&params);
        assert_eq!(case_id, "prefill_uniform:rwkv7_0.1b:wgpu:Vulkan:bs4:c2048:len512");
    }

    #[test]
    fn test_generate_case_id_prefill_mixed() {
        let params = CaseIdParams {
            scenario: Scenario::PrefillMixed,
            model_id: "rwkv7_0.1b",
            backend_id: "wgpu",
            wgpu_backend: "Vulkan",
            batch_size: 8,
            token_chunk_size_effective: 256,
            decode_steps: None,
            seq_len: None,
            mixed_case_id: Some("staircase_8"),
        };

        let case_id = generate_case_id(&params);
        assert_eq!(case_id, "prefill_mixed:rwkv7_0.1b:wgpu:Vulkan:bs8:c256:staircase_8");
    }

    #[test]
    fn test_round_chunk_size() {
        assert_eq!(round_chunk_size(1), 32);
        assert_eq!(round_chunk_size(32), 32);
        assert_eq!(round_chunk_size(33), 64);
        assert_eq!(round_chunk_size(128), 128);
        assert_eq!(round_chunk_size(129), 160);
        assert_eq!(round_chunk_size(256), 256);
    }

    #[test]
    fn test_generate_run_id() {
        let run_id = generate_run_id();
        // Should have format like "20260128T083000Z_a1b2c3"
        assert!(run_id.contains('T'));
        assert!(run_id.contains('Z'));
        assert!(run_id.contains('_'));
    }

    #[test]
    fn test_generate_timestamp_utc() {
        let ts = generate_timestamp_utc();
        // Should have format like "2026-01-28T08:30:00Z"
        assert!(ts.contains('-'));
        assert!(ts.contains('T'));
        assert!(ts.contains(':'));
        assert!(ts.ends_with('Z'));
    }

    #[test]
    fn test_write_run_header() {
        let path = temp_file();

        {
            let mut writer = JsonlWriter::create(&path).unwrap();

            let header = RunHeader {
                run_id: "test_run_123".to_string(),
                started_at_utc: "2026-01-28T08:30:00Z".to_string(),
                git_sha: "abc123".to_string(),
                git_dirty: false,
                crate_version: "0.10.0".to_string(),
                rustc_version: "1.82.0".to_string(),
                host: HostInfo {
                    os: "linux".to_string(),
                    cpu: "Test CPU".to_string(),
                    ram_gb: 32.0,
                },
                gpu: GpuInfo {
                    adapter_name: "Test GPU".to_string(),
                    backend_api: "Vulkan".to_string(),
                    driver_version: Some("1.2.3".to_string()),
                    driver_info: None,
                },
                uname: None,
            };

            writer.write_run_header(&header).unwrap();
            assert_eq!(writer.records_written(), 1);
            assert_eq!(writer.run_id(), Some("test_run_123"));
        }

        // Read and verify
        let content = fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(record["type"], "run");
        assert_eq!(record["schema_version"], 1);
        assert_eq!(record["run_id"], "test_run_123");
        assert_eq!(record["host"]["os"], "linux");
        assert_eq!(record["gpu"]["adapter_name"], "Test GPU");

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_write_measure_record() {
        let path = temp_file();

        {
            let mut writer = JsonlWriter::create(&path).unwrap();

            // Write header first
            let header = RunHeader {
                run_id: "test_run_456".to_string(),
                started_at_utc: "2026-01-28T08:30:00Z".to_string(),
                git_sha: "def456".to_string(),
                git_dirty: true,
                crate_version: "0.10.0".to_string(),
                rustc_version: "1.82.0".to_string(),
                host: HostInfo {
                    os: "linux".to_string(),
                    cpu: "Test CPU".to_string(),
                    ram_gb: 32.0,
                },
                gpu: GpuInfo {
                    adapter_name: "Test GPU".to_string(),
                    backend_api: "Vulkan".to_string(),
                    driver_version: None,
                    driver_info: None,
                },
                uname: None,
            };
            writer.write_run_header(&header).unwrap();

            // Write measure record
            let measure = MeasureRecord {
                run_id: "test_run_456".to_string(),
                case_id: "decode_only:test:wgpu:Vulkan:bs4:c128:steps100".to_string(),
                repeat_index: 0,
                scenario: Scenario::DecodeOnly,
                status: Status::Ok,
                error_kind: None,
                error_message: None,
                case_identity: CaseIdentity {
                    model_id: "sha256:abc".to_string(),
                    model_name: "test_model".to_string(),
                    model_path: "path/to/model.st".to_string(),
                    model_size: "9m".to_string(),
                    rwkv_version: "v7".to_string(),
                    backend_id: "wgpu".to_string(),
                    wgpu_backend: "Vulkan".to_string(),
                    batch_size: 4,
                    token_chunk_size_requested: 128,
                    token_chunk_size_effective: 128,
                },
                scenario_params: ScenarioParams::Decode { decode_steps: 100 },
                metrics: Some(Metrics::Decode(DecodeMetrics {
                    decode_total_ms: 245.67,
                    decode_steps: 100,
                    decode_tokens: 400,
                    decode_tok_per_s: 1628.5,
                    decode_step_ms_p50: Some(2.41),
                    decode_step_ms_p95: Some(2.89),
                })),
            };

            writer.write_measure(&measure).unwrap();
            assert_eq!(writer.records_written(), 2);
        }

        // Read and verify
        let file = fs::File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
        assert_eq!(lines.len(), 2);

        // Verify measure record
        let measure: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(measure["type"], "measure");
        assert_eq!(measure["schema_version"], 1);
        assert_eq!(measure["scenario"], "decode_only");
        assert_eq!(measure["status"], "ok");
        assert_eq!(measure["batch_size"], 4);
        assert_eq!(measure["decode_steps"], 100);
        assert_eq!(measure["decode_tokens"], 400);
        assert!((measure["decode_tok_per_s"].as_f64().unwrap() - 1628.5).abs() < 0.01);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_write_error_record() {
        let path = temp_file();

        {
            let mut writer = JsonlWriter::create(&path).unwrap();

            let measure = MeasureRecord {
                run_id: "test_run_err".to_string(),
                case_id: "decode_only:test:wgpu:Vulkan:bs64:c128:steps100".to_string(),
                repeat_index: 0,
                scenario: Scenario::DecodeOnly,
                status: Status::Error,
                error_kind: Some(ErrorKind::OutOfMemory),
                error_message: Some("Allocation failed".to_string()),
                case_identity: CaseIdentity {
                    model_id: "sha256:abc".to_string(),
                    model_name: "test_model".to_string(),
                    model_path: "path/to/model.st".to_string(),
                    model_size: "9m".to_string(),
                    rwkv_version: "v7".to_string(),
                    backend_id: "wgpu".to_string(),
                    wgpu_backend: "Vulkan".to_string(),
                    batch_size: 64,
                    token_chunk_size_requested: 128,
                    token_chunk_size_effective: 128,
                },
                scenario_params: ScenarioParams::Decode { decode_steps: 100 },
                metrics: None,
            };

            writer.write_measure(&measure).unwrap();
        }

        // Read and verify
        let content = fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(record["status"], "error");
        assert_eq!(record["error_kind"], "out_of_memory");
        assert_eq!(record["error_message"], "Allocation failed");
        // No metrics should be present
        assert!(record.get("decode_total_ms").is_none());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_write_prefill_metrics() {
        let path = temp_file();

        {
            let mut writer = JsonlWriter::create(&path).unwrap();

            let measure = MeasureRecord {
                run_id: "test_run_prefill".to_string(),
                case_id: "prefill_uniform:test:wgpu:Vulkan:bs4:c256:len512".to_string(),
                repeat_index: 0,
                scenario: Scenario::PrefillUniform,
                status: Status::Ok,
                error_kind: None,
                error_message: None,
                case_identity: CaseIdentity {
                    model_id: "sha256:abc".to_string(),
                    model_name: "test_model".to_string(),
                    model_path: "path/to/model.st".to_string(),
                    model_size: "9m".to_string(),
                    rwkv_version: "v7".to_string(),
                    backend_id: "wgpu".to_string(),
                    wgpu_backend: "Vulkan".to_string(),
                    batch_size: 4,
                    token_chunk_size_requested: 256,
                    token_chunk_size_effective: 256,
                },
                scenario_params: ScenarioParams::PrefillUniform { seq_len: 512 },
                metrics: Some(Metrics::Prefill(PrefillMetrics {
                    prefill_total_ms: 89.45,
                    total_prompt_tokens: 2048,
                    prefill_tok_per_s: 22897.1,
                    num_infer_calls: 2,
                    ttft_ms_local: vec![89.12, 89.23, 89.34, 89.45],
                    ttft_min_ms: 89.12,
                    ttft_p50_ms: 89.28,
                    ttft_max_ms: 89.45,
                })),
            };

            writer.write_measure(&measure).unwrap();
        }

        // Read and verify
        let content = fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(record["scenario"], "prefill_uniform");
        assert_eq!(record["seq_len"], 512);
        assert_eq!(record["num_infer_calls"], 2);
        assert_eq!(record["ttft_ms_local"].as_array().unwrap().len(), 4);
        assert!((record["ttft_min_ms"].as_f64().unwrap() - 89.12).abs() < 0.01);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_run_id_mismatch() {
        let path = temp_file();

        let mut writer = JsonlWriter::create(&path).unwrap();

        // Write header
        let header = RunHeader {
            run_id: "run_A".to_string(),
            started_at_utc: "2026-01-28T08:30:00Z".to_string(),
            git_sha: "abc".to_string(),
            git_dirty: false,
            crate_version: "0.10.0".to_string(),
            rustc_version: "1.82.0".to_string(),
            host: HostInfo {
                os: "linux".to_string(),
                cpu: "CPU".to_string(),
                ram_gb: 32.0,
            },
            gpu: GpuInfo {
                adapter_name: "GPU".to_string(),
                backend_api: "Vulkan".to_string(),
                driver_version: None,
                driver_info: None,
            },
            uname: None,
        };
        writer.write_run_header(&header).unwrap();

        // Try to write measure with different run_id
        let measure = MeasureRecord {
            run_id: "run_B".to_string(), // Different!
            case_id: "test".to_string(),
            repeat_index: 0,
            scenario: Scenario::DecodeOnly,
            status: Status::Ok,
            error_kind: None,
            error_message: None,
            case_identity: CaseIdentity {
                model_id: "id".to_string(),
                model_name: "name".to_string(),
                model_path: "path".to_string(),
                model_size: "9m".to_string(),
                rwkv_version: "v7".to_string(),
                backend_id: "wgpu".to_string(),
                wgpu_backend: "Vulkan".to_string(),
                batch_size: 1,
                token_chunk_size_requested: 128,
                token_chunk_size_effective: 128,
            },
            scenario_params: ScenarioParams::Decode { decode_steps: 10 },
            metrics: None,
        };

        let result = writer.write_measure(&measure);
        assert!(result.is_err());
        if let Err(JsonlError::InvalidState(msg)) = result {
            assert!(msg.contains("run_B"));
            assert!(msg.contains("run_A"));
        } else {
            panic!("Expected InvalidState error");
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_scenario_as_str() {
        assert_eq!(Scenario::DecodeOnly.as_str(), "decode_only");
        assert_eq!(Scenario::PrefillUniform.as_str(), "prefill_uniform");
        assert_eq!(Scenario::PrefillMixed.as_str(), "prefill_mixed");
    }

    #[test]
    fn test_append_mode() {
        let path = temp_file();

        // Write first record
        {
            let mut writer = JsonlWriter::create(&path).unwrap();
            let header = RunHeader {
                run_id: "run_1".to_string(),
                started_at_utc: "2026-01-28T08:30:00Z".to_string(),
                git_sha: "abc".to_string(),
                git_dirty: false,
                crate_version: "0.10.0".to_string(),
                rustc_version: "1.82.0".to_string(),
                host: HostInfo {
                    os: "linux".to_string(),
                    cpu: "CPU".to_string(),
                    ram_gb: 32.0,
                },
                gpu: GpuInfo {
                    adapter_name: "GPU".to_string(),
                    backend_api: "Vulkan".to_string(),
                    driver_version: None,
                    driver_info: None,
                },
                uname: None,
            };
            writer.write_run_header(&header).unwrap();
        }

        // Append second run
        {
            let mut writer = JsonlWriter::open_append(&path).unwrap();
            let header = RunHeader {
                run_id: "run_2".to_string(),
                started_at_utc: "2026-01-28T09:30:00Z".to_string(),
                git_sha: "def".to_string(),
                git_dirty: true,
                crate_version: "0.10.0".to_string(),
                rustc_version: "1.82.0".to_string(),
                host: HostInfo {
                    os: "linux".to_string(),
                    cpu: "CPU".to_string(),
                    ram_gb: 32.0,
                },
                gpu: GpuInfo {
                    adapter_name: "GPU".to_string(),
                    backend_api: "Vulkan".to_string(),
                    driver_version: None,
                    driver_info: None,
                },
                uname: None,
            };
            writer.write_run_header(&header).unwrap();
        }

        // Verify both records
        let file = fs::File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
        assert_eq!(lines.len(), 2);

        let rec1: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        let rec2: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(rec1["run_id"], "run_1");
        assert_eq!(rec2["run_id"], "run_2");

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_prefill_result_to_prefill_metrics() {
        use crate::prefill_uniform::PrefillResult;

        let result = PrefillResult::from_ttft(
            vec![10.0, 20.0, 30.0, 40.0],
            512,
            2,
        );

        let metrics: PrefillMetrics = result.into();

        assert_eq!(metrics.prefill_total_ms, 40.0);
        assert_eq!(metrics.total_prompt_tokens, 512);
        assert_eq!(metrics.num_infer_calls, 2);
        assert_eq!(metrics.ttft_ms_local, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(metrics.ttft_min_ms, 10.0);
        assert_eq!(metrics.ttft_max_ms, 40.0);
        assert_eq!(metrics.ttft_p50_ms, 25.0); // (20 + 30) / 2
    }
}
