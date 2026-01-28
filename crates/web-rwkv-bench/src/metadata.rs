//! Environment and build metadata collection for benchmark run headers.
//!
//! This module collects information about the build environment, host system,
//! and GPU adapter for inclusion in benchmark JSONL output. All collection
//! functions are designed to fail gracefully when information is unavailable.
//!
//! # Example
//!
//! ```no_run
//! use web_rwkv_bench::metadata::{collect_run_metadata, RunMetadata};
//!
//! let metadata = collect_run_metadata(None);
//! println!("Git SHA: {:?}", metadata.git.sha);
//! println!("Rustc version: {:?}", metadata.build.rustc_version);
//! println!("CPU: {:?}", metadata.host.cpu);
//! ```

use serde::{Deserialize, Serialize};
use std::process::Command;

/// Complete run metadata for benchmark output.
///
/// This struct aggregates all environment metadata for inclusion in the
/// JSONL run header record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Git repository information
    pub git: GitInfo,
    /// Build/compile information
    pub build: BuildInfo,
    /// Host system information
    pub host: HostInfo,
    /// GPU adapter information (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuInfo>,
}

/// Git repository information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    /// Git commit SHA (short or long form)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
    /// Whether the working directory has uncommitted changes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
}

/// Build/compile information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Crate version from Cargo.toml
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crate_version: Option<String>,
    /// Rust compiler version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc_version: Option<String>,
}

/// Host system information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Operating system name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    /// CPU model/name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    /// Total RAM in gigabytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_gb: Option<f64>,
    /// Full uname string (Unix-like systems)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uname: Option<String>,
}

/// GPU adapter information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU adapter name (e.g., "NVIDIA GeForce RTX 3080")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_name: Option<String>,
    /// Backend API (e.g., "Vulkan", "Metal", "DirectX 12", "Hip")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_api: Option<String>,
    /// Driver version (if obtainable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
}

/// Collect git repository information.
///
/// Runs `git rev-parse HEAD` for SHA and `git status --porcelain` for dirty check.
/// Returns partial info if some commands fail (e.g., not in a git repo).
pub fn collect_git_info() -> GitInfo {
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| !s.trim().is_empty());

    GitInfo { sha, dirty }
}

/// Collect build information.
///
/// Gets crate version from CARGO_PKG_VERSION and rustc version from `rustc --version`.
pub fn collect_build_info() -> BuildInfo {
    // Crate version from compile-time env var
    let crate_version = option_env!("CARGO_PKG_VERSION").map(String::from);

    // Rustc version from command
    let rustc_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    BuildInfo {
        crate_version,
        rustc_version,
    }
}

/// Collect host system information.
///
/// Gathers OS name, CPU model, RAM size, and uname output.
/// Platform-specific implementations for Linux, macOS, and Windows.
pub fn collect_host_info() -> HostInfo {
    let os = get_os_name();
    let cpu = get_cpu_model();
    let ram_gb = get_ram_gb();
    let uname = get_uname();

    HostInfo {
        os,
        cpu,
        ram_gb,
        uname,
    }
}

/// Get the operating system name.
fn get_os_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        // Try /etc/os-release first
        if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
                    return Some(name.trim_matches('"').to_string());
                }
            }
        }
        Some("Linux".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("sw_vers")
            .args(["-productName"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|name| {
                let version = Command::new("sw_vers")
                    .args(["-productVersion"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|v| v.trim().to_string())
                    .unwrap_or_default();
                format!("{} {}", name.trim(), version).trim().to_string()
            })
            .or(Some("macOS".to_string()))
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "ver"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or(Some("Windows".to_string()))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Some(std::env::consts::OS.to_string())
    }
}

/// Get the CPU model name.
fn get_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("model name") || line.starts_with("Model") {
                        if let Some((_key, value)) = line.split_once(':') {
                            return Some(value.trim().to_string());
                        }
                    }
                }
                None
            })
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("wmic")
            .args(["cpu", "get", "name"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .nth(1)
                    .map(|line| line.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Get total RAM in gigabytes.
fn get_ram_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if let Some(rest) = line.strip_prefix("MemTotal:") {
                        // Parse "12345678 kB"
                        let parts: Vec<&str> = rest.trim().split_whitespace().collect();
                        if let Some(kb_str) = parts.first() {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Some(kb as f64 / 1024.0 / 1024.0);
                            }
                        }
                    }
                }
                None
            })
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("wmic")
            .args(["computersystem", "get", "TotalPhysicalMemory"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .nth(1)
                    .and_then(|line| line.trim().parse::<u64>().ok())
            })
            .map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Get full uname string.
fn get_uname() -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        Command::new("uname")
            .arg("-a")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "windows")]
    {
        // Windows equivalent: systeminfo one-liner
        Command::new("cmd")
            .args(["/C", "systeminfo | findstr /B /C:\"OS\""])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Collect GPU information.
///
/// This function accepts optional pre-collected GPU info (e.g., from wgpu adapter).
/// If not provided, it attempts to detect GPU info via system commands.
///
/// # Arguments
///
/// * `adapter_info` - Optional pre-collected adapter info (adapter_name, backend_api, driver_version)
pub fn collect_gpu_info(
    adapter_info: Option<(&str, &str, Option<&str>)>,
) -> Option<GpuInfo> {
    if let Some((adapter_name, backend_api, driver_version)) = adapter_info {
        return Some(GpuInfo {
            adapter_name: Some(adapter_name.to_string()),
            backend_api: Some(backend_api.to_string()),
            driver_version: driver_version.map(String::from),
        });
    }

    // Try to detect GPU info from system
    detect_gpu_from_system()
}

/// Detect GPU information from system commands.
///
/// Tries nvidia-smi for NVIDIA GPUs, rocm-smi for AMD GPUs.
fn detect_gpu_from_system() -> Option<GpuInfo> {
    // Try nvidia-smi first
    if let Some(info) = try_nvidia_smi() {
        return Some(info);
    }

    // Try rocm-smi for AMD GPUs
    if let Some(info) = try_rocm_smi() {
        return Some(info);
    }

    // No GPU detected
    None
}

/// Try to get GPU info from nvidia-smi.
fn try_nvidia_smi() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())?;

    let stdout = String::from_utf8(output.stdout).ok()?;
    let line = stdout.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 2 {
        Some(GpuInfo {
            adapter_name: Some(parts[0].to_string()),
            backend_api: Some("CUDA/Vulkan".to_string()), // NVIDIA supports both
            driver_version: Some(parts[1].to_string()),
        })
    } else if !parts.is_empty() {
        Some(GpuInfo {
            adapter_name: Some(parts[0].to_string()),
            backend_api: Some("CUDA/Vulkan".to_string()),
            driver_version: None,
        })
    } else {
        None
    }
}

/// Extract GPU name from a rocm-smi output line.
///
/// Handles formats like:
/// - "GPU[0]		: Card Series: 		AMD Radeon 8060S"
/// - "Card series:		AMD Radeon RX 7900 XTX"
fn extract_gpu_name_from_rocm_line(line: &str) -> Option<String> {
    // Look for "Card Series:" or "Card series:" marker
    let lower = line.to_lowercase();
    if let Some(pos) = lower.find("card series:") {
        let after_marker = &line[pos + "card series:".len()..];
        let name = after_marker.trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }

    // Look for AMD/Radeon/RX/MI identifiers
    // Note: "MI" is used for AMD Instinct accelerators (MI100, MI250, MI300, etc.)
    if line.contains("Radeon") || line.contains("RX ") || line.contains(" MI") {
        // Try to extract just the GPU part
        // Find the start of the GPU name (typically "AMD" or "Radeon")
        if let Some(pos) = line.find("AMD ") {
            return Some(line[pos..].trim().to_string());
        }
        if let Some(pos) = line.find("Radeon ") {
            return Some(format!("AMD {}", line[pos..].trim()));
        }
    }

    None
}

/// Try to get GPU info from rocm-smi (AMD ROCm).
fn try_rocm_smi() -> Option<GpuInfo> {
    // rocm-smi --showproductname
    let name_output = Command::new("rocm-smi")
        .args(["--showproductname"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;

    let stdout = String::from_utf8(name_output.stdout).ok()?;

    // Parse output for GPU name (format varies by rocm-smi version)
    // Common formats:
    //   "GPU[0]		: Card Series: 		AMD Radeon 8060S"
    //   "Card series:		AMD Radeon RX 7900 XTX"
    let mut adapter_name = None;
    for line in stdout.lines() {
        let line = line.trim();
        // Try to extract just the GPU name from various formats
        if let Some(name) = extract_gpu_name_from_rocm_line(line) {
            adapter_name = Some(name);
            break;
        }
    }

    // Get driver version
    let driver_version = Command::new("rocm-smi")
        .args(["--showdriverversion"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("Driver version:") {
                    return Some(rest.trim().to_string());
                }
                // Fallback: first line with version-like content
                if line.contains('.') && line.chars().any(|c| c.is_ascii_digit()) {
                    return Some(line.trim().to_string());
                }
            }
            None
        });

    Some(GpuInfo {
        adapter_name,
        backend_api: Some("Hip/Vulkan".to_string()),
        driver_version,
    })
}

/// Collect all run metadata.
///
/// This is the main entry point for metadata collection. It gathers git, build,
/// host, and GPU information into a single `RunMetadata` struct.
///
/// # Arguments
///
/// * `gpu_adapter_info` - Optional pre-collected GPU adapter info from wgpu/hip runtime
///   as a tuple of (adapter_name, backend_api, optional driver_version)
///
/// # Example
///
/// ```no_run
/// use web_rwkv_bench::metadata::collect_run_metadata;
///
/// // Without GPU info (will try to detect from system)
/// let metadata = collect_run_metadata(None);
///
/// // With GPU info from wgpu adapter
/// let metadata = collect_run_metadata(Some(("NVIDIA GeForce RTX 3080", "Vulkan", Some("535.104.05"))));
/// ```
pub fn collect_run_metadata(
    gpu_adapter_info: Option<(&str, &str, Option<&str>)>,
) -> RunMetadata {
    RunMetadata {
        git: collect_git_info(),
        build: collect_build_info(),
        host: collect_host_info(),
        gpu: collect_gpu_info(gpu_adapter_info),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_git_info() {
        let info = collect_git_info();
        // Should at least not panic
        // If in a git repo, sha should be Some
        println!("Git SHA: {:?}", info.sha);
        println!("Git dirty: {:?}", info.dirty);

        // If sha is present, it should be a valid hex string
        if let Some(sha) = &info.sha {
            assert!(sha.len() >= 7, "Git SHA should be at least 7 chars");
            assert!(
                sha.chars().all(|c| c.is_ascii_hexdigit()),
                "Git SHA should be hex"
            );
        }
    }

    #[test]
    fn test_collect_build_info() {
        let info = collect_build_info();

        // Crate version should be available at compile time
        assert!(info.crate_version.is_some(), "Crate version should be set");
        println!("Crate version: {:?}", info.crate_version);

        // Rustc version should be available if rustc is in PATH
        println!("Rustc version: {:?}", info.rustc_version);
        if let Some(rustc) = &info.rustc_version {
            assert!(rustc.contains("rustc"), "Should contain 'rustc'");
        }
    }

    #[test]
    fn test_collect_host_info() {
        let info = collect_host_info();

        // OS should always be detectable
        assert!(info.os.is_some(), "OS should be detected");
        println!("OS: {:?}", info.os);

        // CPU might not be available on all systems
        println!("CPU: {:?}", info.cpu);

        // RAM should be available on common platforms
        println!("RAM GB: {:?}", info.ram_gb);
        if let Some(ram_gb) = info.ram_gb {
            assert!(ram_gb > 0.0, "RAM should be positive");
            assert!(ram_gb < 10000.0, "RAM should be reasonable (< 10TB)");
        }

        // Uname available on Unix-like systems
        println!("Uname: {:?}", info.uname);
    }

    #[test]
    fn test_collect_gpu_info_with_adapter() {
        let info = collect_gpu_info(Some((
            "Test GPU",
            "Vulkan",
            Some("1.2.3"),
        )));

        assert!(info.is_some());
        let gpu = info.unwrap();
        assert_eq!(gpu.adapter_name, Some("Test GPU".to_string()));
        assert_eq!(gpu.backend_api, Some("Vulkan".to_string()));
        assert_eq!(gpu.driver_version, Some("1.2.3".to_string()));
    }

    #[test]
    fn test_collect_gpu_info_without_adapter() {
        // This will try system detection - might return None or Some
        let info = collect_gpu_info(None);
        println!("System GPU info: {:?}", info);
    }

    #[test]
    fn test_collect_run_metadata() {
        let metadata = collect_run_metadata(None);

        // Verify structure
        println!("Full metadata: {:#?}", metadata);

        // Should serialize to JSON without error
        let json = serde_json::to_string_pretty(&metadata);
        assert!(json.is_ok(), "Should serialize to JSON");
        println!("JSON output:\n{}", json.unwrap());
    }

    #[test]
    fn test_metadata_serialization_skips_none() {
        let metadata = RunMetadata {
            git: GitInfo {
                sha: Some("abc123".to_string()),
                dirty: None, // Should be skipped in JSON
            },
            build: BuildInfo {
                crate_version: Some("0.1.0".to_string()),
                rustc_version: None, // Should be skipped
            },
            host: HostInfo {
                os: Some("Linux".to_string()),
                cpu: None,
                ram_gb: None,
                uname: None,
            },
            gpu: None, // Should be skipped entirely
        };

        let json = serde_json::to_string(&metadata).unwrap();

        // Verify None fields are not present
        assert!(!json.contains("dirty"));
        assert!(!json.contains("rustc_version"));
        assert!(!json.contains("cpu"));
        assert!(!json.contains("ram_gb"));
        assert!(!json.contains("uname"));
        assert!(!json.contains("gpu"));

        // Verify Some fields are present
        assert!(json.contains("sha"));
        assert!(json.contains("abc123"));
        assert!(json.contains("crate_version"));
        assert!(json.contains("0.1.0"));
        assert!(json.contains("os"));
        assert!(json.contains("Linux"));
    }

    #[test]
    fn test_extract_gpu_name_from_rocm_line() {
        // Test format: "GPU[0]		: Card Series: 		AMD Radeon 8060S"
        let line1 = "GPU[0]\t\t: Card Series: \t\tAMD Radeon 8060S";
        assert_eq!(
            extract_gpu_name_from_rocm_line(line1),
            Some("AMD Radeon 8060S".to_string())
        );

        // Test format: "Card series:		AMD Radeon RX 7900 XTX"
        let line2 = "Card series:\t\tAMD Radeon RX 7900 XTX";
        assert_eq!(
            extract_gpu_name_from_rocm_line(line2),
            Some("AMD Radeon RX 7900 XTX".to_string())
        );

        // Test case insensitivity
        let line3 = "card SERIES:\tAMD Radeon Pro W7900";
        assert_eq!(
            extract_gpu_name_from_rocm_line(line3),
            Some("AMD Radeon Pro W7900".to_string())
        );

        // Test AMD identifier fallback
        let line4 = "Something AMD Radeon RX 6800 XT";
        assert_eq!(
            extract_gpu_name_from_rocm_line(line4),
            Some("AMD Radeon RX 6800 XT".to_string())
        );

        // Test MI accelerator
        let line5 = "GPU[0]: AMD MI250X";
        assert_eq!(
            extract_gpu_name_from_rocm_line(line5),
            Some("AMD MI250X".to_string())
        );

        // Test line with no GPU info
        let line6 = "Some random line";
        assert_eq!(extract_gpu_name_from_rocm_line(line6), None);

        // Test empty line
        let line7 = "";
        assert_eq!(extract_gpu_name_from_rocm_line(line7), None);
    }
}
