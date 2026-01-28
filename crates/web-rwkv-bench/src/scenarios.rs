//! Benchmark scenario implementations.
//!
//! This module provides scenario runners that execute specific benchmark patterns:
//! - [`DecodeScenario`]: Decode-only benchmark (seq_len=1 per call, N steps)
//!
//! Each scenario takes care of:
//! - Warmup runs (untimed)
//! - Measurement repeats (timed)
//! - Deterministic token generation from a seed
//! - Metric calculation
//!
//! # Usage
//!
//! ```rust,ignore
//! use web_rwkv_bench::scenarios::{DecodeScenario, DecodeConfig};
//!
//! let config = DecodeConfig {
//!     decode_steps: 512,
//!     batch_size: 4,
//!     seed: 42,
//!     warmup_runs: 1,
//!     warmup_steps: 64,
//!     repeats: 5,
//!     prime_prefill_len: None,
//!     settle_ms: Some(50),
//! };
//!
//! let scenario = DecodeScenario::new(config);
//!
//! // Execute with your inference function
//! let results = scenario.run(|tokens| {
//!     // tokens: &[Vec<u16>] - one vec of tokens per batch
//!     // Call infer() and return Result<(), String>
//!     Ok(())
//! })?;
//!
//! println!("Throughput: {} tok/s", results.decode_tok_per_s);
//! ```

use std::time::{Duration, Instant};

use crate::jsonl::DecodeMetrics;

/// Configuration for the decode-only benchmark scenario.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Number of decode steps to execute in the measured loop.
    ///
    /// Recommended set: `[128, 512, 2048]`
    /// - 128: fast smoke test
    /// - 512: development runs
    /// - 2048: stable "full" benchmark runs
    pub decode_steps: u32,

    /// Batch size (number of parallel sequences).
    pub batch_size: u32,

    /// PRNG seed for deterministic token generation.
    ///
    /// Default: 42
    pub seed: u64,

    /// Number of untimed warmup runs before measurement.
    ///
    /// Default: 1
    pub warmup_runs: u32,

    /// Number of warmup steps per warmup run.
    ///
    /// Default: 64
    pub warmup_steps: u32,

    /// Number of recorded repeats for statistical significance.
    ///
    /// Default: 5
    pub repeats: u32,

    /// Optional prime prefill length per batch.
    ///
    /// If set, runs an untimed prefill of this length before the decode loop.
    /// This simulates "post-prefill steady-state decode" conditions.
    pub prime_prefill_len: Option<u32>,

    /// Optional settle delay between repeats in milliseconds.
    ///
    /// Helps reduce jitter by allowing the device to settle.
    /// Default: None (no delay)
    pub settle_ms: Option<u64>,

    /// Whether to collect per-step latencies for p50/p95 calculation.
    ///
    /// Disabled by default to reduce overhead.
    pub collect_per_step_latencies: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            decode_steps: 128,
            batch_size: 1,
            seed: 42,
            warmup_runs: 1,
            warmup_steps: 64,
            repeats: 5,
            prime_prefill_len: None,
            settle_ms: None,
            collect_per_step_latencies: false,
        }
    }
}

/// Result of a single decode benchmark repeat.
#[derive(Debug, Clone)]
pub struct DecodeRepeatResult {
    /// Zero-based repeat index.
    pub repeat_index: u32,

    /// Total time for all decode steps in this repeat.
    pub decode_total_ms: f64,

    /// Number of decode steps executed.
    pub decode_steps: u32,

    /// Total tokens generated (batch_size * decode_steps).
    pub decode_tokens: u32,

    /// Decode throughput in tokens per second.
    pub decode_tok_per_s: f64,

    /// Per-step latencies in milliseconds (if collected).
    pub step_latencies_ms: Option<Vec<f64>>,
}

/// Aggregated results from all repeats of a decode benchmark.
#[derive(Debug, Clone)]
pub struct DecodeResults {
    /// Configuration used for this benchmark.
    pub config: DecodeConfig,

    /// Results from each individual repeat.
    pub repeats: Vec<DecodeRepeatResult>,

    /// Median throughput across all repeats (tok/s).
    pub median_tok_per_s: f64,

    /// Mean throughput across all repeats (tok/s).
    pub mean_tok_per_s: f64,

    /// Minimum throughput across all repeats (tok/s).
    pub min_tok_per_s: f64,

    /// Maximum throughput across all repeats (tok/s).
    pub max_tok_per_s: f64,
}

impl DecodeResults {
    /// Convert the first repeat result to JSONL DecodeMetrics format.
    ///
    /// Use `to_metrics_for_repeat` if you need a specific repeat.
    pub fn to_metrics(&self) -> DecodeMetrics {
        self.to_metrics_for_repeat(0)
    }

    /// Convert a specific repeat result to JSONL DecodeMetrics format.
    pub fn to_metrics_for_repeat(&self, repeat_index: usize) -> DecodeMetrics {
        let repeat = &self.repeats[repeat_index];

        // Calculate p50 and p95 if step latencies were collected
        let (p50, p95) = if let Some(ref latencies) = repeat.step_latencies_ms {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            (
                Some(sorted.get(p50_idx).copied().unwrap_or(0.0)),
                Some(sorted.get(p95_idx.min(sorted.len() - 1)).copied().unwrap_or(0.0)),
            )
        } else {
            (None, None)
        };

        DecodeMetrics {
            decode_total_ms: repeat.decode_total_ms,
            decode_steps: repeat.decode_steps,
            decode_tokens: repeat.decode_tokens,
            decode_tok_per_s: repeat.decode_tok_per_s,
            decode_step_ms_p50: p50,
            decode_step_ms_p95: p95,
        }
    }
}

/// Deterministic pseudo-random number generator for token generation.
///
/// Uses xorshift64 for fast, reproducible token sequences.
#[derive(Debug, Clone)]
pub struct TokenRng {
    state: u64,
}

impl TokenRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state (xorshift requirement)
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate the next pseudo-random u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a token ID in the range [0, vocab_size).
    #[inline]
    pub fn next_token(&mut self, vocab_size: u32) -> u16 {
        (self.next_u64() % vocab_size as u64) as u16
    }

    /// Generate a vector of tokens for a batch.
    pub fn generate_batch_tokens(&mut self, batch_size: u32, vocab_size: u32) -> Vec<Vec<u16>> {
        (0..batch_size)
            .map(|_| vec![self.next_token(vocab_size)])
            .collect()
    }

    /// Generate tokens for prime prefill.
    pub fn generate_prefill_tokens(
        &mut self,
        batch_size: u32,
        prefill_len: u32,
        vocab_size: u32,
    ) -> Vec<Vec<u16>> {
        (0..batch_size)
            .map(|_| {
                (0..prefill_len)
                    .map(|_| self.next_token(vocab_size))
                    .collect()
            })
            .collect()
    }
}

/// Decode-only benchmark scenario.
///
/// This scenario measures decode throughput in steady-state "one token per step" mode.
/// Each inference call processes exactly one token per batch element (seq_len=1).
#[derive(Debug, Clone)]
pub struct DecodeScenario {
    config: DecodeConfig,
}

impl DecodeScenario {
    /// Create a new decode scenario with the given configuration.
    pub fn new(config: DecodeConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DecodeConfig {
        &self.config
    }

    /// Execute the decode benchmark.
    ///
    /// # Type Parameters
    ///
    /// - `F`: Inference function that takes `&[Vec<u16>]` tokens and returns `Result<(), E>`
    /// - `P`: Optional prefill function that takes `&[Vec<u16>]` and returns `Result<(), E>`
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Vocabulary size for token generation
    /// * `infer` - Inference function called for each decode step
    /// * `prefill` - Optional prefill function for prime prefill step
    ///
    /// # Returns
    ///
    /// Returns `Ok(DecodeResults)` with all repeat results, or an error if any step fails.
    pub fn run<F, E>(
        &self,
        vocab_size: u32,
        mut infer: F,
    ) -> Result<DecodeResults, E>
    where
        F: FnMut(&[Vec<u16>]) -> Result<(), E>,
    {
        self.run_with_prefill(vocab_size, &mut infer, None::<fn(&[Vec<u16>]) -> Result<(), E>>)
    }

    /// Execute the decode benchmark with optional prefill support.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Vocabulary size for token generation
    /// * `infer` - Inference function called for each decode step
    /// * `prefill` - Optional prefill function for prime prefill step
    pub fn run_with_prefill<F, P, E>(
        &self,
        vocab_size: u32,
        infer: &mut F,
        mut prefill: Option<P>,
    ) -> Result<DecodeResults, E>
    where
        F: FnMut(&[Vec<u16>]) -> Result<(), E>,
        P: FnMut(&[Vec<u16>]) -> Result<(), E>,
    {
        let mut rng = TokenRng::new(self.config.seed);
        let mut results = Vec::with_capacity(self.config.repeats as usize);

        // === Warmup Phase ===
        for _ in 0..self.config.warmup_runs {
            // Prime prefill if configured (also during warmup)
            if let Some(prime_len) = self.config.prime_prefill_len {
                if let Some(ref mut pf) = prefill {
                    let prefill_tokens = rng.generate_prefill_tokens(
                        self.config.batch_size,
                        prime_len,
                        vocab_size,
                    );
                    pf(&prefill_tokens)?;
                }
            }

            // Warmup decode steps
            for _ in 0..self.config.warmup_steps {
                let tokens = rng.generate_batch_tokens(self.config.batch_size, vocab_size);
                infer(&tokens)?;
            }
        }

        // === Measurement Phase ===
        for repeat_idx in 0..self.config.repeats {
            // Optional settle delay
            if let Some(settle_ms) = self.config.settle_ms {
                if repeat_idx > 0 {
                    std::thread::sleep(Duration::from_millis(settle_ms));
                }
            }

            // Reset RNG for reproducibility across repeats
            rng = TokenRng::new(self.config.seed);

            // Prime prefill if configured (untimed)
            if let Some(prime_len) = self.config.prime_prefill_len {
                if let Some(ref mut pf) = prefill {
                    let prefill_tokens = rng.generate_prefill_tokens(
                        self.config.batch_size,
                        prime_len,
                        vocab_size,
                    );
                    pf(&prefill_tokens)?;
                }
            }

            // Pre-generate all tokens for this repeat
            // (minimizes overhead during timed loop)
            let all_tokens: Vec<Vec<Vec<u16>>> = (0..self.config.decode_steps)
                .map(|_| rng.generate_batch_tokens(self.config.batch_size, vocab_size))
                .collect();

            // Timed decode loop
            let mut step_latencies = if self.config.collect_per_step_latencies {
                Some(Vec::with_capacity(self.config.decode_steps as usize))
            } else {
                None
            };

            let start = Instant::now();

            for step_tokens in &all_tokens {
                let step_start = if step_latencies.is_some() {
                    Some(Instant::now())
                } else {
                    None
                };

                infer(step_tokens)?;

                if let (Some(ref mut latencies), Some(ss)) = (&mut step_latencies, step_start) {
                    latencies.push(ss.elapsed().as_secs_f64() * 1000.0);
                }
            }

            let elapsed = start.elapsed();
            let decode_total_ms = elapsed.as_secs_f64() * 1000.0;
            let decode_tokens = self.config.batch_size * self.config.decode_steps;
            let decode_tok_per_s = decode_tokens as f64 / elapsed.as_secs_f64();

            results.push(DecodeRepeatResult {
                repeat_index: repeat_idx,
                decode_total_ms,
                decode_steps: self.config.decode_steps,
                decode_tokens,
                decode_tok_per_s,
                step_latencies_ms: step_latencies,
            });
        }

        // Calculate aggregate statistics
        let throughputs: Vec<f64> = results.iter().map(|r| r.decode_tok_per_s).collect();
        let (median, mean, min, max) = Self::calculate_stats(&throughputs);

        Ok(DecodeResults {
            config: self.config.clone(),
            repeats: results,
            median_tok_per_s: median,
            mean_tok_per_s: mean,
            min_tok_per_s: min,
            max_tok_per_s: max,
        })
    }

    /// Calculate statistics from a slice of values.
    fn calculate_stats(values: &[f64]) -> (f64, f64, f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();

        (median, mean, min, max)
    }
}

/// Builder pattern for DecodeConfig.
#[derive(Debug, Clone, Default)]
pub struct DecodeConfigBuilder {
    config: DecodeConfig,
}

impl DecodeConfigBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of decode steps.
    pub fn decode_steps(mut self, steps: u32) -> Self {
        self.config.decode_steps = steps;
        self
    }

    /// Set the batch size.
    pub fn batch_size(mut self, size: u32) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set the PRNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the number of warmup runs.
    pub fn warmup_runs(mut self, runs: u32) -> Self {
        self.config.warmup_runs = runs;
        self
    }

    /// Set the number of warmup steps per run.
    pub fn warmup_steps(mut self, steps: u32) -> Self {
        self.config.warmup_steps = steps;
        self
    }

    /// Set the number of recorded repeats.
    pub fn repeats(mut self, repeats: u32) -> Self {
        self.config.repeats = repeats;
        self
    }

    /// Set the prime prefill length.
    pub fn prime_prefill_len(mut self, len: Option<u32>) -> Self {
        self.config.prime_prefill_len = len;
        self
    }

    /// Set the settle delay between repeats.
    pub fn settle_ms(mut self, ms: Option<u64>) -> Self {
        self.config.settle_ms = ms;
        self
    }

    /// Enable per-step latency collection.
    pub fn collect_per_step_latencies(mut self, collect: bool) -> Self {
        self.config.collect_per_step_latencies = collect;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> DecodeConfig {
        self.config
    }
}

/// Standard decode step sets as defined in the benchmark plan.
pub mod decode_step_sets {
    /// Fast smoke test (128 steps).
    pub const SMOKE: u32 = 128;

    /// Development benchmark (512 steps).
    pub const DEV: u32 = 512;

    /// Full stable benchmark (2048 steps).
    pub const FULL: u32 = 2048;

    /// All recommended decode step values.
    pub const ALL: [u32; 3] = [SMOKE, DEV, FULL];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_rng_deterministic() {
        let mut rng1 = TokenRng::new(42);
        let mut rng2 = TokenRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_token(65536), rng2.next_token(65536));
        }
    }

    #[test]
    fn test_token_rng_different_seeds() {
        let mut rng1 = TokenRng::new(42);
        let mut rng2 = TokenRng::new(43);

        let tokens1: Vec<u16> = (0..10).map(|_| rng1.next_token(65536)).collect();
        let tokens2: Vec<u16> = (0..10).map(|_| rng2.next_token(65536)).collect();

        assert_ne!(tokens1, tokens2);
    }

    #[test]
    fn test_token_rng_batch_generation() {
        let mut rng = TokenRng::new(42);
        let batch = rng.generate_batch_tokens(4, 65536);

        assert_eq!(batch.len(), 4);
        for tokens in &batch {
            assert_eq!(tokens.len(), 1);
        }
    }

    #[test]
    fn test_token_rng_prefill_generation() {
        let mut rng = TokenRng::new(42);
        let prefill = rng.generate_prefill_tokens(4, 256, 65536);

        assert_eq!(prefill.len(), 4);
        for tokens in &prefill {
            assert_eq!(tokens.len(), 256);
        }
    }

    #[test]
    fn test_decode_config_defaults() {
        let config = DecodeConfig::default();
        assert_eq!(config.decode_steps, 128);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.seed, 42);
        assert_eq!(config.warmup_runs, 1);
        assert_eq!(config.warmup_steps, 64);
        assert_eq!(config.repeats, 5);
        assert!(config.prime_prefill_len.is_none());
        assert!(config.settle_ms.is_none());
        assert!(!config.collect_per_step_latencies);
    }

    #[test]
    fn test_decode_config_builder() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(512)
            .batch_size(4)
            .seed(123)
            .warmup_runs(2)
            .warmup_steps(32)
            .repeats(3)
            .prime_prefill_len(Some(256))
            .settle_ms(Some(100))
            .collect_per_step_latencies(true)
            .build();

        assert_eq!(config.decode_steps, 512);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.seed, 123);
        assert_eq!(config.warmup_runs, 2);
        assert_eq!(config.warmup_steps, 32);
        assert_eq!(config.repeats, 3);
        assert_eq!(config.prime_prefill_len, Some(256));
        assert_eq!(config.settle_ms, Some(100));
        assert!(config.collect_per_step_latencies);
    }

    #[test]
    fn test_decode_scenario_basic() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(10)
            .batch_size(2)
            .warmup_runs(0)
            .repeats(3)
            .build();

        let scenario = DecodeScenario::new(config);

        let mut call_count = 0;
        let result = scenario.run(65536, |tokens| {
            // Verify batch structure
            assert_eq!(tokens.len(), 2);
            for batch_tokens in tokens {
                assert_eq!(batch_tokens.len(), 1); // seq_len=1
            }
            call_count += 1;
            Ok::<_, String>(())
        });

        assert!(result.is_ok());
        let results = result.unwrap();

        // 3 repeats * 10 steps = 30 calls
        assert_eq!(call_count, 30);

        // Verify results
        assert_eq!(results.repeats.len(), 3);
        for repeat in &results.repeats {
            assert_eq!(repeat.decode_steps, 10);
            assert_eq!(repeat.decode_tokens, 20); // 2 * 10
            assert!(repeat.decode_tok_per_s > 0.0);
        }
    }

    #[test]
    fn test_decode_scenario_with_warmup() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(5)
            .batch_size(1)
            .warmup_runs(2)
            .warmup_steps(3)
            .repeats(2)
            .build();

        let scenario = DecodeScenario::new(config);

        let mut call_count = 0;
        let result = scenario.run(65536, |_| {
            call_count += 1;
            Ok::<_, String>(())
        });

        assert!(result.is_ok());

        // 2 warmup_runs * 3 warmup_steps + 2 repeats * 5 steps = 6 + 10 = 16
        assert_eq!(call_count, 16);
    }

    #[test]
    fn test_decode_scenario_with_prefill() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(5)
            .batch_size(2)
            .warmup_runs(1)
            .warmup_steps(2)
            .repeats(2)
            .prime_prefill_len(Some(100))
            .build();

        let scenario = DecodeScenario::new(config);

        let mut decode_count = 0;
        let mut prefill_count = 0;

        let result = scenario.run_with_prefill(
            65536,
            &mut |tokens: &[Vec<u16>]| {
                // Decode: each batch element has 1 token
                assert_eq!(tokens.len(), 2);
                for batch_tokens in tokens {
                    assert_eq!(batch_tokens.len(), 1);
                }
                decode_count += 1;
                Ok::<_, String>(())
            },
            Some(|tokens: &[Vec<u16>]| {
                // Prefill: each batch element has 100 tokens
                assert_eq!(tokens.len(), 2);
                for batch_tokens in tokens {
                    assert_eq!(batch_tokens.len(), 100);
                }
                prefill_count += 1;
                Ok::<_, String>(())
            }),
        );

        assert!(result.is_ok());

        // 1 warmup prefill + 2 measured prefills = 3 prefills
        assert_eq!(prefill_count, 3);

        // 1 warmup_runs * 2 warmup_steps + 2 repeats * 5 steps = 2 + 10 = 12
        assert_eq!(decode_count, 12);
    }

    #[test]
    fn test_decode_scenario_per_step_latencies() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(10)
            .batch_size(1)
            .warmup_runs(0)
            .repeats(1)
            .collect_per_step_latencies(true)
            .build();

        let scenario = DecodeScenario::new(config);

        let result = scenario.run(65536, |_| Ok::<_, String>(()));
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.repeats.len(), 1);

        let repeat = &results.repeats[0];
        assert!(repeat.step_latencies_ms.is_some());
        assert_eq!(repeat.step_latencies_ms.as_ref().unwrap().len(), 10);
    }

    #[test]
    fn test_decode_scenario_error_propagation() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(10)
            .batch_size(1)
            .warmup_runs(0)
            .repeats(1)
            .build();

        let scenario = DecodeScenario::new(config);

        let mut call_count = 0;
        let result = scenario.run(65536, |_| {
            call_count += 1;
            if call_count >= 5 {
                Err("simulated error")
            } else {
                Ok(())
            }
        });

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "simulated error");
    }

    #[test]
    fn test_decode_results_to_metrics() {
        let config = DecodeConfigBuilder::new()
            .decode_steps(100)
            .batch_size(4)
            .warmup_runs(0)
            .repeats(2)
            .collect_per_step_latencies(true)
            .build();

        let scenario = DecodeScenario::new(config);

        let result = scenario.run(65536, |_| {
            std::thread::sleep(std::time::Duration::from_micros(10));
            Ok::<_, String>(())
        });

        assert!(result.is_ok());
        let results = result.unwrap();

        // Test to_metrics() for first repeat
        let metrics = results.to_metrics();
        assert_eq!(metrics.decode_steps, 100);
        assert_eq!(metrics.decode_tokens, 400); // 4 * 100
        assert!(metrics.decode_tok_per_s > 0.0);
        assert!(metrics.decode_step_ms_p50.is_some());
        assert!(metrics.decode_step_ms_p95.is_some());
    }

    #[test]
    fn test_decode_step_sets() {
        assert_eq!(decode_step_sets::SMOKE, 128);
        assert_eq!(decode_step_sets::DEV, 512);
        assert_eq!(decode_step_sets::FULL, 2048);
        assert_eq!(decode_step_sets::ALL, [128, 512, 2048]);
    }

    #[test]
    fn test_calculate_stats() {
        // Test with odd number of values
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (median, mean, min, max) = DecodeScenario::calculate_stats(&values);
        assert_eq!(median, 3.0);
        assert_eq!(mean, 3.0);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);

        // Test with even number of values
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let (median, mean, min, max) = DecodeScenario::calculate_stats(&values);
        assert_eq!(median, 2.5);
        assert_eq!(mean, 2.5);
        assert_eq!(min, 1.0);
        assert_eq!(max, 4.0);

        // Test with empty slice
        let values: Vec<f64> = vec![];
        let (median, mean, min, max) = DecodeScenario::calculate_stats(&values);
        assert_eq!(median, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_deterministic_across_repeats() {
        // Verify that tokens are deterministic but different across batch elements
        let config = DecodeConfigBuilder::new()
            .decode_steps(5)
            .batch_size(3)
            .seed(42)
            .warmup_runs(0)
            .repeats(2)
            .build();

        let scenario = DecodeScenario::new(config);

        let mut all_tokens: Vec<Vec<Vec<u16>>> = Vec::new();

        let result = scenario.run(65536, |tokens| {
            all_tokens.push(tokens.to_vec());
            Ok::<_, String>(())
        });

        assert!(result.is_ok());

        // Each repeat should start fresh with the same seed
        // So repeat 0 tokens should equal repeat 1 tokens
        assert_eq!(all_tokens.len(), 10); // 2 repeats * 5 steps

        // Compare first 5 (repeat 0) with last 5 (repeat 1)
        for i in 0..5 {
            assert_eq!(all_tokens[i], all_tokens[i + 5]);
        }
    }
}
