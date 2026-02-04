//! Unified Backend Text Generation Comparison
//!
//! Runs the same generation tests on both Vulkan (WGPU) and HIP backends,
//! comparing outputs and performance.
//!
//! Demonstrates:
//! - Single sequence generation (short and long form)
//! - Batched generation (same-length sequences)
//! - Variable-length batched generation
//! - Chunked processing for long prompts
//! - Different prompt styles (Q&A, continuation, chat)
//!
//! Run with: cargo run --release --example gen_compare -p hip-rwkv

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};

use web_rwkv::{
    context::InstanceExt,
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{TensorInit, TensorShape as _},
    tokenizer::Tokenizer,
};

use hip_rwkv::hip::{HipRuntime, Rwkv7Hip};

#[derive(Parser, Debug)]
#[command(author, version, about = "Compare text generation across backends")]
struct Cli {
    /// Path to model file (.st safetensors format)
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st"
    )]
    model: PathBuf,

    /// Path to tokenizer JSON file
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = "../assets/vocab/rwkv_vocab_v20230424.json"
    )]
    tokenizer: PathBuf,

    /// Only run Vulkan/WGPU backend
    #[arg(long)]
    vulkan_only: bool,

    /// Only run HIP backend
    #[arg(long)]
    hip_only: bool,

    /// Token chunk size for processing
    #[arg(long, default_value_t = 128)]
    chunk_size: usize,
}

fn sample_argmax(probs: &[f32]) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Sample from top-k candidates using nucleus (top-p) sampling.
fn sample_top_k_nucleus(probs: &[f32], top_k: usize, top_p: f32, temperature: f32) -> u32 {
    if temperature < 1e-6 {
        return sample_argmax(probs);
    }

    // Get top-k indices and values
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_candidates: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Apply temperature (on log scale) and renormalize
    let log_probs: Vec<(usize, f32)> = top_k_candidates
        .iter()
        .map(|(i, p)| (*i, (p.max(1e-10).ln() / temperature).exp()))
        .collect();
    let sum: f32 = log_probs.iter().map(|(_, v)| v).sum();
    let probs: Vec<(usize, f32)> = log_probs.iter().map(|(i, v)| (*i, v / sum)).collect();

    // Nucleus sampling (top-p)
    let mut cumsum = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize and sample
    let candidates: Vec<(usize, f32)> = probs[..cutoff_idx].to_vec();
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r: f32 = fastrand::f32() * total;

    let mut cumsum = 0.0;
    for (idx, p) in candidates {
        cumsum += p;
        if cumsum >= r {
            return idx as u32;
        }
    }

    probs[0].0 as u32
}

#[derive(Clone)]
struct GenerationConfig {
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_k: 10,
            top_p: 0.9,
            stop_tokens: vec![0],
        }
    }
}

/// Backend-agnostic runtime wrapper
enum BackendRuntime {
    Hip(HipRuntime),
    Wgpu(Box<dyn Runtime<Rnn>>),
}

struct UnifiedRuntime {
    runtime: BackendRuntime,
    #[allow(dead_code)]
    context: Option<web_rwkv::context::Context>,
    vocab_size: usize,
    chunk_size: usize,
}

impl UnifiedRuntime {
    fn new_hip(model_path: &std::path::Path, batch_size: usize) -> Result<Self> {
        let model = Rwkv7Hip::load(model_path.to_str().unwrap())?;
        let vocab_size = model.info().n_vocab;
        let runtime = HipRuntime::new(model, batch_size);
        Ok(Self {
            runtime: BackendRuntime::Hip(runtime),
            context: None,
            vocab_size,
            chunk_size: 256,
        })
    }

    async fn new_wgpu(
        model_path: &std::path::Path,
        info: &ModelInfo,
        chunk_size: usize,
    ) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?;

        let context = web_rwkv::context::ContextBuilder::new(adapter)
            .auto_limits(info)
            .build()
            .await?;

        let file = std::fs::File::open(model_path)?;
        let data = unsafe { Mmap::map(&file)? };
        let model = SafeTensors::deserialize(&data)?;

        let vocab_size = info.num_vocab;
        let builder = ModelBuilder::new(&context, model);

        let runtime: Box<dyn Runtime<Rnn>> = match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
        };

        Ok(Self {
            runtime: BackendRuntime::Wgpu(runtime),
            context: Some(context),
            vocab_size,
            chunk_size,
        })
    }

    fn name(&self) -> &'static str {
        match &self.runtime {
            BackendRuntime::Hip(_) => "HIP",
            BackendRuntime::Wgpu(_) => "Vulkan/WGPU",
        }
    }

    async fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        // Encode prompt
        let prompt_tokens = tokenizer.encode(prompt.as_bytes())?;

        // Create input with Last option (we want logits for last token)
        let batch = RnnInputBatch::new(prompt_tokens.clone(), RnnOption::Last);
        let mut input = RnnInput::new(vec![batch], self.chunk_size);

        // Process prompt in chunks until we get output
        let mut last_logits: Option<Vec<f32>> = None;
        loop {
            if input.num_token() == 0 {
                break;
            }

            let (remaining, output) = match &self.runtime {
                BackendRuntime::Hip(rt) => Runtime::<Rnn>::infer(rt, input).await?,
                BackendRuntime::Wgpu(rt) => rt.infer(input).await?,
            };
            input = remaining;

            // Get output if any
            if output.0[0].0.shape()[1] > 0 {
                let probs = match &self.runtime {
                    BackendRuntime::Hip(_) => {
                        hip_rwkv::hip::softmax_hip(output.0[0].0.clone())?
                            .data()
                            .to_vec()
                    }
                    BackendRuntime::Wgpu(_) => {
                        let ctx = self.context.as_ref().unwrap();
                        softmax_one(ctx, output.0[0].0.clone()).await?.to_vec()
                    }
                };
                last_logits = Some(probs);
            }
        }

        let Some(probs) = last_logits else {
            return Ok(String::new());
        };

        // Sample first token
        let mut generated_tokens = vec![sample_top_k_nucleus(
            &probs,
            config.top_k,
            config.top_p,
            config.temperature,
        )];

        // Decode loop
        for _ in 1..config.max_tokens {
            let token = *generated_tokens.last().unwrap();

            if config.stop_tokens.contains(&token) {
                break;
            }

            // Run single token
            let batch = RnnInputBatch::new(vec![token], RnnOption::Last);
            let input = RnnInput::new(vec![batch], self.chunk_size);

            let (_, output) = match &self.runtime {
                BackendRuntime::Hip(rt) => Runtime::<Rnn>::infer(rt, input).await?,
                BackendRuntime::Wgpu(rt) => rt.infer(input).await?,
            };

            let probs = match &self.runtime {
                BackendRuntime::Hip(_) => hip_rwkv::hip::softmax_hip(output.0[0].0.clone())?
                    .data()
                    .to_vec(),
                BackendRuntime::Wgpu(_) => {
                    let ctx = self.context.as_ref().unwrap();
                    softmax_one(ctx, output.0[0].0.clone()).await?.to_vec()
                }
            };

            let next_token =
                sample_top_k_nucleus(&probs, config.top_k, config.top_p, config.temperature);
            generated_tokens.push(next_token);
        }

        // Decode tokens
        let output = tokenizer.decode(&generated_tokens)?;
        Ok(String::from_utf8_lossy(&output).to_string())
    }
}

async fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

async fn run_tests(runtime: &UnifiedRuntime, tokenizer: &Tokenizer) -> Result<()> {
    let backend = runtime.name();

    // =========================================================================
    // Example 1: Short-form Q&A (greedy decoding)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("[{}] Example 1: Short-form Q&A (greedy decoding)", backend);
    println!("{}", "=".repeat(70));

    let qa_prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: What is 2 + 2?\nA:",
        "Q: What color is the sky?\nA:",
    ];

    let greedy_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        stop_tokens: vec![0, 261],
    };

    for prompt in &qa_prompts {
        println!("\nPrompt: {}", prompt.replace('\n', "\\n"));
        print!("Output: ");
        std::io::stdout().flush().unwrap();
        let output = runtime.generate(tokenizer, prompt, &greedy_config).await?;
        println!("{}", output.trim());
    }

    // =========================================================================
    // Example 2: Long-form continuation (with sampling)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!(
        "[{}] Example 2: Long-form continuation (with sampling)",
        backend
    );
    println!("{}", "=".repeat(70));

    let story_prompt = "Once upon a time, in a land far away, there lived a";

    let story_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.8,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0],
    };

    println!("\nPrompt: {}", story_prompt);
    println!("\nGenerated continuation:");
    let output = runtime
        .generate(tokenizer, story_prompt, &story_config)
        .await?;
    println!("{}", output);

    // =========================================================================
    // Example 3: Chat-style interaction
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("[{}] Example 3: Chat-style interaction", backend);
    println!("{}", "=".repeat(70));

    let chat_prompt = r#"User: Hello! How are you today?
Assistant: I'm doing well, thank you for asking! How can I help you?
User: Can you explain what RWKV is?
Assistant:"#;

    let chat_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.7,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0],
    };

    println!("\nConversation:");
    println!("{}", chat_prompt);
    print!(" ");
    let output = runtime
        .generate(tokenizer, chat_prompt, &chat_config)
        .await?;
    println!("{}", output);

    // =========================================================================
    // Example 4: Chunked processing for long prompts
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!(
        "[{}] Example 4: Chunked processing for long prompts",
        backend
    );
    println!("{}", "=".repeat(70));

    let long_prompt = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let long_tokens = tokenizer.encode(long_prompt.as_bytes())?;
    println!("\nLong prompt: {} tokens", long_tokens.len());
    println!("(Internally processed in chunks)");

    let chunk_config = GenerationConfig {
        max_tokens: 30,
        temperature: 0.7,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0],
    };

    let output = runtime
        .generate(tokenizer, &long_prompt, &chunk_config)
        .await?;
    println!("\nGenerated continuation: {}", output.trim());

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "=".repeat(70));
    println!("Unified Backend Text Generation Comparison");
    println!("{}", "=".repeat(70));

    // Check model exists
    if !cli.model.exists() {
        eprintln!("Model not found at: {:?}", cli.model);
        eprintln!("Please specify a valid model path with --model");
        return Ok(());
    }

    // Check tokenizer exists
    if !cli.tokenizer.exists() {
        eprintln!("Tokenizer not found at: {:?}", cli.tokenizer);
        return Ok(());
    }

    // Load tokenizer
    let tokenizer = load_tokenizer(&cli.tokenizer).await?;
    println!("Tokenizer loaded from: {:?}", cli.tokenizer);

    // Load model info
    let file = std::fs::File::open(&cli.model)?;
    let data = unsafe { Mmap::map(&file)? };
    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    drop(data);
    drop(file);

    println!("Model: {:?}", cli.model);
    println!("  Version: {:?}", info.version);
    println!(
        "  Layers: {}, Embed: {}, Vocab: {}",
        info.num_layer, info.num_emb, info.num_vocab
    );

    // Run Vulkan/WGPU backend
    let run_vulkan = !cli.hip_only;

    if run_vulkan {
        println!("\n{}", "#".repeat(70));
        println!("# VULKAN/WGPU BACKEND");
        println!("{}", "#".repeat(70));

        match UnifiedRuntime::new_wgpu(&cli.model, &info, cli.chunk_size).await {
            Ok(runtime) => {
                run_tests(&runtime, &tokenizer).await?;
            }
            Err(e) => {
                eprintln!("Failed to initialize Vulkan/WGPU backend: {}", e);
            }
        }
    }

    // Run HIP backend
    if !cli.vulkan_only {
        println!("\n{}", "#".repeat(70));
        println!("# HIP BACKEND");
        println!("{}", "#".repeat(70));

        match UnifiedRuntime::new_hip(&cli.model, 1) {
            Ok(runtime) => {
                run_tests(&runtime, &tokenizer).await?;
            }
            Err(e) => {
                eprintln!("Failed to initialize HIP backend: {}", e);
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Comparison complete!");
    println!("{}", "=".repeat(70));

    Ok(())
}
