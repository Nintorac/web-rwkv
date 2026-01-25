//! HIP Backend Text Generation Examples
//!
//! Demonstrates the HIP backend with various inference scenarios:
//! - Single sequence generation (short and long form)
//! - Batched generation (same-length sequences)
//! - Different prompt styles (Q&A, continuation, chat)
//!
//! Run with: cargo run --example hip_gen --features hip
//!
//! TODO: Add variable-length batch examples once length-masked kernel is implemented
//! TODO: Add streaming/chunked generation examples
//! TODO: Add state save/restore examples

#[cfg(feature = "hip")]
use std::io::Write;

#[cfg(feature = "hip")]
use web_rwkv::hip::{HipRuntime, Rwkv7Hip};

#[cfg(feature = "hip")]
fn sample_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Sample from top-k logits using nucleus (top-p) sampling.
/// Only considers the top-k candidates for efficiency.
#[cfg(feature = "hip")]
fn sample_top_k_nucleus(logits: &[f32], top_k: usize, top_p: f32, temperature: f32) -> u32 {
    if temperature < 1e-6 {
        return sample_argmax(logits);
    }

    // Get top-k indices and values
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_candidates: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Apply temperature and softmax only to top-k
    let max_val = top_k_candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<(usize, f32)> = top_k_candidates
        .iter()
        .map(|(i, v)| (*i, ((v - max_val) / temperature).exp()))
        .collect();
    let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();
    let probs: Vec<(usize, f32)> = exp_vals.iter().map(|(i, v)| (*i, v / sum)).collect();

    // Nucleus sampling (top-p) within top-k
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

    probs[0].0 as u32 // Fallback to top-1
}

#[cfg(feature = "hip")]
struct GenerationConfig {
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    stop_tokens: Vec<u32>,
}

#[cfg(feature = "hip")]
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_k: 10,  // Only consider top-10 logits
            top_p: 0.9,
            stop_tokens: vec![0], // EOS token
        }
    }
}

#[cfg(feature = "hip")]
fn generate_single(
    runtime: &HipRuntime,
    tokenizer: &web_rwkv::tokenizer::Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> String {
    // Reset state for fresh generation
    runtime.reset_state();

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt.as_bytes()).expect("Failed to encode prompt");
    println!("  Prompt tokens: {} tokens", prompt_tokens.len());

    // Prefill: process entire prompt
    let logits = runtime
        .infer_one(&prompt_tokens)
        .expect("Failed to run prefill");

    // Get logits for last token position
    let vocab_size = runtime.info().n_vocab;
    let last_token_start = (prompt_tokens.len() - 1) * vocab_size;
    let last_logits = &logits.data()[last_token_start..last_token_start + vocab_size];

    // Sample first generated token
    let mut generated_tokens = vec![sample_top_k_nucleus(last_logits, config.top_k, config.top_p, config.temperature)];

    // Decode loop
    for _ in 1..config.max_tokens {
        let token = *generated_tokens.last().unwrap();

        // Check stop condition
        if config.stop_tokens.contains(&token) {
            break;
        }

        // Run single token through model
        let logits = runtime.infer_one(&[token]).expect("Failed to run decode");
        let next_token = sample_top_k_nucleus(logits.data(), config.top_k, config.top_p, config.temperature);
        generated_tokens.push(next_token);
    }

    // Decode tokens to text
    let output = tokenizer
        .decode(&generated_tokens)
        .expect("Failed to decode tokens");
    String::from_utf8_lossy(&output).to_string()
}

#[cfg(feature = "hip")]
fn generate_batched(
    model_path: &str,
    tokenizer: &web_rwkv::tokenizer::Tokenizer,
    prompts: &[&str],
    config: &GenerationConfig,
) -> Vec<String> {
    let batch_size = prompts.len();

    // Load model with correct batch size
    let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
    let runtime = HipRuntime::new(model, batch_size);

    // Encode all prompts
    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p.as_bytes()).expect("Failed to encode"))
        .collect();

    // Check all same length (current limitation)
    let max_len = prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
    let all_same_len = prompt_tokens.iter().all(|t| t.len() == max_len);
    if !all_same_len {
        panic!("Variable-length batching not yet implemented");
    }

    println!("  Batch size: {}, Sequence length: {}", batch_size, max_len);

    // Prefill
    let token_refs: Vec<&[u32]> = prompt_tokens.iter().map(|t| t.as_slice()).collect();
    let logits = runtime.infer(&token_refs).expect("Failed to run batched prefill");
    let vocab_size = runtime.info().n_vocab;

    // Get last token logits for each sequence
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
    for b in 0..batch_size {
        let last_pos = (b * max_len + max_len - 1) * vocab_size;
        let last_logits = &logits.data()[last_pos..last_pos + vocab_size];
        let token = sample_top_k_nucleus(last_logits, config.top_k, config.top_p, config.temperature);
        generated[b].push(token);
    }

    // Decode loop
    for _ in 1..config.max_tokens {
        let tokens: Vec<u32> = generated.iter().map(|g| *g.last().unwrap()).collect();

        if tokens.iter().all(|t| config.stop_tokens.contains(t)) {
            break;
        }

        let token_slices: Vec<&[u32]> = tokens.iter().map(|t| std::slice::from_ref(t)).collect();
        let logits = runtime.infer(&token_slices).expect("Failed to run batched decode");

        for b in 0..batch_size {
            if !config.stop_tokens.contains(generated[b].last().unwrap()) {
                let batch_logits = &logits.data()[b * vocab_size..(b + 1) * vocab_size];
                let next_token = sample_top_k_nucleus(batch_logits, config.top_k, config.top_p, config.temperature);
                generated[b].push(next_token);
            }
        }
    }

    // Decode all sequences
    generated
        .iter()
        .map(|tokens| {
            let output = tokenizer.decode(tokens).expect("Failed to decode");
            String::from_utf8_lossy(&output).to_string()
        })
        .collect()
}

#[cfg(feature = "hip")]
fn main() {
    use std::path::Path;

    println!("{}", "=".repeat(70));
    println!("HIP Backend Text Generation Examples");
    println!("{}", "=".repeat(70));

    // Model path
    let model_path = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at: {}", model_path);
        eprintln!("Please download a RWKV7 model first.");
        return;
    }

    // Load tokenizer
    let tokenizer_path = "assets/vocab/rwkv_vocab_v20230424.json";
    let tokenizer = if Path::new(tokenizer_path).exists() {
        let contents = std::fs::read_to_string(tokenizer_path).expect("Failed to read tokenizer");
        web_rwkv::tokenizer::Tokenizer::new(&contents).expect("Failed to create tokenizer")
    } else {
        eprintln!("Tokenizer not found at: {}", tokenizer_path);
        eprintln!("Skipping text decode, showing token IDs only.");
        // We'll handle this below
        return;
    };

    // Load model
    println!("\nLoading model from: {}", model_path);
    let model = Rwkv7Hip::load(model_path).expect("Failed to load model");
    println!(
        "Model loaded: {} layers, {} embed, {} vocab",
        model.info.n_layer, model.info.n_embd, model.info.n_vocab
    );

    // Create runtime with batch_size=1 for single-sequence examples
    let runtime = HipRuntime::new(model, 1);
    println!("Runtime created with batch size: {}", runtime.num_batch());

    // =========================================================================
    // Example 1: Short-form Q&A (greedy decoding)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Example 1: Short-form Q&A (greedy decoding)");
    println!("{}", "=".repeat(70));

    let qa_prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: What is 2 + 2?\nA:",
        "Q: What color is the sky?\nA:",
    ];

    let greedy_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0, // greedy (ignores top_k/top_p)
        top_k: 1,
        top_p: 1.0,
        stop_tokens: vec![0, 261], // EOS and newline
    };

    for prompt in &qa_prompts {
        println!("\nPrompt: {}", prompt.replace('\n', "\\n"));
        print!("Output: ");
        std::io::stdout().flush().unwrap();
        let output = generate_single(&runtime, &tokenizer, prompt, &greedy_config);
        println!("{}", output.trim());
    }

    // =========================================================================
    // Example 2: Long-form continuation (with sampling)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Example 2: Long-form continuation (with sampling)");
    println!("{}", "=".repeat(70));

    let story_prompt = "Once upon a time, in a land far away, there lived a";

    let story_config = GenerationConfig {
        max_tokens: 100,
        temperature: 0.8,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0],
    };

    println!("\nPrompt: {}", story_prompt);
    println!("\nGenerated continuation:");
    let output = generate_single(&runtime, &tokenizer, story_prompt, &story_config);
    println!("{}", output);

    // =========================================================================
    // Example 3: Batched generation (same-length prompts)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Example 3: Batched generation (same-length prompts)");
    println!("{}", "=".repeat(70));

    let batch_prompts = ["The sun is", "The moon is"];

    let batch_config = GenerationConfig {
        max_tokens: 30,
        temperature: 0.7,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0, 261],
    };

    println!("\nPrompts:");
    for (i, p) in batch_prompts.iter().enumerate() {
        println!("  [{}] {}", i, p);
    }

    println!("\nLoading model with batch_size=2...");
    let outputs = generate_batched(model_path, &tokenizer, &batch_prompts, &batch_config);

    println!("\nOutputs:");
    for (i, (prompt, output)) in batch_prompts.iter().zip(outputs.iter()).enumerate() {
        println!("  [{}] {} -> {}", i, prompt, output.trim());
    }

    // =========================================================================
    // Example 4: Chat-style interaction
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Example 4: Chat-style interaction");
    println!("{}", "=".repeat(70));

    let chat_prompt = r#"User: Hello! How are you today?
Assistant: I'm doing well, thank you for asking! How can I help you?
User: Can you explain what RWKV is?
Assistant:"#;

    let chat_config = GenerationConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_k: 10,
        top_p: 0.9,
        stop_tokens: vec![0],
    };

    println!("\nConversation:");
    println!("{}", chat_prompt);
    print!(" ");
    let output = generate_single(&runtime, &tokenizer, chat_prompt, &chat_config);
    println!("{}", output);

    // =========================================================================
    // TODO: Example 5: Variable-length batched generation
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("TODO: Example 5: Variable-length batched generation");
    println!("{}", "=".repeat(70));
    println!("\nNot yet implemented. Requires:");
    println!("  - Length-masked WKV7 kernel (bd-2sh.7.2)");
    println!("  - Rust wrapper for masked forward pass (bd-2sh.7.3)");
    println!("  - Variable-length batching support (bd-2sh.7.6)");
    println!("\nExample prompts that would be batched:");
    println!("  [0] \"Hi\" (2 tokens)");
    println!("  [1] \"Hello, how are you?\" (5 tokens)");
    println!("  [2] \"What is the meaning of life?\" (7 tokens)");

    // =========================================================================
    // TODO: Example 6: Streaming/chunked generation
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("TODO: Example 6: Streaming/chunked generation");
    println!("{}", "=".repeat(70));
    println!("\nNot yet implemented. Would demonstrate:");
    println!("  - Processing long prompts in chunks");
    println!("  - State persistence across chunks");
    println!("  - v_first handling for RWKV7 value residual");

    // =========================================================================
    // TODO: Example 7: State save/restore
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("TODO: Example 7: State save/restore");
    println!("{}", "=".repeat(70));
    println!("\nNot yet implemented. Would demonstrate:");
    println!("  - Saving state after processing a prompt");
    println!("  - Restoring state to continue from checkpoint");
    println!("  - Multiple conversation branches from same state");

    println!("\n{}", "=".repeat(70));
    println!("Examples complete!");
    println!("{}", "=".repeat(70));
}

#[cfg(not(feature = "hip"))]
fn main() {
    println!("HIP feature not enabled. Run with: cargo run --example hip_gen --features hip");
}
