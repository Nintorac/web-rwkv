# Plan: Extract hip-rwkv Crate + Integrate with ai00 Server

## Overview

1. **Create `hip-rwkv` crate** — extracts the HIP module from this repo into its own crate that **depends on `web-rwkv`** for shared types (`Runtime<Rnn>`, `TensorCpu`, `State`, `ModelInfo`, etc.)
2. **ai00 keeps its existing `web-rwkv` dep** — both ai00 and hip-rwkv resolve to the **same** web-rwkv via `[patch.crates-io]`
3. **ai00 adds `hip-rwkv`** as a new optional dependency (local path) for HIP backend
4. **Backend selection** via `Config.toml` — since both crates share `web-rwkv` types, the existing `run.rs` and `State` trait can be reused
5. **GPU softmax on HIP** — a real HIP kernel replaces `softmax_one_cpu`; ai00 dispatches via `SoftmaxBackend` enum (no CPU fallback, no `Option<Context>`)

Key insight: because `hip-rwkv` depends on `web-rwkv`, `HipRuntime` implements the **same** `web_rwkv::Runtime<Rnn>` trait that ai00 already uses. Types like `TensorCpu`, `ModelInfo`, and `State` are shared. This means ai00 can use `dyn Runtime<Rnn>` for both backends and reuse `run.rs`.

---

## Part 1: Create `hip-rwkv` Crate

### Step 1: New crate + build script

Create `hip-rwkv/` as a workspace member of web-rwkv. The crate contains:
- The contents of `src/hip/` as its primary module
- `Cargo.toml` depending on `web-rwkv` via **workspace version inheritance**
- `build.rs` migrated from the root build script's `#[cfg(feature = "hip")]` block

```
hip-rwkv/
  Cargo.toml
  build.rs            # kernel compilation (moved from root build.rs)
  src/
    lib.rs            # re-exports from hip module
    hip/              # moved from src/hip/
      mod.rs
      runtime.rs
      ffi.rs
      device.rs
      buffer.rs
      kernels/
        softmax.hip   # NEW: GPU softmax kernel
        ...
```

**`hip-rwkv/Cargo.toml`:**
```toml
[package]
name = "hip-rwkv"
version = "0.1.0"
edition = "2021"

[dependencies]
web-rwkv = { version = "0.10.19", default-features = false }
# Only the types: TensorCpu, Shape, TensorInit, Runtime, Rnn, etc.
# No wgpu, no WebGPU features needed.
half = { workspace = true }
# ... other deps the HIP code currently uses from the root Cargo.toml

[features]
default = []
hip-probes = []
```

**`hip-rwkv/build.rs`:**
Move the entire `#[cfg(feature = "hip")] mod hip { ... }` block from the root `build.rs` into `hip-rwkv/build.rs`, but **unconditionally** (this crate always builds HIP kernels):
```rust
fn main() {
    build_hip_kernels();
}

fn build_hip_kernels() {
    // Same logic as current root build.rs hip::build():
    // - println!("cargo:rerun-if-changed=src/hip/kernels/")
    // - Compile all .hip files via hipcc
    // - Link libhip_kernels.a, amdhip64, rocblas, hipblaslt
}
```

### Step 2: Clean up root web-rwkv crate

After moving `src/hip/` into hip-rwkv, the root crate must be updated:

**`Cargo.toml`:**
- Add `"hip-rwkv"` to `[workspace].members`
- Remove `hip = []` and `hip-probes = ["hip"]` from `[features]`

**`build.rs`:**
- Remove the entire `#[cfg(feature = "hip")] mod hip { ... }` block
- build.rs becomes just `fn main() {}`  (or delete it if empty)

**`src/lib.rs`:**
- Remove `#[cfg(feature = "hip")] pub mod hip;`

### Step 2b: Migrate tests, examples, and clean up root crate references

After removing `src/hip/` and the `hip` feature, everything in the root crate that imports `web_rwkv::hip::*` will break. These all need to move to `hip-rwkv` or be updated.

**Tests to move into `hip-rwkv/tests/`** (8 files — they all `use web_rwkv::hip::*`):

| Root file | Action |
|-----------|--------|
| `tests/hip_validation.rs` | Move → `hip-rwkv/tests/` ; change `web_rwkv::hip::` → `hip_rwkv::hip::` |
| `tests/hip_probes.rs` | Move ; change imports |
| `tests/hip_layer_validation.rs` | Move ; change imports |
| `tests/hip_fla_validation.rs` | Move ; change imports |
| `tests/hip_profiling.rs` | Move ; change imports |
| `tests/hipblaslt_benchmark.rs` | Move ; change imports |
| `tests/multi_batch_stress.rs` | Move ; change imports ; `#[cfg(feature = "hip")]` → unconditional |
| `tests/ground_truth.rs` | Move ; change imports ; HIP section becomes unconditional |

**Tests that stay in root but lose HIP sections:**

| Root file | Action |
|-----------|--------|
| `tests/benchmarks.rs` | Remove `#[cfg(feature = "hip")]` sections (lines ~44, 761–783) |
| `tests/functional_metrics.rs` | Remove `#[cfg(feature = "hip")]` sections (lines ~106–246) |

**Examples to move into `hip-rwkv/examples/`** (2 files):

| Root file | Action |
|-----------|--------|
| `examples/hip_test.rs` | Move ; `web_rwkv::hip::` → `hip_rwkv::hip::` |
| `examples/hip_gen.rs` | Move ; change imports ; remove `#[cfg(feature = "hip")]` gates (unconditional in new crate) |

**Examples that stay but need changes:**

| Root file | Action |
|-----------|--------|
| `examples/gen_compare.rs` | Remove entirely or split: HIP path can't import `web_rwkv::hip` anymore. The comparison needs both `web-rwkv` and `hip-rwkv` as deps — move to `hip-rwkv/examples/` with both deps, or delete (the parity test in `runtime.rs` covers the same ground). Also has 2 `softmax_one_cpu` calls (lines 275, 320) → `softmax_hip`. |

**Bench crate (`crates/web-rwkv-bench/`):**

String references to `"hip"` in config/skip/sweep logic (config.rs, skip.rs, sweep.rs, jsonl.rs) are just backend ID strings — they don't import the hip module and don't break. But to actually *run* HIP benchmarks, the sweep runner would need `hip-rwkv` as an optional dependency. This can be deferred.

**`tests/common/` directory:**
Check if any shared test helpers reference hip types — if so, split or move.

### Step 3: Update imports in HIP code

In all `src/hip/*.rs` files that use `crate::` to reach web-rwkv types, change to `web_rwkv::`:
```rust
// Before (in runtime.rs):
use crate::runtime::{infer::{Rnn, RnnInput, ...}, Runtime, RuntimeError, JobInput};
use crate::tensor::{TensorCpu, Shape, TensorInit, ...};

// After:
use web_rwkv::runtime::{infer::{Rnn, RnnInput, ...}, Runtime, RuntimeError, JobInput};
use web_rwkv::tensor::{TensorCpu, Shape, TensorInit, ...};
```

Only `runtime.rs` has these `crate::` imports; all other HIP files use `super::` within the hip module.

### Step 4: `hip-rwkv/src/lib.rs`

```rust
pub mod hip;  // unconditional — this crate IS the HIP backend
```

Re-exports the same public API that `web-rwkv`'s `hip` module currently provides.

### Step 5: Add load_state/get_state to HipRuntime

File: `hip-rwkv/src/hip/runtime.rs`

Add convenience methods (thin wrappers over existing `HipPrefill` methods):
```rust
pub fn load_state(&self, state: &HipState) -> Result<(), HipErrorKind>
pub fn get_state(&self) -> Result<HipState, HipErrorKind>
```

These are needed for the State adapter in ai00.

### Step 6: GPU softmax kernel + drop CPU softmax

**New file: `hip-rwkv/src/hip/kernels/softmax.hip`**

Three-pass numerically-stable softmax using the same reduction primitives as `norm.hip`:

```c
#include <hip/hip_runtime.h>

extern "C" {

#define SOFTMAX_BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ float block_reduce_max(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Per-token softmax over the vocab dimension.
// Input layout: [V, T, 1, 1] — V contiguous elements per token, T tokens.
// One block per token.
__global__ void kernel_softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int V    // vocab size (elements per token)
) {
    int token = blockIdx.x;
    int base = token * V;

    __shared__ float smem[SOFTMAX_BLOCK_SIZE / 32];
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Pass 1: find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        local_max = fmaxf(local_max, input[base + i]);
    }
    float global_max = block_reduce_max(local_max, smem);
    if (threadIdx.x == 0) shared_max = global_max;
    __syncthreads();
    float max_val = shared_max;

    // Pass 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        local_sum += expf(input[base + i] - max_val);
    }
    float global_sum = block_reduce_sum(local_sum, smem);
    if (threadIdx.x == 0) shared_sum = global_sum;
    __syncthreads();
    float inv_sum = 1.0f / shared_sum;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        output[base + i] = expf(input[base + i] - max_val) * inv_sum;
    }
}

hipError_t launch_softmax_f32(
    const float* input,
    float* output,
    int V,          // vocab size
    int T,          // number of tokens
    hipStream_t stream
) {
    if (V <= 0 || T <= 0) return hipSuccess;
    hipLaunchKernelGGL(kernel_softmax_f32,
        dim3(T), dim3(SOFTMAX_BLOCK_SIZE), 0, stream,
        input, output, V);
    return hipGetLastError();
}

} // extern "C"
```

**`hip-rwkv/src/hip/ffi.rs`** — add:
```rust
pub fn launch_softmax_f32(
    input: *const f32,
    output: *mut f32,
    vocab_size: c_int,
    num_tokens: c_int,
    stream: HipStream,
) -> HipError;
```

**`hip-rwkv/src/hip/kernels/` (Rust wrapper)** — add `softmax_f32()`:
```rust
pub fn softmax_f32(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    vocab_size: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        check(launch_softmax_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            vocab_size as c_int,
            num_tokens as c_int,
            stream.handle(),
        ))
    }
}
```

**`hip-rwkv/src/hip/runtime.rs`** — replace `softmax_one_cpu` with GPU version:
```rust
/// GPU softmax for HIP backend.
///
/// Uploads input to GPU, runs softmax kernel, downloads result.
/// Input shape: [vocab_size, num_tokens, 1, 1].
pub fn softmax_hip(input: TensorCpu<f32>) -> Result<TensorCpu<f32>, HipErrorKind> {
    let shape = input.shape();
    if shape.len() == 0 {
        return Ok(input);
    }
    let vocab_size = shape[0];
    let num_tokens = input.data().len() / vocab_size;

    let stream = Stream::new()?;
    let mut d_input = DeviceBuffer::<f32>::new(input.data().len())?;
    let mut d_output = DeviceBuffer::<f32>::new(input.data().len())?;

    d_input.copy_from_host(input.data(), &stream)?;
    softmax_f32(&d_input, &mut d_output, vocab_size, num_tokens, &stream)?;

    let mut output = vec![0.0f32; input.data().len()];
    d_output.copy_to_host(&mut output, &stream)?;
    stream.synchronize()?;

    TensorInit::from_data(shape, output).map_err(|e| HipErrorKind {
        code: -1,
        message: format!("softmax output tensor: {}", e),
    })
}

/// Batched GPU softmax — processes multiple tensors in one kernel launch.
pub fn softmax_hip_batch(inputs: Vec<TensorCpu<f32>>) -> Result<Vec<TensorCpu<f32>>, HipErrorKind> {
    inputs.into_iter().map(softmax_hip).collect()
}
```

**Delete** `softmax_one_cpu` and remove its export from `mod.rs`.

**`hip-rwkv/src/hip/mod.rs`** — change export:
```rust
// Before:
pub use runtime::{softmax_one_cpu, HipRuntime};
// After:
pub use runtime::{softmax_hip, softmax_hip_batch, HipRuntime};
```

**Migrate existing tests** — all `softmax_one_cpu` calls in tests become `softmax_hip`:

| Test | Change |
|------|--------|
| `test_softmax_cpu_basic` → `test_softmax_hip_basic` | Call `softmax_hip` instead; same assertions (sum ≈ 1.0, monotonicity) |
| `test_softmax_cpu_numerical_stability` → `test_softmax_hip_numerical_stability` | Same: large inputs (1000+), check no NaN/Inf |
| `test_softmax_cpu_multiple_tokens` → `test_softmax_hip_multiple_tokens` | Same: [3, 2, 1, 1] shape, per-token sum ≈ 1.0 |
| `test_softmax_cpu_empty` → `test_softmax_hip_empty` | Same: empty tensor returns empty |
| `test_hip_runtime_infer_proof_of_life` (line 725) | Swap `softmax_one_cpu(logits)` → `softmax_hip(logits)` |

These tests now exercise the GPU kernel and require a HIP device to run. The parity test's private `stable_softmax` helper (line 1068) is unaffected — it's a standalone function in the test module, not `softmax_one_cpu`.

**Note on error types:** `softmax_one_cpu` returned `Result<_, TensorError>` while `softmax_hip` returns `Result<_, HipErrorKind>`. The test call sites just `.unwrap()`, so the type change is transparent. If any non-test code pattern-matches on the error type, update it.

---

## Part 2: ai00 Server Integration

### Step 7: Dependency resolution — single web-rwkv

**`repos/ai00_server/Cargo.toml`** (workspace root):
```toml
[workspace.dependencies.web-rwkv]
# path = "../web-rwkv"        # uncomment for local dev
default-features = false
features = ["native"]
version = "0.10.19"            # match web-rwkv workspace version

[workspace.dependencies.hip-rwkv]
path = "../../hip-rwkv"

# Ensure hip-rwkv resolves to the SAME web-rwkv:
[patch.crates-io]
hf-hub = { git = "https://github.com/cgisky1980/hf-hub.git", branch = "main" }
web-rwkv = { path = "../web-rwkv" }   # single source of truth
```

The `[patch.crates-io]` entry forces **both** ai00's direct `web-rwkv` dep and hip-rwkv's transitive `web-rwkv` dep to resolve to the same local checkout. This eliminates duplicate types and trait mismatches.

**`repos/ai00_server/crates/ai00-core/Cargo.toml`:**
```toml
[features]
default = []
hip = ["dep:hip-rwkv"]

[dependencies]
hip-rwkv = { workspace = true, optional = true }
```

**`repos/ai00_server/crates/ai00-server/Cargo.toml`:**
```toml
[features]
default = ["embed"]
embed = [...]
hip = ["ai00-core/hip"]
```

### Step 8: Backend Config

**`repos/ai00_server/crates/ai00-core/src/reload.rs`:**
```rust
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    WebGpu,
    Hip,
}
```

Add `#[serde(default)] backend: Backend` to the `Model` struct.
The `#[serde(default)]` ensures existing Config.toml files without a `backend` field deserialize as `Backend::WebGpu` instead of failing.

**`repos/ai00_server/crates/ai00-core/src/lib.rs`:**
- Add `backend: reload::Backend` to `ReloadRequest`

**`repos/ai00_server/crates/ai00-server/src/config.rs`:**
- Pass `backend` through `TryFrom<Config> for ReloadRequest`

**`repos/ai00_server/assets/configs/Config.toml`:**
```toml
[model]
# backend = "WebGpu"  # or "Hip" for AMD GPUs via ROCm
# Omitting defaults to WebGpu.
```

### Step 9: HipStateAdapter — New File

**`repos/ai00_server/crates/ai00-core/src/hip_state.rs`**

Since `hip-rwkv` uses `web-rwkv` types, implement `web_rwkv::runtime::model::State` directly:

- `num_batch()`: return configured batch size
- `init_shape()`: `[n_embd, head_size + 2, n_layer, 1]` (matches v7 WebGPU format)
- `init()`: zero tensor
- `load(tensor, batch)`: convert `TensorCpu` → `HipState`, call `runtime.load_state()`
- `back(batch)`: call `runtime.get_state()`, convert `HipState` → `TensorCpu`
- `att()`, `ffn()`, `write()`, `read()`: return `Err` (unsupported — Choose path won't work)
- `embed(layer, backed)`: slice backed state tensor

### Step 10: HIP Load Path

**`repos/ai00_server/crates/ai00-core/src/lib.rs`:**

- Add `#[cfg(feature = "hip")] mod hip_state;`
- Add `#[cfg(feature = "hip")] async fn load_runtime_hip(...)`:
  - Validates V7 model version
  - `hip_rwkv::hip::Rwkv7Hip::load(path)` via `spawn_blocking`
  - `hip_rwkv::hip::HipRuntime::with_config(model, config)`
  - Creates `HipStateAdapter` wrapping the runtime
  - Returns `(states, runtime, state)` — same types as `load_runtime()`
- Branch in `ThreadRequest::Reload` on `request.backend`:
  - `Backend::WebGpu` → existing path (unchanged)
  - `Backend::Hip` → `load_runtime_hip()` + build `SoftmaxBackend::Hip`

### Step 11: Softmax backend dispatch — no CPU fallback, no Option\<Context\>

**`repos/ai00_server/crates/ai00-core/src/run.rs`:**

Add a `SoftmaxBackend` enum that owns its resources:
```rust
#[derive(Clone)]
enum SoftmaxBackend {
    WebGpu(Context),
    #[cfg(feature = "hip")]
    Hip,
}
```

Change the `softmax()` async task to dispatch on the enum:
```rust
async fn softmax(
    reload: Arc<ReloadRequest>,
    backend: SoftmaxBackend,
    receiver: Receiver<SoftmaxBatch>,
) -> Result<()> {
    let mut batches = Vec::with_capacity(reload.max_batch);

    while let Ok(batch) = receiver.recv_async().await {
        batches.push(batch);
        for batch in receiver.drain() {
            batches.push(batch);
        }

        let input: Vec<TensorCpu<f32>> =
            batches.iter().map(|b| b.input.clone()).collect();

        let output = match &backend {
            SoftmaxBackend::WebGpu(ctx) => {
                web_rwkv::runtime::softmax::softmax(ctx, input).await?
            }
            #[cfg(feature = "hip")]
            SoftmaxBackend::Hip => {
                // GPU softmax on HIP device — synchronous but runs in its own task
                tokio::task::spawn_blocking(move || {
                    hip_rwkv::hip::softmax_hip_batch(input)
                }).await??
            }
        };

        for (batch, tensor) in batches.iter().zip_eq(output.into_iter()) {
            let _ = batch.sender.send(tensor);
        }
        batches.clear();
    }
    Ok(())
}
```

Change `run()` signature: `context: Context` → `softmax_backend: SoftmaxBackend`.
Extract Context from the enum where still needed (e.g., `tensor_from_data` in `sample()`):
```rust
// In sample(), replace:
//   self.context.tensor_from_data([num_vocab, 1, 1, 1], data)?
// with:
//   TensorInit::from_data([num_vocab, 1, 1, 1], data)?
// (Context was not needed here — tensor_from_data just wraps TensorInit::from_data)
```

The `CoreRuntime` struct changes:
```rust
struct CoreRuntime {
    softmax_backend: SoftmaxBackend,  // replaces `context: Context`
    // ... rest unchanged
}
```

In `run()` initialization:
```rust
let softmax = {
    let (sender, receiver) = flume::unbounded();
    tokio::spawn(softmax(reload.clone(), softmax_backend.clone(), receiver));
    sender
};
```

Callers in `lib.rs` build the appropriate variant:
```rust
// WebGPU path (existing):
let softmax_backend = SoftmaxBackend::WebGpu(context.clone());
run(softmax_backend, runtime, state, receiver, info).await;

// HIP path (new):
#[cfg(feature = "hip")]
let softmax_backend = SoftmaxBackend::Hip;
run(softmax_backend, runtime, state, receiver, info).await;
```

### Step 12: Adapter Listing + ModelInfo Bridge

In `lib.rs`:
- Extend `list_adapters()` with HIP device enumeration via `hip_rwkv::hip::get_device_count/name()`
- Add `hip_to_model_info()` converting `Rwkv7ModelInfo` → `ModelInfo`
- Make `Environment::Loaded.model` optional (HIP doesn't support Save)

---

## Files Modified

| File | Change | Description |
|------|--------|-------------|
| `hip-rwkv/Cargo.toml` | **New** | New crate depending on web-rwkv 0.10.19 |
| `hip-rwkv/build.rs` | **New** | HIP kernel compilation (moved from root) |
| `hip-rwkv/src/lib.rs` | **New** | Re-exports hip module |
| `hip-rwkv/src/hip/*` | **Moved** | From src/hip/, `crate::` → `web_rwkv::` |
| `hip-rwkv/src/hip/kernels/softmax.hip` | **New** | GPU softmax kernel |
| `Cargo.toml` (root) | Modify | Add hip-rwkv to workspace members, remove hip/hip-probes features |
| `build.rs` (root) | Modify | Remove HIP kernel build logic |
| `src/lib.rs` (root) | Modify | Remove `#[cfg(feature = "hip")] pub mod hip` |
| `tests/hip_validation.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/hip_probes.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/hip_layer_validation.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/hip_fla_validation.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/hip_profiling.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/hipblaslt_benchmark.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/multi_batch_stress.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports, remove cfg gates |
| `tests/ground_truth.rs` | **Move** | → `hip-rwkv/tests/`, rewrite imports |
| `tests/benchmarks.rs` | Modify | Remove `#[cfg(feature = "hip")]` sections |
| `tests/functional_metrics.rs` | Modify | Remove `#[cfg(feature = "hip")]` sections |
| `examples/hip_test.rs` | **Move** | → `hip-rwkv/examples/`, rewrite imports |
| `examples/hip_gen.rs` | **Move** | → `hip-rwkv/examples/`, rewrite imports, remove cfg gates |
| `examples/gen_compare.rs` | **Move** | → `hip-rwkv/examples/`, add both deps, `softmax_one_cpu` → `softmax_hip` |
| `repos/ai00_server/Cargo.toml` | Modify | Add hip-rwkv workspace dep, `[patch.crates-io]` web-rwkv path |
| `repos/ai00_server/crates/ai00-core/Cargo.toml` | Modify | Add hip feature + hip-rwkv dep |
| `repos/ai00_server/crates/ai00-server/Cargo.toml` | Modify | Forward hip feature |
| `repos/ai00_server/crates/ai00-core/src/reload.rs` | Modify | Backend enum with `#[serde(default)]` |
| `repos/ai00_server/crates/ai00-core/src/lib.rs` | Modify | HIP load path, backend dispatch, SoftmaxBackend construction |
| `repos/ai00_server/crates/ai00-core/src/hip_state.rs` | **New** | HipStateAdapter implementing State trait |
| `repos/ai00_server/crates/ai00-core/src/run.rs` | Modify | SoftmaxBackend enum dispatch, drop Context field |
| `repos/ai00_server/crates/ai00-server/src/config.rs` | Modify | Pass backend through |
| `repos/ai00_server/assets/configs/Config.toml` | Modify | Document backend field (commented out, defaults to WebGpu) |

## Known Limitations

- HIP backend: V7 models only (error for V4/V5/V6)
- No LoRA, no .state file loading, no model Save on HIP path
- Choose/perplexity endpoint not supported (State.write/read unsupported)

## Verification

1. `cargo check -p hip-rwkv` — new crate compiles with web-rwkv dependency
2. `cargo check -p web-rwkv` — root crate still compiles (no hip module)
3. `cargo check -p ai00-server` — WebGPU path still compiles unchanged
4. `cargo check -p ai00-server --features hip` — HIP path compiles
5. Unit test: `softmax_hip` on known inputs matches expected probabilities
6. Set `backend = "Hip"` in Config.toml, run with V7 .st model
7. `curl -X POST http://localhost:65530/api/oai/v1/completions -d '{"prompt":"Hello","max_tokens":16}'`
8. Compare outputs between WebGpu and Hip backends on same model
