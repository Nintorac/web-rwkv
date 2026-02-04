# RWKV7 HIP Backend Implementation Plan (TDD Approach)

## Philosophy: Test-Driven Development

This plan follows a strict **test-first** methodology:

1. **Red**: Write a failing test that defines the expected behavior
2. **Green**: Write the minimum code to make the test pass
3. **Refactor**: Clean up while keeping tests green

Every task's acceptance criteria includes passing tests. No task is complete until its tests pass.

---

## Executive Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Scope** | Inference only | Simplifies implementation, no backward kernels |
| **Precision** | FP16 I/O, FP32 state | Native FP16 on gfx1151, FP32 for accumulation accuracy |
| **Integration** | Feature flag in web-rwkv | `--features hip` in existing crate |
| **Build** | Raw FFI + build.rs | hipcc compilation, no external binding crates |
| **Memory** | Unified (APU zero-copy) | Weights use `hipMallocManaged`, activations device-only |
| **BLAS** | rocBLAS (hipBLASLt broken) | hipBLASLt unsupported on gfx1151, rocBLAS works |
| **Testing** | TDD - tests written first | Correctness guarantee via Python fixtures |
| **Model Format** | SafeTensors | Reuse existing 0.1B model for testing |
| **Batch Size** | Variable B=1-32 | Optimize for both streaming and throughput |
| **Sequence Length** | Up to 32k+ tokens | Configurable chunking + streaming mode |
| Priority | Correctness over performance | Get it right first |

### Target Hardware

- **GPU**: AMD Radeon 8060S (gfx1151) - Strix Halo
- **Architecture**: RDNA 3.5, Wave32
- **ROCm**: TheRock 7.12.0a20260122 (or validate against official ROCm 7.2+ if available on your distro)

### Key Design Decisions

1. **FP16 over BF16**: gfx1151 has native FP16 support; BF16 may be slower
2. **Unified Memory**: APU allows zero-copy weight access, reducing load time
3. **rocBLAS**: hipBLASLt has [known issues on gfx1151](https://github.com/ROCm/ROCm/issues/5643)
4. **Streaming Mode**: Token-by-token generation with persistent state
5. **Chunked Processing**: Long sequences processed in configurable chunks

### Critical Gaps / Decisions to Confirm Before Coding

1. **Backend abstraction**: Do we want HIP as a peer backend to the existing WGPU/Vulkan path (shared high-level API), or a separate "native" inference entrypoint?
2. **Tensor layout contract (DECIDED)**: Use **rocBLAS-native column-major** storage for GEMM/GEMV (and prefer feature-major tensors where `shape[0]` is the fastest axis, matching existing `web-rwkv` tensor conventions). This avoids per-call row/col mapping in rocBLAS and keeps matmul callsites simple. See rocBLAS docs ("rocBLAS is column-major"): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/what-is-rocblas.html#rocblas
3. **Cooperative launch usage**: Are we committing to grid-sync/cooperative launches (can be limited by device/runtime), or focusing on conventional kernels + graphs/streams?

---

## Phase -1: Backend Abstraction (Enables Future CUDA / Alternate ROCm Backends)

This phase exists to satisfy the "add new backends in the future" requirement without rewriting the model later.

**Goal**: Define a small, explicit backend surface area and implement HIP behind it.

**Deliverables**:
- A backend trait (or set of traits) that covers:
  - device/context + stream lifecycle
  - buffer allocation (device + managed)
  - kernel launch
  - GEMM/GEMV entrypoints (backed by rocBLAS for HIP)
  - basic tensor ops needed by RWKV7 (elementwise ops + norms)
- A directory/crate layout that allows adding `cuda` later without entangling HIP build tooling.

**Design Options (pick one):**
1) **In-crate backends**: `src/backend/{wgpu,hip}/...` behind feature flags.
2) **Separate crates**: `crates/rwkv7-core` (math + traits), `crates/rwkv7-hip` (HIP impl), keep existing WGPU engine separate.

**Acceptance Criteria**:
- [ ] A "dummy" backend implementation can compile (even if not functional) to prove the interface is stable.
- [ ] HIP backend can be swapped in without changing model math/tests (beyond selecting backend).

---

## Phase 0: Test Infrastructure (PREREQUISITE)

This phase establishes the foundation for all testing. It must be completed first.

### 0.1 Reference Data Generation

**Objective**: Generate golden reference data from the Python implementation.

**Why first**: All HIP kernel tests compare against these references. Without reference data, we cannot write meaningful tests.

**Deliverables**:
```
tests/fixtures/
├── kernels/
│   ├── sigmoid/           # Input/output pairs for sigmoid
│   ├── squared_relu/
│   ├── decay_exp/
│   ├── lerp/
│   ├── layer_norm/
│   ├── group_norm/
│   ├── matmul/
│   ├── token_shift/
│   └── wkv7/              # Multiple test cases
├── layers/
│   ├── time_mix/          # Full time-mixing block I/O
│   └── channel_mix/       # Full channel-mixing block I/O
└── model/
    ├── 0.1b_short.npz     # Full model inference test case
    └── 0.1b_batched.npz
```

**Script**: `scripts/generate_test_fixtures.py`
- Uses RWKV-LM-V7 Python implementation
- Captures kernel-level inputs/outputs via hooks
- Saves as `.npz` (NumPy savez) for portability and easy inspection
- Documents exact tensor shapes and dtypes

**Acceptance Criteria**:
- [ ] Script runs successfully with RWKV7 0.1B model
- [ ] Fixtures generated for all kernel types
- [ ] Fixture format documented
- [ ] Fixtures loadable from Rust (verified with stub test)

### 0.2 Test Harness Setup

**Objective**: Create Rust infrastructure for loading fixtures and comparing tensors.

**Components**:

```rust
// tests/common/mod.rs
pub struct TestFixture {
    /// Raw NPZ arrays. Convention: for each tensor key `x`, also store `x_shape` (i64[4]).
    pub data: HashMap<String, FixtureArray>,
}

impl TestFixture {
    pub fn load(path: &str) -> Self;
    pub fn shape4(&self, name: &str) -> [usize; 4]; // reads `{name}_shape`
    pub fn f16(&self, name: &str) -> &[f16];        // reads `{name}`
    pub fn f32(&self, name: &str) -> &[f32];
}

pub fn assert_tensors_close(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,  // relative tolerance
    atol: f32,  // absolute tolerance
) -> Result<(), String>;

pub fn assert_tensors_close_f16(
    actual: &[f16],
    expected: &[f16],
    rtol: f32,
    atol: f32,
) -> Result<(), String>;
```

**Acceptance Criteria**:
- [ ] `TestFixture::load()` works with generated fixtures
- [ ] `assert_tensors_close()` correctly identifies matching/mismatching tensors
- [ ] Clear error messages showing where mismatches occur
- [ ] Test for the test harness itself passes

---

## Phase 1: Foundation (with Tests)

### 1.1 Project Structure + Build System

**Test First**:
```rust
#[test]
fn test_hip_feature_compiles() {
    // This test exists - if it compiles with --features hip,
    // the build system works
}

#[test]
fn test_simple_kernel_runs() {
    // A trivial kernel that just copies data
    let input = vec![1.0f32, 2.0, 3.0];
    let output = hip_copy_kernel(&input);
    assert_eq!(input, output);
}
```

**Implementation**:
- `Cargo.toml` with `hip` feature
- `build.rs` that compiles `.hip` files
- Simple copy kernel to verify build chain

**Acceptance Criteria**:
- [ ] `cargo build --features hip` succeeds
- [ ] `cargo test --features hip test_simple_kernel_runs` passes
- [ ] Kernel executes on GPU (not just compiles)

### 1.2 HIP Context Management

**Test First**:
```rust
#[test]
fn test_hip_context_creation() {
    let ctx = HipContext::new().expect("Failed to create context");
    assert!(ctx.device_id() >= 0);
    assert!(ctx.device_name().contains("gfx1151") || ctx.device_name().contains("Radeon"));
}

#[test]
fn test_hip_memory_alloc_free() {
    let ctx = HipContext::new().unwrap();
    let ptr = ctx.alloc::<f32>(1024).expect("Alloc failed");
    assert!(!ptr.is_null());
    ctx.free(ptr).expect("Free failed");
}

#[test]
fn test_hip_memcpy_roundtrip() {
    let ctx = HipContext::new().unwrap();
    let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();

    let device_ptr = ctx.alloc::<f32>(1024).unwrap();
    ctx.copy_to_device(&host_data, device_ptr).unwrap();

    let mut result = vec![0.0f32; 1024];
    ctx.copy_to_host(device_ptr, &mut result).unwrap();

    assert_eq!(host_data, result);
    ctx.free(device_ptr).unwrap();
}
```

**Implementation**:
- `HipContext` struct wrapping device/stream
- `alloc`, `free`, `copy_to_device`, `copy_to_host`
- Error handling with descriptive messages

**Acceptance Criteria**:
- [ ] All three tests pass
- [ ] No memory leaks (verified with `rocprof` or similar)

### 1.3 HIP Tensor Abstraction

**Test First**:
```rust
#[test]
fn test_tensor_creation() {
    let ctx = HipContext::new().unwrap();
    let tensor = TensorHip::<f32>::zeros(&ctx, &[2, 3, 4]).unwrap();
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_tensor_from_data() {
    let ctx = HipContext::new().unwrap();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = TensorHip::from_slice(&ctx, &data, &[2, 3, 4]).unwrap();

    let result = tensor.to_vec().unwrap();
    assert_eq!(data, result);
}

#[test]
fn test_tensor_view() {
    let ctx = HipContext::new().unwrap();
    let tensor = TensorHip::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let view = tensor.view(&[0..1, 0..2]).unwrap();  // First row
    assert_eq!(view.shape(), &[1, 2]);
}
```

**Acceptance Criteria**:
- [ ] All tensor tests pass
- [ ] Tensor can be created, populated, read back
- [ ] Views work correctly

---

## Phase 2: Core Kernels (TDD for Each)

### 2.1 Sigmoid Kernel

**Test First** (write before any kernel code):
```rust
#[test]
fn test_sigmoid_basic() {
    let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_sigmoid(&ctx, &input).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}

#[test]
fn test_sigmoid_edge_cases() {
    let fixture = TestFixture::load("tests/fixtures/kernels/sigmoid/edge_cases.npz");
    // Tests: -inf, +inf, 0, very large, very small
    // ...
}
```

**Implementation**:
- `hip/kernels/sigmoid.hip`
- Rust wrapper `fn hip_sigmoid(ctx, input) -> TensorHip`

**Acceptance Criteria**:
- [ ] `test_sigmoid_basic` passes (< 1e-3 relative error)
- [ ] `test_sigmoid_edge_cases` passes
- [ ] Matches Python `torch.sigmoid()` output

### 2.2 Squared ReLU Kernel

**Test First**:
```rust
#[test]
fn test_squared_relu() {
    let fixture = TestFixture::load("tests/fixtures/kernels/squared_relu/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_squared_relu(&ctx, &input).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Matches `F.relu(x) ** 2` from PyTorch

### 2.3 Decay Exponential Kernel

**Test First**:
```rust
#[test]
fn test_decay_exp() {
    // exp(-exp(x)) - critical for RWKV7 time decay
    let fixture = TestFixture::load("tests/fixtures/kernels/decay_exp/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_decay_exp(&ctx, &input).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}

#[test]
fn test_decay_exp_numerical_stability() {
    // Test with values that could cause overflow/underflow
    let fixture = TestFixture::load("tests/fixtures/kernels/decay_exp/stability.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] Both tests pass
- [ ] Handles extreme values without NaN/Inf

### 2.4 Lerp Kernel

**Test First**:
```rust
#[test]
fn test_lerp() {
    let fixture = TestFixture::load("tests/fixtures/kernels/lerp/basic.npz");
    let ctx = HipContext::new().unwrap();

    let a = TensorHip::from_fixture(&ctx, &fixture, "a").unwrap();
    let b = TensorHip::from_fixture(&ctx, &fixture, "b").unwrap();
    let t = TensorHip::from_fixture(&ctx, &fixture, "t").unwrap();
    let output = hip_lerp(&ctx, &a, &b, &t).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Matches `torch.lerp(a, b, t)`

### 2.5 Layer Normalization

**Test First**:
```rust
#[test]
fn test_layer_norm_basic() {
    let fixture = TestFixture::load("tests/fixtures/kernels/layer_norm/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let weight = TensorHip::from_fixture(&ctx, &fixture, "weight").unwrap();
    let bias = TensorHip::from_fixture(&ctx, &fixture, "bias").unwrap();

    let output = hip_layer_norm(&ctx, &input, &weight, &bias, 1e-5).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}

#[test]
fn test_layer_norm_rwkv7_epsilon() {
    // RWKV7 uses different epsilon values
    let fixture = TestFixture::load("tests/fixtures/kernels/layer_norm/rwkv7_eps.npz");
    // eps = 1e-5 for LayerNorm
    // ...
}
```

**Acceptance Criteria**:
- [ ] Both tests pass
- [ ] Numerical stability (Welford's algorithm)

### 2.6 Group Normalization

**Test First**:
```rust
#[test]
fn test_group_norm() {
    let fixture = TestFixture::load("tests/fixtures/kernels/group_norm/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let weight = TensorHip::from_fixture(&ctx, &fixture, "weight").unwrap();
    let bias = TensorHip::from_fixture(&ctx, &fixture, "bias").unwrap();

    // RWKV7 uses eps = 64e-5 for GroupNorm
    let output = hip_group_norm(&ctx, &input, &weight, &bias, 32, 64e-5).unwrap();

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes with RWKV7's epsilon (64e-5)
- [ ] Per-head normalization works correctly

### 2.7 L2 Normalization (Key Normalization)

**Test First**:
```rust
#[test]
fn test_l2_normalize() {
    let fixture = TestFixture::load("tests/fixtures/kernels/l2_norm/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_l2_normalize(&ctx, &input, 64).unwrap();  // per head_size=64

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-3, 1e-4
    ).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Per-head normalization correct

### 2.8 Matrix Multiply (GEMV/GEMM)

**Test First**:
```rust
#[test]
fn test_gemv() {
    let fixture = TestFixture::load("tests/fixtures/kernels/matmul/gemv.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();   // [1, K]
    let weight = TensorHip::from_fixture(&ctx, &fixture, "weight").unwrap(); // [N, K]
    let output = hip_gemv(&ctx, &input, &weight).unwrap();                    // [1, N]

    assert_tensors_close_f16(
        &output.to_vec().unwrap(),
        fixture.get_f16("expected"),
        1e-2, 1e-3  // Slightly looser for matmul
    ).unwrap();
}

#[test]
fn test_gemm_batched() {
    let fixture = TestFixture::load("tests/fixtures/kernels/matmul/gemm_batched.npz");
    // [B, T, K] @ [N, K]^T -> [B, T, N]
    // ...
}
```

**Acceptance Criteria**:
- [ ] GEMV test passes
- [ ] GEMM test passes
- [ ] rocBLAS used when available, fallback works

### 2.9 Token Shift

**Test First**:
```rust
#[test]
fn test_token_shift_single_sequence() {
    let fixture = TestFixture::load("tests/fixtures/kernels/token_shift/single.npz");
    let ctx = HipContext::new().unwrap();

    let x = TensorHip::from_fixture(&ctx, &fixture, "x").unwrap();
    let state = TensorHip::from_fixture(&ctx, &fixture, "state").unwrap();
    let mix = TensorHip::from_fixture(&ctx, &fixture, "mix").unwrap();

    let (output, new_state) = hip_token_shift(&ctx, &x, &state, &mix).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
    assert_tensors_close_f16(&new_state.to_vec().unwrap(), fixture.get_f16("expected_state"), 1e-3, 1e-4).unwrap();
}

#[test]
fn test_token_shift_batched() {
    let fixture = TestFixture::load("tests/fixtures/kernels/token_shift/batched.npz");
    // ...
}

#[test]
fn test_token_shift_state_continuity() {
    // Verify that processing tokens one-by-one matches processing all at once
    // This is critical for streaming inference
    // ...
}
```

**Acceptance Criteria**:
- [ ] Single sequence test passes
- [ ] Batched test passes
- [ ] State continuity verified

### 2.10 WKV7 Core Kernel (Critical)

**What is `w`? (DECIDED)**:
- In RWKV7, the model produces a *soft-clamped* `w` via `w = -softplus(-(w_pre)) - 0.5` (so `w <= -0.5`).
- The WKV kernel then computes the actual per-channel decay as `w_decay = exp(-exp(w))` internally (FP32), matching the reference HIP kernel style.
- This is also mathematically equivalent to the WGSL fused form `exp(-exp(-0.5) * sigmoid(w_pre))` used in `src/shaders/time_mix_v7.wgsl`.

**Test First** (extensive testing due to complexity):
```rust
#[test]
fn test_wkv7_single_token() {
    // Simplest case: T=1
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/single_token.npz");
    let ctx = HipContext::new().unwrap();

    let w = TensorHip::from_fixture(&ctx, &fixture, "w").unwrap();
    let q = TensorHip::from_fixture(&ctx, &fixture, "q").unwrap();
    let k = TensorHip::from_fixture(&ctx, &fixture, "k").unwrap();
    let v = TensorHip::from_fixture(&ctx, &fixture, "v").unwrap();
    let a = TensorHip::from_fixture(&ctx, &fixture, "a").unwrap();
    let b = TensorHip::from_fixture(&ctx, &fixture, "b").unwrap();
    let state_in = TensorHip::from_fixture(&ctx, &fixture, "state_in").unwrap();

    let (output, state_out) = hip_wkv7(&ctx, &w, &q, &k, &v, &a, &b, &state_in).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
    assert_tensors_close(&state_out.to_vec().unwrap(), fixture.get_f32("expected_state"), 1e-5, 1e-6).unwrap();
}

#[test]
fn test_wkv7_short_sequence() {
    // T=16
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/short_sequence.npz");
    // ...
}

#[test]
fn test_wkv7_medium_sequence() {
    // T=128
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/medium_sequence.npz");
    // ...
}

#[test]
fn test_wkv7_batched() {
    // B=4, T=64
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/batched.npz");
    // ...
}

#[test]
fn test_wkv7_state_evolution() {
    // Verify state accumulates correctly over multiple calls
    // Process T=32 as 4x T=8 calls, compare final state
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/state_evolution.npz");
    // ...
}

#[test]
fn test_wkv7_numerical_stability() {
    // Test with values near the edge of FP16 range
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv7/stability.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] All WKV7 tests pass
- [ ] State evolution matches Python exactly
- [ ] Numerically stable
- [ ] FP32 accumulation working (state is f32)

### 2.11 WKV Bonus Kernel (time_first)

The WKV bonus adds extra attention on the current token without storing in state.

**Test First**:
```rust
#[test]
fn test_wkv_bonus() {
    // u_t = (r · (ρ ⊙ k̃)^T) v
    let fixture = TestFixture::load("tests/fixtures/kernels/wkv_bonus/basic.npz");
    let ctx = HipContext::new().unwrap();

    let r_k = TensorHip::from_fixture(&ctx, &fixture, "r_k").unwrap();  // [H, N]
    let r = TensorHip::from_fixture(&ctx, &fixture, "r").unwrap();      // [B, T, H, N]
    let k = TensorHip::from_fixture(&ctx, &fixture, "k").unwrap();      // [B, T, H, N]
    let v = TensorHip::from_fixture(&ctx, &fixture, "v").unwrap();      // [B, T, H, N]

    let output = hip_wkv_bonus(&ctx, &r_k, &r, &k, &v).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected"), 1e-3, 1e-4).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Per-head reduction correct

### 2.12 Control K Kernel (Replacement Key)

Computes the replacement key: `k̃ = k * (1 + (a - 1) * k_a)`

**Test First**:
```rust
#[test]
fn test_control_k() {
    let fixture = TestFixture::load("tests/fixtures/kernels/control_k/basic.npz");
    let ctx = HipContext::new().unwrap();

    let k_a = TensorHip::from_fixture(&ctx, &fixture, "k_a").unwrap();  // [C]
    let a = TensorHip::from_fixture(&ctx, &fixture, "a").unwrap();      // [B, T, C]
    let k = TensorHip::from_fixture(&ctx, &fixture, "k").unwrap();      // [B, T, C]

    let output = hip_control_k(&ctx, &k_a, &a, &k).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected"), 1e-3, 1e-4).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Element-wise computation correct

### 2.13 Tanh Activation Kernel

**Test First**:
```rust
#[test]
fn test_tanh() {
    let fixture = TestFixture::load("tests/fixtures/kernels/tanh/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_tanh(&ctx, &input).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected"), 1e-3, 1e-4).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] Matches `torch.tanh()`

### 2.14 Softplus Decay Kernel

Computes: `out = -log(1 + exp(-x)) - 0.5` (numerically stable softplus for decay)

**Test First**:
```rust
#[test]
fn test_softplus_decay() {
    let fixture = TestFixture::load("tests/fixtures/kernels/softplus_decay/basic.npz");
    let ctx = HipContext::new().unwrap();

    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let output = hip_softplus_decay(&ctx, &input).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected"), 1e-3, 1e-4).unwrap();
}

#[test]
fn test_softplus_decay_stability() {
    // Test large positive/negative values don't overflow
    let fixture = TestFixture::load("tests/fixtures/kernels/softplus_decay/stability.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] Both tests pass
- [ ] Numerically stable for extreme values

### 2.15 Channel Mix State Kernel

Handles FFN token shift state management.

**Test First**:
```rust
#[test]
fn test_channel_mix_state() {
    let fixture = TestFixture::load("tests/fixtures/kernels/channel_mix_state/basic.npz");
    let ctx = HipContext::new().unwrap();

    let cursors = TensorHip::from_fixture(&ctx, &fixture, "cursors").unwrap();
    let state = TensorHip::from_fixture(&ctx, &fixture, "state_in").unwrap();
    let x = TensorHip::from_fixture(&ctx, &fixture, "x").unwrap();

    let (output, new_state) = hip_channel_mix_state(&ctx, &cursors, &state, &x).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
    assert_tensors_close(&new_state.to_vec().unwrap(), fixture.get_f32("expected_state"), 1e-5, 1e-6).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] State updates correctly

---

## Phase 3: Layer Integration (TDD)

### 3.1 Time-Mixing Block

**Test First**:
```rust
#[test]
fn test_time_mix_layer_0() {
    // Layer 0 has ln0, no value residual
    let fixture = TestFixture::load("tests/fixtures/layers/time_mix/layer_0.npz");
    let ctx = HipContext::new().unwrap();

    let block = TimeMixBlock::from_fixture(&ctx, &fixture).unwrap();
    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let state = TimeMixState::from_fixture(&ctx, &fixture, "state_in").unwrap();

    let (output, new_state, v_first) = block.forward(&ctx, &input, &state, None).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
}

#[test]
fn test_time_mix_layer_n() {
    // Layer N has value residual from layer 0
    let fixture = TestFixture::load("tests/fixtures/layers/time_mix/layer_n.npz");
    // ...
}

#[test]
fn test_time_mix_all_components() {
    // Verify each intermediate: r, k, v, w, a, g, kk, wkv_out, etc.
    // Use hooks in Python to capture these
    let fixture = TestFixture::load("tests/fixtures/layers/time_mix/intermediates.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] Layer 0 test passes
- [ ] Layer N test passes (with value residual)
- [ ] All intermediate values match reference

### 3.2 Channel-Mixing Block

**Test First**:
```rust
#[test]
fn test_channel_mix() {
    let fixture = TestFixture::load("tests/fixtures/layers/channel_mix/basic.npz");
    let ctx = HipContext::new().unwrap();

    let block = ChannelMixBlock::from_fixture(&ctx, &fixture).unwrap();
    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let state = TensorHip::from_fixture(&ctx, &fixture, "state_in").unwrap();

    let (output, new_state) = block.forward(&ctx, &input, &state).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Test passes
- [ ] State management correct

### 3.3 Full Block (Time-Mix + Channel-Mix)

**Test First**:
```rust
#[test]
fn test_full_block() {
    let fixture = TestFixture::load("tests/fixtures/layers/full_block.npz");
    let ctx = HipContext::new().unwrap();

    let block = Block::from_fixture(&ctx, &fixture).unwrap();
    let input = TensorHip::from_fixture(&ctx, &fixture, "input").unwrap();
    let state = BlockState::from_fixture(&ctx, &fixture, "state_in").unwrap();

    let (output, new_state) = block.forward(&ctx, &input, &state).unwrap();

    assert_tensors_close_f16(&output.to_vec().unwrap(), fixture.get_f16("expected_output"), 1e-3, 1e-4).unwrap();
}
```

**Acceptance Criteria**:
- [ ] Full block output matches reference

### 3.4 Model Loading

**Test First**:
```rust
#[test]
fn test_load_safetensors() {
    let ctx = HipContext::new().unwrap();
    let model = Rwkv7Hip::load(&ctx, "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").unwrap();

    assert_eq!(model.n_layer, 12);
    assert_eq!(model.n_embd, 768);
    assert_eq!(model.n_head, 12);
}

#[test]
fn test_model_weights_match() {
    let ctx = HipContext::new().unwrap();
    let model = Rwkv7Hip::load(&ctx, "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").unwrap();

    // Spot check a few weights against Python-loaded values
    let fixture = TestFixture::load("tests/fixtures/model/weights_spot_check.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] Model loads without error
- [ ] Dimensions correct
- [ ] Spot-checked weights match

### 3.5 Full Model Forward Pass

**Test First**:
```rust
#[test]
fn test_model_forward_deterministic() {
    let ctx = HipContext::new().unwrap();
    let model = Rwkv7Hip::load(&ctx, "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").unwrap();

    let tokens = vec![0u32, 1, 2, 3];
    let output1 = model.forward(&ctx, &tokens, &mut ModelState::new(&ctx, &model)).unwrap();
    let output2 = model.forward(&ctx, &tokens, &mut ModelState::new(&ctx, &model)).unwrap();

    assert_eq!(output1.to_vec().unwrap(), output2.to_vec().unwrap());
}

#[test]
fn test_model_forward_matches_python() {
    let fixture = TestFixture::load("tests/fixtures/model/forward_0.1b.npz");
    let ctx = HipContext::new().unwrap();
    let model = Rwkv7Hip::load(&ctx, "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st").unwrap();

    let tokens: Vec<u32> = fixture.get_u32("tokens").to_vec();
    let output = model.forward(&ctx, &tokens, &mut ModelState::new(&ctx, &model)).unwrap();

    assert_tensors_close(
        &output.to_vec().unwrap(),
        fixture.get_f32("expected_logits"),
        1e-2, 1e-3  // Looser tolerance for full model
    ).unwrap();
}

#[test]
fn test_model_streaming_inference() {
    // Token-by-token matches batch processing
    let fixture = TestFixture::load("tests/fixtures/model/streaming.npz");
    // ...
}
```

**Acceptance Criteria**:
- [ ] Deterministic output
- [ ] Matches Python reference within tolerance
- [ ] Streaming inference works

---

## Phase 4: Optimization (After All Tests Pass)

Only proceed here after all correctness tests pass.

### 4.1 Performance Benchmarks (Not Tests)

```rust
#[bench]
fn bench_wkv7_throughput() {
    // Measure tokens/second
}

#[bench]
fn bench_full_model_latency() {
    // Measure time per forward pass
}
```

### 4.2 Profiling & Optimization

- Use `rocprof` to identify bottlenecks
- Optimize while keeping tests green
- Add regression tests for any performance fixes

---

## Task Summary with Acceptance Criteria

| Task | Test Files | Pass Criteria |
|------|-----------|---------------|
| **Phase 0: Test Infrastructure** |||
| 0.1 Reference Data Gen | N/A (generates fixtures) | Script runs, fixtures loadable |
| 0.2 Test Harness | `tests/common_test.rs` | Harness itself tested |
| **Phase 1: Foundation** |||
| 1.1 Project Structure | `test_hip_feature_compiles`, `test_simple_kernel_runs` | Both pass |
| 1.2 HIP Context | `test_hip_context_*` (3 tests) | All pass |
| 1.3 HIP Tensor | `test_tensor_*` (3 tests) | All pass |
| **Phase 2: Kernels** |||
| 2.1 Sigmoid | `test_sigmoid_*` (2 tests) | All pass, < 1e-3 error |
| 2.2 Squared ReLU | `test_squared_relu` | Pass, < 1e-3 error |
| 2.3 Decay Exp | `test_decay_exp_*` (2 tests) | All pass, stable |
| 2.4 Lerp | `test_lerp` | Pass, < 1e-3 error |
| 2.5 Layer Norm | `test_layer_norm_*` (2 tests) | All pass |
| 2.6 Group Norm | `test_group_norm` | Pass with eps=64e-5 |
| 2.7 L2 Norm | `test_l2_normalize` | Pass |
| 2.8 MatMul | `test_gemv`, `test_gemm_batched` | Both pass, rocBLAS used |
| 2.9 Token Shift | `test_token_shift_*` (3 tests) | All pass, state correct |
| 2.10 WKV7 | `test_wkv7_*` (6 tests) | All pass, state exact |
| 2.11 WKV Bonus | `test_wkv_bonus` | Pass, reduction correct |
| 2.12 Control K | `test_control_k` | Pass |
| 2.13 Tanh | `test_tanh` | Pass |
| 2.14 Softplus Decay | `test_softplus_decay_*` (2 tests) | All pass, stable |
| 2.15 Channel Mix State | `test_channel_mix_state` | Pass, state correct |
| **Phase 3: Integration** |||
| 3.1 Time-Mix | `test_time_mix_*` (3 tests) | All pass |
| 3.2 Channel-Mix | `test_channel_mix` | Pass |
| 3.3 Full Block | `test_full_block` | Pass |
| 3.4 Model Load | `test_load_*` (2 tests) | Both pass |
| 3.5 Model Forward | `test_model_*` (3 tests) | All pass |
| 3.6 Streaming Mode | `test_streaming_*` (2 tests) | Token-by-token matches batch |
| 3.7 Chunked Processing | `test_chunked_*` (2 tests) | Long sequences work |

---

## Definition of Done

A task is **complete** when:

1. ✅ Test written and initially fails (red)
2. ✅ Implementation makes test pass (green)
3. ✅ Code reviewed/cleaned up (refactor)
4. ✅ All existing tests still pass
5. ✅ No memory leaks
6. ✅ Documented in code comments

---

## Dependency Graph (Revised)

```
Phase 0: Test Infrastructure
├── 0.1 Reference Data Generation  ←── Must be FIRST
└── 0.2 Test Harness Setup        ←── Depends on 0.1

Phase 1: Foundation
├── 1.1 Project Structure         ←── Depends on 0.2
├── 1.2 HIP Context              ←── Depends on 1.1
└── 1.3 HIP Tensor               ←── Depends on 1.2

Phase 2: Kernels (can parallelize many)
├── 2.1 Sigmoid                  ←── Depends on 1.3, 0.1
├── 2.2 Squared ReLU             ←── Depends on 1.3, 0.1
├── 2.3 Decay Exp                ←── Depends on 1.3, 0.1
├── 2.4 Lerp                     ←── Depends on 1.3, 0.1
├── 2.5 Layer Norm               ←── Depends on 1.3, 0.1
├── 2.6 Group Norm               ←── Depends on 1.3, 0.1
├── 2.7 L2 Norm                  ←── Depends on 1.3, 0.1
├── 2.8 MatMul (rocBLAS)         ←── Depends on 1.3, 0.1
├── 2.9 Token Shift              ←── Depends on 1.3, 0.1
├── 2.10 WKV7 Core               ←── Depends on 1.3, 0.1, 2.3
├── 2.11 WKV Bonus               ←── Depends on 1.3, 0.1
├── 2.12 Control K               ←── Depends on 1.3, 0.1
├── 2.13 Tanh                    ←── Depends on 1.3, 0.1
├── 2.14 Softplus Decay          ←── Depends on 1.3, 0.1
└── 2.15 Channel Mix State       ←── Depends on 1.3, 0.1

Phase 3: Integration
├── 3.1 Time-Mix Block           ←── Depends on 2.5-2.14
├── 3.2 Channel-Mix Block        ←── Depends on 2.1, 2.2, 2.4, 2.8, 2.9, 2.15
├── 3.3 Full Block               ←── Depends on 3.1, 3.2
├── 3.4 Model Loading            ←── Depends on 1.3 (unified memory)
├── 3.5 Model Forward            ←── Depends on 3.3, 3.4
├── 3.6 Streaming Mode           ←── Depends on 3.5
└── 3.7 Chunked Processing       ←── Depends on 3.5

Phase 4: Optimization           ←── All Phase 3 tests must pass
```

---

## Streaming and Chunked Processing

### Streaming Mode (Token-by-Token Generation)

For interactive use, process one token at a time with persistent state:

```rust
pub struct StreamingContext {
    state: ModelState,
    chunk_size: usize,  // Configurable: 4096 default
}

impl StreamingContext {
    pub fn new(model: &Rwkv7Hip, chunk_size: usize) -> Self;

    /// Process a single token, return logits for next token
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>>;

    /// Process multiple tokens (prompt), return final logits
    pub fn prompt(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Reset state to initial
    pub fn reset(&mut self);
}
```

### Chunked Processing (Long Context)

For sequences > chunk_size, process in chunks while maintaining state:

```rust
pub fn forward_chunked(
    &self,
    tokens: &[u32],
    state: &mut ModelState,
    chunk_size: usize,  // Default: 4096
) -> Result<Vec<f32>> {
    let mut logits = vec![];

    for chunk in tokens.chunks(chunk_size) {
        logits = self.forward(chunk, state)?;
        // State automatically carries over
    }

    Ok(logits)
}
```

### Memory Considerations for 32k+ Context

Avoid hardcoding memory numbers without specifying **batch size** and whether you're doing **streaming** vs **prefill** (materializing large `[B, T, C]` buffers).

**State size (dominant term)** is roughly:

```
bytes ~= B * n_layer * (H * head_size * head_size) * sizeof(f32)
      ~= B * n_layer * (n_embd * head_size) * 4
```

For RWKV7 0.1B (n_layer=12, n_embd=768, head_size=64):
- WKV state per layer per batch: `768 * 64 * 4 = 196,608 bytes` (~192 KiB)
- Total WKV state per batch: `~2.25 MiB` (12 layers), plus token-shift vectors.

**Prefill buffers** can dominate if you choose to store `[B, T, C]` activations in FP16:
- `[B, T, 768]` FP16 bytes = `B * T * 768 * 2`
- Examples:
  - B=1, T=32k: ~47 MiB
  - B=32, T=4096: ~192 MiB
  - B=32, T=32k: ~1.5 GiB

Design for streaming/chunking so memory scales with state, not `T`.

**APU Constraint**: Strix Halo has ~96GB shared RAM, but GPU-accessible portion may be limited. Chunking essential for larger models.
