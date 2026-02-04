# Test Fixture Generation Specification

This document specifies how to generate test fixtures from the Python RWKV-LM-V7 implementation for validating the HIP backend.

## Overview

Test fixtures are binary files containing input/output pairs captured from the Python reference implementation. These serve as ground truth for HIP kernel correctness testing.

## Directory Structure

```
tests/fixtures/
├── kernels/
│   ├── sigmoid/
│   │   ├── basic.npz           # Standard inputs
│   │   └── edge_cases.npz      # -inf, +inf, 0, extremes
│   ├── squared_relu/
│   │   └── basic.npz
│   ├── decay_exp/
│   │   ├── basic.npz
│   │   └── stability.npz       # Values near overflow/underflow
│   ├── tanh/
│   │   └── basic.npz
│   ├── softplus_decay/
│   │   ├── basic.npz
│   │   └── stability.npz
│   ├── lerp/
│   │   └── basic.npz
│   ├── layer_norm/
│   │   ├── basic.npz
│   │   └── rwkv7_eps.npz       # With eps=1e-5
│   ├── group_norm/
│   │   └── basic.npz           # With eps=64e-5
│   ├── l2_norm/
│   │   └── basic.npz           # Per-head normalization
│   ├── matmul/
│   │   ├── gemv.npz            # [1, K] @ [N, K]^T
│   │   └── gemm_batched.npz    # [B, T, K] @ [N, K]^T
│   ├── token_shift/
│   │   ├── single.npz          # Single sequence
│   │   ├── batched.npz         # B > 1
│   │   └── streaming.npz       # Token-by-token continuity
│   ├── wkv7/
│   │   ├── single_token.npz    # T=1
│   │   ├── short_sequence.npz  # T=16
│   │   ├── medium_sequence.npz # T=128
│   │   ├── batched.npz         # B=4, T=64
│   │   ├── state_evolution.npz # Multi-call state test
│   │   └── stability.npz       # Edge values
│   ├── wkv_bonus/
│   │   └── basic.npz
│   ├── control_k/
│   │   └── basic.npz
│   └── channel_mix_state/
│       └── basic.npz
├── layers/
│   ├── time_mix/
│   │   ├── layer_0.npz         # First layer (has ln0, no value residual)
│   │   ├── layer_n.npz         # Later layer (has value residual)
│   │   └── intermediates.npz   # All intermediate tensors
│   ├── channel_mix/
│   │   └── basic.npz
│   └── full_block.npz
└── model/
    ├── weights_spot_check.npz  # Sample weights for verification
    ├── forward_0.1b.npz        # Full forward pass
    └── streaming.npz           # Token-by-token verification
```

## NPZ File Format

Each `.npz` file contains numpy arrays with standardized naming.

### Canonical Tensor Storage Order (HIP / `web-rwkv` compatible)

For parity and to avoid per-test transposes, fixtures should store tensors in the same **linear memory order** that the HIP backend will use:
- `shape[0]` is the fastest axis (contiguous), matching `web-rwkv`'s `Shape` convention.
- Store tensors as **flattened 1D** arrays plus an explicit `{name}_shape` of length 4.

This avoids ambiguity around NumPy's C-order vs Fortran-order flags inside `.npz`.

### Kernel Fixtures

```python
# Example: sigmoid/basic.npz
{
    'input': np.array(..., dtype=np.float16),           # 1D, flattened in backend order
    'input_shape': np.array([X, Y, Z, W], dtype=np.int64),
    'expected': np.array(..., dtype=np.float16),        # 1D, same length
    'expected_shape': np.array([X, Y, Z, W], dtype=np.int64),
}
```

### Layer Fixtures

```python
# Example: time_mix/layer_0.npz
{
    # Inputs
    'input': np.array(..., dtype=np.float16),          # 1D, Shape(C, A, 1, 1) flattened
    'input_shape': np.array([C, A, 1, 1], dtype=np.int64),
    'cursors': np.array(..., dtype=np.uint32),         # packed per-token cursors, length A
    'cursors_shape': np.array([A, 1, 1, 1], dtype=np.int64),
    'state_in': np.array(..., dtype=np.float32),       # 1D, Shape(C, N+2, B, 1) flattened
    'state_in_shape': np.array([C, N+2, B, 1], dtype=np.int64),

    # Weights (or use model path)
    'x_r': np.array(...), 'x_w': np.array(...), ...

    # Expected outputs
    'expected_output': np.array(..., dtype=np.float16),        # 1D, Shape(C, A, 1, 1)
    'expected_output_shape': np.array([C, A, 1, 1], dtype=np.int64),
    'expected_state': np.array(..., dtype=np.float32),         # 1D, Shape(C, N+2, B, 1)
    'expected_state_shape': np.array([C, N+2, B, 1], dtype=np.int64),
    'expected_v_first': np.array(..., dtype=np.float16),       # Layer 0 only

    # Intermediates (for debugging)
    'r': np.array(...), 'k': np.array(...), 'v': np.array(...),
    'w': np.array(...), 'a': np.array(...), 'g': np.array(...),
    'kk': np.array(...), 'wkv_out': np.array(...),
}
```

## Generation Script

### `scripts/generate_test_fixtures.py`

```python
#!/usr/bin/env python3
"""
Generate test fixtures from RWKV-LM-V7 Python implementation.

Usage:
    python scripts/generate_test_fixtures.py \
        --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st \
        --output tests/fixtures/

Requirements:
    pip install torch numpy safetensors
    # RWKV-LM-V7 in path or installed
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def generate_activation_fixtures(output_dir: Path):
    """Generate fixtures for simple activation kernels."""

    def save_tensor(npz_path: Path, **tensors):
        """
        Save flattened tensors + explicit 4D shapes to NPZ.
        Convention: for each key 'x', store 'x' (1D) and 'x_shape' (i64[4]).
        """
        payload = {}
        for name, (arr_1d, shape4) in tensors.items():
            payload[name] = arr_1d
            payload[f"{name}_shape"] = np.array(shape4, dtype=np.int64)
        np.savez(npz_path, **payload)

    # NOTE: The rest of this file shows illustrative fixture generation. When you implement the
    # real generator, ensure *all* fixtures follow the same "flattened + {name}_shape" convention.

    # Sigmoid
    sigmoid_dir = output_dir / "kernels" / "sigmoid"
    sigmoid_dir.mkdir(parents=True, exist_ok=True)

    # Basic
    # Use packed-token layout: Shape(C, A, 1, 1) with A = B*T.
    B, T, C = 4, 128, 768
    A = B * T
    x = torch.randn(A, C, dtype=torch.float16).contiguous()  # token-major, C contiguous
    y = torch.sigmoid(x)
    save_tensor(sigmoid_dir / "basic.npz",
                input=(x.cpu().numpy().reshape(-1), (C, A, 1, 1)),
                expected=(y.cpu().numpy().reshape(-1), (C, A, 1, 1)))

    # Edge cases
    x_edge = torch.tensor([-100, -10, -1, 0, 1, 10, 100, float('inf'), float('-inf')],
                          dtype=torch.float16).contiguous()
    y_edge = torch.sigmoid(x_edge)
    save_tensor(sigmoid_dir / "edge_cases.npz",
                input=(x_edge.cpu().numpy().reshape(-1), (x_edge.numel(), 1, 1, 1)),
                expected=(y_edge.cpu().numpy().reshape(-1), (x_edge.numel(), 1, 1, 1)))

    # Squared ReLU
    sqrelu_dir = output_dir / "kernels" / "squared_relu"
    sqrelu_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(4, 128, 3072, dtype=torch.float16)  # FFN hidden dim
    np.savez(sqrelu_dir / "basic.npz",
             input=x.numpy(),
             expected=(F.relu(x) ** 2).numpy(),
             shape=np.array(x.shape))

    # Decay exponential: exp(-exp(x))
    decay_dir = output_dir / "kernels" / "decay_exp"
    decay_dir.mkdir(parents=True, exist_ok=True)

    # After softplus, values are in range (-inf, -0.5)
    x = torch.randn(4, 128, 768, dtype=torch.float32) * 2 - 3  # Range roughly [-7, 1]
    expected = torch.exp(-torch.exp(x))
    np.savez(decay_dir / "basic.npz",
             input=x.to(torch.float16).numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array(x.shape))

    # Stability test
    x_extreme = torch.tensor([-50, -10, -5, -1, 0, 1, 5], dtype=torch.float32)
    np.savez(decay_dir / "stability.npz",
             input=x_extreme.to(torch.float16).numpy(),
             expected=torch.exp(-torch.exp(x_extreme)).to(torch.float16).numpy(),
             shape=np.array(x_extreme.shape))

    # Tanh
    tanh_dir = output_dir / "kernels" / "tanh"
    tanh_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(4, 128, 64, dtype=torch.float16)  # LoRA hidden dim
    np.savez(tanh_dir / "basic.npz",
             input=x.numpy(),
             expected=torch.tanh(x).numpy(),
             shape=np.array(x.shape))

    # Softplus decay: -log(1 + exp(-x)) - 0.5
    softplus_dir = output_dir / "kernels" / "softplus_decay"
    softplus_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(4, 128, 768, dtype=torch.float32)
    expected = -F.softplus(-x) - 0.5
    np.savez(softplus_dir / "basic.npz",
             input=x.to(torch.float16).numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array(x.shape))

    # Stability
    x_extreme = torch.tensor([-100, -10, 0, 10, 100], dtype=torch.float32)
    expected = -F.softplus(-x_extreme) - 0.5
    np.savez(softplus_dir / "stability.npz",
             input=x_extreme.to(torch.float16).numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array(x_extreme.shape))

    # Lerp
    lerp_dir = output_dir / "kernels" / "lerp"
    lerp_dir.mkdir(parents=True, exist_ok=True)

    a = torch.randn(4, 128, 768, dtype=torch.float16)
    b = torch.randn(4, 128, 768, dtype=torch.float16)
    t = torch.rand(4, 128, 768, dtype=torch.float16)
    np.savez(lerp_dir / "basic.npz",
             a=a.numpy(),
             b=b.numpy(),
             t=t.numpy(),
             expected=torch.lerp(a, b, t).numpy(),
             shape=np.array(a.shape))

    print("Generated activation fixtures")


def generate_norm_fixtures(output_dir: Path):
    """Generate fixtures for normalization kernels."""

    # Layer Norm
    ln_dir = output_dir / "kernels" / "layer_norm"
    ln_dir.mkdir(parents=True, exist_ok=True)

    B, T, C = 4, 128, 768
    x = torch.randn(B, T, C, dtype=torch.float16)
    weight = torch.randn(C, dtype=torch.float16)
    bias = torch.randn(C, dtype=torch.float16)

    # Convert to float32 for accurate computation
    expected = F.layer_norm(x.float(), (C,), weight.float(), bias.float(), eps=1e-5)

    np.savez(ln_dir / "basic.npz",
             input=x.numpy(),
             weight=weight.numpy(),
             bias=bias.numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array([B, T, C]))

    # RWKV7 epsilon
    np.savez(ln_dir / "rwkv7_eps.npz",
             input=x.numpy(),
             weight=weight.numpy(),
             bias=bias.numpy(),
             expected=F.layer_norm(x.float(), (C,), weight.float(), bias.float(),
                                   eps=1e-5).to(torch.float16).numpy(),
             shape=np.array([B, T, C]),
             eps=np.array([1e-5]))

    # Group Norm (per-head)
    gn_dir = output_dir / "kernels" / "group_norm"
    gn_dir.mkdir(parents=True, exist_ok=True)

    H = 12  # heads for 0.1B
    N = C // H  # head_size = 64
    x = torch.randn(B * T, C, dtype=torch.float16)
    weight = torch.randn(C, dtype=torch.float16)
    bias = torch.randn(C, dtype=torch.float16)

    # RWKV7 uses eps=64e-5
    expected = F.group_norm(x.float(), H, weight.float(), bias.float(), eps=64e-5)

    np.savez(gn_dir / "basic.npz",
             input=x.numpy(),
             weight=weight.numpy(),
             bias=bias.numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array([B * T, C]),
             num_groups=np.array([H]),
             eps=np.array([64e-5]))

    # L2 Norm (per-head)
    l2_dir = output_dir / "kernels" / "l2_norm"
    l2_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(B, T, H, N, dtype=torch.float16)
    expected = F.normalize(x.float(), dim=-1, p=2.0)

    np.savez(l2_dir / "basic.npz",
             input=x.numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array([B, T, H, N]),
             head_size=np.array([N]))

    print("Generated normalization fixtures")


def generate_matmul_fixtures(output_dir: Path):
    """Generate fixtures for matrix multiplication."""

    mm_dir = output_dir / "kernels" / "matmul"
    mm_dir.mkdir(parents=True, exist_ok=True)

    # GEMV (column-major): Y = W * X
    # W: Shape(N, K), X: Shape(K, A=1) => Y: Shape(N, A=1)
    K, N = 768, 768
    A = 1
    input_gemv = torch.randn(A, K, dtype=torch.float16)   # token-major
    weight_gemv = torch.randn(N, K, dtype=torch.float16)  # PyTorch layout
    expected_gemv = F.linear(input_gemv.float(), weight_gemv.float())  # [A, N]

    np.savez(mm_dir / "gemv.npz",
             input=input_gemv.numpy().reshape(-1),
             input_shape=np.array([K, A, 1, 1], dtype=np.int64),
             # Store W in backend order (column-major Shape(N, K)) by flattening W^T in C-order.
             weight=weight_gemv.t().contiguous().numpy().reshape(-1),
             weight_shape=np.array([N, K, 1, 1], dtype=np.int64),
             expected=expected_gemv.to(torch.float16).numpy().reshape(-1),
             expected_shape=np.array([N, A, 1, 1], dtype=np.int64))

    # GEMM batched (column-major): Y = W * X, where X packs B*T tokens.
    B, T = 4, 64
    A = B * T
    input_gemm = torch.randn(A, K, dtype=torch.float16)   # token-major packed
    weight_gemm = torch.randn(N, K, dtype=torch.float16)
    expected_gemm = F.linear(input_gemm.float(), weight_gemm.float())  # [A, N]

    np.savez(mm_dir / "gemm_batched.npz",
             input=input_gemm.numpy().reshape(-1),
             input_shape=np.array([K, A, 1, 1], dtype=np.int64),
             weight=weight_gemm.t().contiguous().numpy().reshape(-1),
             weight_shape=np.array([N, K, 1, 1], dtype=np.int64),
             expected=expected_gemm.to(torch.float16).numpy().reshape(-1),
             expected_shape=np.array([N, A, 1, 1], dtype=np.int64))

    print("Generated matmul fixtures")


def generate_wkv7_fixtures(output_dir: Path):
    """Generate fixtures for WKV7 kernel."""

    wkv_dir = output_dir / "kernels" / "wkv7"
    wkv_dir.mkdir(parents=True, exist_ok=True)

    H, N = 12, 64  # 0.1B config
    C = H * N

    def run_wkv7_reference(B, T, state_in=None):
        """Reference WKV7 implementation matching HIP kernel."""
        # IMPORTANT: Be explicit about what "w" means for the WKV kernel.
        #
        # The reference HIP kernel computes:
        #   w_decay = exp(-exp(w_raw))
        # inside the kernel. Some implementations may choose to precompute w_decay
        # in a separate kernel for simplicity/testing.
        #
        # Fixtures should therefore store BOTH:
        #   - w_raw   (pre exp(-exp))
        #   - w_decay (post exp(-exp))
        #
        # Choose "w" (the array fed into the tested kernel) based on your kernel API.
        w_raw = (torch.randn(B, T, H, N, dtype=torch.float32) * 2.0 - 3.0)  # ~[-7, 1]
        w_decay = torch.exp(-torch.exp(w_raw))

        # For now, define fixture field "w" = w_decay (post-transform) to keep the
        # WKV fixture aligned with the math-only reference below.
        w = w_decay
        q = torch.randn(B, T, H, N, dtype=torch.float16)
        k = torch.randn(B, T, H, N, dtype=torch.float16)
        v = torch.randn(B, T, H, N, dtype=torch.float16)
        a = torch.randn(B, T, H, N, dtype=torch.float16)  # -kk (normalized removal key)
        b = torch.randn(B, T, H, N, dtype=torch.float16)  # kk * a_t

        if state_in is None:
            state = torch.zeros(B, H, N, N, dtype=torch.float32)
        else:
            state = state_in.clone()

        expected_output = torch.empty(B, T, H, N, dtype=torch.float16)

        for t in range(T):
            # Per-head, per-batch processing
            for batch in range(B):
                for head in range(H):
                    s = state[batch, head]  # [N, N]

                    q_t = q[batch, t, head].float()  # [N]
                    w_t = w[batch, t, head]  # [N]
                    k_t = k[batch, t, head].float()  # [N]
                    v_t = v[batch, t, head].float()  # [N]
                    a_t = a[batch, t, head].float()  # [N]
                    b_t = b[batch, t, head].float()  # [N]

                    # sa = dot(state, a)
                    sa = (s * a_t.unsqueeze(0)).sum(dim=1)  # [N]

                    # State update: s = s * w + sa * b + k * v
                    s = s * w_t.unsqueeze(0) + sa.unsqueeze(1) * b_t.unsqueeze(0) + \
                        v_t.unsqueeze(1) * k_t.unsqueeze(0)

                    # Output: y = dot(s, q)
                    y = (s * q_t.unsqueeze(0)).sum(dim=1)  # [N]
                    expected_output[batch, t, head] = y.to(torch.float16)

                    state[batch, head] = s

        return {
            'w_raw': w_raw.to(torch.float16),
            'w_decay': w_decay.to(torch.float16),
            'w': w.to(torch.float16),
            'q': q,
            'k': k,
            'v': v,
            'a': a,
            'b': b,
            'expected_output': expected_output,
            'state_in': state_in if state_in is not None else torch.zeros(B, H, N, N, dtype=torch.float32),
            'expected_state': state,
        }

    def save_case(filename: str, B: int, T: int):
        res = run_wkv7_reference(B, T)

        out = {}

        def put(name: str, t: torch.Tensor, shape4):
            out[name] = t.contiguous().cpu().numpy().reshape(-1)
            out[f"{name}_shape"] = np.array(shape4, dtype=np.int64)

        # WKV kernel tensors are naturally generated as [B, T, H, N] in PyTorch.
        # Flattening them gives the same underlying buffer as backend Shape(N, H, T, B).
        for name in ["w_raw", "w_decay", "w", "q", "k", "v", "a", "b", "expected_output"]:
            put(name, res[name], (N, H, T, B))

        # State is [B, H, N, N] in PyTorch; flattening matches backend Shape(N, N, H, B).
        put("state_in", res["state_in"], (N, N, H, B))
        put("expected_state", res["expected_state"], (N, N, H, B))

        np.savez(wkv_dir / filename, **out)

    save_case("single_token.npz", 1, 1)
    save_case("short_sequence.npz", 1, 16)
    save_case("medium_sequence.npz", 1, 128)
    save_case("batched.npz", 4, 64)

    print("Generated WKV7 fixtures")


def generate_control_k_fixtures(output_dir: Path):
    """Generate fixtures for control_k kernel."""

    ck_dir = output_dir / "kernels" / "control_k"
    ck_dir.mkdir(parents=True, exist_ok=True)

    B, T, C = 4, 128, 768

    k_a = torch.randn(C, dtype=torch.float16)  # Replacement rate booster
    a = torch.rand(B, T, C, dtype=torch.float16)  # In-context learning rate (0-1)
    k = torch.randn(B, T, C, dtype=torch.float16)  # Key

    # k_tilde = k * (1 + (a - 1) * k_a)
    expected = k * (1.0 + (a - 1.0) * k_a)

    np.savez(ck_dir / "basic.npz",
             k_a=k_a.numpy(),
             a=a.numpy(),
             k=k.numpy(),
             expected=expected.numpy(),
             shape=np.array([B, T, C]))

    print("Generated control_k fixtures")


def generate_wkv_bonus_fixtures(output_dir: Path):
    """Generate fixtures for WKV bonus (time_first) kernel."""

    bonus_dir = output_dir / "kernels" / "wkv_bonus"
    bonus_dir.mkdir(parents=True, exist_ok=True)

    B, T, H, N = 4, 128, 12, 64

    r_k = torch.randn(H, N, dtype=torch.float16)  # Per-head bonus weight
    r = torch.randn(B, T, H, N, dtype=torch.float16)  # Receptance
    k = torch.randn(B, T, H, N, dtype=torch.float16)  # Replacement key
    v = torch.randn(B, T, H, N, dtype=torch.float16)  # Value

    # u = (r * k * r_k).sum(dim=-1, keepdim=True) * v
    expected = (r.float() * k.float() * r_k.float()).sum(dim=-1, keepdim=True) * v.float()

    np.savez(bonus_dir / "basic.npz",
             r_k=r_k.numpy(),
             r=r.numpy(),
             k=k.numpy(),
             v=v.numpy(),
             expected=expected.to(torch.float16).numpy(),
             shape=np.array([B, T, H, N]))

    print("Generated WKV bonus fixtures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures'))
    parser.add_argument('--model', type=Path, help='Path to model for layer fixtures')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating fixtures to {args.output}")

    generate_activation_fixtures(args.output)
    generate_norm_fixtures(args.output)
    generate_matmul_fixtures(args.output)
    generate_wkv7_fixtures(args.output)
    generate_control_k_fixtures(args.output)
    generate_wkv_bonus_fixtures(args.output)

    print("\nFixture generation complete!")
    print(f"Total files: {sum(1 for _ in args.output.rglob('*.npz'))}")


if __name__ == '__main__':
    main()
```

## Rust Loading

### NPZ Reader

```rust
// tests/common/mod.rs
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use half::f16;
use npyz::npz::NpzArchive;

pub struct TestFixture {
    pub data: HashMap<String, FixtureArray>,
}

pub enum FixtureArray {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
}

impl TestFixture {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut npz = NpzArchive::open(path)?;
        let mut npz = NpzReader::new(file)?;

        let mut data = HashMap::new();

        for name in npz.names()? {
            // Read array and convert to appropriate type
            // ...
        }

        Ok(Self { data })
    }

    pub fn get_f16(&self, name: &str) -> &[f16] {
        match self.data.get(name) {
            Some(FixtureArray::F16(v)) => v,
            _ => panic!("Expected f16 array for {}", name),
        }
    }

    pub fn get_f32(&self, name: &str) -> &[f32] {
        match self.data.get(name) {
            Some(FixtureArray::F32(v)) => v,
            _ => panic!("Expected f32 array for {}", name),
        }
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        // Get shape from 'shape' or '{name}_shape' array
        self.get_i64("shape").iter().map(|&x| x as usize).collect()
    }
}

/// Compare tensors with tolerance
pub fn assert_tensors_close(
    actual: &[f32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        ));
    }

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let threshold = atol + rtol * e.abs();

        if diff > threshold {
            return Err(format!(
                "Mismatch at index {}: {} vs {} (diff={}, threshold={})",
                i, a, e, diff, threshold
            ));
        }
    }

    Ok(())
}

pub fn assert_tensors_close_f16(
    actual: &[f16],
    expected: &[f16],
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    let actual_f32: Vec<f32> = actual.iter().map(|x| x.to_f32()).collect();
    let expected_f32: Vec<f32> = expected.iter().map(|x| x.to_f32()).collect();
    assert_tensors_close(&actual_f32, &expected_f32, rtol, atol)
}
```

## Layer Fixtures with Model

For layer-level testing, load the actual model weights:

```python
def generate_layer_fixtures(model_path: Path, output_dir: Path):
    """Generate layer fixtures using actual model weights."""
    from safetensors import safe_open

    # Load model
    tensors = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Create model with hooks
    # ... (model-specific code)

    # Run inference and capture intermediates
    # ...
```

## Tolerance Guidelines

| Tensor Type | rtol | atol | Notes |
|-------------|------|------|-------|
| FP16 activations | 1e-3 | 1e-4 | Standard precision |
| FP32 state | 1e-5 | 1e-6 | High precision accumulation |
| Normalized values | 1e-3 | 1e-4 | After normalization |
| MatMul outputs | 1e-2 | 1e-3 | Accumulation error |
| Full model logits | 1e-2 | 1e-3 | Accumulated errors |

## Running Generation

```bash
# Install dependencies
pip install torch numpy safetensors

# Generate all fixtures
python scripts/generate_test_fixtures.py \
    --output tests/fixtures/ \
    --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st

# Verify fixtures are loadable
python -c "import numpy as np; print(np.load('tests/fixtures/kernels/sigmoid/basic.npz').files)"
```
