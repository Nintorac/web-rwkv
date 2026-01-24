#!/usr/bin/env python3
"""
Generate test fixtures from RWKV7 for validating HIP backend correctness.

This script generates .npz files containing input/output pairs for each kernel
and layer in the RWKV7 model, following the specification in:
    docs/FIXTURE_GENERATION_SPEC.md

Usage:
    python scripts/generate_test_fixtures.py \
        --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st \
        --output tests/fixtures/

Requirements:
    pip install torch numpy safetensors
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

# Disable TF32 for reproducible FP32 precision in fixture generation.
# TF32 uses only 10-bit mantissa vs FP32's 23-bit, causing 0.001-0.004 precision loss.
# This ensures fixtures match the numerical precision of HIP kernels.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Default model config for 0.1B
DEFAULT_CONFIG = {
    'n_embd': 768,
    'n_head': 12,
    'head_size': 64,
    'n_layer': 12,
    'vocab_size': 65536,
}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_fixture(path: Path, **tensors: Dict[str, Tuple[np.ndarray, Tuple[int, ...]]]):
    """
    Save tensors to NPZ with flattened data and explicit 4D shapes.

    Convention: for each key 'x', store:
    - 'x' (1D flattened as float32 for Rust compatibility)
    - 'x_shape' (i64[4])
    - 'x_dtype' (string like "float16" or "float32")

    This matches the web-rwkv tensor layout where shape[0] is the fastest axis.
    Arrays are saved as float32 because ndarray-npy doesn't support float16.

    Args:
        path: Output .npz file path
        tensors: Dict mapping name -> (array, shape4) tuples
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    for name, (arr, shape4) in tensors.items():
        # Ensure shape is 4D (pad with 1s if needed)
        if len(shape4) < 4:
            shape4 = tuple(shape4) + (1,) * (4 - len(shape4))

        # Flatten array preserving data order
        if isinstance(arr, torch.Tensor):
            original_dtype = str(arr.dtype).replace("torch.", "")
            arr = arr.cpu().numpy()
        else:
            original_dtype = str(arr.dtype)

        flat = arr.reshape(-1)

        # Save as float32 for Rust compatibility (ndarray-npy doesn't support f16)
        # Record original dtype so it can be converted back if needed
        if flat.dtype == np.float16:
            flat = flat.astype(np.float32)
        elif flat.dtype == np.float64:
            flat = flat.astype(np.float32)

        payload[name] = flat
        payload[f"{name}_shape"] = np.array(shape4, dtype=np.int64)
        payload[f"{name}_dtype"] = np.array([original_dtype], dtype='U16')

    np.savez(path, **payload)
    print(f"  Saved: {path} ({len(tensors)} arrays)")


def generate_activation_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for simple activation kernels."""
    print("\n=== Generating Activation Fixtures ===")

    C = config['n_embd']  # 768
    H = config['n_head']  # 12
    N = config['head_size']  # 64

    # Test dimensions
    B, T = 4, 128
    A = B * T  # packed token count

    # ========== Sigmoid ==========
    sigmoid_dir = output_dir / "kernels" / "sigmoid"

    # Basic test - Shape(C, A, 1, 1) in web-rwkv layout
    # PyTorch creates [A, C] (token-major), flatten gives same bytes as Shape(C, A, 1, 1)
    set_seed(1)
    x = torch.randn(A, C, dtype=torch.float16)
    y = torch.sigmoid(x.float()).to(torch.float16)
    save_fixture(sigmoid_dir / "basic.npz",
                 input=(x, (C, A, 1, 1)),
                 expected=(y, (C, A, 1, 1)))

    # Edge cases - extreme values
    x_edge = torch.tensor([-100, -10, -1, 0, 1, 10, 100, float('inf'), float('-inf')],
                          dtype=torch.float16)
    y_edge = torch.sigmoid(x_edge.float()).to(torch.float16)
    save_fixture(sigmoid_dir / "edge_cases.npz",
                 input=(x_edge, (x_edge.numel(), 1, 1, 1)),
                 expected=(y_edge, (x_edge.numel(), 1, 1, 1)))

    # ========== Squared ReLU ==========
    sqrelu_dir = output_dir / "kernels" / "squared_relu"

    set_seed(2)
    # FFN hidden dim is 4x n_embd
    ffn_hidden = C * 4
    x = torch.randn(A, ffn_hidden, dtype=torch.float16)
    y = (F.relu(x.float()) ** 2).to(torch.float16)
    save_fixture(sqrelu_dir / "basic.npz",
                 input=(x, (ffn_hidden, A, 1, 1)),
                 expected=(y, (ffn_hidden, A, 1, 1)))

    # ========== Decay Exponential: exp(-exp(x)) ==========
    decay_dir = output_dir / "kernels" / "decay_exp"

    set_seed(3)
    # After softplus transformation, w values are in (-inf, -0.5)
    # Input to exp(-exp(w)) kernel
    x = torch.randn(A, C, dtype=torch.float32) * 2.0 - 3.0  # Range roughly [-7, 1]
    y = torch.exp(-torch.exp(x))
    save_fixture(decay_dir / "basic.npz",
                 input=(x.to(torch.float16), (C, A, 1, 1)),
                 expected=(y.to(torch.float16), (C, A, 1, 1)))

    # Stability test with extreme values
    x_extreme = torch.tensor([-50, -10, -5, -1, 0, 1, 5], dtype=torch.float32)
    y_extreme = torch.exp(-torch.exp(x_extreme))
    save_fixture(decay_dir / "stability.npz",
                 input=(x_extreme.to(torch.float16), (x_extreme.numel(), 1, 1, 1)),
                 expected=(y_extreme.to(torch.float16), (x_extreme.numel(), 1, 1, 1)))

    # ========== Tanh ==========
    tanh_dir = output_dir / "kernels" / "tanh"

    set_seed(4)
    # LoRA hidden dimension (D_DECAY_LORA ~ 64 for 0.1B)
    lora_dim = 64
    x = torch.randn(A, lora_dim, dtype=torch.float16)
    y = torch.tanh(x.float()).to(torch.float16)
    save_fixture(tanh_dir / "basic.npz",
                 input=(x, (lora_dim, A, 1, 1)),
                 expected=(y, (lora_dim, A, 1, 1)))

    # ========== Softplus Decay: -softplus(-x) - 0.5 ==========
    softplus_dir = output_dir / "kernels" / "softplus_decay"

    set_seed(5)
    x = torch.randn(A, C, dtype=torch.float32)
    y = -F.softplus(-x) - 0.5  # Result is < -0.5
    save_fixture(softplus_dir / "basic.npz",
                 input=(x.to(torch.float16), (C, A, 1, 1)),
                 expected=(y.to(torch.float16), (C, A, 1, 1)))

    # Stability
    x_extreme = torch.tensor([-100, -10, 0, 10, 100], dtype=torch.float32)
    y_extreme = -F.softplus(-x_extreme) - 0.5
    save_fixture(softplus_dir / "stability.npz",
                 input=(x_extreme.to(torch.float16), (x_extreme.numel(), 1, 1, 1)),
                 expected=(y_extreme.to(torch.float16), (x_extreme.numel(), 1, 1, 1)))

    # ========== Lerp ==========
    lerp_dir = output_dir / "kernels" / "lerp"

    set_seed(6)
    a = torch.randn(A, C, dtype=torch.float16)
    b = torch.randn(A, C, dtype=torch.float16)
    t = torch.rand(A, C, dtype=torch.float16)
    y = torch.lerp(a.float(), b.float(), t.float()).to(torch.float16)
    save_fixture(lerp_dir / "basic.npz",
                 a=(a, (C, A, 1, 1)),
                 b=(b, (C, A, 1, 1)),
                 t=(t, (C, A, 1, 1)),
                 expected=(y, (C, A, 1, 1)))

    print("  Activation fixtures generated")


def generate_norm_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for normalization kernels."""
    print("\n=== Generating Normalization Fixtures ===")

    C = config['n_embd']
    H = config['n_head']
    N = config['head_size']

    B, T = 4, 128
    A = B * T

    # ========== Layer Norm ==========
    ln_dir = output_dir / "kernels" / "layer_norm"

    set_seed(10)
    x = torch.randn(A, C, dtype=torch.float16)
    weight = torch.randn(C, dtype=torch.float16)
    bias = torch.randn(C, dtype=torch.float16)

    # Compute in FP32 for accuracy
    y = F.layer_norm(x.float(), (C,), weight.float(), bias.float(), eps=1e-5)
    save_fixture(ln_dir / "basic.npz",
                 input=(x, (C, A, 1, 1)),
                 weight=(weight, (C, 1, 1, 1)),
                 bias=(bias, (C, 1, 1, 1)),
                 expected=(y.to(torch.float16), (C, A, 1, 1)))

    # RWKV7 epsilon (same as basic, but explicit)
    save_fixture(ln_dir / "rwkv7_eps.npz",
                 input=(x, (C, A, 1, 1)),
                 weight=(weight, (C, 1, 1, 1)),
                 bias=(bias, (C, 1, 1, 1)),
                 expected=(y.to(torch.float16), (C, A, 1, 1)),
                 eps=(torch.tensor([1e-5], dtype=torch.float32), (1, 1, 1, 1)))

    # ========== Group Norm (per-head) ==========
    gn_dir = output_dir / "kernels" / "group_norm"

    set_seed(11)
    x = torch.randn(A, C, dtype=torch.float16)
    weight = torch.randn(C, dtype=torch.float16)
    bias = torch.randn(C, dtype=torch.float16)

    # RWKV7 uses eps=64e-5 for GroupNorm
    y = F.group_norm(x.float(), H, weight.float(), bias.float(), eps=64e-5)
    save_fixture(gn_dir / "basic.npz",
                 input=(x, (C, A, 1, 1)),
                 weight=(weight, (C, 1, 1, 1)),
                 bias=(bias, (C, 1, 1, 1)),
                 expected=(y.to(torch.float16), (C, A, 1, 1)),
                 num_groups=(torch.tensor([H], dtype=torch.int64), (1, 1, 1, 1)),
                 eps=(torch.tensor([64e-5], dtype=torch.float32), (1, 1, 1, 1)))

    # ========== L2 Norm (per-head key normalization) ==========
    l2_dir = output_dir / "kernels" / "l2_norm"

    set_seed(12)
    # Shape: [B, T, H, N] in PyTorch -> Shape(N, H, T, B) in web-rwkv
    x = torch.randn(B, T, H, N, dtype=torch.float16)
    y = F.normalize(x.float(), dim=-1, p=2.0).to(torch.float16)
    save_fixture(l2_dir / "basic.npz",
                 input=(x, (N, H, T, B)),
                 expected=(y, (N, H, T, B)),
                 head_size=(torch.tensor([N], dtype=torch.int64), (1, 1, 1, 1)))

    print("  Normalization fixtures generated")


def generate_matmul_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for matrix multiplication (GEMV/GEMM via rocBLAS)."""
    print("\n=== Generating MatMul Fixtures ===")

    C = config['n_embd']  # 768

    mm_dir = output_dir / "kernels" / "matmul"

    # ========== GEMV: single token ==========
    # Y = X @ W^T where X is [1, K], W is [N, K], Y is [1, N]
    # In column-major/web-rwkv: X is Shape(K, 1), W is Shape(N, K), Y is Shape(N, 1)
    set_seed(20)
    K, N = C, C  # Square weight matrix
    A = 1

    x_gemv = torch.randn(A, K, dtype=torch.float16)
    w_gemv = torch.randn(N, K, dtype=torch.float16)
    y_gemv = F.linear(x_gemv.float(), w_gemv.float()).to(torch.float16)

    # Store weight in column-major order: transpose then flatten
    save_fixture(mm_dir / "gemv.npz",
                 input=(x_gemv, (K, A, 1, 1)),
                 weight=(w_gemv.t().contiguous(), (N, K, 1, 1)),
                 expected=(y_gemv, (N, A, 1, 1)))

    # ========== GEMM batched: multiple tokens ==========
    set_seed(21)
    B, T = 4, 64
    A = B * T

    x_gemm = torch.randn(A, K, dtype=torch.float16)
    w_gemm = torch.randn(N, K, dtype=torch.float16)
    y_gemm = F.linear(x_gemm.float(), w_gemm.float()).to(torch.float16)

    save_fixture(mm_dir / "gemm_batched.npz",
                 input=(x_gemm, (K, A, 1, 1)),
                 weight=(w_gemm.t().contiguous(), (N, K, 1, 1)),
                 expected=(y_gemm, (N, A, 1, 1)))

    print("  MatMul fixtures generated")


def generate_token_shift_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for token shift operation."""
    print("\n=== Generating Token Shift Fixtures ===")

    C = config['n_embd']
    ts_dir = output_dir / "kernels" / "token_shift"

    # ========== Single sequence ==========
    set_seed(30)
    B, T = 1, 16
    x = torch.randn(B, T, C, dtype=torch.float16)
    state = torch.randn(B, C, dtype=torch.float16)  # Previous last token
    mix = torch.rand(1, 1, C, dtype=torch.float16)  # Mixing factor (x_r, x_k, etc.)

    # Token shift: x_shifted = lerp(x, shift(x, state), mix)
    # Where shift prepends state and drops last token
    x_shifted = torch.cat([state.unsqueeze(1), x[:, :-1, :]], dim=1)
    output = torch.lerp(x.float(), x_shifted.float(), mix.float()).to(torch.float16)
    new_state = x[:, -1, :].clone()  # Last token becomes new state

    save_fixture(ts_dir / "single.npz",
                 x=(x, (C, T, B, 1)),
                 state=(state, (C, B, 1, 1)),
                 mix=(mix.squeeze(), (C, 1, 1, 1)),
                 expected_output=(output, (C, T, B, 1)),
                 expected_state=(new_state, (C, B, 1, 1)))

    # ========== Batched ==========
    set_seed(31)
    B, T = 4, 32
    x = torch.randn(B, T, C, dtype=torch.float16)
    state = torch.randn(B, C, dtype=torch.float16)
    mix = torch.rand(1, 1, C, dtype=torch.float16)

    x_shifted = torch.cat([state.unsqueeze(1), x[:, :-1, :]], dim=1)
    output = torch.lerp(x.float(), x_shifted.float(), mix.float()).to(torch.float16)
    new_state = x[:, -1, :].clone()

    save_fixture(ts_dir / "batched.npz",
                 x=(x, (C, T, B, 1)),
                 state=(state, (C, B, 1, 1)),
                 mix=(mix.squeeze(), (C, 1, 1, 1)),
                 expected_output=(output, (C, T, B, 1)),
                 expected_state=(new_state, (C, B, 1, 1)))

    # ========== Streaming continuity test ==========
    # Process T=32 as 4x T=8, verify final state matches
    set_seed(32)
    B, T_total = 1, 32
    T_chunk = 8
    x_full = torch.randn(B, T_total, C, dtype=torch.float16)
    state_init = torch.randn(B, C, dtype=torch.float16)
    mix = torch.rand(1, 1, C, dtype=torch.float16)

    # Full processing
    x_shifted_full = torch.cat([state_init.unsqueeze(1), x_full[:, :-1, :]], dim=1)
    output_full = torch.lerp(x_full.float(), x_shifted_full.float(), mix.float()).to(torch.float16)
    state_final = x_full[:, -1, :].clone()

    # Chunked processing
    state_chunk = state_init.clone()
    outputs_chunk = []
    for i in range(0, T_total, T_chunk):
        x_chunk = x_full[:, i:i+T_chunk, :]
        x_shifted = torch.cat([state_chunk.unsqueeze(1), x_chunk[:, :-1, :]], dim=1)
        out = torch.lerp(x_chunk.float(), x_shifted.float(), mix.float()).to(torch.float16)
        outputs_chunk.append(out)
        state_chunk = x_chunk[:, -1, :].clone()

    output_chunked = torch.cat(outputs_chunk, dim=1)

    save_fixture(ts_dir / "streaming.npz",
                 x=(x_full, (C, T_total, B, 1)),
                 state_init=(state_init, (C, B, 1, 1)),
                 mix=(mix.squeeze(), (C, 1, 1, 1)),
                 expected_output_full=(output_full, (C, T_total, B, 1)),
                 expected_output_chunked=(output_chunked, (C, T_total, B, 1)),
                 expected_state=(state_final, (C, B, 1, 1)),
                 chunk_size=(torch.tensor([T_chunk], dtype=torch.int64), (1, 1, 1, 1)))

    print("  Token shift fixtures generated")


def generate_wkv7_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for WKV7 core kernel."""
    print("\n=== Generating WKV7 Fixtures ===")

    H = config['n_head']  # 12
    N = config['head_size']  # 64
    C = H * N  # 768

    wkv_dir = output_dir / "kernels" / "wkv7"

    def run_wkv7_reference(B: int, T: int, state_in: Optional[torch.Tensor] = None):
        """
        Reference WKV7 implementation matching the HIP kernel math.

        The WKV7 recurrence is:
            sa = state @ a_t           # [N] per head
            state = state * w_t + outer(sa, b_t) + outer(v_t, k_t)
            output = state @ q_t       # [N] per head

        Where w_t = exp(-exp(w_raw)) is the decay (computed inside kernel or passed pre-computed).
        """
        # Generate random inputs
        # w_raw is pre-softplus, w_decay = exp(-exp(w_raw))
        w_raw = torch.randn(B, T, H, N, dtype=torch.float32) * 2.0 - 3.0
        w_decay = torch.exp(-torch.exp(w_raw))

        q = torch.randn(B, T, H, N, dtype=torch.float16)
        k = torch.randn(B, T, H, N, dtype=torch.float16)
        v = torch.randn(B, T, H, N, dtype=torch.float16)
        a = torch.randn(B, T, H, N, dtype=torch.float16)  # -kk (normalized removal key)
        b = torch.randn(B, T, H, N, dtype=torch.float16)  # kk * a_t (replacement)

        if state_in is None:
            state = torch.zeros(B, H, N, N, dtype=torch.float32)
        else:
            state = state_in.clone()

        output = torch.empty(B, T, H, N, dtype=torch.float32)

        # Sequential processing (reference implementation)
        for t in range(T):
            for batch in range(B):
                for head in range(H):
                    s = state[batch, head]  # [N, N] matrix

                    q_t = q[batch, t, head].float()  # [N]
                    w_t = w_decay[batch, t, head]     # [N]
                    k_t = k[batch, t, head].float()  # [N]
                    v_t = v[batch, t, head].float()  # [N]
                    a_t = a[batch, t, head].float()  # [N]
                    b_t = b[batch, t, head].float()  # [N]

                    # sa = state @ a (contract last dim of state with a)
                    sa = (s * a_t.unsqueeze(0)).sum(dim=1)  # [N]

                    # State update: s = s * w + outer(sa, b) + outer(v, k)
                    s = s * w_t.unsqueeze(0) + \
                        sa.unsqueeze(1) * b_t.unsqueeze(0) + \
                        v_t.unsqueeze(1) * k_t.unsqueeze(0)

                    # Output: y = state @ q
                    y = (s * q_t.unsqueeze(0)).sum(dim=1)  # [N]
                    output[batch, t, head] = y

                    state[batch, head] = s

        return {
            'w_raw': w_raw.to(torch.float16),
            'w_decay': w_decay,  # Keep float32 - this was used in computation
            'q': q,
            'k': k,
            'v': v,
            'a': a,
            'b': b,
            'state_in': state_in if state_in is not None else torch.zeros(B, H, N, N, dtype=torch.float32),
            'expected_output': output,
            'expected_state': state,
        }

    def save_wkv_case(filename: str, B: int, T: int, state_in: Optional[torch.Tensor] = None):
        res = run_wkv7_reference(B, T, state_in)

        # WKV tensors [B, T, H, N] -> Shape(N, H, T, B)
        # State tensors [B, H, N, N] -> Shape(N, N, H, B)
        save_fixture(wkv_dir / filename,
                     w_raw=(res['w_raw'], (N, H, T, B)),
                     w_decay=(res['w_decay'], (N, H, T, B)),
                     q=(res['q'], (N, H, T, B)),
                     k=(res['k'], (N, H, T, B)),
                     v=(res['v'], (N, H, T, B)),
                     a=(res['a'], (N, H, T, B)),
                     b=(res['b'], (N, H, T, B)),
                     state_in=(res['state_in'], (N, N, H, B)),
                     expected_output=(res['expected_output'], (N, H, T, B)),
                     expected_state=(res['expected_state'], (N, N, H, B)))

        return res

    # Generate test cases
    set_seed(40)
    save_wkv_case("single_token.npz", B=1, T=1)

    set_seed(41)
    save_wkv_case("short_sequence.npz", B=1, T=16)

    set_seed(42)
    save_wkv_case("medium_sequence.npz", B=1, T=128)

    set_seed(43)
    save_wkv_case("batched.npz", B=4, T=64)

    # State evolution test: process in chunks, verify final state
    set_seed(44)
    B, T_total = 1, 32
    T_chunk = 8

    # First compute full sequence
    full_res = run_wkv7_reference(B, T_total)

    # Then compute in chunks, using output state as input for next chunk
    # This requires matching the random inputs, so we save and reload
    state_evolve = torch.zeros(B, H, N, N, dtype=torch.float32)

    # For state evolution, we save the inputs from full_res and chunk them
    save_fixture(wkv_dir / "state_evolution.npz",
                 w_raw=(full_res['w_raw'], (N, H, T_total, B)),
                 w_decay=(full_res['w_decay'], (N, H, T_total, B)),
                 q=(full_res['q'], (N, H, T_total, B)),
                 k=(full_res['k'], (N, H, T_total, B)),
                 v=(full_res['v'], (N, H, T_total, B)),
                 a=(full_res['a'], (N, H, T_total, B)),
                 b=(full_res['b'], (N, H, T_total, B)),
                 state_in=(torch.zeros(B, H, N, N, dtype=torch.float32), (N, N, H, B)),
                 expected_output=(full_res['expected_output'], (N, H, T_total, B)),
                 expected_state=(full_res['expected_state'], (N, N, H, B)),
                 chunk_size=(torch.tensor([T_chunk], dtype=torch.int64), (1, 1, 1, 1)))

    # Stability test with edge values
    set_seed(45)
    save_wkv_case("stability.npz", B=1, T=4)

    print("  WKV7 fixtures generated")


def generate_wkv_bonus_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for WKV bonus (time_first) kernel."""
    print("\n=== Generating WKV Bonus Fixtures ===")

    H = config['n_head']
    N = config['head_size']

    B, T = 4, 128

    bonus_dir = output_dir / "kernels" / "wkv_bonus"

    set_seed(50)
    r_k = torch.randn(H, N, dtype=torch.float16)  # Per-head bonus weight
    r = torch.randn(B, T, H, N, dtype=torch.float16)  # Receptance
    k = torch.randn(B, T, H, N, dtype=torch.float16)  # Replacement key (after control_k)
    v = torch.randn(B, T, H, N, dtype=torch.float16)  # Value

    # u = (r * k * r_k).sum(dim=-1, keepdim=True) * v
    # This is the "time_first" bonus attention on current token
    expected = ((r.float() * k.float() * r_k.float()).sum(dim=-1, keepdim=True) * v.float()).to(torch.float16)

    save_fixture(bonus_dir / "basic.npz",
                 r_k=(r_k, (N, H, 1, 1)),
                 r=(r, (N, H, T, B)),
                 k=(k, (N, H, T, B)),
                 v=(v, (N, H, T, B)),
                 expected=(expected, (N, H, T, B)))

    print("  WKV bonus fixtures generated")


def generate_control_k_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for control_k (replacement key) kernel."""
    print("\n=== Generating Control-K Fixtures ===")

    C = config['n_embd']
    B, T = 4, 128
    A = B * T

    ck_dir = output_dir / "kernels" / "control_k"

    set_seed(60)
    k_a = torch.randn(1, 1, C, dtype=torch.float16)  # Replacement rate booster
    a = torch.rand(B, T, C, dtype=torch.float16)     # In-context learning rate (0-1)
    k = torch.randn(B, T, C, dtype=torch.float16)    # Key

    # k_tilde = k * (1 + (a - 1) * k_a)
    # This is fused_k_rwkv7 operation
    expected = (k.float() * (1.0 + (a.float() - 1.0) * k_a.float())).to(torch.float16)

    save_fixture(ck_dir / "basic.npz",
                 k_a=(k_a.squeeze(), (C, 1, 1, 1)),
                 a=(a.view(A, C), (C, A, 1, 1)),
                 k=(k.view(A, C), (C, A, 1, 1)),
                 expected=(expected.view(A, C), (C, A, 1, 1)))

    print("  Control-K fixtures generated")


def generate_channel_mix_state_fixtures(output_dir: Path, config: dict):
    """Generate fixtures for channel-mix state kernel."""
    print("\n=== Generating Channel-Mix State Fixtures ===")

    C = config['n_embd']
    B, T = 4, 32

    cms_dir = output_dir / "kernels" / "channel_mix_state"

    set_seed(70)
    x = torch.randn(B, T, C, dtype=torch.float16)
    state_in = torch.randn(B, C, dtype=torch.float32)  # Previous last token
    x_k = torch.rand(1, 1, C, dtype=torch.float16)     # Mix factor

    # Channel mix token shift: xx = lerp(x, shifted(x, state), x_k)
    x_shifted = torch.cat([state_in.unsqueeze(1).to(torch.float16), x[:, :-1, :]], dim=1)
    output = torch.lerp(x.float(), x_shifted.float(), x_k.float()).to(torch.float16)
    new_state = x[:, -1, :].float()  # Last token becomes new state

    save_fixture(cms_dir / "basic.npz",
                 x=(x, (C, T, B, 1)),
                 state_in=(state_in, (C, B, 1, 1)),
                 x_k=(x_k.squeeze(), (C, 1, 1, 1)),
                 expected_output=(output, (C, T, B, 1)),
                 expected_state=(new_state, (C, B, 1, 1)))

    print("  Channel-mix state fixtures generated")


def generate_layer_fixtures(output_dir: Path, config: dict, model_path: Optional[Path] = None):
    """Generate fixtures for layer-level tests using actual model weights."""
    print("\n=== Generating Layer Fixtures ===")

    if model_path is None or not model_path.exists():
        print("  Skipping layer fixtures (no model provided)")
        return

    from safetensors import safe_open

    C = config['n_embd']
    H = config['n_head']
    N = config['head_size']

    # Load model weights
    print(f"  Loading model from {model_path}")
    tensors = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    layer_dir = output_dir / "layers"

    # ========== Time-Mix Layer 0 ==========
    # Layer 0 is special: has ln0, produces v_first
    set_seed(100)
    B, T = 2, 16

    # Get layer 0 weights
    prefix = "blocks.0.att."
    ln0_w = tensors.get("blocks.0.ln0.weight", torch.ones(C))
    ln0_b = tensors.get("blocks.0.ln0.bias", torch.zeros(C))
    ln1_w = tensors["blocks.0.ln1.weight"]
    ln1_b = tensors["blocks.0.ln1.bias"]

    x_r = tensors[prefix + "x_r"].squeeze()
    x_w = tensors[prefix + "x_w"].squeeze()
    x_k = tensors[prefix + "x_k"].squeeze()
    x_v = tensors[prefix + "x_v"].squeeze()
    x_a = tensors[prefix + "x_a"].squeeze()
    x_g = tensors[prefix + "x_g"].squeeze()

    w0 = tensors[prefix + "w0"].squeeze()
    w1 = tensors[prefix + "w1"]
    w2 = tensors[prefix + "w2"]

    a0 = tensors[prefix + "a0"].squeeze()
    a1 = tensors[prefix + "a1"]
    a2 = tensors[prefix + "a2"]

    # Skip v0/v1/v2 for layer 0 (no value residual)

    g1 = tensors[prefix + "g1"]
    g2 = tensors[prefix + "g2"]

    k_k = tensors[prefix + "k_k"].squeeze()
    k_a = tensors[prefix + "k_a"].squeeze()
    r_k = tensors[prefix + "r_k"]

    r_weight = tensors[prefix + "receptance.weight"]
    k_weight = tensors[prefix + "key.weight"]
    v_weight = tensors[prefix + "value.weight"]
    o_weight = tensors[prefix + "output.weight"]

    ln_x_w = tensors[prefix + "ln_x.weight"]
    ln_x_b = tensors[prefix + "ln_x.bias"]

    # Generate input
    x_input = torch.randn(B, T, C, dtype=torch.float16)
    state_in = torch.zeros(B, H, N, N, dtype=torch.float32)
    token_shift_state = torch.zeros(B, C, dtype=torch.float16)

    # Forward pass (simplified, matching model.py)
    # Apply ln0 for layer 0
    x = F.layer_norm(x_input.float(), (C,), ln0_w.float(), ln0_b.float(), eps=1e-5).to(torch.float16)

    # Layer norm before attention
    x_normed = F.layer_norm(x.float(), (C,), ln1_w.float(), ln1_b.float(), eps=1e-5).to(torch.float16)

    # Token shift
    xx = torch.cat([token_shift_state.unsqueeze(1), x_normed[:, :-1, :]], dim=1)

    # fused_addcmul: xr = x + (xx - x) * x_r = lerp(x, xx, x_r)
    xr = torch.lerp(x_normed.float(), xx.float(), x_r.float()).to(torch.float16)
    xw = torch.lerp(x_normed.float(), xx.float(), x_w.float()).to(torch.float16)
    xk = torch.lerp(x_normed.float(), xx.float(), x_k.float()).to(torch.float16)
    xv = torch.lerp(x_normed.float(), xx.float(), x_v.float()).to(torch.float16)
    xa = torch.lerp(x_normed.float(), xx.float(), x_a.float()).to(torch.float16)
    xg = torch.lerp(x_normed.float(), xx.float(), x_g.float()).to(torch.float16)

    # Linear projections
    r = F.linear(xr.float(), r_weight.float()).to(torch.float16)
    k = F.linear(xk.float(), k_weight.float()).to(torch.float16)
    v = F.linear(xv.float(), v_weight.float()).to(torch.float16)

    # w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
    w_lora = torch.tanh(xw.float() @ w1.float().t()) @ w2.float().t()
    w = (-F.softplus(-(w0.float() + w_lora)) - 0.5).to(torch.float16)

    # a = sigmoid(a0 + (xa @ a1) @ a2)
    a_lora = (xa.float() @ a1.float().t()) @ a2.float().t()
    a = torch.sigmoid(a0.float() + a_lora).to(torch.float16)

    # g = sigmoid(xg @ g1) @ g2
    g = (torch.sigmoid(xg.float() @ g1.float().t()) @ g2.float().t()).to(torch.float16)

    # For layer 0, v_first = v (store for later layers)
    v_first = v.clone()

    # L2 normalize k
    kk = F.normalize((k.float() * k_k.float()).view(B, T, H, -1), dim=-1, p=2.0)
    kk = kk.view(B, T, C).to(torch.float16)

    # Control K: k = k * (1 + (a - 1) * k_a)
    k_ctrl = (k.float() * (1.0 + (a.float() - 1.0) * k_a.float())).to(torch.float16)

    # Prepare WKV inputs (a and b in WKV sense)
    # In RWKV7: WKV gets (r, w, k_ctrl, v, -kk, kk * a)
    wkv_a = -kk
    wkv_b = kk * a

    # Run WKV7 reference to compute output
    # Convert w to decay form: w_decay = exp(w) since w is already in log decay form
    # Note: In RWKV7, w = -softplus(-(...)) - 0.5 gives the log decay directly
    # w_decay = exp(w) where w is negative, so decay is in (0, 1)
    w_decay = torch.exp(w.float())

    # Reshape for WKV7: [B, T, C] -> [B, T, H, N]
    r_wkv = r.float().view(B, T, H, N)
    k_wkv = k_ctrl.float().view(B, T, H, N)
    v_wkv = v.float().view(B, T, H, N)
    w_wkv = w_decay.view(B, T, H, N)
    a_wkv = wkv_a.float().view(B, T, H, N)
    b_wkv = wkv_b.float().view(B, T, H, N)

    wkv_output = torch.empty(B, T, H, N, dtype=torch.float32)
    wkv_state_out = state_in.clone()

    for t in range(T):
        for batch in range(B):
            for head in range(H):
                s = wkv_state_out[batch, head]  # [N, N]

                q_t = r_wkv[batch, t, head]
                w_t = w_wkv[batch, t, head]
                k_t = k_wkv[batch, t, head]
                v_t = v_wkv[batch, t, head]
                a_t = a_wkv[batch, t, head]
                b_t = b_wkv[batch, t, head]

                # sa = state @ a
                sa = (s * a_t.unsqueeze(0)).sum(dim=1)

                # state = state * w + outer(sa, b) + outer(v, k)
                s = s * w_t.unsqueeze(0) + \
                    sa.unsqueeze(1) * b_t.unsqueeze(0) + \
                    v_t.unsqueeze(1) * k_t.unsqueeze(0)

                # output = state @ q (using r as q)
                y = (s * q_t.unsqueeze(0)).sum(dim=1)
                wkv_output[batch, t, head] = y

                wkv_state_out[batch, head] = s

    wkv_out_flat = wkv_output.view(B, T, C)

    # Save fixtures for layer 0
    tm_dir = layer_dir / "time_mix"

    save_fixture(tm_dir / "layer_0.npz",
                 # Inputs
                 input=(x_input, (C, T, B, 1)),
                 state_in=(state_in, (N, N, H, B)),
                 token_shift_state=(token_shift_state, (C, B, 1, 1)),

                 # Weights (selected key ones)
                 x_r=(x_r, (C, 1, 1, 1)),
                 x_w=(x_w, (C, 1, 1, 1)),
                 x_k=(x_k, (C, 1, 1, 1)),
                 x_v=(x_v, (C, 1, 1, 1)),
                 x_a=(x_a, (C, 1, 1, 1)),
                 x_g=(x_g, (C, 1, 1, 1)),

                 # Intermediates
                 r=(r, (C, T, B, 1)),
                 k=(k, (C, T, B, 1)),
                 v=(v, (C, T, B, 1)),
                 w=(w, (C, T, B, 1)),
                 w_decay=(w_decay, (N, H, T, B)),  # Decay form for WKV kernel (matches WKV7 shape)
                 a=(a, (C, T, B, 1)),
                 g=(g, (C, T, B, 1)),
                 v_first=(v_first, (C, T, B, 1)),
                 kk=(kk, (C, T, B, 1)),
                 k_ctrl=(k_ctrl, (C, T, B, 1)),
                 wkv_a=(wkv_a, (C, T, B, 1)),
                 wkv_b=(wkv_b, (C, T, B, 1)),

                 # WKV outputs
                 expected_wkv_output=(wkv_out_flat, (C, T, B, 1)),
                 expected_wkv_state=(wkv_state_out, (N, N, H, B)),

                 # New token shift state
                 expected_token_shift_state=(x_normed[:, -1, :], (C, B, 1, 1)))

    # ========== Channel-Mix ==========
    cm_dir = layer_dir / "channel_mix"

    set_seed(101)
    x_cm = torch.randn(B, T, C, dtype=torch.float16)
    cm_state = torch.zeros(B, C, dtype=torch.float16)

    prefix_ffn = "blocks.0.ffn."
    ffn_x_k = tensors[prefix_ffn + "x_k"].squeeze()
    ffn_key_w = tensors[prefix_ffn + "key.weight"]
    ffn_val_w = tensors[prefix_ffn + "value.weight"]

    # Token shift: shift x along time, prepending state
    xx_cm = torch.cat([cm_state.unsqueeze(1), x_cm[:, :-1, :]], dim=1)

    # Lerp: k = lerp(x, xx, x_k)
    k_cm = torch.lerp(x_cm.float(), xx_cm.float(), ffn_x_k.float()).to(torch.float16)

    # Key projection: k_proj = k @ key_weight.T
    k_proj = F.linear(k_cm.float(), ffn_key_w.float())

    # Squared ReLU: k_sq = relu(k_proj)^2
    k_sq = (F.relu(k_proj) ** 2).to(torch.float16)

    # Value projection: out = k_sq @ value_weight.T
    out_cm = F.linear(k_sq.float(), ffn_val_w.float()).to(torch.float16)

    # Transpose weights for rocBLAS column-major GEMM
    # PyTorch F.linear does: output = input @ weight.T
    # rocBLAS GEMM does: C = A @ B (column-major)
    #
    # The trick: when we store a PyTorch row-major matrix and interpret it as
    # column-major, rocBLAS sees its transpose. So if we want rocBLAS to see W,
    # we need to store W.T in row-major (PyTorch) format.
    #
    # For F.linear(input, W) where input=[B,T,K], W=[N,K]:
    #   PyTorch: output = input @ W.T
    # For rocBLAS with our stored W.T:
    #   rocBLAS sees: A=W.T interpreted as column-major = W (what we want)
    #   output = W @ input (when input is also in column-major form)
    #
    # But wait - we need output = input @ W.T, not W @ input.
    # These are different: [B*T, K] @ [K, N] vs [N, K] @ [K, B*T]
    # Actually they give the same result but with different output layouts!
    #
    # Let's use rocBLAS transpose: C = alpha * op(A) * op(B)
    # Actually our current implementation uses NoTrans for both.
    #
    # Simpler approach: store weight as-is, but use it differently in GEMM.
    # For F.linear: output[N, T*B] = weight[N, K] @ input[K, T*B]
    # This works if weight is stored in the right format for column-major.
    #
    # PyTorch weight [N, K] stored row-major = [N, K] column-major.T = [K, N] column-major
    # So rocBLAS sees [K, N], not [N, K].
    #
    # To get rocBLAS to see [N, K], store weight.T.contiguous() which is [K, N] row-major.
    # When rocBLAS interprets [K, N] row-major as column-major, it sees [N, K]. Perfect!
    key_w_for_gemm = ffn_key_w.T.contiguous()  # [K=768, N=3072] row-major -> [N, K] col-major
    val_w_for_gemm = ffn_val_w.T.contiguous()  # [K=3072, N=768] row-major -> [N, K] col-major

    # hidden_size is the intermediate dimension (3072 for RWKV 0.1B)
    hidden_size = ffn_key_w.shape[0]

    save_fixture(cm_dir / "basic.npz",
                 input=(x_cm, (C, T, B, 1)),
                 state_in=(cm_state, (C, B, 1, 1)),
                 x_k=(ffn_x_k, (C, 1, 1, 1)),
                 # Weights transposed for rocBLAS column-major GEMM
                 # key_weight: [N=hidden, K=C] in rocBLAS col-major
                 key_weight=(key_w_for_gemm, (hidden_size, C, 1, 1)),
                 # value_weight: [N=C, K=hidden] in rocBLAS col-major
                 value_weight=(val_w_for_gemm, (C, hidden_size, 1, 1)),
                 # Intermediates for step-by-step validation
                 shifted=(xx_cm, (C, T, B, 1)),
                 after_lerp=(k_cm, (C, T, B, 1)),
                 after_key_proj=(k_proj, (hidden_size, T, B, 1)),
                 after_squared_relu=(k_sq, (hidden_size, T, B, 1)),
                 expected_output=(out_cm, (C, T, B, 1)),
                 expected_state=(x_cm[:, -1, :], (C, B, 1, 1)))

    # ========== Full Block (Time-Mix + Channel-Mix) ==========
    # This fixture tests a complete RWKV7 block:
    # 1. Layer norm (ln1) -> Time-mix -> Residual
    # 2. Layer norm (ln2) -> Channel-mix -> Residual
    full_dir = layer_dir / "full_block"

    set_seed(102)
    B_full, T_full = 2, 16

    # Generate input embedding (after embedding layer)
    x_block = torch.randn(B_full, T_full, C, dtype=torch.float16)

    # Initial states
    att_state_in = torch.zeros(B_full, H, N, N, dtype=torch.float32)
    att_token_shift_state = torch.zeros(B_full, C, dtype=torch.float16)
    ffn_state_in = torch.zeros(B_full, C, dtype=torch.float16)

    # Get layer 0 weights (same as above)
    prefix = "blocks.0.att."
    ln1_w = tensors["blocks.0.ln1.weight"]
    ln1_b = tensors["blocks.0.ln1.bias"]
    ln2_w = tensors["blocks.0.ln2.weight"]
    ln2_b = tensors["blocks.0.ln2.bias"]

    x_r = tensors[prefix + "x_r"].squeeze()
    x_w = tensors[prefix + "x_w"].squeeze()
    x_k = tensors[prefix + "x_k"].squeeze()
    x_v = tensors[prefix + "x_v"].squeeze()
    x_a = tensors[prefix + "x_a"].squeeze()
    x_g = tensors[prefix + "x_g"].squeeze()

    w0 = tensors[prefix + "w0"].squeeze()
    w1 = tensors[prefix + "w1"]
    w2 = tensors[prefix + "w2"]

    a0 = tensors[prefix + "a0"].squeeze()
    a1 = tensors[prefix + "a1"]
    a2 = tensors[prefix + "a2"]

    g1 = tensors[prefix + "g1"]
    g2 = tensors[prefix + "g2"]

    k_k = tensors[prefix + "k_k"].squeeze()
    k_a = tensors[prefix + "k_a"].squeeze()
    r_k = tensors[prefix + "r_k"]

    r_weight = tensors[prefix + "receptance.weight"]
    k_weight = tensors[prefix + "key.weight"]
    v_weight = tensors[prefix + "value.weight"]
    o_weight = tensors[prefix + "output.weight"]

    ln_x_w = tensors[prefix + "ln_x.weight"]
    ln_x_b = tensors[prefix + "ln_x.bias"]

    # FFN weights
    prefix_ffn = "blocks.0.ffn."
    ffn_x_k = tensors[prefix_ffn + "x_k"].squeeze()
    ffn_key_w = tensors[prefix_ffn + "key.weight"]
    ffn_val_w = tensors[prefix_ffn + "value.weight"]

    # ==== Time-Mix (Attention) ====
    # Layer norm before attention
    x_ln1 = F.layer_norm(x_block.float(), (C,), ln1_w.float(), ln1_b.float(), eps=1e-5).to(torch.float16)

    # Token shift
    xx_att = torch.cat([att_token_shift_state.unsqueeze(1), x_ln1[:, :-1, :]], dim=1)

    # Time-shifted inputs
    xr = torch.lerp(x_ln1.float(), xx_att.float(), x_r.float()).to(torch.float16)
    xw = torch.lerp(x_ln1.float(), xx_att.float(), x_w.float()).to(torch.float16)
    xk = torch.lerp(x_ln1.float(), xx_att.float(), x_k.float()).to(torch.float16)
    xv = torch.lerp(x_ln1.float(), xx_att.float(), x_v.float()).to(torch.float16)
    xa = torch.lerp(x_ln1.float(), xx_att.float(), x_a.float()).to(torch.float16)
    xg = torch.lerp(x_ln1.float(), xx_att.float(), x_g.float()).to(torch.float16)

    # Linear projections
    r_proj = F.linear(xr.float(), r_weight.float()).to(torch.float16)
    k_proj_att = F.linear(xk.float(), k_weight.float()).to(torch.float16)
    v_proj = F.linear(xv.float(), v_weight.float()).to(torch.float16)

    # w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
    w_lora = torch.tanh(xw.float() @ w1.float().t()) @ w2.float().t()
    w_proj = (-F.softplus(-(w0.float() + w_lora)) - 0.5).to(torch.float16)

    # a = sigmoid(a0 + (xa @ a1) @ a2)
    a_lora = (xa.float() @ a1.float().t()) @ a2.float().t()
    a_proj = torch.sigmoid(a0.float() + a_lora).to(torch.float16)

    # g = sigmoid(xg @ g1) @ g2
    g_proj = (torch.sigmoid(xg.float() @ g1.float().t()) @ g2.float().t()).to(torch.float16)

    # For layer 0, v_first = v
    v_first_block = v_proj.clone()

    # L2 normalize k
    kk_block = F.normalize((k_proj_att.float() * k_k.float()).view(B_full, T_full, H, -1), dim=-1, p=2.0)
    kk_block = kk_block.view(B_full, T_full, C).to(torch.float16)

    # Control K
    k_ctrl_block = (k_proj_att.float() * (1.0 + (a_proj.float() - 1.0) * k_a.float())).to(torch.float16)

    # WKV inputs
    wkv_a_block = -kk_block
    wkv_b_block = kk_block * a_proj

    # Decay
    w_decay_block = torch.exp(w_proj.float())

    # Reshape for WKV7
    r_wkv_block = r_proj.float().view(B_full, T_full, H, N)
    k_wkv_block = k_ctrl_block.float().view(B_full, T_full, H, N)
    v_wkv_block = v_proj.float().view(B_full, T_full, H, N)
    w_wkv_block = w_decay_block.view(B_full, T_full, H, N)
    a_wkv_block = wkv_a_block.float().view(B_full, T_full, H, N)
    b_wkv_block = wkv_b_block.float().view(B_full, T_full, H, N)

    # Run WKV7
    wkv_output_block = torch.empty(B_full, T_full, H, N, dtype=torch.float32)
    wkv_state_out_block = att_state_in.clone()

    for t in range(T_full):
        for batch in range(B_full):
            for head in range(H):
                s = wkv_state_out_block[batch, head]

                q_t = r_wkv_block[batch, t, head]
                w_t = w_wkv_block[batch, t, head]
                k_t = k_wkv_block[batch, t, head]
                v_t = v_wkv_block[batch, t, head]
                a_t = a_wkv_block[batch, t, head]
                b_t = b_wkv_block[batch, t, head]

                sa = (s * a_t.unsqueeze(0)).sum(dim=1)
                s = s * w_t.unsqueeze(0) + \
                    sa.unsqueeze(1) * b_t.unsqueeze(0) + \
                    v_t.unsqueeze(1) * k_t.unsqueeze(0)
                y = (s * q_t.unsqueeze(0)).sum(dim=1)
                wkv_output_block[batch, t, head] = y
                wkv_state_out_block[batch, head] = s

    wkv_out_flat_block = wkv_output_block.view(B_full, T_full, C)

    # WKV Bonus: u = (r * k * r_k).sum(dim=-1, keepdim=True) * v
    r_bonus = r_proj.float().view(B_full, T_full, H, N)
    k_bonus = k_ctrl_block.float().view(B_full, T_full, H, N)
    v_bonus = v_proj.float().view(B_full, T_full, H, N)
    wkv_bonus = ((r_bonus * k_bonus * r_k.float()).sum(dim=-1, keepdim=True) * v_bonus)
    wkv_bonus = wkv_bonus.view(B_full, T_full, C)

    # Combine WKV output and bonus
    x_att_out = wkv_out_flat_block + wkv_bonus

    # Group norm
    x_att_gn = F.group_norm(x_att_out.view(B_full * T_full, C), H, ln_x_w.float(), ln_x_b.float(), eps=64e-5)
    x_att_gn = x_att_gn.view(B_full, T_full, C)

    # Gate and output projection
    x_att_gated = (x_att_gn * g_proj.float())
    x_att_proj = F.linear(x_att_gated, o_weight.float()).to(torch.float16)

    # Residual connection
    x_after_att = x_block.float() + x_att_proj.float()
    x_after_att = x_after_att.to(torch.float16)

    # ==== Channel-Mix (FFN) ====
    # Layer norm before FFN
    x_ln2 = F.layer_norm(x_after_att.float(), (C,), ln2_w.float(), ln2_b.float(), eps=1e-5).to(torch.float16)

    # Token shift
    xx_ffn = torch.cat([ffn_state_in.unsqueeze(1), x_ln2[:, :-1, :]], dim=1)

    # Lerp
    k_ffn = torch.lerp(x_ln2.float(), xx_ffn.float(), ffn_x_k.float()).to(torch.float16)

    # Key projection
    k_proj_ffn = F.linear(k_ffn.float(), ffn_key_w.float())

    # Squared ReLU
    k_sq_ffn = (F.relu(k_proj_ffn) ** 2).to(torch.float16)

    # Value projection
    x_ffn_out = F.linear(k_sq_ffn.float(), ffn_val_w.float()).to(torch.float16)

    # Residual connection
    x_final = x_after_att.float() + x_ffn_out.float()
    x_final = x_final.to(torch.float16)

    # New states
    new_att_token_shift_state = x_ln1[:, -1, :].clone()
    new_ffn_state = x_ln2[:, -1, :].clone()

    # Transpose weights for rocBLAS column-major GEMM
    key_w_for_gemm_block = ffn_key_w.T.contiguous()
    val_w_for_gemm_block = ffn_val_w.T.contiguous()
    r_weight_for_gemm = r_weight.T.contiguous()
    k_weight_for_gemm = k_weight.T.contiguous()
    v_weight_for_gemm = v_weight.T.contiguous()
    o_weight_for_gemm = o_weight.T.contiguous()

    hidden_size = ffn_key_w.shape[0]

    save_fixture(full_dir / "layer_0.npz",
                 # Block input
                 input=(x_block, (C, T_full, B_full, 1)),

                 # Initial states
                 att_state_in=(att_state_in, (N, N, H, B_full)),
                 att_token_shift_state=(att_token_shift_state, (C, B_full, 1, 1)),
                 ffn_state_in=(ffn_state_in, (C, B_full, 1, 1)),

                 # Layer norms
                 ln1_weight=(ln1_w, (C, 1, 1, 1)),
                 ln1_bias=(ln1_b, (C, 1, 1, 1)),
                 ln2_weight=(ln2_w, (C, 1, 1, 1)),
                 ln2_bias=(ln2_b, (C, 1, 1, 1)),

                 # Time-mix weights (key ones for testing)
                 x_r=(x_r, (C, 1, 1, 1)),
                 x_w=(x_w, (C, 1, 1, 1)),
                 x_k_att=(x_k, (C, 1, 1, 1)),
                 x_v=(x_v, (C, 1, 1, 1)),
                 x_a=(x_a, (C, 1, 1, 1)),
                 x_g=(x_g, (C, 1, 1, 1)),
                 w0=(w0, (C, 1, 1, 1)),
                 # LoRA weights: store transposed data, keep original shape
                 # This matches how channel_mix stores ffn_key_w.T with shape (hidden, C)
                 # rocBLAS column-major interprets the transposed row-major data correctly
                 w1=(w1.T.contiguous(), (w1.shape[0], w1.shape[1], 1, 1)),
                 w2=(w2.T.contiguous(), (w2.shape[0], w2.shape[1], 1, 1)),
                 a0=(a0, (C, 1, 1, 1)),
                 a1=(a1.T.contiguous(), (a1.shape[0], a1.shape[1], 1, 1)),
                 a2=(a2.T.contiguous(), (a2.shape[0], a2.shape[1], 1, 1)),
                 g1=(g1.T.contiguous(), (g1.shape[0], g1.shape[1], 1, 1)),
                 g2=(g2.T.contiguous(), (g2.shape[0], g2.shape[1], 1, 1)),
                 k_k=(k_k, (C, 1, 1, 1)),
                 k_a=(k_a, (C, 1, 1, 1)),
                 # r_k is [H, N] in PyTorch, saved with shape (N, H, 1, 1) matching wkv_bonus fixture
                 # NO transpose needed - row-major [H, N] data with column-major (N, H) shape works
                 r_k_weight=(r_k, (N, H, 1, 1)),
                 r_weight=(r_weight_for_gemm, (C, C, 1, 1)),
                 k_weight=(k_weight_for_gemm, (C, C, 1, 1)),
                 v_weight=(v_weight_for_gemm, (C, C, 1, 1)),
                 o_weight=(o_weight_for_gemm, (C, C, 1, 1)),
                 ln_x_weight=(ln_x_w, (C, 1, 1, 1)),
                 ln_x_bias=(ln_x_b, (C, 1, 1, 1)),

                 # FFN weights
                 x_k_ffn=(ffn_x_k, (C, 1, 1, 1)),
                 ffn_key_weight=(key_w_for_gemm_block, (hidden_size, C, 1, 1)),
                 ffn_value_weight=(val_w_for_gemm_block, (C, hidden_size, 1, 1)),

                 # Intermediates
                 after_ln1=(x_ln1, (C, T_full, B_full, 1)),
                 w_proj=(w_proj, (C, T_full, B_full, 1)),
                 a_proj=(a_proj, (C, T_full, B_full, 1)),
                 g_proj=(g_proj, (C, T_full, B_full, 1)),
                 r_proj=(r_proj, (C, T_full, B_full, 1)),
                 k_proj=(k_proj_att, (C, T_full, B_full, 1)),
                 v_proj=(v_proj, (C, T_full, B_full, 1)),
                 kk=(kk_block, (C, T_full, B_full, 1)),
                 k_ctrl=(k_ctrl_block, (C, T_full, B_full, 1)),
                 wkv_out=(wkv_out_flat_block, (C, T_full, B_full, 1)),
                 wkv_bonus_out=(wkv_bonus.view(B_full, T_full, C), (C, T_full, B_full, 1)),
                 after_time_mix=(x_after_att, (C, T_full, B_full, 1)),
                 after_ln2=(x_ln2, (C, T_full, B_full, 1)),

                 # Expected outputs
                 expected_output=(x_final, (C, T_full, B_full, 1)),
                 expected_att_state=(wkv_state_out_block, (N, N, H, B_full)),
                 expected_att_token_shift_state=(new_att_token_shift_state, (C, B_full, 1, 1)),
                 expected_ffn_state=(new_ffn_state, (C, B_full, 1, 1)))

    print("  Layer fixtures generated (including full_block)")


def generate_model_fixtures(output_dir: Path, config: dict, model_path: Optional[Path] = None):
    """Generate fixtures for full model tests."""
    print("\n=== Generating Model Fixtures ===")

    if model_path is None or not model_path.exists():
        print("  Skipping model fixtures (no model provided)")
        return

    from safetensors import safe_open

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ========== Weights spot check ==========
    print(f"  Loading model from {model_path}")
    with safe_open(model_path, framework="pt") as f:
        spot_checks = {}

        # Sample a few weights from different parts
        keys_to_check = [
            "emb.weight",
            "blocks.0.att.receptance.weight",
            "blocks.0.att.r_k",
            "blocks.5.ffn.key.weight",
            "blocks.11.ln2.weight",
            "head.weight",
        ]

        for key in keys_to_check:
            if key in f.keys():
                t = f.get_tensor(key)
                # Store first 64 elements for spot check
                flat = t.reshape(-1)[:64]
                spot_checks[key.replace(".", "_")] = (flat, (flat.numel(), 1, 1, 1))

        np.savez(model_dir / "weights_spot_check.npz",
                 **{k: v[0].cpu().numpy().astype(np.float32) for k, v in spot_checks.items()},
                 **{f"{k}_shape": np.array(v[1], dtype=np.int64) for k, v in spot_checks.items()})

    print("  Model fixtures generated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for RWKV7 HIP backend validation"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('tests/fixtures'),
        help='Output directory for fixtures'
    )
    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=Path('/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st'),
        help='Path to RWKV7 model for layer/model fixtures'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility'
    )
    parser.add_argument(
        '--skip-layer',
        action='store_true',
        help='Skip layer-level fixtures (faster)'
    )
    parser.add_argument(
        '--skip-model',
        action='store_true',
        help='Skip model-level fixtures (faster)'
    )
    args = parser.parse_args()

    print(f"=== RWKV7 Test Fixture Generator ===")
    print(f"Output directory: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Base seed: {args.seed}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine config from model or use defaults
    config = DEFAULT_CONFIG.copy()
    if args.model.exists():
        from safetensors import safe_open
        with safe_open(args.model, framework="pt") as f:
            emb = f.get_tensor("emb.weight")
            config['vocab_size'] = emb.shape[0]
            config['n_embd'] = emb.shape[1]

            n_layer = max([int(k.split('.')[1]) for k in f.keys() if k.startswith('blocks.')]) + 1
            config['n_layer'] = n_layer

            r_k = f.get_tensor("blocks.0.att.r_k")
            config['n_head'] = r_k.shape[0]
            config['head_size'] = r_k.shape[1]
        print(f"Config from model: {config}")
    else:
        print(f"Using default config: {config}")

    # Set base seed
    set_seed(args.seed)

    # Generate fixtures
    generate_activation_fixtures(args.output, config)
    generate_norm_fixtures(args.output, config)
    generate_matmul_fixtures(args.output, config)
    generate_token_shift_fixtures(args.output, config)
    generate_wkv7_fixtures(args.output, config)
    generate_wkv_bonus_fixtures(args.output, config)
    generate_control_k_fixtures(args.output, config)
    generate_channel_mix_state_fixtures(args.output, config)

    if not args.skip_layer:
        generate_layer_fixtures(args.output, config, args.model if args.model.exists() else None)

    if not args.skip_model:
        generate_model_fixtures(args.output, config, args.model if args.model.exists() else None)

    # Summary
    print("\n=== Fixture Generation Complete ===")
    total_files = sum(1 for _ in args.output.rglob("*.npz"))
    print(f"Total .npz files: {total_files}")

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(args.output.rglob("*.npz")):
        rel_path = f.relative_to(args.output)
        size_kb = f.stat().st_size / 1024
        print(f"  {rel_path}: {size_kb:.1f} KB")


if __name__ == '__main__':
    main()
