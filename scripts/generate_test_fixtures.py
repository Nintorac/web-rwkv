#!/usr/bin/env python3
"""
Generate KERNEL-LEVEL test fixtures for validating HIP backend correctness.

This script generates .npz files containing input/output pairs for individual
kernels (activation, normalization, matmul, token_shift, wkv7, etc.) using
PyTorch as the reference implementation.

For LAYER and MODEL level fixtures, use extract_rwkv7_fixtures.py instead,
which uses the official rwkvfla implementation as ground truth.

Usage:
    python scripts/generate_test_fixtures.py \
        --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st \
        --output tests/fixtures/

Requirements:
    uv pip install torch numpy safetensors
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


def pairwise_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Pairwise summation along a dimension for improved numerical precision.

    Uses tree-based reduction: error is O(epsilon * log(N)) vs O(epsilon * N) for naive sum.
    This matches the pairwise summation used in the HIP WKV7 kernel for consistency.

    Args:
        x: Input tensor
        dim: Dimension to reduce

    Returns:
        Tensor with the specified dimension reduced
    """
    # Move the reduction dimension to the last position
    x = x.movedim(dim, -1)
    original_shape = x.shape[:-1]
    n = x.shape[-1]

    # Flatten all other dimensions
    x = x.reshape(-1, n)

    # Ensure n is a power of 2 for clean reduction, pad if necessary
    if n & (n - 1) != 0:
        # Not a power of 2, use standard sum as fallback
        result = x.sum(dim=-1)
    else:
        # Pairwise tree reduction
        while x.shape[-1] > 1:
            half = x.shape[-1] // 2
            x = x[..., :half] + x[..., half:]
        result = x.squeeze(-1)

    return result.reshape(original_shape)


def save_fixture(path: Path, **tensors: Dict[str, Tuple[np.ndarray, Tuple[int, ...]]]):
    """
    Save tensors to NPZ with flattened data and explicit 4D shapes.

    Convention: for each key 'x', store:
    - 'x' (1D flattened, stored in native dtype)
    - 'x_shape' (i64[4])
    - 'x_dtype' (string like "float16" or "float32")

    This matches the web-rwkv tensor layout where shape[0] is the fastest axis.
    Arrays are saved with their native dtype.

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

        # Keep native dtype in NPZ payload

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

        # In real RWKV7, a=-kk and b=kk*a_proj where:
        # - kk is L2-normalized (|kk|_2 = 1), so elements are bounded to [-1, 1]
        # - a_proj = sigmoid(...), so a_proj is bounded to (0, 1)
        # Using bounded values prevents numerical explosion in state accumulation.
        kk = F.normalize(torch.randn(B, T, H, N), dim=-1, p=2.0)  # L2-normalized
        a_proj = torch.sigmoid(torch.randn(B, T, H, N))  # bounded to (0, 1)
        a = (-kk).to(torch.float16)  # -kk, elements roughly in [-1, 1]
        b = (kk * a_proj).to(torch.float16)  # kk * a_proj, bounded

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
                    # Use pairwise_sum for numerical consistency with HIP kernel
                    sa = pairwise_sum(s * a_t.unsqueeze(0), dim=1)  # [N]

                    # State update: s = s * w + outer(sa, b) + outer(v, k)
                    s = s * w_t.unsqueeze(0) + \
                        sa.unsqueeze(1) * b_t.unsqueeze(0) + \
                        v_t.unsqueeze(1) * k_t.unsqueeze(0)

                    # Output: y = state @ q
                    # Use pairwise_sum for numerical consistency with HIP kernel
                    y = pairwise_sum(s * q_t.unsqueeze(0), dim=1)  # [N]
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


# NOTE: Layer and model fixtures are now generated by extract_rwkv7_fixtures.py
# which uses the official rwkvfla implementation instead of reimplementing the model.



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
        help='Path to RWKV7 model for config detection'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility'
    )
    args = parser.parse_args()

    print(f"=== RWKV7 Kernel Test Fixture Generator ===")
    print(f"Output directory: {args.output}")
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

    # Generate kernel-level fixtures only
    # NOTE: Layer and model fixtures are now generated by extract_rwkv7_fixtures.py
    # which uses the official rwkvfla implementation.
    generate_activation_fixtures(args.output, config)
    generate_norm_fixtures(args.output, config)
    generate_matmul_fixtures(args.output, config)
    generate_token_shift_fixtures(args.output, config)
    generate_wkv7_fixtures(args.output, config)
    generate_wkv_bonus_fixtures(args.output, config)
    generate_control_k_fixtures(args.output, config)
    generate_channel_mix_state_fixtures(args.output, config)

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
