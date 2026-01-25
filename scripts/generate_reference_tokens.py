#!/usr/bin/env python3
"""
Generate reference tokens for top-k accuracy testing.

This script runs the Python RWKV7 model on a prompt and saves:
1. The input prompt tokens
2. The expected top-k tokens at each generation step
3. The full logits at each step (for debugging)

Usage:
    python scripts/generate_reference_tokens.py \
        --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st \
        --output tests/fixtures/model/generation_reference.npz \
        --num_tokens 100
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open

# Disable TF32 for reproducible precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def pairwise_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pairwise summation for numerical precision."""
    while x.shape[dim] > 1:
        half = x.shape[dim] // 2
        if x.shape[dim] % 2 == 1:
            x = torch.cat([
                x.narrow(dim, 0, half) + x.narrow(dim, half, half),
                x.narrow(dim, 2*half, 1)
            ], dim=dim)
        else:
            x = x.narrow(dim, 0, half) + x.narrow(dim, half, half)
    return x.squeeze(dim)


class RWKV7Reference:
    """Reference RWKV7 implementation for generating test data."""

    def __init__(self, model_path: str):
        self.tensors = {}
        with safe_open(model_path, framework="pt") as f:
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key).float()  # Convert to FP32

        # Extract model info
        emb_weight = self.tensors['emb.weight']
        self.vocab_size, self.n_embd = emb_weight.shape

        # Count layers
        self.n_layer = 0
        while f'blocks.{self.n_layer}.att.r_k' in self.tensors:
            self.n_layer += 1

        # Get head info from first layer
        r_k = self.tensors['blocks.0.att.r_k']
        self.n_head = r_k.shape[0]
        self.head_size = r_k.shape[1]

        print(f"Loaded model: vocab={self.vocab_size}, embd={self.n_embd}, "
              f"layers={self.n_layer}, heads={self.n_head}, head_size={self.head_size}")

        # Initialize state
        self.att_states = [torch.zeros(1, self.n_head, self.head_size, self.head_size,
                                       dtype=torch.float32)
                          for _ in range(self.n_layer)]
        self.ffn_states = [torch.zeros(1, 1, self.n_embd, dtype=torch.float32)
                          for _ in range(self.n_layer)]
        self.att_token_shift_states = [torch.zeros(1, 1, self.n_embd, dtype=torch.float32)
                                       for _ in range(self.n_layer)]
        # v_first is set in layer 0 and used by all other layers
        # (reset at the start of each forward pass)

    def layer_norm(self, x, weight, bias, eps=1e-5):
        """Layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return weight * (x - mean) / torch.sqrt(var + eps) + bias

    def forward_token(self, token_id: int) -> torch.Tensor:
        """Forward pass for a single token, returning logits."""
        # Embedding
        x = self.tensors['emb.weight'][token_id].unsqueeze(0).unsqueeze(0).float()  # [1, 1, C]

        # ln0 (embedding layer norm)
        x = self.layer_norm(x, self.tensors['blocks.0.ln0.weight'],
                           self.tensors['blocks.0.ln0.bias'])

        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        # v_first is computed fresh each forward call
        v_first = None

        # Process each layer
        for layer_idx in range(self.n_layer):
            prefix = f'blocks.{layer_idx}'

            # Get weights
            ln1_w = self.tensors[f'{prefix}.ln1.weight']
            ln1_b = self.tensors[f'{prefix}.ln1.bias']
            ln2_w = self.tensors[f'{prefix}.ln2.weight']
            ln2_b = self.tensors[f'{prefix}.ln2.bias']

            # Time-mix (attention)
            x_ln1 = self.layer_norm(x, ln1_w, ln1_b)

            # Token shift for attention
            x_r = self.tensors[f'{prefix}.att.x_r'].view(1, 1, C)
            x_w = self.tensors[f'{prefix}.att.x_w'].view(1, 1, C)
            x_k = self.tensors[f'{prefix}.att.x_k'].view(1, 1, C)
            x_v = self.tensors[f'{prefix}.att.x_v'].view(1, 1, C)
            x_a = self.tensors[f'{prefix}.att.x_a'].view(1, 1, C)
            x_g = self.tensors[f'{prefix}.att.x_g'].view(1, 1, C)

            shifted = self.att_token_shift_states[layer_idx]
            self.att_token_shift_states[layer_idx] = x_ln1.clone()

            xr = x_ln1 + (shifted - x_ln1) * x_r
            xw = x_ln1 + (shifted - x_ln1) * x_w
            xk = x_ln1 + (shifted - x_ln1) * x_k
            xv = x_ln1 + (shifted - x_ln1) * x_v
            xa = x_ln1 + (shifted - x_ln1) * x_a
            xg = x_ln1 + (shifted - x_ln1) * x_g

            # Linear projections
            r_weight = self.tensors[f'{prefix}.att.receptance.weight']
            k_weight = self.tensors[f'{prefix}.att.key.weight']
            v_weight = self.tensors[f'{prefix}.att.value.weight']
            o_weight = self.tensors[f'{prefix}.att.output.weight']

            r = F.linear(xr, r_weight)
            k = F.linear(xk, k_weight)
            v = F.linear(xv, v_weight)

            # Value residual (layers > 0)
            if layer_idx > 0 and f'{prefix}.att.v0' in self.tensors and v_first is not None:
                v0 = self.tensors[f'{prefix}.att.v0'].view(1, 1, C)
                v1 = self.tensors[f'{prefix}.att.v1']
                v2 = self.tensors[f'{prefix}.att.v2']
                v_lora = F.linear(F.linear(v, v1), v2)
                v_res = torch.sigmoid(v0 + v_lora)
                v = v + (v_first - v) * v_res
            elif layer_idx == 0:
                # Layer 0: save v_first for use in other layers
                v_first = v.clone()

            # w (decay) LoRA
            w0 = self.tensors[f'{prefix}.att.w0'].view(1, 1, C)
            w1 = self.tensors[f'{prefix}.att.w1']
            w2 = self.tensors[f'{prefix}.att.w2']
            w_lora = torch.tanh(F.linear(xw, w1)) @ w2.t()
            w = -F.softplus(-(w0 + w_lora)) - 0.5

            # a (learning rate) LoRA
            a0 = self.tensors[f'{prefix}.att.a0'].view(1, 1, C)
            a1 = self.tensors[f'{prefix}.att.a1']
            a2 = self.tensors[f'{prefix}.att.a2']
            a_lora = F.linear(F.linear(xa, a1), a2)
            a = torch.sigmoid(a0 + a_lora)

            # g (gate) LoRA
            g1 = self.tensors[f'{prefix}.att.g1']
            g2 = self.tensors[f'{prefix}.att.g2']
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

            # L2 normalize k
            k_k = self.tensors[f'{prefix}.att.k_k'].view(1, 1, C)
            kk = F.normalize(k * k_k, dim=-1, p=2.0)

            # Control k
            k_a = self.tensors[f'{prefix}.att.k_a'].view(1, 1, C)
            k_ctrl = k * (1.0 + (a - 1.0) * k_a)

            # WKV inputs
            wkv_a = -kk
            wkv_b = kk * a
            w_decay = torch.exp(w)

            # Reshape for WKV7
            r_wkv = r.view(B, T, H, N)
            k_wkv = k_ctrl.view(B, T, H, N)
            v_wkv = v.view(B, T, H, N)
            w_wkv = w_decay.view(B, T, H, N)
            a_wkv = wkv_a.view(B, T, H, N)
            b_wkv = wkv_b.view(B, T, H, N)

            # Run WKV7 (single token)
            state = self.att_states[layer_idx]
            wkv_out = torch.zeros(B, T, H, N, dtype=x.dtype)
            for batch in range(B):
                for head in range(H):
                    s = state[batch, head]
                    q_t = r_wkv[batch, 0, head]
                    w_t = w_wkv[batch, 0, head]
                    k_t = k_wkv[batch, 0, head]
                    v_t = v_wkv[batch, 0, head]
                    a_t = a_wkv[batch, 0, head]
                    b_t = b_wkv[batch, 0, head]

                    sa = pairwise_sum(s * a_t.unsqueeze(0), dim=1)
                    s = s * w_t.unsqueeze(0) + sa.unsqueeze(1) * b_t.unsqueeze(0) + v_t.unsqueeze(1) * k_t.unsqueeze(0)
                    y = pairwise_sum(s * q_t.unsqueeze(0), dim=1)

                    self.att_states[layer_idx][batch, head] = s
                    wkv_out[batch, 0, head] = y

            wkv_out = wkv_out.view(B, T, C)

            # WKV bonus (time_first)
            r_k = self.tensors[f'{prefix}.att.r_k'].view(1, 1, H, N)
            r_reshape = r.view(B, T, H, N)
            k_reshape = k_ctrl.view(B, T, H, N)
            v_reshape = v.view(B, T, H, N)
            bonus = (r_reshape * k_reshape * r_k).sum(dim=-1, keepdim=True) * v_reshape
            bonus = bonus.view(B, T, C)

            wkv_combined = wkv_out + bonus

            # Group norm
            ln_x_w = self.tensors[f'{prefix}.att.ln_x.weight']
            ln_x_b = self.tensors[f'{prefix}.att.ln_x.bias']
            wkv_normed = F.group_norm(wkv_combined.view(B*T, C), H, ln_x_w, ln_x_b, eps=64e-5)
            wkv_normed = wkv_normed.view(B, T, C)

            # Gate and output
            wkv_gated = wkv_normed * g
            x_att = F.linear(wkv_gated, o_weight)

            # Residual
            x = x + x_att

            # Channel-mix (FFN)
            x_ln2 = self.layer_norm(x, ln2_w, ln2_b)

            x_k_ffn = self.tensors[f'{prefix}.ffn.x_k'].view(1, 1, C)
            shifted_ffn = self.ffn_states[layer_idx]
            self.ffn_states[layer_idx] = x_ln2.clone()

            xk_ffn = x_ln2 + (shifted_ffn - x_ln2) * x_k_ffn

            ffn_k_weight = self.tensors[f'{prefix}.ffn.key.weight']
            ffn_v_weight = self.tensors[f'{prefix}.ffn.value.weight']

            k_ffn = F.linear(xk_ffn, ffn_k_weight)
            k_ffn_sq = F.relu(k_ffn) ** 2
            x_ffn = F.linear(k_ffn_sq, ffn_v_weight)

            x = x + x_ffn

        # Final layer norm and head
        ln_out_w = self.tensors['ln_out.weight']
        ln_out_b = self.tensors['ln_out.bias']
        x = self.layer_norm(x, ln_out_w, ln_out_b)

        head_weight = self.tensors['head.weight']
        logits = F.linear(x, head_weight)

        return logits.squeeze(0).squeeze(0)  # [vocab_size]

    def generate(self, prompt_tokens: list, num_tokens: int, top_k: int = 10) -> dict:
        """Generate tokens and save top-k predictions at each step."""
        all_top_k_tokens = []
        all_top_k_probs = []
        all_argmax_tokens = []

        # Process prompt
        for token in prompt_tokens[:-1]:
            _ = self.forward_token(token)

        # Generate
        current_token = prompt_tokens[-1]
        for i in range(num_tokens):
            logits = self.forward_token(current_token)

            # Get top-k
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_tokens = probs.topk(top_k)

            all_top_k_tokens.append(top_k_tokens.numpy())
            all_top_k_probs.append(top_k_probs.numpy())
            all_argmax_tokens.append(top_k_tokens[0].item())

            # Sample (argmax for reproducibility)
            current_token = top_k_tokens[0].item()

            if i < 10 or i % 20 == 0:
                print(f"Step {i}: token={current_token}, top_k={top_k_tokens[:3].tolist()}")

        return {
            'prompt_tokens': np.array(prompt_tokens, dtype=np.int32),
            'top_k_tokens': np.array(all_top_k_tokens, dtype=np.int32),
            'top_k_probs': np.array(all_top_k_probs, dtype=np.float32),
            'argmax_tokens': np.array(all_argmax_tokens, dtype=np.int32),
            'top_k': np.array([top_k], dtype=np.int32),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st')
    parser.add_argument('--output', type=str,
                       default='tests/fixtures/model/generation_reference.npz')
    parser.add_argument('--num_tokens', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = RWKV7Reference(args.model)

    # Use a simple prompt
    # "The" in RWKV tokenizer
    prompt_tokens = [1, 510]  # BOS + "The"

    print(f"\nGenerating {args.num_tokens} tokens with top-k={args.top_k}...")
    result = model.generate(prompt_tokens, args.num_tokens, args.top_k)

    # Save
    np.savez(args.output, **result)
    print(f"\nSaved to {args.output}")

    # Print summary
    print(f"\nGenerated {len(result['argmax_tokens'])} tokens")
    print(f"First 20 argmax tokens: {result['argmax_tokens'][:20].tolist()}")


if __name__ == '__main__':
    main()
