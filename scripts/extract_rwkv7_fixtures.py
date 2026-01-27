#!/usr/bin/env python3
"""
Extract test fixtures from RWKV7 model using the official rwkvfla implementation.

This uses rwkvfla.ops.rwkv7.chunk_rwkv7 (the official Triton kernel) to generate
ground truth fixtures for all layers. This ensures fixtures match the real RWKV7
behavior exactly.

Usage:
    python scripts/extract_rwkv7_fixtures.py \
        --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.pth \
        --output tests/fixtures/ground_truth/

Programmatic usage:
    from scripts.extract_rwkv7_fixtures import RWKV7FixtureExtractor

    extractor = RWKV7FixtureExtractor("/workspace/models/model.pth")
    fixtures = extractor.extract_all_layers([1, 510, 47])
    extractor.save(fixtures, "tests/fixtures/ground_truth/")
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Disable TF32 for reproducible precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Import official rwkvfla WKV7 implementation
from rwkvfla.ops.rwkv7 import chunk_rwkv7


@dataclass
class ModelConfig:
    """Auto-detected model configuration."""
    n_embd: int
    n_layer: int
    n_head: int
    head_size: int = 64  # Constant across public models
    vocab_size: int = 65536

    # LoRA dimensions (auto-detected)
    d_decay_lora: int = 64
    d_aaa_lora: int = 64
    d_mv_lora: int = 32
    d_gate_lora: int = 128


class RWKV7FixtureExtractor:
    """
    Extract ground truth fixtures from RWKV7 model using official rwkvfla ops.

    Supports both .pth (torch) and .st (safetensors) model formats.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = Path(model_path)
        self.device = device
        self.tensors = self._load_model()
        self.config = self._detect_config()

        # State tracking for RNN-mode inference
        self._reset_state()

    def _load_model(self) -> Dict[str, torch.Tensor]:
        """Load model weights from .pth or .st file."""
        path = self.model_path

        if path.suffix == '.st':
            from safetensors import safe_open
            tensors = {}
            with safe_open(str(path), framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key).float().to(self.device)
        else:
            # .pth format
            raw_tensors = torch.load(str(path), map_location=self.device, weights_only=True)
            # Convert all tensors to float32
            tensors = {k: v.float() if v.is_floating_point() else v
                       for k, v in raw_tensors.items()}

        print(f"Loaded {len(tensors)} tensors from {path}")
        return tensors

    def _detect_config(self) -> ModelConfig:
        """Auto-detect model configuration from weights."""
        emb = self.tensors['emb.weight']
        vocab_size, n_embd = emb.shape

        # Count layers
        n_layer = 0
        while f'blocks.{n_layer}.att.r_k' in self.tensors:
            n_layer += 1

        # Get head info
        r_k = self.tensors['blocks.0.att.r_k']
        n_head = r_k.shape[0]
        head_size = r_k.shape[1]

        # Get LoRA dimensions
        d_decay_lora = self.tensors['blocks.0.att.w1'].shape[1]
        d_aaa_lora = self.tensors['blocks.0.att.a1'].shape[1]
        d_mv_lora = self.tensors['blocks.0.att.v1'].shape[1]
        d_gate_lora = self.tensors['blocks.0.att.g1'].shape[1]

        config = ModelConfig(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            head_size=head_size,
            vocab_size=vocab_size,
            d_decay_lora=d_decay_lora,
            d_aaa_lora=d_aaa_lora,
            d_mv_lora=d_mv_lora,
            d_gate_lora=d_gate_lora,
        )

        print(f"Detected config: n_embd={n_embd}, n_layer={n_layer}, "
              f"n_head={n_head}, head_size={head_size}")
        return config

    def _reset_state(self):
        """Reset all RNN states."""
        C = self.config.n_embd
        H = self.config.n_head
        N = self.config.head_size
        L = self.config.n_layer

        # WKV state: [B, H, N, N] per layer
        self.wkv_states = [
            torch.zeros(1, H, N, N, dtype=torch.float32, device=self.device)
            for _ in range(L)
        ]

        # Token shift states for attention: [B, 1, C] per layer
        self.att_shift_states = [
            torch.zeros(1, 1, C, dtype=torch.float32, device=self.device)
            for _ in range(L)
        ]

        # Token shift states for FFN: [B, 1, C] per layer
        self.ffn_shift_states = [
            torch.zeros(1, 1, C, dtype=torch.float32, device=self.device)
            for _ in range(L)
        ]

        # v_first: computed in layer 0, used by other layers
        self.v_first = None

    def _layer_norm(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Apply layer normalization."""
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    def _group_norm(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor, num_groups: int, eps: float = 64e-5) -> torch.Tensor:
        """Apply group normalization."""
        B, T, C = x.shape
        return F.group_norm(x.view(B * T, C), num_groups, weight, bias, eps).view(B, T, C)

    def forward_layer(self, layer_idx: int, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through a single layer, capturing all intermediates.

        Returns dict with all intermediate tensors for fixture generation.
        """
        B, T, C = x.shape
        H = self.config.n_head
        N = self.config.head_size
        prefix = f'blocks.{layer_idx}'

        intermediates = {'input': x.clone()}

        # Layer norm weights
        ln1_w = self.tensors[f'{prefix}.ln1.weight']
        ln1_b = self.tensors[f'{prefix}.ln1.bias']
        ln2_w = self.tensors[f'{prefix}.ln2.weight']
        ln2_b = self.tensors[f'{prefix}.ln2.bias']

        # Apply ln0 for layer 0 only
        if layer_idx == 0 and f'{prefix}.ln0.weight' in self.tensors:
            ln0_w = self.tensors[f'{prefix}.ln0.weight']
            ln0_b = self.tensors[f'{prefix}.ln0.bias']
            x = self._layer_norm(x, ln0_w, ln0_b)
            intermediates['after_ln0'] = x.clone()

        # ========== TIME-MIX (Attention) ==========
        x_ln1 = self._layer_norm(x, ln1_w, ln1_b)
        intermediates['after_ln1'] = x_ln1.clone()

        # Token shift
        x_r = self.tensors[f'{prefix}.att.x_r'].view(1, 1, C)
        x_w = self.tensors[f'{prefix}.att.x_w'].view(1, 1, C)
        x_k = self.tensors[f'{prefix}.att.x_k'].view(1, 1, C)
        x_v = self.tensors[f'{prefix}.att.x_v'].view(1, 1, C)
        x_a = self.tensors[f'{prefix}.att.x_a'].view(1, 1, C)
        x_g = self.tensors[f'{prefix}.att.x_g'].view(1, 1, C)

        shifted = self.att_shift_states[layer_idx]
        self.att_shift_states[layer_idx] = x_ln1[:, -1:, :].clone()

        # For T>1, we need to shift within the sequence too
        if T > 1:
            xx = torch.cat([shifted, x_ln1[:, :-1, :]], dim=1)
        else:
            xx = shifted

        xr = x_ln1 + (xx - x_ln1) * x_r
        xw = x_ln1 + (xx - x_ln1) * x_w
        xk = x_ln1 + (xx - x_ln1) * x_k
        xv = x_ln1 + (xx - x_ln1) * x_v
        xa = x_ln1 + (xx - x_ln1) * x_a
        xg = x_ln1 + (xx - x_ln1) * x_g

        intermediates['xr'] = xr.clone()
        intermediates['xw'] = xw.clone()
        intermediates['xk'] = xk.clone()
        intermediates['xv'] = xv.clone()
        intermediates['xa'] = xa.clone()
        intermediates['xg'] = xg.clone()

        # Linear projections
        r_weight = self.tensors[f'{prefix}.att.receptance.weight']
        k_weight = self.tensors[f'{prefix}.att.key.weight']
        v_weight = self.tensors[f'{prefix}.att.value.weight']
        o_weight = self.tensors[f'{prefix}.att.output.weight']

        r = F.linear(xr, r_weight)
        k = F.linear(xk, k_weight)
        v = F.linear(xv, v_weight)

        intermediates['r'] = r.clone()
        intermediates['k'] = k.clone()
        intermediates['v'] = v.clone()

        # Value residual (layers > 0) - uses torch.lerp
        if layer_idx > 0 and f'{prefix}.att.v0' in self.tensors and self.v_first is not None:
            v0 = self.tensors[f'{prefix}.att.v0'].view(1, 1, C)
            v1 = self.tensors[f'{prefix}.att.v1']  # [C, d_lora]
            v2 = self.tensors[f'{prefix}.att.v2']  # [d_lora, C]
            v_lora = (xv @ v1) @ v2
            v_res = torch.sigmoid(v0 + v_lora)
            v = torch.lerp(v, self.v_first, v_res)  # lerp(v, v_first, weight)
            intermediates['v_after_residual'] = v.clone()
        elif layer_idx == 0:
            self.v_first = v.clone()

        intermediates['v_first'] = self.v_first.clone() if self.v_first is not None else v.clone()

        # W (decay) LoRA - uses @ not F.linear
        w0 = self.tensors[f'{prefix}.att.w0'].view(1, 1, C)
        w1 = self.tensors[f'{prefix}.att.w1']  # [C, d_lora]
        w2 = self.tensors[f'{prefix}.att.w2']  # [d_lora, C]
        w_lora = torch.tanh(xw @ w1) @ w2
        w = -F.softplus(-(w0 + w_lora)) - 0.5
        intermediates['w'] = w.clone()

        # A (learning rate) LoRA
        a0 = self.tensors[f'{prefix}.att.a0'].view(1, 1, C)
        a1 = self.tensors[f'{prefix}.att.a1']  # [C, d_lora]
        a2 = self.tensors[f'{prefix}.att.a2']  # [d_lora, C]
        a_lora = (xa @ a1) @ a2
        a = torch.sigmoid(a0 + a_lora)
        intermediates['a'] = a.clone()

        # G (gate) LoRA
        g1 = self.tensors[f'{prefix}.att.g1']  # [C, d_lora]
        g2 = self.tensors[f'{prefix}.att.g2']  # [d_lora, C]
        g = torch.sigmoid(xg @ g1) @ g2
        intermediates['g'] = g.clone()

        # L2 normalize k
        k_k = self.tensors[f'{prefix}.att.k_k'].view(1, 1, C)
        kk = F.normalize((k * k_k).view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        intermediates['kk'] = kk.clone()

        # Control k
        k_a = self.tensors[f'{prefix}.att.k_a'].view(1, 1, C)
        k_ctrl = k * (1.0 + (a - 1.0) * k_a)
        intermediates['k_ctrl'] = k_ctrl.clone()

        # WKV inputs
        wkv_a = -kk
        wkv_b = kk * a
        intermediates['wkv_a'] = wkv_a.clone()
        intermediates['wkv_b'] = wkv_b.clone()

        # Save WKV state before
        intermediates['wkv_state_in'] = self.wkv_states[layer_idx].clone()

        # Run WKV7 using official rwkvfla implementation
        # chunk_rwkv7 signature: (r, k, v, a, b, log_w=None, w=None, ...)
        # Note: RWKV7 uses w = exp(-exp(w_raw)) as decay, but chunk_rwkv7 expects log_w
        # w in our code is already -softplus(-...) - 0.5, which is log(decay)

        r_wkv = r.view(B, T, H, N).to(torch.bfloat16)
        k_wkv = k_ctrl.view(B, T, H, N).to(torch.bfloat16)
        v_wkv = v.view(B, T, H, N).to(torch.bfloat16)
        w_wkv = w.view(B, T, H, N).to(torch.bfloat16)
        a_wkv = wkv_a.view(B, T, H, N).to(torch.bfloat16)
        b_wkv = wkv_b.view(B, T, H, N).to(torch.bfloat16)

        # For initial_state, chunk_rwkv7 expects [B, H, N, N]
        state_in = self.wkv_states[layer_idx].to(torch.bfloat16)

        wkv_out, wkv_state_out = chunk_rwkv7(
            r_wkv, k_wkv, v_wkv, a_wkv, b_wkv,
            log_w=w_wkv,
            initial_state=state_in,
            output_final_state=True,
            head_first=False
        )

        # Update state
        self.wkv_states[layer_idx] = wkv_state_out.float()

        wkv_out = wkv_out.float().view(B, T, C)
        intermediates['wkv_out'] = wkv_out.clone()
        intermediates['wkv_state_out'] = self.wkv_states[layer_idx].clone()

        # Group norm
        ln_x_w = self.tensors[f'{prefix}.att.ln_x.weight']
        ln_x_b = self.tensors[f'{prefix}.att.ln_x.bias']
        wkv_normed = self._group_norm(wkv_out, ln_x_w, ln_x_b, H)
        intermediates['wkv_normed'] = wkv_normed.clone()

        # WKV bonus (time_first)
        r_k = self.tensors[f'{prefix}.att.r_k'].view(1, 1, H, N)
        r_reshape = r.view(B, T, H, N)
        k_reshape = k_ctrl.view(B, T, H, N)
        v_reshape = v.view(B, T, H, N)
        bonus = (r_reshape * k_reshape * r_k).sum(dim=-1, keepdim=True) * v_reshape
        bonus = bonus.view(B, T, C)
        intermediates['wkv_bonus'] = bonus.clone()

        wkv_combined = wkv_normed + bonus
        intermediates['wkv_combined'] = wkv_combined.clone()

        # Gate and output
        wkv_gated = wkv_combined * g
        att_out = F.linear(wkv_gated, o_weight)
        intermediates['att_out'] = att_out.clone()

        # Residual
        x = x + att_out
        intermediates['after_att_residual'] = x.clone()

        # ========== CHANNEL-MIX (FFN) ==========
        x_ln2 = self._layer_norm(x, ln2_w, ln2_b)
        intermediates['after_ln2'] = x_ln2.clone()

        # Token shift for FFN
        x_k_ffn = self.tensors[f'{prefix}.ffn.x_k'].view(1, 1, C)
        shifted_ffn = self.ffn_shift_states[layer_idx]
        self.ffn_shift_states[layer_idx] = x_ln2[:, -1:, :].clone()

        if T > 1:
            xx_ffn = torch.cat([shifted_ffn, x_ln2[:, :-1, :]], dim=1)
        else:
            xx_ffn = shifted_ffn

        xk_ffn = x_ln2 + (xx_ffn - x_ln2) * x_k_ffn
        intermediates['xk_ffn'] = xk_ffn.clone()

        # FFN projections
        ffn_k_weight = self.tensors[f'{prefix}.ffn.key.weight']
        ffn_v_weight = self.tensors[f'{prefix}.ffn.value.weight']

        k_ffn = F.linear(xk_ffn, ffn_k_weight)
        intermediates['k_ffn'] = k_ffn.clone()

        k_ffn_sq = F.relu(k_ffn) ** 2
        intermediates['k_ffn_sq'] = k_ffn_sq.clone()

        ffn_out = F.linear(k_ffn_sq, ffn_v_weight)
        intermediates['ffn_out'] = ffn_out.clone()

        # Residual
        x = x + ffn_out
        intermediates['output'] = x.clone()

        return intermediates

    def forward_token(self, token_id: int) -> Dict[str, Any]:
        """Forward pass for a single token, returning logits and all layer intermediates."""
        # Embedding
        x = self.tensors['emb.weight'][token_id].unsqueeze(0).unsqueeze(0).float()

        all_intermediates = {
            'token_id': token_id,
            'embedding': x.clone(),
        }

        # Process each layer
        for layer_idx in range(self.config.n_layer):
            layer_ints = self.forward_layer(layer_idx, x)

            # Prefix all keys with layer index
            for key, val in layer_ints.items():
                all_intermediates[f'layer_{layer_idx}_{key}'] = val

            x = layer_ints['output']

        # Final layer norm and head
        ln_out_w = self.tensors['ln_out.weight']
        ln_out_b = self.tensors['ln_out.bias']
        x = self._layer_norm(x, ln_out_w, ln_out_b)
        all_intermediates['after_ln_out'] = x.clone()

        head_weight = self.tensors['head.weight']
        logits = F.linear(x, head_weight)
        all_intermediates['logits'] = logits.clone()

        return all_intermediates

    def extract_sequence(self, tokens: List[int]) -> Dict[str, Dict[str, Any]]:
        """
        Extract fixtures for a sequence of tokens.

        Returns dict mapping step index to intermediates dict.
        """
        self._reset_state()
        self._tokens = tokens  # Save for config

        all_steps = {}
        for step, token_id in enumerate(tokens):
            print(f"  Step {step}: token {token_id}")
            intermediates = self.forward_token(token_id)
            all_steps[f'step_{step}'] = intermediates

        return all_steps

    def save_fixtures(self, fixtures: Dict[str, Dict[str, Any]], output_dir: Path):
        """Save extracted fixtures to NPZ files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, intermediates in fixtures.items():
            output_file = output_dir / f'{name}.npz'

            # Convert tensors to numpy
            np_data = {}
            for key, val in intermediates.items():
                if isinstance(val, torch.Tensor):
                    np_data[key] = val.detach().cpu().float().numpy()
                else:
                    np_data[key] = np.array(val)

            np.savez(output_file, **np_data)
            print(f"Saved {len(np_data)} tensors to {output_file}")

        # Also save config including token sequence
        config_file = output_dir / 'config.npz'
        np.savez(config_file,
                 n_embd=self.config.n_embd,
                 n_layer=self.config.n_layer,
                 n_head=self.config.n_head,
                 head_size=self.config.head_size,
                 vocab_size=self.config.vocab_size,
                 tokens=np.array(self._tokens, dtype=np.int64),
                 n_steps=len(self._tokens))
        print(f"Saved config to {config_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract RWKV7 fixtures using official rwkvfla')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.pth or .st)')
    parser.add_argument('--output', type=str, default='tests/fixtures/ground_truth/',
                        help='Output directory for fixtures')
    # Default: "<|endoftext|>User: what is the capital of France?\nAssistant:"
    # Token 0 = <|endoftext|> is required per RWKV-7 Goose paper for proper eval
    parser.add_argument('--tokens', type=str,
                        default='0,24281,59,32464,4600,22590,51128,4706,44312,64,11,5585,41693,59',
                        help='Comma-separated token IDs to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    args = parser.parse_args()

    tokens = [int(t) for t in args.tokens.split(',')]

    print(f"Loading model from {args.model}")
    extractor = RWKV7FixtureExtractor(args.model, device=args.device)

    print(f"\nExtracting fixtures for tokens: {tokens}")
    fixtures = extractor.extract_sequence(tokens)

    print(f"\nSaving fixtures to {args.output}")
    extractor.save_fixtures(fixtures, Path(args.output))

    print("\nDone!")


if __name__ == '__main__':
    main()
