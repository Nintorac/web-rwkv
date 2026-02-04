# RWKV7 "Goose" Architecture: Complete Implementation Guide

This document provides a comprehensive technical specification of the RWKV7 architecture, derived from the official implementation and mathematical foundations. It is intended for implementers who need to understand every detail of the architecture.

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Model Architecture](#3-model-architecture)
4. [Time-Mixing Block (Attention)](#4-time-mixing-block-attention)
5. [Channel-Mixing Block (FFN)](#5-channel-mixing-block-ffn)
6. [WKV7 Kernel](#6-wkv7-kernel)
7. [Token Shift Mechanism](#7-token-shift-mechanism)
8. [Parameter Initialization](#8-parameter-initialization)
9. [State Management](#9-state-management)
10. [Training Considerations](#10-training-considerations)
11. [Implementation Checklist](#11-implementation-checklist)

---

## 1. Overview

### 1.1 Design Philosophy

RWKV7 employs **Dynamic State Evolution** that transcends the fundamental TC0 expressivity limitations of attention/linear attention paradigms. Unlike traditional attention mechanisms that store and retrieve key-value pairs, RWKV7 maintains an internal state matrix that **learns** relationships between keys and values through simulated gradient descent during the forward pass.

Key characteristics:
- **Constant memory usage** per token (O(1) memory complexity)
- **Constant inference time** per token (O(1) time complexity for inference)
- **NC1 expressivity** - can solve problems that standard attention cannot
- **Parallelizable training** like transformers

### 1.2 Core Insight

RWKV7 maintains an internal model approximating `v ≈ k^T S`, where:
- `S` is a state matrix
- `k` are key vectors
- `v` are value vectors

The model continuously updates `S` to minimize an L2 loss, implementing gradient descent in the forward pass itself.

### 1.3 Dimensions Reference

| Symbol | Description | Typical Values |
|--------|-------------|----------------|
| B | Batch size | Variable |
| T | Sequence length | 512, 1024, 2048, 4096 |
| C | Embedding dimension (n_embd) | 768, 1024, 2048, 4096 |
| H | Number of heads | C / head_size |
| N | Head size (head_size) | 64 (recommended) |
| L | Number of layers | 12, 24, 32, 40, 48 |

---

## 2. Mathematical Foundation

### 2.1 Design Motivation

RWKV7's mechanism is derived from stochastic gradient descent on an internal L2 loss. The model maintains a state matrix `S` that learns to map keys to values:

```
L = (1/2) ||v - k^T S||^2
```

The gradient with respect to state S is:

```
∂L/∂S = S k k^T - v k^T
```

This leads to a recurrent update that simulates gradient descent during inference.

### 2.2 The WKV State Update (Paper Formulation)

The official RWKV-7 state evolution from the paper uses a **transition matrix** formulation:

```
wkv_0 = 0

wkv_t = wkv_{t-1} * G_t + v_t^T · k̃_t
```

Where `G_t` is the **transition matrix**:

```
G_t = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)
```

This can be factored as:

```
G_t = (I - κ̂_t^T (a_t/w_t ⊙ κ̂_t)) * diag(w_t)
```

**Key insight**: G_t approximates a **scaled Householder matrix**, with eigenvalues guaranteed in [-1, 1] for stability. This allows both:
- Dynamic state evolution (information transformation)
- Forget gate behavior (information decay)

### 2.3 Parameter Definitions

| Symbol | Name | Definition | Code Variable |
|--------|------|------------|---------------|
| `k_t` | Key precursor | `x_k_t @ W_k` | `k = self.key(xk)` |
| `κ_t` | Removal key | `k_t ⊙ ξ` | `kk = k * self.k_k` |
| `κ̂_t` | Normalized removal key | `κ_t / ||κ_t||_2` | `kk = normalize(kk)` |
| `k̃_t` | Replacement key | `k_t ⊙ lerp(1, a_t, α)` | `k = k * (1 + (a-1) * k_a)` |
| `a_t` | In-context learning rate | `sigmoid(loramlp_a(...))` | `a = sigmoid(a0 + ...)` |
| `w_t` | Decay | `exp(-exp(raw_w))` | After softplus transform |
| `ξ` | Removal key multiplier | Learned, range ~[-5.3, 9.4] | `self.k_k` |
| `α` | Replacement rate booster | Learned | `self.k_a` |

### 2.4 State Update Components

The state update has three terms:

1. **Decay term**: `wkv_{t-1} * diag(w_t)` - forgetting old information
2. **Removal term**: `-wkv_{t-1} * κ̂_t^T (a_t ⊙ κ̂_t)` - selective erasure
3. **Addition term**: `v_t^T · k̃_t` - adding new key-value association

Combined: The transition matrix `G_t` performs decay AND selective removal in one operation.

### 2.5 Output Computation

```
p_t = LayerNorm(r_t · wkv_t^T) + u_t
o_t = (g_t ⊙ p_t) @ W_o
```

Where `u_t` is the **WKV bonus** (see Section 4.5).

### 2.6 Efficient Kernel Implementation

The matrix formulation is computed efficiently per-element:

```python
# Per head, per batch:
κ̂ = normalize(k * ξ)           # Normalized removal key
sa = sum(state * (-κ̂))          # State weighted by -κ̂ (the 'a' input)
b = κ̂ * a                       # The 'b' input is κ̂ scaled by learning rate

# State update (equivalent to matrix form)
state = state * w + sa * b + k̃ * v

# Output
output = sum(state * r)
```

The kernel receives: `(r, w, k̃, v, -κ̂, κ̂*a)` as `(q, w, k, v, a, b)`.

---

## 3. Model Architecture

### 3.1 Overall Structure

```
Input: token_ids [B, T]
         ↓
    Embedding [B, T, C]
         ↓
    Layer 0: ln0(x) → Block
         ↓
    Layers 1 to L-1: Block (with value residual)
         ↓
    LayerNorm (ln_out)
         ↓
    Linear Head → logits [B, T, vocab_size]
```

### 3.2 Block Structure

Each block contains:
1. **Pre-LayerNorm** (ln1 before attention, ln2 before FFN)
2. **Time-Mixing** (RWKV_Tmix_x070) - the attention mechanism
3. **Channel-Mixing** (RWKV_CMix_x070) - the feed-forward network
4. **Residual connections** around both

```python
class Block:
    def forward(self, x, v_first):
        if layer_id == 0:
            x = self.ln0(x)  # Only layer 0 has ln0

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn  # Residual

        x = x + self.ffn(self.ln2(x))  # Residual
        return x, v_first
```

### 3.3 Model Sizes

| Model | Layers (L) | Dim (C) | Heads (H) | FFN Dim | Params |
|-------|------------|---------|-----------|---------|--------|
| 0.1B  | 12         | 768     | 12        | 3072    | ~0.1B  |
| 0.4B  | 16         | 1024    | 16        | 4096    | ~0.4B  |
| 1.5B  | 24         | 2048    | 32        | 8192    | ~1.5B  |
| 2.9B  | 32         | 4096    | 64        | 16384   | ~2.9B  |
| 7.2B  | 40         | 5120    | 80        | 20480   | ~7.2B  |
| 13.3B | 48         | 8192    | 128       | 32768   | ~13.3B |

---

## 4. Time-Mixing Block (Attention)

### 4.1 Parameters

The time-mixing block has the following learnable parameters:

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| **Token Shift** |||
| x_r | [1, 1, C] | Shift mix for receptance input |
| x_w | [1, 1, C] | Shift mix for decay input |
| x_k | [1, 1, C] | Shift mix for key input |
| x_v | [1, 1, C] | Shift mix for value input |
| x_a | [1, 1, C] | Shift mix for learning rate input |
| x_g | [1, 1, C] | Shift mix for gate input |
| **Decay (w)** |||
| w0 | [1, 1, C] | Decay bias |
| w1 | [C, D_DECAY_LORA] | Decay LoRA down projection |
| w2 | [D_DECAY_LORA, C] | Decay LoRA up projection |
| **Learning Rate (a)** |||
| a0 | [1, 1, C] | Learning rate bias |
| a1 | [C, D_AAA_LORA] | Learning rate LoRA down |
| a2 | [D_AAA_LORA, C] | Learning rate LoRA up |
| **Value Residual (v)** - layers 1+ only |||
| v0 | [1, 1, C] | Value interpolation bias |
| v1 | [C, D_MV_LORA] | Value LoRA down |
| v2 | [D_MV_LORA, C] | Value LoRA up |
| **Gate (g)** |||
| g1 | [C, D_GATE_LORA] | Gate LoRA down |
| g2 | [D_GATE_LORA, C] | Gate LoRA up |
| **Key Parameters** |||
| k_k (ξ) | [1, 1, C] | Removal key multiplier - transforms k into removal key κ |
| k_a (α) | [1, 1, C] | Replacement rate booster - controls amount added to state |
| **WKV Bonus** |||
| r_k (ρ) | [H, N] | Bonus weight - extra attention on current token |
| **Linear Projections** |||
| receptance.weight | [C, C] | Query projection |
| key.weight | [C, C] | Key projection |
| value.weight | [C, C] | Value projection |
| output.weight | [C, C] | Output projection |
| **Normalization** |||
| ln_x | GroupNorm(H, C, eps=64e-5) | Per-head normalization |

### 4.2 LoRA Dimensions

The LoRA dimensions scale with model size:

| Model | D_DECAY_LORA | D_AAA_LORA | D_MV_LORA | D_GATE_LORA |
|-------|--------------|------------|-----------|-------------|
| 0.1B  | 64           | 64         | 32        | 128         |
| 0.4B  | 64           | 64         | 32        | 128         |
| 1.5B  | 96           | 96         | 64        | 256         |
| 2.9B  | 96           | 96         | 64        | 320         |
| 7.2B  | 128          | 128        | 96        | 480         |
| 13.3B | 192          | 192        | 128       | 384         |

Calculation formula:
```python
factor = head_size / 64
D_DECAY_LORA = max(32, int(round((2.5 * sqrt(C)) * factor / 32) * 32))
D_AAA_LORA = max(32, int(round((2.5 * sqrt(C)) * factor / 32) * 32))
D_MV_LORA = max(32, int(round((1.7 * sqrt(C)) * factor / 32) * 32))
D_GATE_LORA = max(32, int(round((5 * sqrt(C)) / 32) * 32))
```

### 4.3 Forward Pass

```python
def forward(self, x, v_first):
    B, T, C = x.size()
    H = self.n_head
    N = self.head_size

    # Step 1: Token Shift
    xx = token_shift(x)  # x shifted by one position in time

    # Step 2: Token Shift Blending
    # Each component gets a different blend of current and shifted
    xr = x + xx * self.x_r  # For receptance
    xw = x + xx * self.x_w  # For decay
    xk = x + xx * self.x_k  # For key
    xv = x + xx * self.x_v  # For value
    xa = x + xx * self.x_a  # For learning rate
    xg = x + xx * self.x_g  # For gate

    # Step 3: Linear Projections
    r = self.receptance(xr)  # Query/receptance [B, T, C]
    k = self.key(xk)         # Key [B, T, C]
    v = self.value(xv)       # Value [B, T, C]

    # Step 4: Compute Decay
    # w is in log-log space, softplus constrains to (-inf, -0.5)
    w = -F.softplus(-(self.w0 + tanh(xw @ self.w1) @ self.w2)) - 0.5

    # Step 5: Compute In-Context Learning Rate
    a = sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # Range [0, 1]

    # Step 6: Value Residual
    if self.layer_id == 0:
        v_first = v  # Store first layer's values
    else:
        # Interpolate between current v and first layer's v
        mix = sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        v = lerp(v, v_first, mix)

    # Step 7: Output Gate
    g = sigmoid(xg @ self.g1) @ self.g2

    # Step 8: Compute Removal Key (κ)
    # κ = k ⊙ ξ, then normalize per head
    # ξ (k_k) transforms key into removal key
    kk = k * self.k_k  # κ = k ⊙ ξ
    kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)  # κ̂

    # Step 9: Compute Replacement Key (k̃)
    # k̃ = k ⊙ lerp(1, a, α) = k * (1 + (a - 1) * α)
    # α (k_a) is the replacement rate booster
    k_tilde = k * (1.0 + (a - 1.0) * self.k_a)

    # Step 10: WKV7 Kernel
    # Core attention computation with transition matrix
    # Kernel inputs: (r, w, k̃, v, -κ̂, κ̂*a)
    # Maps to:       (q, w, k, v, a,   b)
    x = RUN_CUDA_RWKV7g(r, w, k_tilde, v, -kk, kk * a)

    # Step 11: LayerNorm per head (implemented as GroupNorm)
    # Paper: p_t = LayerNorm(r_t · wkv_t^T) + u_t
    x = self.ln_x(x.view(B * T, C)).view(B, T, C)

    # Step 12: WKV Bonus (u_t)
    # Paper: u_t = (r_t · (ρ ⊙ k̃_t)^T) v_t
    # Allows extra attention on current token without storing in state
    # ρ (r_k) is per-head bonus weight
    u = (
        (r.view(B, T, H, N) * k_tilde.view(B, T, H, N) * self.r_k)
        .sum(dim=-1, keepdim=True) * v.view(B, T, H, N)
    ).view(B, T, C)
    x = x + u

    # Step 13: Output Gate and Projection
    # Paper: o_t = (g_t ⊙ p_t) @ W_o
    x = self.output(x * g)

    return x, v_first
```

---

## 5. Channel-Mixing Block (FFN)

### 5.1 Parameters

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| x_k | [1, 1, C] | Token shift mix for key |
| key.weight | [4*C, C] | Expand projection |
| value.weight | [C, 4*C] | Contract projection |

### 5.2 Forward Pass

```python
def forward(self, x):
    # Token shift blending
    xx = token_shift(x)
    k = x + xx * self.x_k

    # Expand with ReLU^2 activation
    k = relu(self.key(k)) ** 2

    # Contract back to embedding dimension
    return self.value(k)
```

### 5.3 Key Design Choices

- **4x expansion ratio** (vs 3.5x in RWKV6)
- **ReLU^2 activation** - provides:
  - Non-linearity
  - Sparsity
  - Smoother gradients than ReLU alone
- **Zero-initialized value weights** - gradual learning

---

## 6. WKV7 Kernel

### 6.1 Input/Output Specification

**Inputs (after reshaping to per-head):**
| Name | Shape | Paper Symbol | Description |
|------|-------|--------------|-------------|
| q | [B, T, H, N] | r_t | Receptance/query |
| w | [B, T, H, N] | (raw) | Log-decay, kernel applies exp(-exp(w)) |
| k | [B, T, H, N] | k̃_t | Replacement key |
| v | [B, T, H, N] | v_t | Value |
| a | [B, T, H, N] | -κ̂_t | Negative normalized removal key |
| b | [B, T, H, N] | κ̂_t ⊙ a_t | Removal key scaled by learning rate |

**Outputs:**
| Name | Shape | Description |
|------|-------|-------------|
| y | [B, T, H, N] | r_t · wkv_t^T (before LayerNorm) |
| s | [B, H, T/CHUNK_LEN, N, N] | Checkpointed states |
| sa | [B, T, H, N] | Saved intermediate for backward |

### 6.2 Relationship to Paper Formulation

The kernel implements the state update:

```
wkv_t = wkv_{t-1} * G_t + v_t^T · k̃_t
```

Where `G_t = diag(w_t) - κ̂_t^T (a_t ⊙ κ̂_t)`

The kernel receives `-κ̂` as `a` and `κ̂ * a_t` as `b`, so:

```
# In kernel notation:
sa = sum(state * a)        # = sum(state * (-κ̂)) = -state · κ̂
state = state * w + sa * b + k * v
      = state * w + (-state · κ̂) * (κ̂ * a_t) + k̃ * v
      = state * w - (state · κ̂) * κ̂ * a_t + k̃ * v
```

This is equivalent to the matrix form `wkv * G_t + v^T k̃` when computed per-element.

### 6.3 Forward Kernel Algorithm

```cpp
// Grid: (H, B) blocks - one block per (head, batch) pair
// Block: (N) threads - one thread per head dimension element
// Each thread maintains its own row of the state matrix

float state[N] = {0};  // Per-thread state vector (N = head_size = 64)

for (int t = 0; t < T; t++) {
    // Load inputs to shared memory (synchronized across block)
    __syncthreads();
    q[i] = query[t, i];
    w[i] = exp(-exp(decay[t, i]));  // Double exp: w ∈ (0, 1)
    k[i] = key[t, i];      // This is k̃ (replacement key)
    a[i] = alpha[t, i];    // This is -κ̂ (negated normalized removal key)
    b[i] = beta[t, i];     // This is κ̂ * a_t
    __syncthreads();

    // Compute sa = sum(state * a) = -sum(state * κ̂)
    // This is the inner product for the removal term
    float sa = 0;
    for (int j = 0; j < N; j++) {
        sa += a[j] * state[j];
    }

    float v = value[t, i];
    float y = 0;

    // State update: implements G_t multiplication + outer product
    for (int j = 0; j < N; j++) {
        // state[j] = state[j] * w[j] + sa * b[j] + k[j] * v
        // Decay      + removal term  + addition term
        state[j] = state[j] * w[j] + sa * b[j] + k[j] * v;

        // Output: r · wkv^T (inner product)
        y += state[j] * q[j];
    }

    output[t, i] = y;

    // Checkpoint every CHUNK_LEN timesteps for gradient computation
    if ((t + 1) % CHUNK_LEN == 0) {
        save_state(state);
    }
}
```

### 6.4 State Update in Matrix Form

The per-element kernel update implements the matrix equation:

```
wkv_t = wkv_{t-1} * (diag(w) - κ̂^T (a ⊙ κ̂)) + v^T · k̃
```

In element form for state[i][j]:
```
state[i][j] = state[i][j] * w[j] + (sum_k state[i][k] * (-κ̂[k])) * (κ̂[j] * a[j]) + k̃[j] * v[i]
```

The `-κ̂` and `κ̂ * a` formulation allows efficient computation without explicit matrix multiplication.

### 6.4 Backward Kernel

The backward pass computes gradients using checkpointed states:

```cpp
for (int t = T-1; t >= 0; t--) {
    // Load from checkpoint if available
    if ((t + 1) % CHUNK_LEN == 0) {
        load_state(stateT);
    }

    // Gradient w.r.t. query
    dq[t] = sum(stateT * dy);

    // Reconstruct previous state
    stateT = (stateT - k * v - b * sa) / w;

    // Accumulate gradient contributions
    dstate += q * dy;

    // Gradients w.r.t. other parameters
    dw[t] = -sum(dstate * prev_state) * w * exp(w_raw);
    dk[t] = sum(dstate * v);
    dv[t] = sum(dstate * k);
    db[t] = sum(dstate * sa);

    // Gradient for sa
    dSb = sum(dstate * b);
    da[t] = sum(prev_state * dSb);

    // Propagate gradient to previous state
    dstate = dstate * w + dSb * a;
}
```

### 6.5 Numerical Considerations

- **BF16 I/O, FP32 state** - Prevents precision loss in accumulation
- **Double exponential for decay** - `exp(-exp(w))` ensures decay in (0, 1)
- **GroupNorm epsilon** - Uses 64e-5 (larger than default 1e-5) for BF16 stability
- **Checkpointing** - Every 16 timesteps balances memory vs recomputation

---

## 7. Token Shift Mechanism

### 7.1 Implementation

```python
self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

def token_shift(x):
    # x: [B, T, C]
    # Pads top with zeros, removes bottom row
    # Result: xx[t] = x[t-1] for t > 0, xx[0] = 0
    return self.time_shift(x)
```

### 7.2 Blending Formula

Each component uses a different blend:
```python
xr = x + xx * x_r  # x + (x_{t-1} - x) * mix = lerp(x, x_{t-1}, mix)
```

Where `x_r` values close to 0 mean less shift (more current), close to 1 means more shift (more previous).

### 7.3 Per-Channel Initialization

```python
# Channel-dependent mix strength
ddd[i] = i / C  # 0 to 1 across channels

# Different exponents for different components
x_r = 1.0 - pow(ddd, 0.2 * ratio_1_to_almost0)  # Weaker shift
x_w = 1.0 - pow(ddd, 0.9 * ratio_1_to_almost0)  # Stronger shift
x_k = 1.0 - pow(ddd, 0.7 * ratio_1_to_almost0)
x_v = 1.0 - pow(ddd, 0.7 * ratio_1_to_almost0)
x_a = 1.0 - pow(ddd, 0.9 * ratio_1_to_almost0)
x_g = 1.0 - pow(ddd, 0.2 * ratio_1_to_almost0)

# ratio_1_to_almost0 = 1 - layer_id / n_layer
# Earlier layers: stronger shift overall
# Later layers: weaker shift overall
```

---

## 8. Parameter Initialization

### 8.1 Decay Parameters (w0, w1, w2)

```python
# Linear component
linear[n] = n / (C - 1) - 0.5  # Range [-0.5, 0.5]

# Zigzag pattern within each head
zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
zigzag[n] = zigzag[n] * abs(zigzag[n])  # Amplified zigzag

# Base decay values
www[n] = -6 + 6 * (n / (C - 1)) ** (1 + ratio_0_to_1 ** 0.3)
# Ranges from -6 to 0 across channels
# Later layers have steeper curve

# Final w0: combines base + offset + zigzag
w0 = www + 0.5 + zigzag * 2.5
# The 0.5 offset compensates for softplus

# LoRA weights
w1 = zeros(C, D_DECAY_LORA)  # Zero-initialized
w2 = ortho_init(zeros(D_DECAY_LORA, C), gain=0.1)
```

### 8.2 Learning Rate Parameters (a0, a1, a2)

```python
a0 = zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
# Base: -0.19 → sigmoid ≈ 0.45 (moderate learning rate)
# Variation from zigzag and linear patterns

a1 = zeros(C, D_AAA_LORA)  # Zero-initialized
a2 = ortho_init(zeros(D_AAA_LORA, C), gain=0.1)
```

### 8.3 Value Residual Parameters (v0, v1, v2)

Only for layers 1+:

```python
v0 = zeros(1, 1, C) + 0.73 - linear * 0.4
# Base: 0.73 → sigmoid ≈ 0.67 (use 67% first layer values)
# Earlier channels: more value residual
# Later channels: less value residual

v1 = zeros(C, D_MV_LORA)
v2 = ortho_init(zeros(D_MV_LORA, C), gain=0.1)
```

### 8.4 Gate Parameters (g1, g2)

```python
g1 = zeros(C, D_GATE_LORA)  # Zero-initialized
g2 = ortho_init(zeros(D_GATE_LORA, C), gain=0.1)
```

### 8.5 Key Parameters (ξ, α, ρ)

```python
# ξ (k_k) - Removal key multiplier
# Transforms key k into removal key κ: κ = k ⊙ ξ
# Paper notes ξ lies in range approximately [-5.3, 9.4] after training
k_k = zeros(1, 1, C) + 0.71 - linear * 0.1

# α (k_a) - Replacement rate booster
# Controls replacement key: k̃ = k ⊙ lerp(1, a, α)
# Adjusts amount added to state after transition
k_a = zeros(1, 1, C) + 1.02

# ρ (r_k) - WKV Bonus weight
# Weights extra attention on current token: u = (r · (ρ ⊙ k̃)^T) v
r_k = zeros(H, N) - 0.04
```

### 8.6 Linear Layers

```python
receptance.weight = uniform(-0.5 / sqrt(C), 0.5 / sqrt(C))
key.weight = uniform(-0.05 / sqrt(C), 0.05 / sqrt(C))  # 10x smaller
value.weight = uniform(-0.5 / sqrt(C), 0.5 / sqrt(C))
output.weight = zeros(C, C)  # Zero-initialized (gradual learning)
```

### 8.7 Embedding and Head

```python
emb.weight = uniform(-1e-4, 1e-4)  # Very small

# Head initialization depends on vocab/emb ratio
if vocab_size > n_embd:
    scale = 0.5 * sqrt(vocab_size / n_embd)
else:
    scale = 0.5
head.weight = ortho_init(gain=scale)
```

### 8.8 GroupNorm Weights

```python
# Scale by layer depth
layer_scale = (1 + layer_id) / n_layer
ln_x.weight = layer_scale ** 0.7
ln_x.bias = 0
```

---

## 9. State Management

### 9.1 State Representation

The state is a matrix per head:

```
State shape: [B, H, V, K] where V = K = head_size (N)
```

For a 1.5B model (H=32, N=64):
- State per sample: 32 × 64 × 64 = 131,072 floats = 512KB (FP32)
- State size is **constant** regardless of sequence length

### 9.2 Initial State

State is initialized to zeros:
```python
state = torch.zeros(B, H, V, K, dtype=torch.float32)
```

### 9.3 State Checkpointing

During training, states are saved every `CHUNK_LEN` (default 16) timesteps:

```python
# Total checkpoints for sequence length T
num_checkpoints = T // CHUNK_LEN

# Checkpoint storage shape
checkpoints = [B, H, num_checkpoints, N, N]
```

This enables gradient computation without storing every intermediate state.

### 9.4 Value Residual (v_first)

A special cross-layer state that caches the first layer's values:

```python
v_first = torch.empty(B, T, C)

# Layer 0 stores
v_first = v

# Layers 1+ interpolate
v = lerp(v, v_first, sigmoid(v0 + (xv @ v1) @ v2))
```

---

## 10. Training Considerations

### 10.1 Recommended Learning Rates

| Model Size | Learning Rate |
|------------|---------------|
| L12-D768 (0.1B) | 6e-4 |
| L24-D1024 (0.4B) | 4e-4 |
| L24-D2048 (1.5B) | 3e-4 |

### 10.2 Optimizer Settings

```python
# Adam parameters
beta1 = 0.9
beta2 = 0.99
eps = 1e-18  # Very small epsilon
weight_decay = 0  # or small positive value

# Gradient clipping
grad_clip = 1.0
```

### 10.3 Learning Rate Schedules

- `w0` parameters use **2x learning rate**
- Weight decay applied to: embeddings, attention weights, key weights, value weights, head weights

### 10.4 L2Wrap Regularization

Prevents overconfidence and BF16 precision loss:

```python
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # Encourage logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy
```

### 10.5 Precision Requirements

- **Training**: BF16 for most operations, FP32 for state accumulation
- **State**: Always FP32 to prevent precision loss
- **Gradients**: BF16 acceptable with gradient scaling

---

## 11. Implementation Checklist

### 11.1 Core Components

- [ ] Embedding layer with small initialization
- [ ] Token shift mechanism (ZeroPad2d)
- [ ] Time-mixing block with all 6 shift variants
- [ ] Channel-mixing block with ReLU^2
- [ ] WKV7 kernel with correct state update
- [ ] GroupNorm with per-head grouping
- [ ] Value residual passing between layers
- [ ] Output head with orthogonal init

### 11.2 Time-Mixing Details

- [ ] LoRA for decay (w0, w1, w2)
- [ ] LoRA for learning rate (a0, a1, a2)
- [ ] LoRA for value residual (v0, v1, v2) - layers 1+ only
- [ ] LoRA for gate (g1, g2)
- [ ] Key normalization (k_k, L2 normalize per head)
- [ ] Key adaptation (k_a)
- [ ] Per-head r_k scaling
- [ ] Softplus clamping for decay

### 11.3 WKV7 Kernel

- [ ] Double exponential for decay: `exp(-exp(w))`
- [ ] sa computation: `sum(state * a)`
- [ ] State update: `state = w * state + sa * b + k * v`
- [ ] Output: `sum(state * q)`
- [ ] Checkpointing every CHUNK_LEN steps
- [ ] BF16 I/O, FP32 state
- [ ] Backward with state reconstruction

### 11.4 Initialization

- [ ] Depth-aware token shift (ratio_1_to_almost0)
- [ ] Zigzag pattern for decay
- [ ] Zero init for output weights
- [ ] Orthogonal init for LoRA up-projections
- [ ] Per-layer GroupNorm scaling

### 11.5 Numerical Stability

- [ ] Large epsilon (64e-5) for GroupNorm
- [ ] FP32 state accumulation
- [ ] Gradient clipping
- [ ] L2Wrap regularization

---

## Appendix A: Complete Forward Pass Pseudocode

```python
def rwkv7_forward(tokens):
    # Embedding
    x = embedding(tokens)  # [B, T, C]

    # Initialize value residual storage
    v_first = empty_like(x)

    for layer_id, block in enumerate(blocks):
        # Layer 0 only: initial layer norm
        if layer_id == 0:
            x = layer_norm_0(x)

        # === TIME MIXING ===
        x_normed = layer_norm_1(x)

        # Token shift: x_{t-1} for lerp blending
        xx = token_shift(x_normed) - x_normed  # Difference for lerp

        # Token shift blending (Eq. 3 in paper)
        # lerp(x_t, x_{t-1}, μ) = x_t + (x_{t-1} - x_t) * μ
        xr = x_normed + xx * μ_r
        xw = x_normed + xx * μ_w  # Paper calls this x_d
        xk = x_normed + xx * μ_k
        xv = x_normed + xx * μ_v
        xa = x_normed + xx * μ_a
        xg = x_normed + xx * μ_g

        # Receptance (query) - Eq. 13
        r = xr @ W_r

        # Key precursor - Eq. 5
        k = xk @ W_k

        # Value computation with residual - Eq. 9-10
        v_prime = xv @ W_v
        if layer_id == 0:
            v_first = v_prime
            v = v_prime
        else:
            # ν = sigmoid(loramlp_v(...)) - Eq. 9
            ν = sigmoid(v0 + (xv @ v1) @ v2)
            # v = lerp(v'_{t,0}, v'_{t,l}, ν) - Eq. 10
            v = v_prime + (v_first - v_prime) * ν

        # In-context learning rate - Eq. 4
        a = sigmoid(a0 + (xa @ a1) @ a2)

        # Decay - Eq. 11-12
        # Paper: w = exp(-e^{-0.5} * sigmoid(d))
        # Code uses softplus for numerical stability
        d = w0 + tanh(xw @ w1) @ w2
        w = -softplus(-d) - 0.5  # Then kernel applies exp(-exp(w))

        # Gate - Eq. 14
        g = sigmoid(xg @ g1) @ g2

        # Removal key κ - Eq. 6
        # κ = k ⊙ ξ, then normalize
        κ = k * ξ  # ξ is k_k
        κ_hat = normalize(κ, dim=-1, per_head=True)  # Eq. 15

        # Replacement key k̃ - Eq. 7
        # k̃ = k ⊙ lerp(1, a, α)
        k_tilde = k * (1 + (a - 1) * α)  # α is k_a

        # WKV7 kernel - Eq. 16-17
        # wkv_t = wkv_{t-1} * G_t + v^T · k̃
        # where G_t = diag(w) - κ̂^T (a ⊙ κ̂)
        wkv_out = wkv7_kernel(r, w, k_tilde, v, -κ_hat, κ_hat * a)

        # LayerNorm per head + WKV bonus - Eq. 20-21
        # p = LayerNorm(r · wkv^T) + u
        p = layer_norm_per_head(wkv_out)
        # u = (r · (ρ ⊙ k̃)^T) v - Eq. 20
        u = (r * k_tilde * ρ).sum(dim=-1, keepdim=True) * v  # ρ is r_k
        p = p + u

        # Output - Eq. 22
        # o = (g ⊙ p) @ W_o
        att_out = (g * p) @ W_o

        x = x + att_out

        # === CHANNEL MIXING (MLP) - Eq. 23-24 ===
        x_normed = layer_norm_2(x)
        xx = token_shift(x_normed) - x_normed

        # k' = lerp(x', x'_{t-1}, μ'_k) @ W_k'
        k_prime = (x_normed + xx * μ_k_prime) @ W_k_prime

        # o' = ReLU(k')^2 @ W_v'
        ffn_out = relu(k_prime) ** 2 @ W_v_prime

        x = x + ffn_out

    # Output
    x = layer_norm_out(x)
    logits = head(x)
    return logits
```

---

## Appendix B: Glossary

| Term | Paper Symbol | Code Variable | Definition |
|------|--------------|---------------|------------|
| **Receptance** | r_t | `r` | Query vector that selects from state |
| **Key precursor** | k_t | `k` | Raw key before transformation |
| **Removal key** | κ_t | `kk` (pre-normalize) | Key for selective state erasure |
| **Normalized removal key** | κ̂_t | `kk` (post-normalize) | L2-normalized removal key per head |
| **Replacement key** | k̃_t | `k` (after k_a multiply) | Key for adding to state |
| **In-context learning rate** | a_t | `a` | Per-channel, per-timestep learning rate |
| **Decay** | w_t | `w` (after exp(-exp())) | Diagonal forget gate values |
| **Removal key multiplier** | ξ | `k_k` | Transforms k → κ |
| **Replacement rate booster** | α | `k_a` | Controls amount added after transition |
| **WKV bonus weight** | ρ | `r_k` | Extra attention on current token |
| **Value residual mix** | ν_t | (computed inline) | Interpolation weight for v_first |
| **Transition matrix** | G_t | (computed in kernel) | diag(w) - κ̂^T(a ⊙ κ̂) |
| **Token shift mix** | μ_□ | `x_r`, `x_w`, etc. | Per-channel lerp weights |
| **WKV state** | wkv_t | `state` (in kernel) | Matrix mapping keys to values |
| **WKV bonus** | u_t | `u` | (r · (ρ ⊙ k̃)^T) v |
| **LoRA** | loramlp | `*1`, `*2` params | 2-layer MLP with small hidden dim |

---

## Appendix C: Parallel Formulation

The recurrent WKV update can alternatively be written in parallel form for efficient training:

```
wkv_t = Σ_{i=1}^{t} (v_i^T k̃_i · Π_{j=i+1}^{t} G_j)
```

Where `G_j = diag(w_j) - κ̂_j^T (a_j ⊙ κ̂_j)` is the transition matrix.

This allows chunked parallel computation during training while maintaining the recurrent formulation for inference.

---

## Appendix D: Paper vs Code Notation

| Paper | Code | Notes |
|-------|------|-------|
| x_d | xw | Decay input (paper uses d for decay precursor) |
| d_t | d (intermediate) | Decay precursor, before exp transform |
| w_t | w (after softplus) | Final decay values |
| κ_t | kk (before normalize) | Removal key |
| κ̂_t | kk (after normalize) | Normalized removal key |
| k̃_t | k (after k_a transform) | Replacement key |
| ξ | k_k | Removal key multiplier |
| α | k_a | Replacement rate booster |
| ρ | r_k | WKV bonus weight |
| ν_t | (computed inline) | Value residual mix weight |
| loramlp_□ | □0, □1, □2 | Bias + two linear layers |
| μ_□ | x_□ | Token shift mix parameters |

**Key difference**: The paper presents clean mathematical notation, while code uses fused operations and different variable names for clarity/efficiency.

---

## References

1. RWKV-7 Official Implementation: https://github.com/BlinkDL/RWKV-LM
2. RWKV-LM-V7 Training Repository: https://github.com/RWKV-Vibe/RWKV-LM-V7
3. Flash Linear Attention (Triton kernels): https://github.com/fla-org/flash-linear-attention
4. RWKV-7 "Goose" Paper: https://arxiv.org/abs/2503.14456
