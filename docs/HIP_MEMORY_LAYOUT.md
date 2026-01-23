# HIP Memory Layout Specification

This document specifies memory layouts for tensors and state in the RWKV7 HIP backend.

## Design Principles

1. **rocBLAS-Native GEMM**: Store GEMM matrices in **column-major** layout so rocBLAS calls are direct (no row/col mapping at callsites). rocBLAS is column-major by default. See: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/what-is-rocblas.html#rocblas
2. **Match `web-rwkv` Tensor Convention**: This repo's `Shape` uses `shape[0]` as the **fastest-moving axis** (opposite of PyTorch). That makes `[rows, cols] = [shape[0], shape[1]]` naturally column-major for GEMM.
3. **FP16 I/O, FP32 State**: Activations/weights in FP16 (or BF16 if needed), recurrent state in FP32.
4. **Coalesced Access**: Keep per-token features contiguous (channel-first) for efficient vectorized loads/stores.
5. **Managed vs Device Memory**: Weights can be managed (`hipMallocManaged`) on APUs, but the default should remain device memory unless profiling shows managed memory is beneficial.

## Tensor Layouts

### General Convention

**Important**: `web-rwkv` tensors are **not** "torch-style row-major `[B, T, C]`". The internal convention is:
- `shape[0]` is the **fastest** axis (contiguous).
- A 2D matrix stored as `Shape(rows, cols, 1, 1)` is therefore in **column-major** memory layout with `lda = rows`.

```
Shape(x, y, z, w)  ->  linear_index(x, y, z, w) = x + y*X + z*X*Y + w*X*Y*Z
where X=shape[0], Y=shape[1], Z=shape[2]
```

### Activation Tensors

| Tensor | Shape | Dtype | Layout |
|--------|-------|-------|--------|
| Packed activations (`x`) | `[C, A, 1, 1]` | f16 | Channel-first (C contiguous) |
| Packed attention vectors (`r,w,k,v,a,kk, ...`) | `[N, H, A, 1]` | f16 | Head-split, channel-first |
| Fused attention bundle (optional, like WGSL `att_n`) | `[N, H, A, 4]` | f16 | Stores k,v,a,kk as 4 "planes" |

Where:
- `C` = embedding dimension (`n_embd`)
- `H` = number of heads (`n_head`)
- `N` = head size (`head_size`, usually 64)
- `A` = packed token count = sum over active batches of `T_b` (see `TensorStack` + packed `cursors` in this repo)

### Memory Strides (Generic)

For a tensor with `Shape(X, Y, Z, W)`, the linear element offset is:

```
offset(x, y, z, w) = x + y*X + z*X*Y + w*X*Y*Z
```

This means:
- Iterating `x` is contiguous.
- Iterating `y` has stride `X`.
- Iterating `z` has stride `X*Y`.

## WKV State Layout

For parity with the existing `web-rwkv` v7 implementation, store per-layer state as:

```
layer_state: Shape(C, N+2, B, 1)  dtype=f32
```

Semantic meaning of the second dimension (`y`):
- `y = 0`: attention token-shift state (previous `x` for time-mix token shift)
- `y = 1..N`: attention WKV time-state (flattened across heads inside the `C` dimension)
- `y = N+1`: FFN token-shift state (previous `x` for channel-mix token shift)

This matches how `src/runtime/v7.rs` slices state via `state.att(layer)` and `state.ffn(layer)`.

### State Access Patterns

**Basic addressing (for a given layer tensor)**:
```cpp
// layer_state is Shape(C, N+2, B, 1) with C contiguous.
//
// Access element at (c, y, b, 0):
//   offset = c + y*C + b*C*(N+2)
```

### Why This Layout?

1. **Unit-Stride Per-Thread Access**: Each thread loads/stores its own `N`-element row with unit stride. Across lanes this is strided by `N`, but the state load/store happens once per call/chunk and the hot loop keeps state in registers.

2. **LDS Efficiency**: The state row fits in registers (N=64 → 64 floats = 256 bytes), avoiding LDS for state storage

3. **Batch Independence**: Each (batch, head) block operates on its own state region with no cross-block dependencies

## Weight Layout

### Unified Memory Allocation

Weights use `hipMallocManaged` for zero-copy APU access:

```cpp
struct ModelWeights {
    void* data;         // Managed memory pointer
    size_t total_size;  // Total allocation

    // Weight regions (offsets into data)
    size_t embed_offset;
    size_t layers_offset;
    size_t head_offset;
};

ModelWeights allocate_weights(const ModelConfig& config) {
    size_t total = compute_total_size(config);

    void* ptr;
    hipMallocManaged(&ptr, total);
    hipMemAdvise(ptr, total, hipMemAdviseSetReadMostly, 0);

    return ModelWeights{ptr, total, ...};
}
```

### Weight Tensor Layout

For HIP + rocBLAS, store all GEMM weights as **column-major** 2D matrices:

```
W: Shape(out_features, in_features, 1, 1)  dtype=f16 (or bf16)
```

This enables direct rocBLAS calls for linear layers:

```
Y = W * X
X: Shape(in_features,  A, 1, 1)
Y: Shape(out_features, A, 1, 1)
```

Note: safetensors/PyTorch stores linear weights as `[out_features, in_features]` but `web-rwkv`'s loader reverses dimensions for its own shaders. The HIP backend should either:
- load weights into a HIP-specific tensor with `Shape(out, in, 1, 1)`, or
- transpose once at load time into that layout.

| Weight | Shape | Notes |
|--------|-------|-------|
| `embed.w` | [vocab, C] | Embedding table |
| `att.w_r` | [C, C] | Receptance projection |
| `att.w_k` | [C, C] | Key projection |
| `att.w_v` | [C, C] | Value projection |
| `att.w_o` | [C, C] | Output projection |
| `ffn.w_k` | [C_hidden, C] | FFN expand |
| `ffn.w_v` | [C, C_hidden] | FFN contract |

### LoRA Weights

LoRA matrices for decay/gate computation:

| Weight | Shape | Notes |
|--------|-------|-------|
| `att.w1` | [lora_dim, C] | Decay LoRA down |
| `att.w2` | [C, lora_dim] | Decay LoRA up |
| `att.a1` | [lora_dim, C] | Learning rate LoRA down |
| `att.a2` | [C, lora_dim] | Learning rate LoRA up |
| `att.g1` | [lora_dim, C] | Gate LoRA down |
| `att.g2` | [C, lora_dim] | Gate LoRA up |

## Kernel Memory Patterns

### WKV7 Forward Kernel

```cpp
__global__ void wkv7_forward(
    int T, int H,
    const half* __restrict__ w,   // [B, T, H, N] - decay
    const half* __restrict__ q,   // [B, T, H, N] - receptance
    const half* __restrict__ k,   // [B, T, H, N] - key
    const half* __restrict__ v,   // [B, T, H, N] - value
    const half* __restrict__ a,   // [B, T, H, N] - removal key (-kk)
    const half* __restrict__ b,   // [B, T, H, N] - removal rate (kk*a)
    half* __restrict__ y,         // [B, T, H, N] - output
    float* __restrict__ state     // [B, H, N+1, N] - state
) {
    constexpr int N = 64;  // Head size

    int bb = blockIdx.y;   // Batch
    int hh = blockIdx.x;   // Head
    int i = threadIdx.x;   // Row (0..N-1)

    // Per-thread local state
    float local_state[N] = {0};

    // Shared memory for input vectors
    __shared__ float sq[N], sw[N], sk[N], sa[N], sb[N];

    for (int t = 0; t < T; t++) {
        // Input offset for this (batch, time, head)
        int base = bb * T * H * N + t * H * N + hh * N;

        // Load inputs to shared (coalesced)
        __syncthreads();
        if (i < N) {
            sq[i] = __half2float(q[base + i]);
            sw[i] = __expf(-__expf(__half2float(w[base + i])));
            sk[i] = __half2float(k[base + i]);
            sa[i] = __half2float(a[base + i]);
            sb[i] = __half2float(b[base + i]);
        }
        __syncthreads();

        // Compute sa_dot = dot(local_state, sa)
        float sa_dot = 0;
        #pragma unroll
        for (int j = 0; j < N; j++) {
            sa_dot += sa[j] * local_state[j];
        }

        // Load v (single value per thread)
        float v_val = __half2float(v[base + i]);

        // State update + output
        float y_val = 0;
        #pragma unroll
        for (int j = 0; j < N; j++) {
            float& s = local_state[j];
            s = s * sw[j] + sa_dot * sb[j] + sk[j] * v_val;
            y_val += s * sq[j];
        }

        // Write output
        y[base + i] = __float2half(y_val);
    }

    // Store final state
    int state_base = (bb * H + hh) * (N + 1) * N + N;  // Skip token shift row
    #pragma unroll
    for (int j = 0; j < N; j++) {
        state[state_base + i * N + j] = local_state[j];
    }
}
```

### MatMul Memory Access

rocBLAS is **column-major**. With the layouts above, GEMM is direct:

```cpp
// W: Shape(out, in)
// X: Shape(in,  A)   (A = packed tokens)
// Y: Shape(out, A)
rocblas_gemm_ex(handle,
                rocblas_operation_none, rocblas_operation_none,
                /*m=*/out, /*n=*/A, /*k=*/in,
                &alpha,
                W, rocblas_datatype_f16_r, /*lda=*/out,
                X, rocblas_datatype_f16_r, /*ldb=*/in,
                &beta,
                Y, rocblas_datatype_f16_r, /*ldc=*/out,
                Y, rocblas_datatype_f16_r, /*ldd=*/out,
                rocblas_datatype_f32_r,
                rocblas_gemm_algo_standard,
                0, 0);
```

## FFN State Layout

The FFN (channel-mix) uses a simpler state:

```
ffn_state: [B, C]  dtype=f32
```

This stores the previous token's embedding for token shifting in the FFN.

### Combined State

For inference, the complete state per layer is:

```
layer_state: [B, H, N+2, N]  dtype=f32

Where:
  [:, :, 0, :]      = token_shift (attention)   [B, H, N]
  [:, :, 1:N+1, :]  = wkv_state                 [B, H, N, N]
  [:, :, N+1, :]    = ffn_token_shift           [B, H, N]
```

**Alternative** (matching web-rwkv Vulkan):
```
layer_state: [B, C, N+2]  dtype=f32

Reshaped from [B, H, N+2, N] for compatibility with Vulkan implementation.
```

## Alignment Requirements

| Data Type | Alignment | Notes |
|-----------|-----------|-------|
| f16 | 2 bytes | Minimum |
| f32 | 4 bytes | Minimum |
| float4 | 16 bytes | For vectorized loads |
| LDS | 128 bytes | For bank conflict avoidance |

### Ensuring Alignment

```cpp
// Allocate aligned memory
void* alloc_aligned(size_t size, size_t alignment) {
    void* ptr;
    hipMalloc(&ptr, size);  // hipMalloc guarantees 256-byte alignment
    return ptr;
}

// For managed memory
void* alloc_managed_aligned(size_t size) {
    void* ptr;
    hipMallocManaged(&ptr, size);  // Also 256-byte aligned
    return ptr;
}
```

## Rust Tensor Abstraction

```rust
#[repr(C)]
pub struct TensorHip<T: Copy> {
    ptr: *mut T,
    shape: Vec<usize>,
    strides: Vec<usize>,
    managed: bool,  // true = unified memory
}

impl<T: Copy> TensorHip<T> {
    pub fn zeros(ctx: &HipContext, shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let ptr = ctx.alloc(size)?;
        ctx.memset(ptr, 0, size)?;

        let strides = compute_strides(shape);
        Ok(Self { ptr: ptr as *mut T, shape: shape.to_vec(), strides, managed: false })
    }

    pub fn managed(ctx: &HipContext, shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let ptr = ctx.alloc_managed(size)?;

        let strides = compute_strides(shape);
        Ok(Self { ptr: ptr as *mut T, shape: shape.to_vec(), strides, managed: true })
    }

    #[inline]
    pub fn offset(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
```
