// FLA (Flash Linear Attention) kernels for RWKV7 HIP backend.
//
// Stage 1: kernel_fla_cumsum — cumulative decay scan within each chunk.
//
// All tensors use web-rwkv column-major layout [K, H, T, B] with K fastest.
// For packed/varlen sequences, B=1 and T is total packed length.
// Position t in sequence starting at bos, head h, element k:
//   index = k + h * K + (bos + t) * H * K
//
// Reference: repos/flash-linear-attention/fla/ops/rwkv6/chunk.py
//            chunk_rwkv6_fwd_cumsum_kernel (lines 41-75)
#include <hip/hip_runtime.h>

extern "C" {

// ============================================================================
// Stage 1: Cumulative decay scan (kernel_fla_cumsum)
//
// Computes intra-chunk inclusive and exclusive cumulative sums of log-decay gk.
//
// For each chunk:
//   gi[t] = sum(gk[chunk_start..=t])   (inclusive: add current before store)
//   ge[t] = sum(gk[chunk_start..t])    (exclusive: store before adding current)
//
// The Triton reference computes this as matrix-vector products:
//   oi = lower_triangular_inclusive @ s
//   oe = lower_triangular_exclusive @ s
//
// For C=16 with K=64 threads, each thread handles one element of the head
// dimension and loops sequentially over the C=16 positions in the chunk.
// This is equivalent but simpler for small C.
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(K) where K=64 (one thread per head-size element)
// ============================================================================

__global__ __launch_bounds__(64, 8)
void kernel_fla_cumsum(
    const float* __restrict__ gk,           // [K, H, T_packed, 1] input log-decay
    float* __restrict__ gi,                 // [K, H, T_packed, 1] inclusive cumsum output
    float* __restrict__ ge,                 // [K, H, T_packed, 1] exclusive cumsum output
    const int* __restrict__ chunk_indices,  // [total_chunks * 2] flat: (seq_id, local_chunk_id)
    const int* __restrict__ cu_seqlens,     // [N+1] cumulative sequence lengths
    int K,                                  // head_size (64)
    int H,                                  // n_heads
    int C                                   // chunk_size (16)
) {
    const int chunk_flat = blockIdx.x;  // which chunk (0..total_chunks-1)
    const int h = blockIdx.y;           // which head (0..H-1)
    const int k = threadIdx.x;          // which head-size element (0..K-1)

    if (h >= H || k >= K) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = cu_seqlens[seq_id];
    const int eos = cu_seqlens[seq_id + 1];
    const int T   = eos - bos;  // sequence length

    // Stride for advancing one token in column-major [K, H, T, B] layout
    const int token_stride = H * K;

    // Base offset for head h, element k, at sequence start
    const int base = k + h * K + bos * token_stride;

    // Sequential prefix sum over C positions in this chunk.
    // C=16 is tiny so a simple loop is optimal (no shared memory needed).
    float accum = 0.0f;

    #pragma unroll
    for (int t = 0; t < C; t++) {
        const int t_abs = local_chunk * C + t;  // position within sequence

        if (t_abs < T) {
            const int idx = base + t_abs * token_stride;
            const float val = gk[idx];

            // Exclusive: store BEFORE adding current value
            ge[idx] = accum;

            accum += val;

            // Inclusive: store AFTER adding current value
            gi[idx] = accum;
        }
    }
}

// ============================================================================
// Host launch wrapper for kernel_fla_cumsum
// ============================================================================

hipError_t launch_fla_cumsum(
    const float* gk,
    float* gi,
    float* ge,
    const int* chunk_indices,
    const int* cu_seqlens,
    int K,
    int H,
    int C,
    int total_chunks,
    hipStream_t stream
) {
    if (K <= 0 || H <= 0 || C <= 0 || total_chunks <= 0) {
        return hipSuccess;
    }

    dim3 grid(total_chunks, H, 1);
    dim3 block(K);

    hipLaunchKernelGGL(kernel_fla_cumsum, grid, block, 0, stream,
                       gk, gi, ge, chunk_indices, cu_seqlens,
                       K, H, C);

    return hipGetLastError();
}

} // extern "C"
