// FLA (Flash Linear Attention) kernels for RWKV7 HIP backend.
//
// Stage 1: kernel_fla_cumsum — cumulative decay scan within each chunk.
// Stage 2: kernel_fla_intra_gate + kernel_fla_intra_attn — intra-chunk attention.
//
// All per-token tensors use web-rwkv column-major layout [K, H, T, B] with K fastest.
// For packed/varlen sequences, B=1 and T is total packed length.
// Position t in sequence starting at bos, head h, element k:
//   index = k + h * K + (bos + t) * H * K
//
// Attention matrices use layout [C, C, H, total_chunks]:
//   A[i, j, h, chunk] = i + j*C + h*C*C + chunk*H*C*C
//
// Reference: repos/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

// ============================================================================
// Stage 2, Part A: Decay-scaled gating (kernel_fla_intra_gate)
//
// Computes gated versions of q, k, a, b using cumulative decays gi, ge:
//   g_last = gi[last_valid_position_in_chunk]  (per K element)
//   qg[t] = q[t] * exp(gi[t])
//   kg[t] = k[t] * exp(-gi[t] + g_last)
//   ag[t] = a[t] * exp(ge[t])
//   bg[t] = b[t] * exp(-gi[t] + g_last)
//
// Inputs q, k, a, b are f16 [K, H, T, B].
// Inputs gi, ge are f32 [K, H, T, B] (from Stage 1).
// Outputs qg, kg, ag, bg are f32 [K, H, T, B].
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(K) where K=64 (one thread per head-size element)
// ============================================================================

__global__ __launch_bounds__(64, 8)
void kernel_fla_intra_gate(
    const __half* __restrict__ q,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ k,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ a,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ b,              // [K, H, T_packed, 1] f16
    const float*  __restrict__ gi,             // [K, H, T_packed, 1] inclusive cumsum
    const float*  __restrict__ ge,             // [K, H, T_packed, 1] exclusive cumsum
    float* __restrict__ qg_out,                // [K, H, T_packed, 1] decay-scaled query
    float* __restrict__ kg_out,                // [K, H, T_packed, 1] decay-scaled key
    float* __restrict__ ag_out,                // [K, H, T_packed, 1] decay-scaled a
    float* __restrict__ bg_out,                // [K, H, T_packed, 1] decay-scaled b
    const int* __restrict__ chunk_indices,     // [total_chunks * 2]
    const int* __restrict__ cu_seqlens,        // [N+1]
    int K_dim,                                 // head_size (64)
    int H,                                     // n_heads
    int C                                      // chunk_size (16)
) {
    const int chunk_flat = blockIdx.x;
    const int h = blockIdx.y;
    const int kk = threadIdx.x;  // head-size element index

    if (h >= H || kk >= K_dim) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = cu_seqlens[seq_id];
    const int eos = cu_seqlens[seq_id + 1];
    const int T   = eos - bos;

    // Early exit for chunks entirely beyond sequence
    if (local_chunk * C >= T) return;

    // Stride for advancing one token in [K, H, T, B] layout
    const int token_stride = H * K_dim;

    // Base offset for head h, element kk, at sequence start
    const int base = kk + h * K_dim + bos * token_stride;

    // Compute g_last: inclusive cumsum at last valid position in this chunk
    const int last_t_abs = min((local_chunk + 1) * C, T) - 1;
    const float g_last = gi[base + last_t_abs * token_stride];

    // Process each position in the chunk
    #pragma unroll
    for (int t = 0; t < C; t++) {
        const int t_abs = local_chunk * C + t;

        if (t_abs < T) {
            const int idx = base + t_abs * token_stride;

            // Load f16 inputs and convert to f32
            const float q_val = __half2float(q[idx]);
            const float k_val = __half2float(k[idx]);
            const float a_val = __half2float(a[idx]);
            const float b_val = __half2float(b[idx]);

            // Load f32 cumulative decays
            const float gi_val = gi[idx];
            const float ge_val = ge[idx];

            // Compute decay-scaled outputs
            const float exp_gi     = expf(gi_val);
            const float exp_neg_gi = expf(-gi_val + g_last);
            const float exp_ge     = expf(ge_val);

            qg_out[idx] = q_val * exp_gi;
            kg_out[idx] = k_val * exp_neg_gi;
            ag_out[idx] = a_val * exp_ge;
            bg_out[idx] = b_val * exp_neg_gi;
        }
    }
}

// ============================================================================
// Stage 2, Part B: Intra-chunk attention matrices (kernel_fla_intra_attn)
//
// Computes 4 CxC attention matrices per chunk via dot products of gated vectors:
//   A_qk[i,j] = sum_k(qg[i,k] * kg[j,k])  for j <= i  (lower triangular)
//   A_qb[i,j] = sum_k(qg[i,k] * bg[j,k])  for j <= i
//   A_ak[i,j] = sum_k(ag[i,k] * kg[j,k])  for j < i   (strict lower triangular)
//   A_ab[i,j] = sum_k(ag[i,k] * bg[j,k])  for j < i
//
// Inputs qg, kg, ag, bg are f32 [K, H, T, B] (from Part A).
// Outputs are f32 [C, C, H, total_chunks] (column-major: i fastest).
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(C) where C=16 (one thread per row i of the attention matrices)
// ============================================================================

__global__ __launch_bounds__(16, 16)
void kernel_fla_intra_attn(
    const float* __restrict__ qg,              // [K, H, T_packed, 1]
    const float* __restrict__ kg,              // [K, H, T_packed, 1]
    const float* __restrict__ ag,              // [K, H, T_packed, 1]
    const float* __restrict__ bg,              // [K, H, T_packed, 1]
    float* __restrict__ A_qk,                  // [C, C, H, total_chunks]
    float* __restrict__ A_qb,                  // [C, C, H, total_chunks]
    float* __restrict__ A_ak,                  // [C, C, H, total_chunks]
    float* __restrict__ A_ab,                  // [C, C, H, total_chunks]
    const int* __restrict__ chunk_indices,     // [total_chunks * 2]
    const int* __restrict__ cu_seqlens,        // [N+1]
    int K_dim,                                 // head_size (64)
    int H,                                     // n_heads
    int C                                      // chunk_size (16)
) {
    const int chunk_flat = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;  // row index within the CxC attention matrix

    if (h >= H || i >= C) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = cu_seqlens[seq_id];
    const int eos = cu_seqlens[seq_id + 1];
    const int T   = eos - bos;

    // Absolute position of row i
    const int i_abs = local_chunk * C + i;

    // Token stride and base offset
    const int token_stride = H * K_dim;
    const int base_h = h * K_dim + bos * token_stride;

    // Output base: [C, C, H, total_chunks], element (i, j, h, chunk)
    //   = i + j*C + h*C*C + chunk*H*C*C
    const int out_base = i + h * C * C + chunk_flat * H * C * C;

    // Number of valid positions in this chunk
    const int chunk_start = local_chunk * C;
    const int valid_len = min(C, T - chunk_start);

    // If row i is beyond sequence length, zero out the entire row
    if (i_abs >= T) {
        for (int j = 0; j < C; j++) {
            const int out_idx = out_base + j * C;
            A_qk[out_idx] = 0.0f;
            A_qb[out_idx] = 0.0f;
            A_ak[out_idx] = 0.0f;
            A_ab[out_idx] = 0.0f;
        }
        return;
    }

    // Preload row i vectors: qg[i,:] and ag[i,:] across all K elements
    // i_abs is valid, so we can safely load
    const int row_i_base = base_h + i_abs * token_stride;

    // Loop over columns j
    for (int j = 0; j < C; j++) {
        const int out_idx = out_base + j * C;
        const int j_abs = chunk_start + j;

        // Boundary check: j beyond valid positions
        if (j >= valid_len) {
            A_qk[out_idx] = 0.0f;
            A_qb[out_idx] = 0.0f;
            A_ak[out_idx] = 0.0f;
            A_ab[out_idx] = 0.0f;
            continue;
        }

        // Causal masks
        const bool inclusive_mask = (i >= j);  // j <= i for A_qk, A_qb
        const bool strict_mask   = (i > j);    // j < i  for A_ak, A_ab

        // If both masks fail, all four are zero
        if (!inclusive_mask) {
            A_qk[out_idx] = 0.0f;
            A_qb[out_idx] = 0.0f;
            A_ak[out_idx] = 0.0f;
            A_ab[out_idx] = 0.0f;
            continue;
        }

        // Compute dot products across K dimension
        const int row_j_base = base_h + j_abs * token_stride;

        float dot_qk = 0.0f;
        float dot_qb = 0.0f;
        float dot_ak = 0.0f;
        float dot_ab = 0.0f;

        #pragma unroll 16
        for (int kk = 0; kk < K_dim; kk++) {
            const float qg_ik = qg[row_i_base + kk];
            const float ag_ik = ag[row_i_base + kk];
            const float kg_jk = kg[row_j_base + kk];
            const float bg_jk = bg[row_j_base + kk];

            dot_qk += qg_ik * kg_jk;
            dot_qb += qg_ik * bg_jk;
            dot_ak += ag_ik * kg_jk;
            dot_ab += ag_ik * bg_jk;
        }

        // Store with causal masking applied
        A_qk[out_idx] = dot_qk;  // inclusive_mask is true here
        A_qb[out_idx] = dot_qb;

        if (strict_mask) {
            A_ak[out_idx] = dot_ak;
            A_ab[out_idx] = dot_ab;
        } else {
            // j == i: strict lower triangular means zero
            A_ak[out_idx] = 0.0f;
            A_ab[out_idx] = 0.0f;
        }
    }
}

// ============================================================================
// Host launch wrapper for Stage 2 (both parts)
// ============================================================================

hipError_t launch_fla_intra(
    const __half* q,
    const __half* k,
    const __half* a,
    const __half* b,
    const float* gi,
    const float* ge,
    float* qg,
    float* kg,
    float* ag,
    float* bg,
    float* A_qk,
    float* A_qb,
    float* A_ak,
    float* A_ab,
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

    // Part A: Compute decay-scaled gated vectors
    {
        dim3 grid(total_chunks, H, 1);
        dim3 block(K);

        hipLaunchKernelGGL(kernel_fla_intra_gate, grid, block, 0, stream,
                           q, k, a, b, gi, ge, qg, kg, ag, bg,
                           chunk_indices, cu_seqlens,
                           K, H, C);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) return err;
    }

    // Part B: Compute CxC attention matrices
    {
        dim3 grid(total_chunks, H, 1);
        dim3 block(C);

        hipLaunchKernelGGL(kernel_fla_intra_attn, grid, block, 0, stream,
                           qg, kg, ag, bg,
                           A_qk, A_qb, A_ak, A_ab,
                           chunk_indices, cu_seqlens,
                           K, H, C);

        return hipGetLastError();
    }
}

} // extern "C"
