use half::f16;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

/// HIP error codes
pub type HipError = c_int;

/// HIP stream handle (opaque pointer)
pub type HipStream = *mut c_void;

/// HIP event handle (opaque pointer)
pub type HipEvent = *mut c_void;

/// rocBLAS handle type (opaque pointer)
pub type RocblasHandle = *mut c_void;

/// rocBLAS status code
pub type RocblasStatus = c_int;

/// rocBLAS success status
pub const ROCBLAS_STATUS_SUCCESS: RocblasStatus = 0;

/// hipBLASLt handle type (opaque pointer)
pub type HipblasLtHandle = *mut c_void;

/// hipBLAS status code (same enum as rocBLAS status for API consistency)
pub type HipblasStatus = c_int;

/// hipBLAS success status
pub const HIPBLAS_STATUS_SUCCESS: HipblasStatus = 0;
/// hipBLAS not supported status
pub const HIPBLAS_STATUS_NOT_SUPPORTED: HipblasStatus = 3;
/// hipBLAS allocation failed status
pub const HIPBLAS_STATUS_ALLOC_FAILED: HipblasStatus = 4;

/// HIP device properties structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProp {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multi_processor_count: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub compute_mode: c_int,
    pub clock_instruction_rate: c_int,
    pub arch: HipDeviceArch,
    pub concurrent_kernels: c_int,
    pub pci_domain_id: c_int,
    pub pci_bus_id: c_int,
    pub pci_device_id: c_int,
    pub max_shared_memory_per_multi_processor: usize,
    pub is_multi_gpu_board: c_int,
    pub can_map_host_memory: c_int,
    pub gcn_arch: c_int,
    pub gcn_arch_name: [c_char; 256],
    pub integrated: c_int,
    pub cooperative_launch: c_int,
    pub cooperative_multi_device_launch: c_int,
    pub max_texture_1d_linear: c_int,
    pub max_texture_1d: c_int,
    pub max_texture_2d: [c_int; 2],
    pub max_texture_3d: [c_int; 3],
    pub mem_pitch: usize,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: c_int,
    pub kernel_exec_timeout_enabled: c_int,
    pub ecc_enabled: c_int,
    pub tcc_driver: c_int,
    pub cooperative_multi_device_unmatched_func: c_int,
    pub cooperative_multi_device_unmatched_grid_dim: c_int,
    pub cooperative_multi_device_unmatched_block_dim: c_int,
    pub cooperative_multi_device_unmatched_shared_mem: c_int,
    pub is_large_bar: c_int,
    pub asic_revision: c_int,
    pub managed_memory: c_int,
    pub direct_managed_mem_access_from_host: c_int,
    pub concurrent_managed_access: c_int,
    pub pageable_memory_access: c_int,
    pub pageable_memory_access_uses_host_page_tables: c_int,
    // Padding for forward compatibility
    _reserved: [c_char; 64],
}

/// HIP device architecture
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HipDeviceArch {
    _unused: [c_int; 28],
}

impl Default for HipDeviceProp {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// FFI declarations for our compiled HIP kernels
extern "C" {
    pub fn hip_get_device_count(count: *mut c_int) -> HipError;
    pub fn hip_get_device_properties(props: *mut HipDeviceProp, device_id: c_int) -> HipError;
    pub fn hip_set_device(device_id: c_int) -> HipError;
    pub fn hip_get_device(device_id: *mut c_int) -> HipError;
    pub fn hip_malloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    pub fn hip_malloc_managed(ptr: *mut *mut c_void, size: usize) -> HipError;
    pub fn hip_free(ptr: *mut c_void) -> HipError;
    pub fn hip_memset(ptr: *mut c_void, value: c_int, size: usize) -> HipError;
    pub fn hip_memcpy_h2d(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: HipStream,
    ) -> HipError;
    pub fn hip_memcpy_d2h(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: HipStream,
    ) -> HipError;
    pub fn hip_stream_create(stream: *mut HipStream) -> HipError;
    pub fn hip_stream_destroy(stream: HipStream) -> HipError;
    pub fn hip_stream_synchronize(stream: HipStream) -> HipError;
    pub fn hip_stream_wait_event(stream: HipStream, event: HipEvent, flags: c_uint) -> HipError;
    pub fn hip_device_synchronize() -> HipError;

    // Pinned (page-locked) host memory
    pub fn hip_host_malloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipError;
    pub fn hip_host_free(ptr: *mut c_void) -> HipError;

    // Events for async synchronization
    pub fn hip_event_create(event: *mut HipEvent) -> HipError;
    pub fn hip_event_destroy(event: HipEvent) -> HipError;
    pub fn hip_event_record(event: HipEvent, stream: HipStream) -> HipError;
    pub fn hip_event_synchronize(event: HipEvent) -> HipError;
    pub fn hip_event_query(event: HipEvent) -> HipError;
    pub fn hip_get_error_string(error: HipError) -> *const c_char;
    pub fn launch_copy_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_copy_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_copy_f16_to_f32(
        input: *const f16,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_decay_exp_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_decay_exp_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_lerp_f32(
        a: *const f32,
        b: *const f32,
        t: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_lerp_f16(
        a: *const f16,
        b: *const f16,
        t: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_sigmoid_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_sigmoid_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_squared_relu_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_squared_relu_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_softplus_decay_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_softplus_decay_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_layer_norm_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_layer_norm_f16(
        input: *const f16,
        weight: *const f16,
        bias: *const f16,
        output: *mut f16,
        c: c_int,
        n: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_group_norm_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        g: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_group_norm_f16(
        input: *const f16,
        weight: *const f16,
        bias: *const f16,
        output: *mut f16,
        c: c_int,
        n: c_int,
        g: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_l2_norm_f32(
        input: *const f32,
        output: *mut f32,
        c: c_int,
        n: c_int,
        head_size: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_l2_norm_f16(
        input: *const f16,
        output: *mut f16,
        c: c_int,
        n: c_int,
        head_size: c_int,
        eps: f32,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_tanh_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_tanh_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_token_shift_f32(
        x: *const f32,
        state_in: *const f32,
        mix: *const f32,
        output: *mut f32,
        state_out: *mut f32,
        c: c_int,
        t: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_channel_mix_state_f32(
        x: *const f32,
        state_in: *const f32,
        x_k: *const f32,
        output: *mut f32,
        state_out: *mut f32,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_channel_mix_state_f16(
        x: *const f16,
        state_in: *const f16,
        x_k: *const f16,
        output: *mut f16,
        state_out: *mut f16,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_channel_mix_state_f32_masked(
        x: *const f32,
        state_in: *const f32,
        x_k: *const f32,
        output: *mut f32,
        state_out: *mut f32,
        lengths: *const c_int,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_channel_mix_state_f16_masked(
        x: *const f16,
        state_in: *const f16,
        x_k: *const f16,
        output: *mut f16,
        state_out: *mut f16,
        lengths: *const c_int,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_wkv_bonus_f32(
        r: *const f32,
        k: *const f32,
        v: *const f32,
        r_k: *const f32,
        output: *mut f32,
        n: c_int, // head_size
        h: c_int, // n_heads
        t: c_int, // tokens
        b: c_int, // batch
        stream: HipStream,
    ) -> HipError;
    pub fn launch_wkv_bonus_f16(
        r: *const f16,
        k: *const f16,
        v: *const f16,
        r_k: *const f16,
        output: *mut f16,
        n: c_int,
        h: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_control_k_f32(
        k_a: *const f32,  // per-channel weight [C, 1, 1, 1]
        a: *const f32,    // attention [C, T, B, 1]
        k: *const f32,    // key [C, T, B, 1]
        output: *mut f32, // output [C, T, B, 1]
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_control_k_f16(
        k_a: *const f16,
        a: *const f16,
        k: *const f16,
        output: *mut f16,
        c: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    /// Wave-cooperative WKV7 with explicit shuffle reductions
    /// Uses 256 threads (8 waves), no atomics
    pub fn launch_wkv7_wave_reduce(
        w_decay: *const f16,
        q: *const f16,
        k: *const f16,
        v: *const f16,
        a: *const f16,
        b: *const f16,
        state_in: *const f32,
        output: *mut f16,
        state_out: *mut f32,
        lengths: *const c_int,
        n: c_int,
        h: c_int,
        t: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_wkv7_fused_t1(
        w_decay: *const f16,
        q: *const f16,
        k: *const f16,
        v: *const f16,
        a: *const f16,
        b: *const f16,
        state: *mut f32,     // in-place
        output: *mut f16,
        lengths: *const c_int,
        n: c_int,
        h: c_int,
        b: c_int,
        stream: HipStream,
    ) -> HipError;

    // Elementwise operations for GPU-native forward pass
    pub fn launch_add_f32(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_add_f16(
        a: *const f16,
        b: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_mul_f32(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_mul_f16(
        a: *const f16,
        b: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_negate_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_negate_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_exp_f32(
        input: *const f32,
        output: *mut f32,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_exp_f16(
        input: *const f16,
        output: *mut f16,
        n: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_broadcast_add_f32(
        input: *const f32,
        bias: *const f32,
        output: *mut f32,
        n: c_int,
        bias_len: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_broadcast_add_f16(
        input: *const f16,
        bias: *const f16,
        output: *mut f16,
        n: c_int,
        bias_len: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_broadcast_mul_f32(
        input: *const f32,
        scale: *const f32,
        output: *mut f32,
        n: c_int,
        scale_len: c_int,
        stream: HipStream,
    ) -> HipError;
    pub fn launch_broadcast_mul_f16(
        input: *const f16,
        scale: *const f16,
        output: *mut f16,
        n: c_int,
        scale_len: c_int,
        stream: HipStream,
    ) -> HipError;

    // Safe property accessors (avoid struct layout issues)
    pub fn hip_get_device_name(device_id: c_int, name: *mut c_char, max_len: c_int) -> HipError;
    pub fn hip_get_device_gcn_arch_name(
        device_id: c_int,
        name: *mut c_char,
        max_len: c_int,
    ) -> HipError;
    pub fn hip_get_device_total_memory(device_id: c_int, total_mem: *mut usize) -> HipError;
    pub fn hip_get_device_mp_count(device_id: c_int, count: *mut c_int) -> HipError;
    pub fn hip_get_device_warp_size(device_id: c_int, warp_size: *mut c_int) -> HipError;
    pub fn hip_get_device_compute_capability(
        device_id: c_int,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> HipError;
    pub fn hip_is_device_integrated(device_id: c_int, integrated: *mut c_int) -> HipError;
    pub fn hip_supports_cooperative_launch(device_id: c_int, supported: *mut c_int) -> HipError;

    // rocBLAS functions
    pub fn rocblas_handle_create(handle: *mut RocblasHandle) -> RocblasStatus;
    pub fn rocblas_handle_destroy(handle: RocblasHandle) -> RocblasStatus;
    pub fn rocblas_set_stream_wrapper(handle: RocblasHandle, stream: HipStream) -> RocblasStatus;
    pub fn launch_hgemm(
        handle: RocblasHandle,
        m: c_int,      // rows of A and C
        n: c_int,      // cols of B and C
        k: c_int,      // cols of A, rows of B
        a: *const u16, // M×K matrix (FP16 as u16)
        b: *const u16, // K×N matrix (FP16 as u16)
        c: *mut u16,   // M×N matrix (FP16 as u16)
    ) -> RocblasStatus;
    pub fn launch_sgemm(
        handle: RocblasHandle,
        m: c_int, // rows of A and C
        n: c_int, // cols of B and C
        k: c_int, // cols of A, rows of B
        alpha: f32,
        a: *const f32, // M×K matrix
        b: *const f32, // K×N matrix
        beta: f32,
        c: *mut f32, // M×N matrix
    ) -> RocblasStatus;
    pub fn launch_sgemm_ta(
        handle: RocblasHandle,
        m: c_int, // rows of output C (output features)
        n: c_int, // cols of B and C (tokens)
        k: c_int, // input features
        alpha: f32,
        a: *const f32, // stored as K×M (row-major [M, K])
        b: *const f32, // K×N matrix
        beta: f32,
        c: *mut f32, // M×N matrix
    ) -> RocblasStatus;
    pub fn launch_hgemm_f32_out(
        handle: RocblasHandle,
        m: c_int,      // rows of A and C (vocab size / output features)
        n: c_int,      // cols of B and C (tokens)
        k: c_int,      // cols of A, rows of B (embedding dim)
        a: *const u16, // M×K matrix (FP16 weights as u16)
        b: *const u16, // K×N matrix (FP16 input as u16)
        c: *mut f32,   // M×N matrix (FP32 output)
    ) -> RocblasStatus;
    pub fn rocblas_to_hip_error(status: RocblasStatus) -> HipError;

    // rocBLAS batched GEMV for WKV7
    pub fn launch_sgemv_strided_batched(
        handle: RocblasHandle,
        m: c_int,           // rows of each A matrix
        n: c_int,           // cols of each A matrix
        a: *const f32,      // batched M×N matrices
        stride_a: i64,      // stride between matrices
        x: *const f32,      // batched length-N vectors
        stride_x: i64,      // stride between x vectors
        y: *mut f32,        // batched length-M output vectors
        stride_y: i64,      // stride between y vectors
        batch_count: c_int, // number of batches
    ) -> RocblasStatus;

    // WKV7 state update kernel
    pub fn launch_wkv7_state_update_flat(
        state: *mut f32,    // [N, N, batch_count] state matrices
        w: *const f32,      // [N, batch_count] decay values
        sa: *const f32,     // [N, batch_count] state @ a result
        b: *const f32,      // [N, batch_count] b vectors
        v: *const f32,      // [N, batch_count] v vectors
        k: *const f32,      // [N, batch_count] k vectors
        n: c_int,           // head size (64)
        batch_count: c_int, // total batches
        stream: HipStream,
    ) -> HipError;

    // Full WKV7 GEMV implementation
    pub fn launch_wkv7_gemv(
        handle: RocblasHandle,
        w_decay: *const f32,  // [N, H, T, B]
        q: *const f32,        // [N, H, T, B]
        k: *const f32,        // [N, H, T, B]
        v: *const f32,        // [N, H, T, B]
        a: *const f32,        // [N, H, T, B]
        b: *const f32,        // [N, H, T, B]
        state_in: *const f32, // [N, N, H, B]
        output: *mut f32,     // [N, H, T, B]
        state_out: *mut f32,  // [N, N, H, B]
        sa_tmp: *mut f32,     // [N, H, B] scratch buffer
        n: c_int,             // head_size (64)
        h: c_int,             // n_heads
        t: c_int,             // tokens
        b_param: c_int,       // batch
        stream: HipStream,
    ) -> HipError;

    // hipBLASLt functions
    pub fn hipblaslt_handle_create(handle: *mut HipblasLtHandle) -> HipblasStatus;
    pub fn hipblaslt_handle_destroy(handle: HipblasLtHandle) -> HipblasStatus;
    pub fn launch_hipblaslt_hgemm_f32_out(
        handle: HipblasLtHandle,
        m: c_int,               // rows of A and C (vocab size / output features)
        n: c_int,               // cols of B and C (tokens)
        k: c_int,               // cols of A, rows of B (embedding dim)
        a: *const u16,          // M×K matrix (FP16 weights as u16)
        b: *const u16,          // K×N matrix (FP16 input as u16)
        c: *mut f32,            // M×N matrix (FP32 output)
        workspace: *mut c_void, // workspace buffer (can be null)
        workspace_size: usize,  // workspace size in bytes (0 for auto)
        stream: HipStream,      // HIP stream
    ) -> HipblasStatus;
    pub fn launch_hipblaslt_hgemm(
        handle: HipblasLtHandle,
        m: c_int,      // rows of A and C
        n: c_int,      // cols of B and C
        k: c_int,      // cols of A, rows of B
        a: *const u16, // M×K matrix (FP16 as u16)
        b: *const u16, // K×N matrix (FP16 as u16)
        c: *mut u16,   // M×N matrix (FP16 as u16)
        workspace: *mut c_void,
        workspace_size: usize,
        stream: HipStream,
    ) -> HipblasStatus;
    pub fn hipblaslt_to_hip_error(status: HipblasStatus) -> HipError;

    // FLA (Flash Linear Attention) kernels
    /// Stage 1: Cumulative decay scan within each chunk.
    /// Computes inclusive (gi) and exclusive (ge) cumulative sums of log-decay gk.
    pub fn launch_fla_cumsum(
        gk: *const f32,
        gi: *mut f32,
        ge: *mut f32,
        chunk_indices: *const c_int,
        cu_seqlens: *const c_int,
        k: c_int,
        h: c_int,
        c: c_int,
        total_chunks: c_int,
        stream: HipStream,
    ) -> HipError;

    /// Stage 2: Intra-chunk attention matrices.
    /// Part A computes decay-scaled vectors qg, kg, ag, bg.
    /// Part B computes 4 CxC attention matrices A_qk, A_qb, A_ak, A_ab.
    pub fn launch_fla_intra(
        q: *const f16,
        k: *const f16,
        a: *const f16,
        b: *const f16,
        gi: *const f32,
        ge: *const f32,
        qg: *mut f32,
        kg: *mut f32,
        ag: *mut f32,
        bg: *mut f32,
        a_qk: *mut f32,
        a_qb: *mut f32,
        a_ak: *mut f32,
        a_ab: *mut f32,
        chunk_indices: *const c_int,
        cu_seqlens: *const c_int,
        k_dim: c_int,
        h: c_int,
        c: c_int,
        total_chunks: c_int,
        stream: HipStream,
    ) -> HipError;

    /// Stage 3: WY representation.
    /// Part A inverts the lower-triangular A_ab matrix per chunk.
    /// Part B computes w = A_ab_inv @ ag and u = (A_ab_inv @ A_ak) @ v.
    pub fn launch_fla_wy_repr(
        a_ab: *const f32,
        a_ak: *const f32,
        a_ab_inv: *mut f32,
        ag: *const f32,
        v: *const f16,
        w_wy: *mut f32,
        u_wy: *mut f32,
        chunk_indices: *const c_int,
        cu_seqlens: *const c_int,
        k: c_int,
        h: c_int,
        c: c_int,
        total_chunks: c_int,
        stream: HipStream,
    ) -> HipError;
}

/// HIP success error code
pub const HIP_SUCCESS: HipError = 0;

/// hipHostMalloc flags
pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0;
pub const HIP_HOST_MALLOC_PORTABLE: c_uint = 1;
pub const HIP_HOST_MALLOC_MAPPED: c_uint = 2;
pub const HIP_HOST_MALLOC_WRITE_COMBINED: c_uint = 4;

/// hipEventCreate flags (default is blocking sync)
pub const HIP_EVENT_DEFAULT: c_uint = 0;

/// hipStreamWaitEvent flags
pub const HIP_STREAM_WAIT_VALUE_EQ: c_uint = 0;

/// hipErrorNotReady - returned by hipEventQuery when event is not complete
pub const HIP_ERROR_NOT_READY: HipError = 600;

/// Convert a HIP error to a human-readable string
pub fn error_string(error: HipError) -> String {
    if error == HIP_SUCCESS {
        return "hipSuccess".to_string();
    }
    unsafe {
        let ptr = hip_get_error_string(error);
        if ptr.is_null() {
            return format!("Unknown HIP error: {}", error);
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Result type for HIP operations
pub type Result<T> = std::result::Result<T, HipErrorKind>;

/// HIP error with context
#[derive(Debug, Clone)]
pub struct HipErrorKind {
    pub code: HipError,
    pub message: String,
}

impl std::fmt::Display for HipErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HIP error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for HipErrorKind {}

/// Check a HIP error code and convert to Result
pub(crate) fn check(error: HipError) -> Result<()> {
    if error == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipErrorKind {
            code: error,
            message: error_string(error),
        })
    }
}

/// Get the number of HIP devices available
pub fn get_device_count() -> Result<i32> {
    let mut count: c_int = 0;
    unsafe { check(hip_get_device_count(&mut count))? };
    Ok(count)
}

/// Get properties for a HIP device
pub fn get_device_properties(device_id: i32) -> Result<HipDeviceProp> {
    let mut props = HipDeviceProp::default();
    unsafe { check(hip_get_device_properties(&mut props, device_id))? };
    Ok(props)
}

/// Get the device name as a String (using safe FFI)
pub fn get_device_name(device_id: i32) -> Result<String> {
    let mut name_buf = [0i8; 256];
    unsafe {
        check(hip_get_device_name(device_id, name_buf.as_mut_ptr(), 256))?;
        let name = CStr::from_ptr(name_buf.as_ptr());
        Ok(name.to_string_lossy().into_owned())
    }
}

/// Get the GCN architecture name as a String (using safe FFI)
pub fn get_gcn_arch_name(device_id: i32) -> Result<String> {
    let mut name_buf = [0i8; 256];
    unsafe {
        check(hip_get_device_gcn_arch_name(
            device_id,
            name_buf.as_mut_ptr(),
            256,
        ))?;
        let name = CStr::from_ptr(name_buf.as_ptr());
        Ok(name.to_string_lossy().into_owned())
    }
}

/// Set the active HIP device
pub fn set_device(device_id: i32) -> Result<()> {
    unsafe { check(hip_set_device(device_id)) }
}

/// Get the current active HIP device
pub fn get_device() -> Result<i32> {
    let mut device_id: c_int = 0;
    unsafe { check(hip_get_device(&mut device_id))? };
    Ok(device_id)
}

/// Get total device memory in bytes
pub fn get_device_total_memory(device_id: i32) -> Result<usize> {
    let mut total_mem: usize = 0;
    unsafe { check(hip_get_device_total_memory(device_id, &mut total_mem))? };
    Ok(total_mem)
}

/// Get device multiprocessor (compute unit) count
pub fn get_device_mp_count(device_id: i32) -> Result<i32> {
    let mut count: c_int = 0;
    unsafe { check(hip_get_device_mp_count(device_id, &mut count))? };
    Ok(count)
}

/// Get device warp (wavefront) size
pub fn get_device_warp_size(device_id: i32) -> Result<i32> {
    let mut warp_size: c_int = 0;
    unsafe { check(hip_get_device_warp_size(device_id, &mut warp_size))? };
    Ok(warp_size)
}

/// Get device compute capability
pub fn get_device_compute_capability(device_id: i32) -> Result<(i32, i32)> {
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    unsafe {
        check(hip_get_device_compute_capability(
            device_id, &mut major, &mut minor,
        ))?
    };
    Ok((major, minor))
}

/// Check if device is integrated (APU)
pub fn is_device_integrated(device_id: i32) -> Result<bool> {
    let mut integrated: c_int = 0;
    unsafe { check(hip_is_device_integrated(device_id, &mut integrated))? };
    Ok(integrated != 0)
}

/// Check if device supports cooperative launch
pub fn device_supports_cooperative_launch(device_id: i32) -> Result<bool> {
    let mut supported: c_int = 0;
    unsafe { check(hip_supports_cooperative_launch(device_id, &mut supported))? };
    Ok(supported != 0)
}
