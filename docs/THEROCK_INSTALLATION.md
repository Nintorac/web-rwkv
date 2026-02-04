# TheRock Installation Guide for gfx1151 (Strix Halo)

This guide documents how to install and configure TheRock (AMD's ROCm nightly builds) for HIP development on gfx1151 GPUs (AMD Radeon 8060S / Strix Halo).

## Why TheRock?

TheRock is useful when:
- Your distro/kernel combo isn't well supported by the "official" ROCm packages yet.
- You need newer gfx1151 fixes than what your distro provides.

Note: As of **2026-01-21**, AMD published ROCm **7.2** release notes describing Radeon/Ryzen support updates (including Strix Halo-class systems). If you're on a supported distro (often Ubuntu LTS variants), it may now be preferable to use official ROCm instead of TheRock. This doc keeps TheRock instructions because they remain practical on fast-moving distros/kernels.

## Current Installation (2026-01-23)

| Component | Version |
|-----------|---------|
| TheRock | 7.12.0a20260122 |
| Target GPU | gfx1151 |
| Kernel | 6.18.5-200.fc43.x86_64 |
| Status | Working |

## Installation Steps

### Step 1: Create Virtual Environment and Install

```bash
# Create virtual environment
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Install TheRock for gfx1151
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"

# Initialize development files
rocm-sdk init
```

This installs:
- `rocm-sdk-core` - Core runtime (HSA, HIP)
- `rocm-sdk-libraries-gfx1151` - GPU libraries (hipBLAS, MIOpen, etc.)
- `rocm-sdk-devel` - Development tools (headers, CMake configs)

### Step 2: Environment Variables

Add to shell or source before building:

```bash
# Activate venv
source /opt/venv/bin/activate

# Get the ROCm SDK root path
export ROCM_SDK_ROOT=$(/opt/venv/bin/rocm-sdk path --root)

# Essential environment variables
export ROCM_PATH=$ROCM_SDK_ROOT
export HIP_PATH=$ROCM_SDK_ROOT
export HIP_CLANG_PATH=$ROCM_SDK_ROOT/lib/llvm/bin
export HIP_DEVICE_LIB_PATH=$ROCM_SDK_ROOT/lib/llvm/amdgcn/bitcode

# Path configuration
export PATH=$ROCM_SDK_ROOT/bin:$ROCM_SDK_ROOT/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_SDK_ROOT/lib:$LD_LIBRARY_PATH

# CMake configuration
export CMAKE_PREFIX_PATH=$(/opt/venv/bin/rocm-sdk path --cmake)
```

### Step 3: Verify Installation

```bash
# Check targets
rocm-sdk targets  # Should show gfx1151

# Check GPU detected
rocminfo | grep gfx1151

# Test HIP
hipcc --version
```

## Compiling HIP Kernels

### Using hipcc

```bash
hipcc mykernel.hip -o mykernel --offload-arch=gfx1151 -O3
```

### Using amdclang++ (recommended for new code)

```bash
$ROCM_SDK_ROOT/lib/llvm/bin/amdclang++ \
    -x hip mykernel.hip \
    -o mykernel \
    --offload-arch=gfx1151 \
    --rocm-path=$ROCM_SDK_ROOT \
    -O3
```

## Directory Structure

```
/opt/venv/lib/python3.12/site-packages/
  _rocm_sdk_core/               # Core runtime
    bin/                        # hipcc, rocminfo, etc.
    lib/                        # libhsa-runtime64.so, libamdhip64.so
      llvm/
        bin/                    # amdclang++, clang, lld
        amdgcn/bitcode/         # Device libraries (.bc)
    include/                    # HIP, HSA headers
  _rocm_sdk_devel/              # Development files (after init)
    lib/cmake/                  # CMake configs
  _rocm_sdk_libraries_gfx1151/  # gfx1151-specific libraries
    lib/                        # librocblas.so, libhipblas.so, etc.
```

## Known Issues

1. **SDMA Artifacts**: For visual artifacts during inference:
   ```bash
   export HSA_ENABLE_SDMA=0
   ```

2. **Performance**: Native gfx1151 kernels may be slower than gfx1100. For benchmarking:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

## Verification Test

Run this to verify HIP works:

```bash
cat > /tmp/hip_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = i * 2.0f;
}

int main() {
    float *d_x;
    hipMalloc(&d_x, 1024 * sizeof(float));
    test_kernel<<<4, 256>>>(d_x, 1024);
    hipError_t err = hipDeviceSynchronize();
    printf("HIP test: %s\n", err == hipSuccess ? "PASSED" : "FAILED");
    hipFree(d_x);
    return err != hipSuccess;
}
EOF

hipcc /tmp/hip_test.cpp -o /tmp/hip_test --offload-arch=gfx1151
/tmp/hip_test
```

## References

- [ROCm/TheRock GitHub](https://github.com/ROCm/TheRock)
- [AMD ROCm Nightlies Index](https://rocm.nightlies.amd.com/v2/gfx1151/)
