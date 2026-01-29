# Test Fixtures

This directory contains test fixtures for validating HIP kernel correctness against the Python reference implementation.

## Regenerating Fixtures

Fixtures are not committed to git (they're ~40MB total). Regenerate them with:

```bash
python scripts/generate_test_fixtures.py \
    --model /workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st \
    --output tests/fixtures/
```

## Structure

```
tests/fixtures/
├── kernels/          # Individual kernel I/O pairs
│   ├── sigmoid/
│   ├── squared_relu/
│   ├── decay_exp/
│   ├── tanh/
│   ├── softplus_decay/
│   ├── lerp/
│   ├── layer_norm/
│   ├── group_norm/
│   ├── l2_norm/
│   ├── matmul/
│   ├── token_shift/
│   ├── wkv7/
│   ├── wkv_bonus/
│   ├── control_k/
│   └── channel_mix_state/
├── layers/           # Layer-level fixtures with actual weights
│   ├── time_mix/
│   └── channel_mix/
└── model/            # Full model fixtures
    └── weights_spot_check.npz
```

## NPZ Format

Each `.npz` file contains NumPy arrays with this convention:
- `{name}`: Flattened 1D array as **float32** (for Rust compatibility)
- `{name}_shape`: 4-element int64 array with web-rwkv Shape(d0, d1, d2, d3)
- `{name}_dtype`: Original dtype string (e.g., "float16", "float32")

**Note**: Arrays are stored in their native dtype (e.g., float16 for activations).
The original dtype is recorded so tests can convert back if needed.

Example loading in Python:
```python
import numpy as np
data = np.load('kernels/sigmoid/basic.npz')
input_flat = data['input']           # float16 array
input_shape = data['input_shape']    # [768, 512, 1, 1] = Shape(C, A, 1, 1)
input_dtype = data['input_dtype'][0] # "float16" - original dtype
```

Example loading in Rust (with npyz):
```rust
use npyz::npz::NpzArchive;

let mut npz = NpzArchive::open("kernels/sigmoid/basic.npz")?;
let input = npz.by_name("input")?.unwrap().into_vec::<half::f16>()?;
let shape = npz.by_name("input_shape")?.unwrap().into_vec::<i64>()?;
```

## Tolerance Guidelines

| Tensor Type | rtol | atol |
|-------------|------|------|
| FP16 activations | 1e-3 | 1e-4 |
| FP32 state | 1e-5 | 1e-6 |
| MatMul outputs | 1e-2 | 1e-3 |
