# Benchmarking + Visualization Plan (Config-Driven)

This doc captures the **planning** for a benchmarking + visualization setup for `web-rwkv`.

## Goals

- Provide **repeatable**, **config-driven** performance benchmarks that sweep:
  - `batch_size × seq_len × model_size × backend`
  - (Plus: record `token_chunk_size`, adapter/device info, and build metadata)
- Cover two benchmark families:
  - **Decode-only**: kernel processes **seq_len = 1** each call; many calls simulate decoding.
  - **Prefill**: prompt processing for lengths `<chunk`, `chunk`, `>chunk`, multiples of `chunk`, and **mixed-size batches**.
- Write results as **append-only JSONL** (`.jsonl`) with a stable schema.
- Provide a **static HTML dashboard** that can load one/many JSONL files and support drilldowns.
- Ensure the dashboard looks like an **engineering tool**, not “AI product UI”.

## Non-goals (v1)

- Perfect cross-machine comparability (GPU clocks, thermals, drivers vary).
- A server-backed app (keep it static HTML + local file loading).

## Key Concepts / Terminology

- **token_chunk_size (C)**: the maximum total tokens processed per `infer()` call (rounded up to a multiple of 32 internally). This strongly shapes prefill behavior.
- **Prefill**: processing a prompt of length `L` until the first logits are returned for each batch (local TTFT).
- **Decode**: repeated `infer()` calls where each batch provides one token per call; each call returns logits for that token.
- **Backend**:
  - `wgpu` (and its backend variants: Vulkan/DX12/Metal/OpenGL depending on OS/device).
  - `hip` (optional feature; AMD-only path).
- **Adapter**: the GPU device the run is executed on (recorded metadata; not a sweep dimension).
  - Example: `Radeon 8060s` (example only; not a required/canonical device)
- **model_size**: a *human* size label (e.g. `0.1b`, `2.9b`) used for grouping/sorting in the dashboard.
  - This should be explicitly provided in config per model (do not infer from filenames in code).

## Architecture Overview

### Runner (benchmarking-as-testing)

Implement the benchmark runner as:

- A Rust **integration test** target (e.g. `tests/benchmarks.rs`) marked `#[ignore]` so it runs on demand:
  - Recommended invocation pattern:
    - `WEB_RWKV_BENCH_CONFIG=benchmarks/config.yaml WEB_RWKV_BENCH_PROFILE=smoke cargo test --release --test benchmarks -- --ignored --nocapture`
    - Use `--features hip` when you want HIP runs included.
  - This aligns with “benchmarking should be testing”.
- Optionally mirror the runner as a small CLI binary (for convenience), but keep the test harness as the canonical entrypoint.

### Config-driven design

All benchmark matrices live in config. Code only understands:

- Scenario types (`decode_only`, `prefill_uniform`, `prefill_mixed`)
- How to execute a scenario from a config record
- How to write JSONL records

Everything else (exact sizes, models, backends to try, repeats, warmups, etc.) must be editable via config only.

Recommended approach:

- `benchmarks/config.yaml` (or `.toml`) describing:
  - `profiles` (smoke/dev/full)
  - `models`
  - `backends`
  - `sweeps` (cartesian products + per-scenario parameter lists)
  - `output` settings

## Config Schema (Proposed)

### Top level

- `schema_version`: integer, bump on breaking changes.
- `profiles`: map of named presets that reference model/backend lists and scenario params.
- `models`: list of model entries (paths, tags).
- `backends`: list of backend entries.
- `scenarios`: definitions of decode/prefill scenario parameter templates.
- `output`: where to write JSONL and how to name runs.

### Models

Each model entry should define:

- `model_id`: sha256 sum of the model weights file (artifact identity; stable across renames).
- `model_name`: name of the weights file without suffix (human display + filtering).
- `model_size`: human size label (e.g. `0.1b`, `2.9b`) for grouping/sorting.
- `path`: safetensors `.st` path (relative or absolute).
- `tags`:
  - `rwkv_version`: `v4|v5|v6|v7`
  - (optional) anything else useful for filtering (quantization tags are **out of scope for v1** and should not be part of the sweep matrix)
- Optional execution overrides:
  - `max_batch_size`
  - `max_token_chunk_size`
  - `skip`: boolean

### Backends

Backend entries:

- `backend_id`: `wgpu` or `hip`
- `wgpu_backends`: list, e.g. `[Vulkan, Dx12]` (runner skips unavailable ones)
- `adapter_selection`:
  - `auto_high_perf` (default)
  - `by_name_regex`
  - `interactive` (avoid for automation; fine for local)
- Optional constraints:
  - `skip`: boolean

### Sweeps

Sweeps should support:

- A cartesian product of:
  - `models[]`
  - `backends[]`
  - `batch_sizes[]`
  - `token_chunk_sizes[]`
  - `seq_lens[]` (for prefill uniform)
- Plus scenario-specific parameters (decode steps, mixed length sets).

The config should also allow:

- `skip_conditions`: e.g. “skip if batch_size > model.max_batch_size”.
- `limits`: max total cases, max runtime, fail-fast toggle.

## Benchmark Scenarios (Exact Specs)

### Shared controls

All scenarios should include these configurable controls:

- `seed`: fixed PRNG seed for token generation (default 42)
- `warmup`:
  - `warmup_runs`: number of full repeats not recorded (default 1)
  - `warmup_steps`: per-repeat warmup steps for decode (default 64)
- `repeats`: recorded repeats per case (default 5)
- `timing`:
  - Use `Instant::now()` around the *measured loop only*.
  - Do not include model file IO / model build time in the timed portion.
  - Run a “device settle” delay optionally between repeats (e.g. 50–200ms) if needed to reduce jitter.
- `error_policy`:
  - Continue-on-error and write `status=error` JSONL records (do not abort the whole run).
  - Classify errors: `oom`, `device_lost`, `unsupported_backend`, `file_missing`, etc.

### Scenario 1: Decode-only

**Purpose**: measure decode throughput/latency in steady-state “one token per step” mode.

**Definition**:

- Kernel effective seq len is always `1` per call.
- Simulate decoding by executing `infer()` repeatedly for `N` steps.

**Axes**:

- `batch_size` (B)
- `backend` (wgpu/hip; and for wgpu, wgpu backend)
- `model` (model size / version)
- (Optional) `token_chunk_size` (C): even though decode uses seq_len=1, chunk size can still affect packing/dispatch overhead; record it for comparability.

**Parameters per case**:

- `decode_steps`: recommended set `[128, 512, 2048]`
  - `128` for fast smoke
  - `512` for dev
  - `2048` for stable “full” runs

**Setup** (per repeat):

1. Create `RnnInput` with `B` batches.
2. For each batch, keep the token buffer length at exactly 1 token per step:
   - Initialize each batch with one token (e.g. `0`).
   - After each `infer()` returns, replace each batch tokens with the next token for the next call.
   - Use a deterministic token sequence (fixed seed). Do not run tokenizer; operate on token ids directly.
3. Use `RnnOption::Last` so each step yields logits for that token.

**Timing loop**:

- Start timer.
- For `step in 0..decode_steps`:
  - Call `infer()` once.
  - Discard logits (do not softmax/sample inside the timed region).
  - Replace each batch’s token with the next token id.
- Stop timer.

**Metrics**:

- `decode_total_ms`
- `decode_steps`
- `decode_tokens = batch_size * decode_steps`
- `decode_tok_per_s = decode_tokens / (decode_total_ms / 1000)`
- Optional: `per_step_ms` sampling for p50/p95 (disabled by default to keep overhead low)

**Notes**:

- If we want “post-prefill steady-state decode”, allow an optional untimed “prime prefill” step:
  - Run a prefill prompt of length `prime_len` (e.g. 256) per batch, untimed, then run the decode loop timed.

### Scenario 2: Prefill-uniform

**Purpose**: measure prompt processing performance as prompt length crosses chunk boundaries.

**Definition**:

- Each batch has the same prompt length `L`.
- Measure time until each batch returns its first logits (local TTFT) and time to completion (all batches finished prompt consumption).

**Axes**:

- `batch_size` (B)
- `seq_len` (L)
- `token_chunk_size` (C)
- `backend`
- `model`

**Seq len sets (recommended canonical)**

For each `C`, define `L` as a list emphasizing:

- `<C` region
- exactly `C`
- `>C` boundaries
- multiples of `C`

Recommended per-`C` list:

- `L = [1, C/4, C/2, C-1, C, C+1, 2C, 4C, 8C]`
  - Round to integers; ensure `>= 1`.
  - If some cases are too large for a given machine/model, rely on skip conditions and/or record an `oom` error (do not silently cap lengths).

Additionally, for multi-batch comparability, include total-token targets:

- Choose total token targets `T = [C/2, C, 2C, 4C, 8C]`
- For each `(B, C, T)`, set `L = ceil(T / B)` so `B*L ≈ T`
  - Record `total_prompt_tokens = B*L`

**Setup** (per repeat):

1. For each batch, generate `L` tokens deterministically.
2. Create `RnnInput` with `B` batches, each with its tokens and `RnnOption::Last`.
3. Call `infer()` repeatedly until all batches have produced logits at least once.

**Timing loop**:

- Start timer at first `infer()`.
- After each `infer()`:
  - Check each batch output tensor size.
  - For each batch, record `ttft_ms_local` the first time its output becomes non-empty.
  - Continue until all batches have `ttft_ms_local` recorded.
- Stop timer.

**Metrics**:

- `prefill_total_ms` (time until all batches reached first logits)
- Per-batch `ttft_ms_local[]` (min/median/max derived for dashboard)
- `num_infer_calls` (important when L > C)
- `total_prompt_tokens = B*L`
- `prefill_tok_per_s = total_prompt_tokens / (prefill_total_ms / 1000)`

### Scenario 3: Prefill-mixed batches

**Purpose**: measure batching behavior when batch elements have different prompt lengths.

**Definition**:

- One `infer()` call processes up to `C` total tokens across all batches; with mixed lengths, some batches finish earlier than others.
- We want distributions of TTFT and total completion time across the mixed batch.

**Axes**:

- `batch_size` (B)
- `token_chunk_size` (C)
- `backend`
- `model`
- `mixed_case_id` (named case describing the lengths)

**Named mixed cases (recommended defaults)**

For each `C`, produce per-batch length vectors by scaling these patterns:

- `staircase_8` (spread from very short to very long):
  - For `B=8`: `[C/16, C/8, C/4, C/2, 3C/4, C, 2C, 4C]`
  - For other B: truncate or repeat the pattern; always keep at least one long and one short.
- `bimodal_half`:
  - Two buckets: short `C/8`, long `4C`
  - If `B` is odd, the “long” side gets the extra element (`ceil(B/2)` long, `floor(B/2)` short).
  - Deterministic ordering: `[long × ceil(B/2)] + [short × floor(B/2)]`
- `one_long_rest_short`:
  - `[8C] + [C/8] * (B-1)`
- `realistic_chat_scaled`:
  - A hand-curated increasing vector (e.g. `[32, 64, 96, 128, 192, 256, 384, 512]` for `C=256`)
  - Scale deterministically to other `C` using integer math:
    - `len_i = max(1, base_i * C / 256)` where `base_i` is from the `C=256` base vector.

**Determinism note (mixed batches)**

- The PRNG `seed` controls **token contents** only.
- The **length vector** for a mixed case must be deterministic solely from `(mixed_case_id, batch_size B, token_chunk_size C)` so different implementations don’t diverge.
- Proposed rule for adapting an 8-element base pattern to arbitrary `B`:
  1. Compute the base integer lengths using integer arithmetic (e.g. `C/16` is floor division).
  2. If `B <= base_len`: take the first `B`.
  3. If `B > base_len`: repeat the base pattern until length ≥ `B`, then truncate to `B`.
  4. Ensure all lengths are `>= 1`.

**Setup** (per repeat):

1. Generate tokens for each batch length `Li` deterministically.
2. Create a `RnnInput` with those batch token vectors.

**Timing loop**:

- Same as uniform prefill, but record:
  - `ttft_ms_local[i]` for each batch
  - `time_to_all_ttft_ms_local = max(ttft_ms_local)`
  - `num_infer_calls`

**Metrics**:

- `prefill_total_ms` (until all batches got logits once)
- `ttft_ms_local[]`
- `ttft_min_ms`, `ttft_p50_ms`, `ttft_max_ms`
- `prefill_tok_per_s` computed over `sum(Li)`

## Benchmark Matrix (Recommended Defaults)

These are suggested starting points; keep them in config so they can be updated without code changes.

### token_chunk_size values

Because the library enforces multiples of 32:

- Smoke: `[128]`
- Dev: `[128, 256]`
- Full: `[32, 128, 256, 512]` (optionally `1024` if hardware allows)

### Batch sizes

- Smoke: `[1, 4]`
- Dev: `[1, 2, 4, 8, 16]`
- Full: `[1, 2, 4, 8, 16, 32]` (and `64` for tiny models only)

### Models

Use the repo’s small models for smoke/dev so it’s runnable by default:

- `assets/models/rwkv7_othello_9m_L10_D256_extended.st` (xs)
- `assets/models/rwkv-puzzle15.st` (xs)

For “full” runs on dedicated machines, config can include larger external models (not stored in repo).

## JSONL Output Schema (Proposed)

Write one JSON object per **(case × repeat)** plus an optional “run header” record.

### Run header record (type = "run")

Fields:

- `type: "run"`
- `schema_version`
- `run_id` (timestamp + random suffix)
- `started_at_utc`
- `git_sha`, `git_dirty` (both)
- `crate_version`, `rustc_version`
- `host`: `os`, `cpu`, `ram_gb`
- `gpu`: `adapter_name`, `backend_api` (wgpu backend), driver info if obtainable
- `uname`: full `uname` string (optional; preferred over duplicating kernel fields)
- `firmware`: firmware versions (optional; OS/vendor-specific)

### Measurement record (type = "measure")

Common fields:

- `type: "measure"`
- `schema_version`, `run_id`, `case_id`, `repeat_index`
- `scenario`: `decode_only | prefill_uniform | prefill_mixed`
- `status`: `ok | skipped | error`
- `error_kind`, `error_message` (only when not ok)

Case identity fields:

- `model_id`, `model_name`, `model_path`, `model_size`, `rwkv_version`
- `backend_id`, `wgpu_backend`
- `batch_size`
- `token_chunk_size_requested`, `token_chunk_size_effective`
- `seq_len` (uniform only) OR `seq_lens[]` + `mixed_case_id` (mixed only)
- `decode_steps` (decode only)

`case_id` definition:

- `case_id` is a stable identifier for a single benchmark “cell” in the matrix (used to group repeats and compare runs).
- It should be computed from **normalized case params only** (exclude `run_id`, timestamps, and adapter name).
- Recommended: a human-readable string or a hash of a canonical JSON encoding of:
  - `scenario`, `model_id` (or `model_name`), `backend_id`, `wgpu_backend`, `batch_size`,
  - `token_chunk_size_effective`,
  - plus scenario params (`decode_steps` OR `seq_len` OR (`mixed_case_id` + `seq_lens[]`)).

Metrics fields:

- Decode:
  - `decode_total_ms`
  - `decode_tokens`
  - `decode_tok_per_s`
  - Optional: `decode_step_ms_p50`, `decode_step_ms_p95`
- Prefill:
  - `prefill_total_ms`
  - `total_prompt_tokens`
  - `prefill_tok_per_s`
  - `num_infer_calls`
  - `ttft_ms_local[]` (array length = batch_size; for uniform, it should be tight; for mixed it will vary)
  - `ttft_min_ms`, `ttft_p50_ms`, `ttft_max_ms`

## Dashboard Plan (Static HTML)

### Data loading

- Primary: drag-and-drop one/many `.jsonl` files into the page.
- Parse line-by-line (stream if possible) and merge into an in-memory dataset.
- Allow exporting filtered subsets as JSONL again (optional).

### Core interactions / drilldowns

- Global filters (multi-select):
  - scenario, model_name, model_size, model_id, backend_id, wgpu_backend, adapter_name
  - batch_size, token_chunk_size, seq_len (uniform), mixed_case_id (mixed)
  - run_id / git_sha / timestamp window
- Views (v1):
  - **Summary table**: sortable by throughput, latency; shows best/worst; click row to drill in.
  - **Heatmap**: `batch_size × seq_len` colored by `tok/s` (prefill uniform).
  - **Line chart**: `tok/s vs seq_len` for fixed batch/model/backend/chunk.
  - **Distribution**: mixed prefill `ttft_ms_local` box/violin per case.
  - **Compare runs**: choose two runs and show % deltas for matched cases (highlight regressions).

### Visual design (“not AI‑y”)

Avoid:

- Big gradient backgrounds, neon accents, glow effects, “glassmorphism”.
- Over-rounded “SaaS template” cards/bento grids.
- Chat-bubble motifs, “assistant” icons, sparkles, generic AI branding.

Prefer:

- Simple neutral background (light or dark mode, but restrained).
- High-information density tables with strong typography hierarchy.
- Subtle borders/dividers; minimal shadows; modest radii.
- One restrained accent color for selection/highlights.
- Charts with engineering-style defaults (legible axes, clear legends, minimal decoration).

### Tech choice (keep it minimal)

- Single static HTML + JS bundle under `docs/benchmarks/` (or `benches/dashboard/`).
- Prefer lightweight charting:
  - Option A: D3 + small helper utilities.
  - Option B: Vega-Lite (more config-driven charts), if bundle size is acceptable.
- Ensure the dashboard also supports config-driven views (e.g. `dashboard_views.json` describing which charts to render).

## Execution Workflow

### Running benchmarks

- “Smoke” locally (fast):
  - uses the included tiny models and a minimal matrix
  - validates runner correctness and schema
- “Dev” locally:
  - broader batch/chunk/seq coverage on one machine
- “Full” on a dedicated benchmark machine:
  - more models/backends, longer repeats, produces the canonical dataset for dashboards

### Storing results

- Store raw JSONL under `benches/reports/`.
- Optionally post-process into a compact `.json` index for faster dashboard load (but keep JSONL as source of truth).

## Roadmap (Phased)

1. **Schema + config**: finalize config schema and JSONL schema; add a parser + validator.
2. **Runner harness**: integration-test runner that can execute a sweep and write JSONL, with skip/error handling.
3. **Decode-only**: implement + validate stable timing and low overhead.
4. **Prefill uniform**: implement length sets relative to chunk size, plus multi-batch total-token targets.
5. **Prefill mixed**: implement named mixed cases + per-batch TTFT stats.
6. **Dashboard v1**: drag/drop JSONL + filters + summary table + 2–3 charts.
7. **Regression compare**: run-to-run diffs with % change and thresholds.

## Ticket Outline (Not Created Yet)

### Epic: Benchmarking + Visualization

**Epic ID**: bd-bench (or similar)
**Type**: epic
**Priority**: P2

**Summary**: Config-driven benchmarking infrastructure with JSONL output and static HTML dashboard for performance analysis.

**Children** (all tickets below are children of this epic):
- Config + Schema (sub-epic or task group)
- Benchmark Runner (sub-epic or task group)
- Decode-only Benchmark
- Prefill-uniform Benchmark
- Prefill-mixed Benchmark
- Dashboard v1 (sub-epic or task group)
- Regression Comparison

**Acceptance Criteria**:
- [ ] Smoke profile runs successfully on included models
- [ ] JSONL output validates against schema
- [ ] Dashboard loads and displays benchmark data
- [ ] At least one regression comparison can be performed

**Non-goals (v1)**: See plan doc non-goals section

---

### Config + Schema
- Define `benchmarks/config.yaml` schema and document it
  - Profiles (smoke/dev/full)
  - Models list with `model_id`, `model_name`, `model_size`, `path`, `tags`
  - Backends list with `backend_id`, `wgpu_backends`, `adapter_selection`
  - Sweeps with cartesian product definitions
- Define JSONL output schema
  - Run header record (`type: "run"`, metadata fields)
  - Measure record (`type: "measure"`, case identity + metrics)
  - `case_id` computation (stable identifier from normalized params)
- Implement skip_conditions + limits logic
  - Skip if `batch_size > model.max_batch_size`
  - Max total cases, max runtime, fail-fast toggle
- Add config validation command/test

### Benchmark Runner
- Implement ignored integration test runner (`tests/benchmarks.rs`)
  - Config loading from `WEB_RWKV_BENCH_CONFIG` env var
  - Profile selection from `WEB_RWKV_BENCH_PROFILE` env var
- Implement sweep execution engine
  - Cartesian product expansion
  - Per-case setup and teardown
- Implement JSONL writer
  - Append-only output
  - Run header on start
  - Measure record per (case × repeat)
- Add robust error classification + skip logic
  - Error kinds: `oom`, `device_lost`, `unsupported_backend`, `file_missing`
  - Continue-on-error with `status=error` records
- Add environment metadata collection
  - GPU adapter name, backend API, driver info
  - `git_sha`, `git_dirty`, `crate_version`, `rustc_version`
  - Host info: `os`, `cpu`, `ram_gb`, `uname`
- (Optional) CLI binary mirror for convenience

### Decode-only Benchmark
- Implement scenario execution
  - Kernel seq_len=1 per call, N steps
  - Deterministic token sequence from seed
  - `RnnOption::Last` for logits
- Add shared controls
  - `seed` (default 42)
  - `warmup_runs`, `warmup_steps`
  - `repeats` (default 5)
  - Optional device settle delay between repeats
- Validate correctness invariants (no NaNs, expected shapes)
- (Optional) Prime prefill step (untimed prefill before timed decode)
- (Optional) Per-step latency sampling for p50/p95

### Prefill-uniform Benchmark
- Implement length sets relative to chunk size
  - `L = [1, C/4, C/2, C-1, C, C+1, 2C, 4C, 8C]`
- Add total-token target mode for multi-batch comparability
  - `T = [C/2, C, 2C, 4C, 8C]`, `L = ceil(T / B)`
- Implement per-batch TTFT tracking
  - Record `ttft_ms_local[]` per batch
  - Derive `ttft_min_ms`, `ttft_p50_ms`, `ttft_max_ms`
- Track `num_infer_calls` (important when L > C)

### Prefill-mixed Benchmark
- Implement named mixed cases
  - `staircase_8`: spread from very short to very long
  - `bimodal_half`: two buckets (short C/8, long 4C)
  - `one_long_rest_short`: `[8C] + [C/8] * (B-1)`
  - `realistic_chat_scaled`: hand-curated, scales with C
- Implement deterministic length vector generation
  - From `(mixed_case_id, B, C)` only
  - Truncate or repeat base pattern for arbitrary B
- Implement per-batch TTFT stats
  - `ttft_ms_local[]`, `ttft_min_ms`, `ttft_p50_ms`, `ttft_max_ms`

### Dashboard v1
- Static HTML + JS bundle setup
  - Location: `benches/dashboard/` or `docs/benchmarks/`
  - Lightweight charting (D3 or Vega-Lite)
- Drag-and-drop JSONL loading
  - Parse line-by-line, merge into in-memory dataset
  - Support loading multiple files
- Filter UI (multi-select)
  - scenario, model_name, model_size, backend_id, wgpu_backend
  - batch_size, token_chunk_size, seq_len, mixed_case_id
  - run_id, git_sha, timestamp window
- Summary table view
  - Sortable by throughput, latency
  - Click row to drill in
- Heatmap view
  - `batch_size × seq_len` colored by tok/s (prefill uniform)
- Line chart view
  - tok/s vs seq_len for fixed batch/model/backend/chunk
- Distribution view
  - Box/violin plot for mixed prefill `ttft_ms_local` per case
- JSONL export
  - Export filtered subsets as JSONL
- Styling pass
  - "Engineering tool" aesthetic, not "AI product"
  - High-information density, minimal decoration

### Regression Comparison
- Run-to-run diff view
  - Choose two runs, show % deltas for matched `case_id`s
  - Highlight regressions above threshold
- Exportable report (JSON or markdown)

### Post-v1 (Optional)
- Compact `.json` index for faster dashboard load
- Config-driven dashboard views (`dashboard_views.json`)

---

## Ticket Dependencies

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        bd-bench (Epic)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
┌───────────────────────────┐             ┌───────────────────────────┐
│   CONFIG + SCHEMA         │             │   DASHBOARD FOUNDATION    │
│   (Foundation Layer)      │             │   (Can start in parallel) │
├───────────────────────────┤             ├───────────────────────────┤
│ 1. Config YAML schema     │             │ 1. Static HTML + JS setup │
│ 2. JSONL output schema ───┼─────────────┼─▶ (needs JSONL schema)    │
│ 3. Skip conditions logic  │             │ 2. Drag-drop loading      │
│ 4. Config validation      │             │ 3. Filter UI skeleton     │
└───────────┬───────────────┘             └───────────┬───────────────┘
            │                                         │
            ▼                                         │
┌───────────────────────────┐                         │
│   BENCHMARK RUNNER        │                         │
│   (Execution Layer)       │                         │
├───────────────────────────┤                         │
│ 1. Integration test runner│                         │
│    (needs config schema)  │                         │
│ 2. Sweep execution engine │                         │
│    (needs skip conditions)│                         │
│ 3. JSONL writer ──────────┼─────────────────────────┤
│    (needs JSONL schema)   │                         │
│ 4. Error classification   │                         │
│ 5. Env metadata collection│                         │
└───────────┬───────────────┘                         │
            │                                         │
            ├──────────────┬──────────────┐           │
            ▼              ▼              ▼           │
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  DECODE-ONLY     │ │ PREFILL-UNIFORM  │ │ PREFILL-MIXED    │
│  (Scenario)      │ │ (Scenario)       │ │ (Scenario)       │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ • Scenario exec  │ │ • Length sets    │ │ • Named cases    │
│ • Shared controls│ │ • Total-token    │ │ • Length vector  │
│ • Validation     │ │   target mode    │ │   generation     │
│ • (Opt) Prime    │ │ • Per-batch TTFT │ │ • Per-batch TTFT │
│ • (Opt) p50/p95  │ │ • num_infer_calls│ │   stats          │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         │    ┌───────────────┴───────────────┐    │
         │    │  (All scenarios produce JSONL) │    │
         │    └───────────────┬───────────────┘    │
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
            ┌───────────────────────────────────────┐
            │         DASHBOARD VIEWS               │
            │         (Needs JSONL data)            │
            ├───────────────────────────────────────┤
            │ • Summary table (needs any scenario)  │
            │ • Heatmap (needs prefill-uniform)     │
            │ • Line chart (needs prefill-uniform)  │
            │ • Distribution (needs prefill-mixed)  │
            │ • JSONL export                        │
            │ • Styling pass                        │
            └───────────────────┬───────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────┐
            │       REGRESSION COMPARISON           │
            │       (Needs dashboard + 2 runs)      │
            ├───────────────────────────────────────┤
            │ • Run-to-run diff view                │
            │ • % delta calculation                 │
            │ • Exportable report                   │
            └───────────────────────────────────────┘
```

### Detailed Dependency List

#### Tier 0: Foundation (No Dependencies)
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Config YAML schema definition | — | Everything |
| JSONL output schema definition | — | JSONL writer, Dashboard loading |
| Static HTML + JS bundle setup | — | All dashboard views |

#### Tier 1: Core Infrastructure
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Skip conditions + limits logic | Config schema | Sweep engine |
| Config validation command | Config schema | Runner (soft) |
| Integration test runner | Config schema | All scenarios |
| Drag-drop JSONL loading | JSONL schema | All dashboard views |
| Filter UI skeleton | HTML setup | All dashboard views |

#### Tier 2: Execution Engine
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Sweep execution engine | Runner, Skip conditions | All scenarios |
| JSONL writer | JSONL schema, Runner | All scenarios |
| Error classification | Runner | All scenarios |
| Env metadata collection | Runner | All scenarios |

#### Tier 3: Scenarios (Can Run in Parallel)
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Decode-only: scenario exec | Sweep engine, JSONL writer | Summary table |
| Decode-only: shared controls | Scenario exec | — |
| Prefill-uniform: length sets | Sweep engine, JSONL writer | Heatmap, Line chart |
| Prefill-uniform: TTFT tracking | Length sets | — |
| Prefill-mixed: named cases | Sweep engine, JSONL writer | Distribution view |
| Prefill-mixed: length vector gen | Named cases | — |

#### Tier 4: Dashboard Views
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Summary table view | JSONL loading, Filter UI, ≥1 scenario | Regression comparison |
| Heatmap view | JSONL loading, Prefill-uniform | — |
| Line chart view | JSONL loading, Prefill-uniform | — |
| Distribution view | JSONL loading, Prefill-mixed | — |
| JSONL export | JSONL loading | — |
| Styling pass | All views | — |

#### Tier 5: Comparison & Polish
| Ticket | Depends On | Blocks |
|--------|------------|--------|
| Run-to-run diff view | Summary table, ≥2 runs | — |
| Exportable report | Diff view | — |

### Critical Path

The minimum path to a working v1:

```
Config schema → JSONL schema → Runner → Sweep engine → JSONL writer
                     ↓                        ↓
              Dashboard setup          Decode-only scenario
                     ↓                        ↓
              JSONL loading ←─────────── (produces data)
                     ↓
              Summary table view
                     ↓
              Styling pass
```

**Estimated ticket count on critical path**: 9 tickets
