# BRA V8 Alignment Progress 2026-03-21

## Scope

This log records the first executable V8-alignment milestone:

- unified BRA core
- V8-style `BRA_zero` math path
- compute logging in processor/eval entrypoints
- minimal VASM artifact pipeline
- key benchmark entry wiring for `FREAK` and `VIDHALLUC`
- minimal `BRA_calib` training/checkpoint scaffold

## Code State

- `bra_logits_processor.py` is now the canonical BRA implementation.
- `bra_operator.py` and `bra_operator_multi.py` are compatibility shims only.
- `BRA_zero` now uses:
  - adaptive visual Top-k
  - candidate softmax relative scores
  - default always-on post-warmup intervention instead of entropy as the default trigger
- processor stats now expose:
  - `avg_candidate_window`
  - `avg_visual_topk`
  - `avg_resonance_time_ms`
  - `intervention_rate`
  - `selected_frame_histogram`
- offline artifact support added:
  - `tools/build_vasm_table.py`
  - `train_bra_calib.py`
  - `bra_projector.py`
  - `bra_vasm.py`

## Remote Validation

### 1. Unified-core smoke test

- Script: `bra_smoke_test.py`
- Model: `Qwen3-VL-2B-Instruct`
- Config:
  - `BRA_SMOKE_SAMPLES=2`
  - `BRA_SMOKE_MAX_NEW_TOKENS=32`
- Result:
  - base AGL: `32.0`
  - BRA AGL: `32.0`
  - AGL delta: `0.00%`
  - VRAM delta: `-0.0MB`
  - outputs changed: `2/2`
  - avg candidate window: `50.0`
  - avg visual Top-k: `3.5`
  - intervention rate: `1.00`
  - avg resonance time: `0.700 ms`

### 2. POPE mini-eval with compute stats

- Script: `run_eval_pipeline.py`
- Method: `bra_zero`
- Split: `POPE random`
- Result snapshot:
  - `n_samples=5`
  - `accuracy=0.8000`
  - `f1=0.8000`
  - `avg_candidate_window=50.0`
  - `avg_visual_topk=3.0`
  - `intervention_rate=1.0`
- Diagnostics sanity check:
  - `run_eval_pipeline.py` now exports `sample_audits`
  - audit payloads include per-step token ids, raw resonance, relative scores, mask weights, and trigger reasons
  - aggregated frame histograms are now available through BRA stats when video samples are runnable

### 3. VASM artifact pipeline

- Built remote artifact:
  - `/root/autodl-tmp/BRA_Project/artifacts/qwen3vl2b_vasm.json`
- Verified runtime path by running:
  - `run_eval_pipeline.py --method bra_zero --vasm_artifact ...`
- Runtime validation succeeded.

### 4. FREAK / VIDHALLUC entry status

- `FREAK`:
  - loader is now wired
  - remote dataset inspection showed `test-00000-of-00005.parquet` is `0 bytes`
  - loader now skips zero-byte shards and uses non-empty parquet files
  - `bra_eval_matrix.py --dataset freak --n_samples 2` runs successfully
- `VIDHALLUC`:
  - loader and result schema are now wired
  - remote JSON schema is parsed successfully
  - current remote `data/*.zip` package is unreadable / not a valid zip
  - evaluation entry no longer crashes, but currently loads `0` runnable samples until data packaging is fixed

### 5. BRA_calib minimal scaffold

- Local synthetic feature training succeeded with:
  - `train_bra_calib.py`
  - checkpoint output: `tmp_bra_calib_projector.pt`
- Checkpoint loading verified through `bra_projector.create_projector(...)`.

## Current Risks

- `VIDHALLUC` still lacks runnable video samples because remote archive packaging is broken.
- `BRA_calib` is scaffolding-level only; no real vision-text feature pairs have been prepared yet.
- `VIDHALLUC` temporal diagnostics will remain empty until runnable video files are restored on the remote dataset.
