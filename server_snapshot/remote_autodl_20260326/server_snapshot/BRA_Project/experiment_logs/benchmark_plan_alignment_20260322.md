# Benchmark Plan Alignment - 2026-03-22

## Scope

Aligned the current BRA codebase to `BRA_标杆实验方案.md` and treated that document as the primary source of truth when it conflicted with earlier V8-alignment defaults.

## Code Changes

- `bra_logits_processor.py`
  - added direct aliases for `bra_max`, `bra_no_vasm`, and `bra_v1_like` through `make_bra_config()`
- `run_eval_pipeline.py`
  - expanded method choices to include `bra_no_vasm`, `bra_max`, and `bra_v1_like`
  - added unified benchmark log fields: `model_family`, `sample_count`, `tokens_per_second`, `notes`
  - raised `CHAIR` generation cap and made it dataset-specific via `--chair_max_new_tokens`
  - added explicit cap warnings such as `chair_agl_near_cap`
- `bra_eval_matrix.py`
  - added framework loaders for `docvqa` and `video_mme`
  - added `bra_no_vasm`/`bra_max` aliases in eval config
  - exported `agl`, `itl_ms_per_token`, `tokens_per_second`, `sample_count`, `model_family`, and `notes`
  - preserved paper diagnostics and temporal histogram export path
- `compare_bra_variants.py`
  - upgraded from internal variant comparer to evidence-chain runner
  - supports chains `a`, `b`, `c`, `video`
  - emits unified JSON and CSV outputs
- `bra_smoke_test.py`
  - now checks required benchmark log fields in BRA processor stats

## Validation

### Smoke test

- Remote `bra_smoke_test.py` passed with:
  - AGL delta: `0.00%`
  - VRAM delta: `+0.0MB`
  - required log fields: `OK`

### Evidence-chain runner pilot

Command family:

- chains: `A + C`
- methods: `base`, `bra_zero`, `bra_meanpool`, `bra_no_vasm`
- sample count: `2`

Artifacts:

- remote: `logs/minitest/benchmark_runner_pilot.json`
- remote: `logs/minitest/benchmark_runner_pilot.csv`
- local mirror:
  - `experiment_logs/remote_mirror/benchmark_runner_pilot.json`
  - `experiment_logs/remote_mirror/benchmark_runner_pilot.csv`

Observed pilot signals:

- `POPE`
  - runner path works
  - unified fields are present
  - `bra_zero` and ablations log `intervention_rate`, candidate statistics, and audits correctly
- `CHAIR`
  - runner path works
  - new cap warning correctly fires: `chair_agl_near_cap`
  - even after increasing from the old default, captions still approach the cap, so the default was raised again after the pilot
- `FREAK`
  - evidence-chain C route is wired and executable
  - current 2-sample pilot returned `accuracy=0.0` for all tested methods, so this remains an active analysis target rather than a claim
- `DocVQA`
  - loader and runner entry are now wired
  - current remote environment returned `no_samples`, so this branch is code-ready but data-blocked

## Current Risks

- `CHAIR` still needs a fresh rerun under the newest higher default cap to verify that AGL is no longer saturating.
- `DocVQA` is not presently runnable on the remote dataset layout available during this pass.
- Evidence chains `B` and `video` are not yet re-piloted under the new benchmark-first runner in this session.
