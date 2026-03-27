# BRA V8 Core Smoke 2026-03-21

## Purpose

Validate the first post-refactor BRA core where:

- `bra_logits_processor.py` becomes the canonical BRA implementation
- `bra_operator.py` and `bra_operator_multi.py` are reduced to compatibility shims
- `BRA_zero` path remains generation-safe on the remote server

## Environment

- Model: `Qwen3-VL-2B-Instruct`
- Script: `bra_smoke_test.py`
- Config:
  - `BRA_SMOKE_SAMPLES=2`
  - `BRA_SMOKE_MAX_NEW_TOKENS=32`

## Result

- Baseline AGL: `32.0`
- BRA AGL: `32.0`
- AGL delta: `0.00%`
- VRAM delta: `-0.0MB`
- Outputs changed: `2/2`
- Avg BRA steps: `32.0`
- Avg vision tokens: `330.0`
- Overall status: `ALL CHECKS PASSED`

## Interpretation

- The canonical BRA core is now live without breaking the generation path.
- The compatibility shims do not introduce a regression in basic Qwen3-VL smoke behavior.
- This validates moving forward with V8-style `BRA_zero` math alignment on top of the unified core.
