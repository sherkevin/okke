# UniGround V2 Migration Boundary

## Purpose

This note freezes the current universal v1 path and defines the only allowed
direction for ongoing development.

## Legacy v1 Files

- `run_uniground_eval.py`
- `uniground_runtime.py`
- `bra_universal_plugin.py`:
  - reusable types and scorer definitions may still be imported
  - the legacy `UniversalSidecarProcessor` remains reference-only

## What Counts As Legacy-Only

The following must not receive new benchmark-specific behavior:

- `POPE`-specific prompt rewrites inside the runtime path
- ad-hoc answer bonuses or extra evidence votes in eval code
- new binary-choice heuristics inside v1 logits processors
- new control-flow branches that exist only to rescue one benchmark

## Allowed Changes In Legacy v1

- bug fixes needed to keep old checkpoints/results readable
- compatibility maintenance for validators and audits
- comments or docs clarifying that the implementation is frozen

## V2 Development Rule

All new universal-path behavior must land in the v2 stack:

- `uniground_v2/`
- `run_uniground_v2_eval.py`

Any feature proposal must answer this first:

1. Is the behavior part of the core universal method?
2. If yes, implement it in v2 only.
3. If no, keep it in a task adapter or benchmark wrapper outside the runtime.

## Immediate Consequence

The repository now treats v1 as a comparison/control route and v2 as the only
active implementation target for the universal plugin line.
