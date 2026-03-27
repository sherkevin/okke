# UniGround Completed Experiments Snapshot (2026-03-22)

## Purpose

This file is a compact snapshot of what has been genuinely completed so far, separated from blocker analysis. It is meant to be compared directly against the total experiment plan.

## A. First-host recovered mini-runs

Host:

- `Qwen3-VL-8B`
- `POPE random`
- `mini_test=32`

Recovered canonical result files:

- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_base_20260322_140701.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_tlra_internal_zero_20260322_140755.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_tlra_internal_calib_20260322_140848.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_external_global_prior_20260322_143555.json`

Recovered metrics:

- `base`
  - `f1=0.9333`
  - `ITL/TPOT=22.17 ms/token`
  - `peak_vram=17.637 GB`
- `tlra_internal_zero`
  - `f1=0.9333`
  - `ITL/TPOT=21.32 ms/token`
  - `peak_vram=17.637 GB`
- `tlra_internal_calib`
  - `f1=0.9333`
  - `ITL/TPOT=21.28 ms/token`
  - `peak_vram=17.637 GB`
- `external_global_prior`
  - `f1=0.9333`
  - `ITL/TPOT=71.9 ms/token`
  - `peak_vram=17.637 GB`
  - `intervention_coverage=1.0`
  - `prefix_ambiguity_rate=0.5574`
  - `span_collapse_errors=103.4062`
  - `abstention_rate=0.0`
  - `abort_trigger_rate=0.0`
  - `candidate_construction_ms=0.9772`
  - `sidecar_scoring_ms=46.7546`
  - `bridge_redistribution_ms=0.0546`
  - `jitter_ms=17.3336`

## B. Universal artifact set now completed

Canonical external encoder:

- `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`

Real shared `Psi_univ` checkpoint:

- `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- sha256:
  - `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`

Checkpoint config:

- `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json`

Training payload:

- `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt`
- payload sha256:
  - `0722c23467469f3dfef7db1216ca0e216d839ea5e2503fb212f3f52828510e67`

Validation status:

- checkpoint validator: pass
- `load_universal_scorer(...)`: pass

## C. Second-host plumbing now completed

Runner support now includes:

- `qwen3-vl-8b`
- `qwen3-vl-4b`
- `qwen3-vl-2b`

Recovered second-host base evidence:

- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_smoke/uniground_qwen3-vl-4b_pope_base_20260322_141148.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_base_20260322_151158.json`

This means the second-host route is no longer hypothetical; it is waiting only for real `TLRA_univ` runs.

## D. Universality audit chain completed

Recovered audit infrastructure:

- `validate_uniground_universality.py`
- `audit_uniground_batch.py`
- `UNIGROUND_UNIVERSALITY_CHECKLIST_20260322.md`
- `UNIVERSAL_CLAIM_FAILURE_AUDIT_20260322.md`
- `experiment_logs/uniground_v6/universality_audit_latest.md`
- `experiment_logs/uniground_v6/universality_audit_latest.json`

Synthetic smoke / local acceptance:

- synthetic checkpoint smoke: pass
- synthetic batch audit: pass

## E. First-host universal batch now completed

Host:

- `Qwen3-VL-8B`
- `POPE random`

Recovered JSONs with the real shared checkpoint:

- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_20260322_154331.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_no_gate_20260322_154721.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_global_only_20260322_155102.json`
- `experiment_logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_no_abstain_20260322_155750.json`

Execution status:

- all four JSONs reported `n_errors: 0`
- all four JSONs include:
  - `intervention_coverage`
  - `prefix_ambiguity_rate`
  - `span_collapse_errors`
  - `abstention_rate`
  - `abort_trigger_rate`
  - `latency_split`

This means the first-host path is no longer blocked at the execution level; it has now produced the main universal row plus the critical ablation rows.

## F. Second-host universal batch partially completed

Host:

- `Qwen3-VL-4B-Instruct`
- `POPE random`

Recovered JSONs:

- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_no_gate_20260322_155450.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json`

Audit outcome:

- validator-approved:
  - `uniground_qwen3-vl-4b_pope_uniground_no_gate_20260322_155450.json`
- structurally rejected:
  - `uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json`
  - `uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json`

Current rejection signature:

- `abort_trigger_rate = 1.0`
- `abort_backoff_verified_steps = 0.0`

This means second-host portability is no longer hypothetical; it has reached the validator stage and is now blocked by one concrete runtime semantics bug rather than by missing assets.

## G. Current global audit snapshot

- accepted:
  - `7`
- rejected:
  - `9`

The accepted/rejected totals now include real second-host universal JSONs rather than only smoke artifacts.

## H. What is still not completed

Still not closed at formal acceptance level:

- validator-approved second-host `uniground`
- validator-approved second-host `uniground_global_only`
- first validator-approved cross-host hot-swap block containing:
  - first-host `uniground`
  - second-host `uniground`
  - same checkpoint sha
- formal first-host main table with interpreted metrics
- formal cross-host portability summary for the abstract / paper
