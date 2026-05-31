# CHORD Actual Evidence Tables, 2026-05-28

Scope: real evidence collected locally during rebuttal execution. This document separates verified results from unavailable or blocked measurements.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record.

## Execution Status

Local GPU status:

- `nvidia-smi` is not available on this machine.
- Therefore, new GPU evaluations cannot be run locally.

Remote status:

- Remote preflight via `remote_exec.py` failed with `Connection closed by 198.18.0.138 port 47559`.
- Therefore, no new remote GPU diagnostics were launched in this pass.

Local code sanity:

- Command: `$env:PYTHONPATH='D:\Shervin\OneDrive\Desktop\breaking\remote_chiro_patch'; python -m pytest remote_chiro_patch/tests/test_chord_config.py remote_chiro_patch/tests/test_chord_current_score.py remote_chiro_patch/tests/test_future_rollout.py -q`
- Result: `8 passed in 2.40s`.

Available diagnostic evidence:

- Existing fixed-slice POPE ablation JSONs under `D:\Shervin\OneDrive\Desktop\breaking\remote_chiro_patch\tests\ablations_random_0_64`.
- Slice: POPE random, offset 0, limit 64.
- Boundary: this is a small fixed-slice diagnostic only. It is not sufficient as primary rebuttal evidence for full benchmark claims.

## Table 1: Future Rollout Mechanism, Existing 64-Sample POPE Diagnostic

| Comparison | Changed answers | Corrected errors | Introduced errors | Neutral changes | Interpretation |
|---|---:|---:|---:|---:|---|
| Full CHORD vs Past+Current | 1 / 64 | 0 | 1 | 0 | On this small POPE random slice, future did not provide positive answer-level evidence over Past+Current. |
| Full CHORD vs OPERA/Past | 0 / 64 | 0 | 0 | 0 | Full matched OPERA/Past answer decisions on this slice. |
| Full CHORD vs Future-only | 0 / 64 | 0 | 0 | 0 | Future-only and Full matched answer decisions on this slice. |
| Full CHORD vs alpha=0/no-GINO-weight | 0 / 64 | 0 | 0 | 0 | Disabling anchor weighting did not change answer decisions on this slice. |

Observed changed case:

| idx | Query | Label | Past+Current | Full CHORD | Effect |
|---:|---|---:|---|---|---|
| 30 | Is there a toaster in the image? | 0 | No, correct | Yes, wrong | Harmful Full-vs-P+C flip |

Boundary:

- This result should not be used as positive proof for the future mechanism.
- It supports a conservative rebuttal stance: Past+Current is the efficient POPE setting; Full CHORD's stronger role should be argued from the submitted CHAIR/open-ended results or from new CHAIR diagnostics, not from this small POPE slice.

## Table 2: Detector / Anchor Attribution, Existing 64-Sample POPE Diagnostic

| Variant | Acc | Precision | Recall | F1 | Yes ratio | TP/TN/FP/FN | Interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| OPERA/Past | 0.8281 | 0.8286 | 0.8529 | 0.8406 | 0.5469 | 29/24/6/5 | Reference on this slice. |
| Past+Current | 0.8438 | 0.8529 | 0.8529 | 0.8529 | 0.5312 | 29/25/5/5 | Best among checked slice variants; one fewer false positive. |
| Future-only | 0.8281 | 0.8286 | 0.8529 | 0.8406 | 0.5469 | 29/24/6/5 | Matches OPERA/Past. |
| Full CHORD | 0.8281 | 0.8286 | 0.8529 | 0.8406 | 0.5469 | 29/24/6/5 | Matches OPERA/Past and is below Past+Current on this slice. |
| Full, alpha=0/no-GINO-weight | 0.8281 | 0.8286 | 0.8529 | 0.8406 | 0.5469 | 29/24/6/5 | Same as Full on this slice; not enough to prove detector attribution either way. |
| Full, horizon=2 | 0.8438 | 0.8529 | 0.8529 | 0.8529 | 0.5312 | 29/25/5/5 | Shorter future horizon matches Past+Current on this slice. |
| Full, lambda_fut=0.05 | 0.8438 | 0.8529 | 0.8529 | 0.8529 | 0.5312 | 29/25/5/5 | Small future weight avoids the harmful flip observed in default Full. |

Anchor availability in Full CHORD slice:

| Number of anchors | Samples |
|---:|---:|
| 0 | 24 |
| 1 | 24 |
| 2 | 10 |
| 3 | 6 |

Boundary:

- This fixed-slice result does not answer reviewers' detector-attribution request completely.
- Missing controls: random anchors, same-anchor non-CHORD baseline, alternate proposer, full split, CHAIR/open-ended detector failure cases.

## Table 3: Cost Accounting, Available vs Missing

| Method / regime | Detector proposal time | Decode ITL | Total answer latency | Peak VRAM | Evidence status |
|---|---:|---:|---:|---:|---|
| Greedy, LLaVA-v1.5 | N/A | 19.73 ms/token | not measured | not measured | Submitted paper table only. |
| OPERA, LLaVA-v1.5 | N/A | 21.69 ms/token | not measured | not measured | Submitted paper table only. |
| CHORD Past+Current, LLaVA-v1.5 | not measured | 27.24 ms/token | not measured | not measured | Submitted paper table only. |
| Full CHORD, LLaVA-v1.5 | not measured | 37.31 ms/token | not measured | not measured | Submitted paper table only. |
| Greedy, InstructBLIP | N/A | 16.47 ms/token | not measured | not measured | Submitted paper table only. |
| OPERA, InstructBLIP | N/A | 16.90 ms/token | not measured | not measured | Submitted paper table only. |
| CHORD Past+Current, InstructBLIP | not measured | 24.51 ms/token | not measured | not measured | Submitted paper table only. |
| Full CHORD, InstructBLIP | not measured | 35.86 ms/token | not measured | not measured | Submitted paper table only. |

Boundary:

- Reviewers asked specifically whether latency is end-to-end and whether Grounding DINO proposal time is included.
- Existing submitted ITL does not fully answer that concern.
- Because local GPU is unavailable and remote SSH failed, no new detector-time, total-latency, or VRAM measurements were produced in this pass.

## Rebuttal Consequence

Safe claims from actual evidence:

- The current code's core CHORD scoring unit tests pass locally.
- Existing fixed-slice POPE diagnostics indicate Past+Current is more stable than default Full on this POPE slice.
- Existing submitted paper tables already support a two-regime interpretation: Past+Current is the latency-oriented setting; Full CHORD is quality-oriented in the main tables.

Unsafe claims:

- Do not claim new full-split future flip/correctness diagnostics.
- Do not claim detector attribution has been resolved.
- Do not claim end-to-end latency, detector proposal time, or VRAM have been measured.
- Do not use the 64-sample slice as primary acceptance evidence.
