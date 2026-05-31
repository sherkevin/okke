# Real Rebuttal Run, 2026-05-30, LIMIT=64

Remote run dir:

```text
/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit64
```

Local mirror:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\real_runs\real_rebuttal_20260530_limit64
```

## Current State

This is a partial mirror while the remote run is active.

- Remote PID: `1274377`.
- Launch time: `2026-05-30_14:39:36_+0800`.
- Target: `LIMIT=64`, `TAG=adv64`.
- GPU assignment: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,2,3,6`.
- Initial preflight: 64 POPE records, missing images `0`.
- Anchor files created: `anchors_pope_adv64.jsonl`, `anchors_pope_adv64_uniform.jsonl`, `anchors_pope_adv64_random_matched.jsonl`.
- Current variant after latest partial sync: `llava_pope_adv64_opera`.
- Current local partial files include `run_extended.log`, `llava_pope_adv64_opera.log`, and the three anchor JSONL files.
- Latest monitor/sync snapshot: `2026-05-30_16:32:30_+0800`, `llava_pope_adv64_opera` reached 52/64 samples without traceback or OOM; no JSON/status file has landed yet.
- Current local counts: 0 non-failed JSON files, 1 log, 0 status files.

## Reason For This Run

The completed `adv8` run does not match `author_response_min_diff_expected_20260529.tex`: LLaVA Full--P+C produced zero flips and zero F1 delta at `n=8`, while the expected table assumes a positive Full-vs-P+C effect on a 3000-sample split.

`adv64` is a predeclared larger LLaVA POPE slice to test whether the `adv8` zero-delta result is a small-sample artifact.

## Boundary

This run must not be used to force exact agreement with hypothetical expected values. It can only provide measured evidence.

Even if completed, it remains a LLaVA POPE slice. It does not measure InstructBLIP POPE, LLaVA/InstructBLIP CHAIR, CHAIR-S, batch scaling, or full 3000/5000-sample statistics.
