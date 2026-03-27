# GPU3 Fast Smoke Loop (2026-03-22)

## Goal

Use `GPU3` as a rapid optimization lane for `UniGround`, with the sole objective of finding a configuration or runtime patch that beats both `base` and `uniground_global_only` on a small but informative slice before scaling up.

## Fixed assets

- host model:
  - `qwen3-vl-4b` preferred
- checkpoint:
  - `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- external encoder:
  - `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`
- runner:
  - `run_uniground_eval.py`
- dataset:
  - `POPE random`

## Smoke phase 0: establish current baseline

Run on the same slice:

1. `base`
2. `uniground`
3. `uniground_global_only`

Recommended command pattern:

```bash
cd /root/autodl-tmp/BRA_Project && python run_uniground_eval.py --model qwen3-vl-4b --dataset pope --pope_split random --mini_test 64 --method uniground --psi_checkpoint /root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt --external_encoder /root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14 --run_tag gpu3_baseline
```

## Smoke phase 1: fast knob sweep

Run these small variants on the same host and same slice:

1. `centered` actuation
   - `--bias_mode centered`
2. `host_relative` actuation
   - `--bias_mode host_relative --host_relative_strength 4.0`
3. wider frontier
   - `--top_k 80`
4. lower ambiguity abort sensitivity
   - `--ambiguity_abort_threshold 0.80`
5. fully disable ambiguity abort for one probe
   - `--no_abort_on_prefix_ambiguity`
6. lower abstention
   - `--abstain_threshold 0.75`
7. stronger redistribution
   - `--bias_scale 0.60`

Recommended tags:

- `gpu3_centered`
- `gpu3_hostrel`
- `gpu3_topk80`
- `gpu3_amb080`
- `gpu3_noambabort`
- `gpu3_abs075`
- `gpu3_bias060`

## Promotion rule from phase 1

Promote a config immediately if it satisfies either:

1. `hallucination_primary_value` is strictly better than both:
   - `base`
   - `uniground_global_only`
2. `hallucination_primary_value` ties the best baseline, but:
   - `intervention_coverage` is meaningfully higher; and
   - `abort_trigger_rate` is meaningfully lower; and
   - the qualitative yes/no outputs differ on at least one audited sample

## Escalation rule to phase 2

Escalate immediately if all phase-1 variants still show:

- the same primary metric as `base`; and
- no clear separation from `uniground_global_only`

## Smoke phase 2: stronger actuation check

Use the best phase-1 config as the base, then test:

1. `--bias_mode host_relative --host_relative_strength 8.0`
2. `--bias_mode host_relative --host_relative_strength 12.0`
3. combine stronger actuation with wider frontier:
   - `--bias_mode host_relative --host_relative_strength 8.0 --top_k 80`
4. combine stronger actuation with reduced aborting:
   - `--bias_mode host_relative --host_relative_strength 8.0 --ambiguity_abort_threshold 0.80`

## Promotion rule from phase 2

Pick the best-performing branch if:

- it beats `uniground_global_only`; or
- it produces the first clear metric separation from `base`

## Escalation rule to phase 3

If phase 2 still fails, the next move is not more random sweeps. It is a stronger code-level branch, such as:

- host-aware threshold calibration
- stronger host-relative redistribution
- hybrid performance-first actuation logic

## Output recording

Every smoke run must preserve:

- JSON output file
- `run_tag`
- `runtime_knobs`

This is required so later comparisons are not lost in timestamp-only filenames.
