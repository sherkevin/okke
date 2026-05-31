# Real Rebuttal Run, 2026-05-30, LIMIT=8

Remote run dir:

```text
/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit8
```

Local mirror:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\real_runs\real_rebuttal_20260530_limit8
```

## Final State

The `adv8` LLaVA POPE run completed on the remote server at `2026-05-30T14:23:38+08:00` and has been fully mirrored locally.

- Target: `LIMIT=8`, `TAG=adv8`.
- GPU assignment: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,2,3,6`.
- Remote PID: `1269526`, now not running.
- Completed variants: `llava_pope_adv8_opera`, `llava_pope_adv8_same_anchor_non_chord`, `llava_pope_adv8_pc_real`, `llava_pope_adv8_full_k5_m3`, `llava_pope_adv8_full_k3_m2`, `llava_pope_adv8_full_k5_m1`, `llava_pope_adv8_full_k5_m2`, `llava_pope_adv8_full_k5_m4`, `llava_pope_adv8_full_k10_m3`, `llava_pope_adv8_pc_uniform`, `llava_pope_adv8_pc_random`, `llava_pope_adv8_past_future`.
- Local counts: 12 non-failed JSON files, 12 logs, 12 status files.
- Status verification: all 12 status files contain `0`; no `*.failed_*` or `*.traceback` files were present after final sync.
- Summary verification: `real_rebuttal_metrics_20260529.json` reports `tag=adv8`, `present_json_count=12`, `missing_json=[]`, and every measured table row has `n=8`.

## Verified Hashes

Remote and local SHA256 matched for the final summary artifacts:

- `real_rebuttal_metrics_20260529.json`: `b2d4e81c7e2f11f2ada5d49b6a8b41b00938c6741921285b6448af9972747178`
- `real_rebuttal_metrics_20260529.md`: `faef1d19cbc8de4ef2150f3f88b78b397b34ebc0d1b4ab5ee667ac3fe9b11105`
- `run_extended.log`: `07d63e5c5d4ae68202efa21c9bb5c6325d9d4e7cac9876b8adf9053c85aa3317`

Remote and local SHA256 also matched incrementally for the mirrored JSON results:

- `llava_pope_adv8_opera.json`: `928af9409b380ed8be10782445863d3488cc4830f2b423d4af6a1c7aca8008c1`
- `llava_pope_adv8_same_anchor_non_chord.json`: `f891b60af41b26c592663cfb0a44fdbf991d855ac1cba47159bb506c8e9c2fea`
- `llava_pope_adv8_pc_real.json`: `c0dd8f60d522a4e140fdd0ed1b0db373f9184768874c3dbcaf26901361ae4e25`
- `llava_pope_adv8_full_k5_m3.json`: `753ea2d2cbc25ec647e8522a055c4f8595ed574b6d91121883081a634cdbf0f0`
- `llava_pope_adv8_full_k3_m2.json`: `25e41ec76a445b160dc9c0800883b2a83aa7b70b790244b2c6489589b5659077`
- `llava_pope_adv8_full_k5_m1.json`: `f6658494f5d0628b567ef2c5efed06859a5d446d611dc3b5abd6923e9f635d87`
- `llava_pope_adv8_full_k5_m2.json`: `629937f7f39d982b61af5b09e222fce6ff5bd20f640a43a71566d235c86d4fc0`
- `llava_pope_adv8_full_k5_m4.json`: `9c6af851ce97c11e3cdd63ba6664d0dc7871cf63c616ddd2709dc9c52901b19d`
- `llava_pope_adv8_full_k10_m3.json`: `3114381d55558afbd62a08866028e603082a429c45a14b82a4d0b118c56aa2d8`
- `llava_pope_adv8_pc_uniform.json`: `0f6e4c133270240ac15915d6cbab6e523fd42b47023c9d554748dd8b373e3619`
- `llava_pope_adv8_pc_random.json`: `7686344e09f50876aef10ec8babcc52710497533b7ab4ebba29866bda308c37d`
- `llava_pope_adv8_past_future.json`: `5c71047a6d5acf6c6a8da92047a89ee8373c6f9d3b142d6effac89bf5a58ffe9`

## Boundary

This is a real 8-sample LLaVA POPE slice, not a replacement for any 3000/5000-sample PDF claim.

The summary intentionally preserves missing evidence boundaries:

- `InstructBLIP POPE-Adv`: not run in this LLaVA-focused remote pass.
- `LLaVA CHAIR`: CHAIR captioning was not measured by `pope_eval` JSON.
- `InstructBLIP CHAIR`: not run in this LLaVA-focused remote pass.

Do not insert CHAIR, CHAIR-S, InstructBLIP, batch-scaling, or full-split statistical claims into the rebuttal as real evidence from this run.
