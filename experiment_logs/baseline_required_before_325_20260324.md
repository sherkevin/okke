# Baseline Gap Audit Before 3.25

Last updated: `2026-03-24`

Source of truth:

- `论文方法章节_完整版_IFCB_20260324.md`
- `experiment_logs/experiment_registry_latest.md`
- `experiment_logs/llava_pope_baseline_main_table_20260323.md`

## 1. What counts as baseline before 3.25

Only the baseline-related parts of the pre-deadline program are counted here.

The required baseline package in the methodology draft is:

- `Base`
- `VCD`
- `DAMO`

For `3.25`, the baseline-relevant requirements are interpreted as:

1. `LLaVA-1.5 / POPE / random+popular+adversarial`:
   `Base + VCD + DAMO`
2. `LLaVA-1.5 / CHAIR`:
   at least `Base`, preferably also `VCD + DAMO`
3. `InstructBLIP / POPE / random+popular+adversarial`:
   at least the core baseline anchor; preferred package is `Base + VCD + DAMO`

Not treated as baseline blockers before `3.25` in this file:

- full `HallusionBench` baseline matrix
- full `MME` / `MMBench` baseline matrix
- optional `DeCo`
- factor ablations
- polished full runtime table

## 2. Current completed baseline assets that still matter

### LLaVA-1.5 / POPE / Base

Already complete on all three official splits:

- `random`  -> `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_140310.json`
- `popular` -> `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_155904.json`
- `adversarial` -> `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_175141.json`

These runs remain valid because `Base` is part of the required baseline package.

### Stopped queue assets recovered from the wrong-direction Qwen run

These are preserved for traceability but are **out of scope** for the `3.25` baseline package:

- `/root/autodl-tmp/BRA_Project/logs/minitest/beam_search_pope_20260323_222013.json`
- `/root/autodl-tmp/BRA_Project/logs/minitest/dola_pope_20260324_022644.json`
- `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260324_041653.json`
- `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260324_054629.json`
- `/root/autodl-tmp/BRA_Project/logs/minitest/beam_search_pope_20260324_073528.json`
- `/root/autodl-tmp/BRA_Project/logs/minitest/dola_pope_20260324_113637.json`
- parent log: `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.log`
- manifest: `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.manifest.tsv`

### DAMO smoke assets

Method integration smoke tests succeeded after GitHub import:

- `llava-v1.5-7b / pope / random / damo / mini_test=2`
  -> `/root/autodl-tmp/BRA_Project/logs/minitest/damo_pope_20260324_123557.json`
- `instructblip-7b / pope / random / damo / mini_test=2`
  -> `/root/autodl-tmp/BRA_Project/logs/minitest/damo_pope_20260324_123619.json`

## 3. Missing baseline cells before 3.25

### A. LLaVA-1.5 / POPE

Completed:

- `Base` on `random`, `popular`, `adversarial`

Still missing:

- `VCD` on `random`, `popular`, `adversarial`
- `DAMO` on `random`, `popular`, `adversarial`

Total missing cells: **6**

### B. LLaVA-1.5 / CHAIR

Completed:

- none under the required `Base + VCD + DAMO` package

Still missing:

- `Base`
- `VCD`
- `DAMO`

Total missing cells: **3**

### C. InstructBLIP / POPE

Completed:

- none under the required package

Still missing:

- `Base` on `random`, `popular`, `adversarial`
- `VCD` on `random`, `popular`, `adversarial`
- `DAMO` on `random`, `popular`, `adversarial`

Total missing cells: **9**

## 4. Immediate GPU0 queue to launch

Recommended requeue after stopping the old wrong-direction baseline jobs:

1. `llava-v1.5-7b / pope / random / vcd`
2. `llava-v1.5-7b / pope / random / damo`
3. `llava-v1.5-7b / pope / popular / vcd`
4. `llava-v1.5-7b / pope / popular / damo`
5. `llava-v1.5-7b / pope / adversarial / vcd`
6. `llava-v1.5-7b / pope / adversarial / damo`
7. `llava-v1.5-7b / chair / base`
8. `llava-v1.5-7b / chair / vcd`
9. `llava-v1.5-7b / chair / damo`
10. `instructblip-7b / pope / random / base`
11. `instructblip-7b / pope / random / vcd`
12. `instructblip-7b / pope / random / damo`
13. `instructblip-7b / pope / popular / base`
14. `instructblip-7b / pope / popular / vcd`
15. `instructblip-7b / pope / popular / damo`
16. `instructblip-7b / pope / adversarial / base`
17. `instructblip-7b / pope / adversarial / vcd`
18. `instructblip-7b / pope / adversarial / damo`

Total queued cells: **18**
