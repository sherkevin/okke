# CHAIR Decoding Baseline — Experiment Matrix

Last updated: `2026-03-24`

**Scope now in queue:** `llava-v1.5-7b`, `qwen3-vl-8b`, `instructblip-7b`  
**Out of scope:** `qwen3-vl-4b`, `qwen3-vl-2b`  
**Methods:** `base`, `beam_search`, `dola`, `opera`  
**Count:** `mini_test=5000`, `chair_max_new_tokens=384`

**Remote parent queue:**

- log: `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.log`
- runner: `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.sh`
- watcher: `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.watch.sh`
- manifest: `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.manifest.tsv`

The watcher is configured to launch automatically after the current POPE GPU0 parent PID `6900` exits.

## Matrix

| Model | base | beam_search | dola | opera | Status |
| --- | --- | --- | --- | --- | --- |
| `llava-v1.5-7b` | ⏳ | ⏳ | ⏳ | ⏳ | scheduled |
| `qwen3-vl-8b` | ⏳ | ⏳ | ⏳ | ⏳ | scheduled |
| `instructblip-7b` | ⏳ | ⏳ | ⏳ | ⏳ | scheduled |

## Notes

- This CHAIR queue intentionally matches the current POPE baseline method set and excludes `vcd`.
- Result JSONs will land under `/root/autodl-tmp/BRA_Project/logs/minitest/` with names like `{method}_chair_<timestamp>.json`.
- Once rows finish, promote each manifest row into child entries in `experiment_logs/experiment_registry_latest.md`.
