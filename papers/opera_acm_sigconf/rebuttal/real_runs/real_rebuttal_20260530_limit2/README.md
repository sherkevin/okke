# Real Rebuttal Run, 2026-05-30, LIMIT=2

Remote run dir:

```text
/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit2
```

Local mirror:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\real_runs\real_rebuttal_20260530_limit2
```

## Configuration

- Model: LLaVA-1.5-7B local checkpoint.
- Task: POPE adversarial, first 2 records only.
- Decode: beam 5, max new tokens 20, batch size 1.
- GPU path: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,2,3,6`.
- Load path: 8-bit LLaVA with `model.device_map=balanced_low_0`.
- Runner: `logs/run_extended_real_rebuttal_experiments_20260529.sh`.
- Summarizer: `logs/summarize_real_rebuttal_metrics_20260529.py --tag adv2`.

## Outputs

- 12 / 12 LLaVA POPE JSON files succeeded after the dtype fix.
- Summary JSON: `real_rebuttal_metrics_20260529.json`.
- Summary Markdown: `real_rebuttal_metrics_20260529.md`.
- Raw logs and status files are mirrored beside the JSON files.
- The first `pc_real` attempt failed with `scatter()` dtype mismatch and is kept as `*.failed_dtype_20260530_0139.*` forensic evidence only.

## Boundary

This run proves the real JSON/log pipeline and fills LLaVA POPE rows for the configured 2-sample slice. It is not a PDF-replacement run for the expected 3000/5000-sample tables.

Do not insert these values into `author_response_min_diff_expected_20260529.tex` as final rebuttal evidence. The missing or non-replaceable parts are:

- InstructBLIP POPE rows.
- LLaVA/InstructBLIP CHAIR rows.
- CHAIR-S columns in detector and k/m tables.
- Any full-split statistical claim beyond `n=2`.

## Verification Snapshot

- `present_json_count`: 12.
- `missing_json`: none for the LLaVA POPE runner.
- LLaVA POPE Full vs P+C: `n=2`, `flip_rate=0.0`, `corrected=0`, `harmful=0`, `metric_delta=0.0`.
- Anchor precompute: `limit=2`, `status=0`, `mean_detector_ms=8407.0`.
