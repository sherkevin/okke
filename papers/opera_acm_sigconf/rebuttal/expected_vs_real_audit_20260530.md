# Expected-vs-Real Audit, 2026-05-30

Scope: compare `author_response_min_diff_expected_20260529.tex` against the completed real run `real_rebuttal_20260530_limit8`, then define the next legitimate rerun.

## Verdict

The completed `adv8` run is not consistent with the expected tables.

The mismatch is structural, not rounding noise:

- The expected PDF table claims LLaVA POPE-Adv Full--P+C at `N=3000`, `flip_rate=4.1%`, corrected/harmful flips `96 / 42`, and `+0.013 Adv. F1`.
- The measured `adv8` summary reports LLaVA POPE-Adv Full--P+C at `n=8`, `flip_rate=0.0`, `flips=0`, `corrected=0`, `harmful=0`, and `metric_delta=0.0`.
- The expected attribution and k/m tables assume a monotonic quality pattern from OPERA/Past to P+C to Full, with distinct F1 and FP-rate values.
- The measured `adv8` summary reports identical POPE F1 `0.888888888888889` and FP rate `0.25` for all measured LLaVA POPE variants.
- Expected CHAIR-S rows are not measured by the current POPE pipeline. The real summary correctly leaves CHAIR-S as `null`.
- Expected InstructBLIP rows are not measured. The real summary correctly marks them absent.
- Expected latency is sub-second per sample. The measured remote 8-bit four-2080Ti path is roughly 128--137 seconds per sample for the LLaVA POPE variants.

## Real Evidence Used

Completed run:

```text
/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit8
```

Local mirror:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\real_runs\real_rebuttal_20260530_limit8
```

Verification:

- 12 non-failed JSON files, 12 logs, and 12 status files.
- All status files contain `0`.
- `real_rebuttal_metrics_20260529.json` reports `tag=adv8`, `present_json_count=12`, `missing_json=[]`.
- All measured rows have `n=8`.
- Remote/local SHA256 matched for all 12 raw JSON files and the summary artifacts.

## What This Means For The Expected PDF

The expected values in `author_response_min_diff_expected_20260529.tex` must not be used as real results.

If later real runs remain inconsistent, the correct rebuttal action is to replace or downgrade the affected claims, not to keep rerunning until a hand-picked run exactly matches the expected table.

## Next Legitimate Run

The next run has been predeclared and launched to test whether `adv8` is only a small-sample artifact:

```text
LIMIT=64
TAG=adv64
RUN_DIR=/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit64
CUDA_VISIBLE_DEVICES=0,2,3,6
PID=1274377
```

Launch time:

```text
2026-05-30_14:39:36_+0800
```

Initial evidence:

- 64 POPE records found; missing images: `0`.
- `anchors_pope_adv64.jsonl` created with 64 entries.
- Uniform and random-matched anchor controls created.
- First variant started: `llava_pope_adv64_opera`.
- Local partial mirror created under:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\real_runs\real_rebuttal_20260530_limit64
```

Acceptance criteria for `adv64`:

- 12 non-failed `llava_pope_adv64_*.json` files.
- 12 corresponding logs and 12 status files, all status `0`.
- `real_rebuttal_metrics_20260529.json/.md` present.
- Summary reports `tag=adv64`, `present_json_count=12`, `missing_json=[]`.
- Local and remote hashes match for summary artifacts and raw JSON files.
- The comparison against expected tables is reported honestly, including any contradiction.

## Boundary

`adv64` is still a LLaVA POPE slice. It does not measure InstructBLIP POPE, LLaVA/InstructBLIP CHAIR, CHAIR-S, batch-scaling, or full 3000/5000-sample statistics.
