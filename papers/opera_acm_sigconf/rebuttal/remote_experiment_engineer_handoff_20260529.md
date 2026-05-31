# Remote Experiment Engineer Handoff, 2026-05-29

Scope: this handoff separates real SSH experiment execution from the author-response PDF optimization thread.

## Ownership

- Engineer window owns remote SSH execution, debugging, measured JSON/log collection, and final metric extraction.
- Current PDF thread owns only author-response wording, expected-result table shape, and replacement-ready LaTeX/PDF structure.

## Remote Access

```bash
ssh dengkw@10.103.16.12
```

Remote workspace:

```bash
/media/data3/dengkw/chord_rebuttal_20260529
```

Remote script:

```bash
/media/data3/dengkw/chord_rebuttal_20260529/logs/run_real_rebuttal_experiments_20260529.sh
```

Local audited copy:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\remote_scripts\run_real_rebuttal_experiments_20260529.sh
```

## Last Known Remote State

The old assistant-side monitor is paused. At last check, the run was downloading LLaVA weights.

PID file:

```bash
/media/data3/dengkw/chord_rebuttal_20260529/logs/run_real_rebuttal_experiments_20260529.pid
```

Main run log:

```bash
/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260529/run.log
```

Quick status command:

```bash
BASE=/media/data3/dengkw/chord_rebuttal_20260529
cd "$BASE"
pid=$(cat logs/run_real_rebuttal_experiments_20260529.pid 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "RUNNING $pid"; else echo "NOT_RUNNING ${pid:-none}"; fi
du -sh assets/models assets/models/llava-v1.5-7b assets/models/llava-v1.5-7b/.cache runs/real_rebuttal_20260529 2>/dev/null || true
tail -n 120 runs/real_rebuttal_20260529/run.log 2>/dev/null || true
```

## Expected Outputs To Produce

Minimum smoke outputs from the current script:

```bash
runs/real_rebuttal_20260529/anchors_pope_adv64.jsonl
runs/real_rebuttal_20260529/llava_pope_adv16_pc.json
runs/real_rebuttal_20260529/llava_pope_adv16_full.json
```

For final rebuttal replacement, the engineer should extend beyond smoke to the measured forms needed by:

1. Future mechanism table: Full vs P+C flip rate, corrected/harmful flips, POPE-Adv delta, CHAIR delta.
2. Detector attribution table: OPERA/Past, same-anchor non-CHORD, uniform/random anchors, Past+Future, P+C real anchors, Full real anchors.
3. Efficiency table: proposal ms, decode ITL, generated tokens, total ms/sample, peak VRAM, batch size.
4. k/m table: P+C, k3m2, k5m1, k5m2, k5m3, k5m4, k10m3 under the same split/protocol.

## Replacement Target In The PDF

Canonical expected-result PDF source:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260529.tex
```

Compatibility copy:

```text
D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\full_rebuttal_draft_20260529.tex
```

When real results arrive:

1. Replace superscript-`E` values in the four tables.
2. Remove all `\expected{}` marks.
3. Delete the four `Expected-result note:` lines under the tables.
4. If measured results do not match the expected pattern, downgrade the corresponding prose instead of forcing the claim.

## Important Boundary

Expected numbers in the current PDF are not measured results. They are a high-quality target/reference for replacement and reviewer-coverage planning only.
