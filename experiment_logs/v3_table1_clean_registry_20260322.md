# Table 1 Clean Asset Registry (2026-03-22)

## Scope

- Goal: close `Table 1` using only the clean `GPU1` Chain A sequence.
- Fixed order:
  1. `POPE base`
  2. `POPE vcd`
  3. `POPE dola`
  4. `POPE tlra_zero`
  5. `CHAIR base`
  6. `CHAIR vcd`
  7. `CHAIR dola`
  8. `CHAIR tlra_zero`
- Policy: `PHI_FROZEN=false`; no `TLRA_calib` main-table runs.
- Policy: previously captured `POPE/CHAIR` JSONs with known OOM contamination are archive-only and cannot be imported into the main table.

## Clean Asset Status

| Slot | Latest file | Clean | Main-table importable | Notes |
| --- | --- | --- | --- | --- |
| `POPE base` | pending clean 8B JSON | no | no | Current clean `GPU1` process is still running; prior `base_pope_20260322_092654.json` is OOM-contaminated. A newly observed `base_pope_20260322_102922.json` is clean but `model=qwen3-vl-2b`, so it is out-of-scope for this 8B Table 1 chain. |
| `POPE vcd` | pending clean JSON | no | no | Prior `vcd_pope_20260322_092715.json` is OOM-contaminated. |
| `POPE dola` | pending clean JSON | no | no | Prior `dola_pope_20260322_092739.json` is OOM-contaminated. |
| `POPE tlra_zero` | pending clean JSON | no | no | Prior `tlra_zero_pope_20260322_092755.json` is OOM-contaminated. |
| `CHAIR base` | pending clean JSON | no | no | Prior `base_chair_20260322_092816.json` is OOM-contaminated. |
| `CHAIR vcd` | pending clean JSON | no | no | Not yet produced by the clean chain. |
| `CHAIR dola` | pending clean JSON | no | no | Not yet produced by the clean chain. |
| `CHAIR tlra_zero` | pending clean JSON | no | no | Not yet produced by the clean chain. |

## Archived Contaminated Files

- `experiment_logs/remote_mirror/v3_engineer_a/base_pope_20260322_092654.json`
- `experiment_logs/remote_mirror/v3_engineer_a/vcd_pope_20260322_092715.json`
- `experiment_logs/remote_mirror/v3_engineer_a/dola_pope_20260322_092739.json`
- `experiment_logs/remote_mirror/v3_engineer_a/tlra_zero_pope_20260322_092755.json`
- `experiment_logs/remote_mirror/v3_engineer_a/base_chair_20260322_092816.json`

Reason: these files were generated while stale parallel launchers were still occupying `GPU1`, and their JSON contents show `sample_count=0` / `n_errors>0` OOM failure traces rather than clean comparable runs.

## Observed But Not Table-1-Importable

- `experiment_logs/remote_mirror/v3_engineer_a/base_pope_20260322_102922.json`

Reason: this file is clean and already contains both `agl_stddev` and `peak_vram_gb`, but it reports `model=qwen3-vl-2b`. The active clean Table 1 chain is the `qwen3-vl-8b` sequence on `GPU1`, so this file cannot be used to certify or populate the 8B main table.

## Non-Table-1 Assets Already Clean

- `stage0_tlra_zero_8b.json`: clean, appendix / Stage 0 evidence.
- `mmbench_tlra_full.json`: clean, can support Chain B evidence.
- `mmbench_tlra_no_vasm.json`: clean, can support Chain B evidence.
- `mme_tlra_full.json`: clean, can support Chain B evidence.
- `mme_tlra_no_vasm.json`: clean, can support Chain B evidence.
- `mmmu_tlra_full.json`: clean, can support Chain B evidence.
- `mmmu_tlra_no_vasm.json`: clean, can support Chain B evidence.

## Table-Field Readiness

- Already present in clean non-Table-1 assets:
  - `accuracy`
  - `agl`
  - `itl_ms_per_token`
  - `tokens_per_second`
  - `avg_candidate_window`
  - `avg_visual_topk`
  - `avg_resonance_time_ms`
  - `intervention_rate`
- Confirmed emitted by the current code path:
  - `agl_stddev`
  - `peak_vram_gb`
- Not yet available from clean `Table 1` runs:
  - clean `POPE` 8B metrics from the active `GPU1` chain
  - `CHAIR` metrics
  - first clean `POPE/CHAIR` 8B JSON proving the active chain is writing these fields

## Live Chain A State

- Clean chain root: `42419`
- Active child at last check: `42422`
- Active step at last clean log snapshot: `POPE base`
- Last confirmed progress line in `chainA_main.log`: `[30/200]`
