# Second-Host UniGround Execution Status (2026-03-22)

## Current rule

- The second host now uses `Qwen3-VL-4B-Instruct` first.
- If `Qwen3-VL-4B-Instruct` becomes unstable or unavailable, fall back to `Qwen3-VL-2B-Instruct`.
- The project should no longer be described as blocked on download.
- The only active primary blocker for formal `TLRA_univ` results is: no real `Psi_univ` checkpoint has been delivered yet.

## Fixed entry

- Runner wrapper: `run_second_host_uniground.py`
- Priority rule:
  - prefer `qwen3-vl-4b`
  - fallback `qwen3-vl-2b`
- Fixed output roots:
  - `logs/uniground_v6/second_host_qwen4b`
  - `logs/uniground_v6/second_host_qwen2b`

## Immediate base-path confirmation

- A fresh tiny second-host base smoke was executed through the fixed wrapper on `GPU3`.
- Result path:
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_base_20260322_151158.json`
- This confirms:
  - the second-host runner entry is live
  - `Qwen3-VL-4B-Instruct` is the active preferred host
  - the fixed second-host output root is stable

## Do not run before checkpoint arrival

- Do not spend `GPU3` on `uniground` long runs before a real `Psi_univ` checkpoint exists.
- Keep only the second-host base path and output root ready.

## Phase-2 command order after checkpoint arrival

Run on the second host in this order:

1. `uniground`
2. `uniground_no_gate`
3. `uniground_global_only` if time allows

Preferred command pattern:

```bash
cd /root/autodl-tmp/BRA_Project
CUDA_VISIBLE_DEVICES=3 python run_second_host_uniground.py \
  --method uniground \
  --psi_checkpoint /root/autodl-tmp/BRA_Project/models/uniground_v6/<REAL_PSI_UNIV>.pt
```

```bash
cd /root/autodl-tmp/BRA_Project
CUDA_VISIBLE_DEVICES=3 python run_second_host_uniground.py \
  --method uniground_no_gate \
  --psi_checkpoint /root/autodl-tmp/BRA_Project/models/uniground_v6/<REAL_PSI_UNIV>.pt
```

```bash
cd /root/autodl-tmp/BRA_Project
CUDA_VISIBLE_DEVICES=3 python run_second_host_uniground.py \
  --method uniground_global_only \
  --psi_checkpoint /root/autodl-tmp/BRA_Project/models/uniground_v6/<REAL_PSI_UNIV>.pt
```

## Mandatory batch audit after every new result batch

```bash
cd /root/autodl-tmp/BRA_Project && python audit_uniground_batch.py \
  --summary-json experiment_logs/uniground_v6/universality_audit_latest.json \
  --status-md experiment_logs/uniground_v6/universality_audit_latest.md
```

Required result fields:

- `universal_claim_manifest`
- `psi_univ checkpoint sha`
- `prefix_ambiguity_rate`
- `span_collapse_errors`
- `suffix_stability_rate`
- `abstention_rate`
- `abort_trigger_rate`
- `abort_backoff_verified_steps`
- `latency_split`

Reject immediately when:

- `abort_trigger_rate > 0` and `abort_backoff_verified_steps == 0`

## Reporting rule

- After every batch, return an explicit pass list and fail list.
- Do not report only an aggregate summary.
