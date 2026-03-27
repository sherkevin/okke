# UniGround Engineer Coordination Protocol (2026-03-22)

## Goal

This protocol removes human relay from the execution loop. Engineers do not need to message each other directly. They communicate only through fixed file locations and status manifests.

## Shared roots

- local result root:
  - `d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\uniground_v6\`
- remote result root:
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/`
- remote audit root:
  - `/root/autodl-tmp/BRA_Project/experiment_logs/uniground_v6/`

## Polling rule

- No engineer waits for a human relay message.
- All watch actions must be implemented as polling the corresponding feed file every `60-120` seconds.
- A feed change is defined as either:
  - new timestamp block appended; or
  - new JSON filename appearing in the latest block.

## Canonical checkpoint

- checkpoint:
  - `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- config:
  - `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json`
- sha256:
  - `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`

## Engineer 1 write contract

Engineer 1 writes first-host result JSONs to:

- remote:
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/`
- local mirror:
  - `d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\uniground_v6\`

After each batch, Engineer 1 must update:

- `experiment_logs/uniground_v6/first_host_batch_ready_20260322.md`

with:

- exact JSON filenames
- host model
- methods completed
- timestamp
- whether mirror to local succeeded

Engineer 1 GPU contract:

- primary GPU:
  - `GPU1`
- queue ownership:
  - first-host long runs only
- do not wait for validator feedback before starting the next queued first-host run unless the latest `audit_pass_fail_feed_20260322.md` explicitly marks a first-host method as structurally rejected for a runtime-contract reason that invalidates all subsequent runs.

## Engineer 3 watch contract

Engineer 3 does not wait for a chat message from Engineer 1.

Engineer 3 watches:

- `experiment_logs/uniground_v6/first_host_batch_ready_20260322.md`
- `experiment_logs/uniground_v6/second_host_batch_ready_20260322.md`

Trigger rule:

- if either file changes and lists new JSON filenames, Engineer 3 immediately runs batch audit on the full result directory
- Engineer 3 also watches:
  - `experiment_logs/uniground_v6/audit_pass_fail_feed_20260322.md`
  - `experiment_logs/uniground_v6/checkpoint_release_feed_20260322.md`

Audit command:

```bash
cd /root/autodl-tmp/BRA_Project && python audit_uniground_batch.py --summary-json experiment_logs/uniground_v6/universality_audit_latest.json --status-md experiment_logs/uniground_v6/universality_audit_latest.md
```

After audit, Engineer 3 must update:

- `experiment_logs/uniground_v6/universality_audit_latest.md`
- `experiment_logs/uniground_v6/universality_audit_latest.json`

and append one short pass/fail note to:

- `experiment_logs/uniground_v6/audit_pass_fail_feed_20260322.md`

Engineer 3 GPU contract:

- primary GPU:
  - `GPU3`
- secondary device:
  - `GPU0` only for very short smoke reruns or validator-coupled reproductions under `15` minutes
- queue ownership:
  - second-host long runs
  - batch audit after every new first-host or second-host batch

## Engineer 2 write contract

Engineer 2 owns canonical model artifacts and writes:

- checkpoint changes or replacement notices to:
  - `experiment_logs/uniground_v6/checkpoint_release_feed_20260322.md`

This file must include:

- checkpoint path
- sha256
- config path
- whether this is a replacement or the canonical unchanged artifact

Engineer 1 and Engineer 3 treat the latest entry in this file as the only valid checkpoint source.

## Engineer 2 watch and recovery contract

Engineer 2 watches:

- `experiment_logs/uniground_v6/audit_pass_fail_feed_20260322.md`
- `experiment_logs/uniground_v6/universality_audit_latest.md`
- `experiment_logs/uniground_v6/checkpoint_release_feed_20260322.md`

Trigger rule:

- if a new fail block appears with a concrete failure signature, Engineer 2 immediately becomes the recovery owner for that signature;
- the current highest-priority failure signature is:
  - `abort_trigger_rate > 0` together with `abort_backoff_verified_steps == 0`

Engineer 2 GPU contract:

- primary GPU:
  - `GPU2`
- secondary GPU:
  - `GPU0` for short reproductions, runtime sanity checks, and threshold / semantic fixes under `15` minutes
- queue ownership:
  - runtime bug recovery
  - checkpoint compatibility recovery
  - emergency reruns of the exact rejected method/host pair after a fix lands
  - only after recovery is stable may GPU2 be used for threshold sweep or light retraining

Engineer 2 must append a recovery note to:

- `experiment_logs/uniground_v6/checkpoint_release_feed_20260322.md`

with:

- whether checkpoint changed
- whether code/runtime changed
- which failure signature is being recovered
- exact rerun target host and method

## Engineer 3 second-host write contract

Engineer 3 writes second-host result JSONs to:

- remote:
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/`
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen2b/`

After each second-host batch, Engineer 3 updates:

- `experiment_logs/uniground_v6/second_host_batch_ready_20260322.md`

with:

- exact JSON filenames
- active second host
- methods completed
- timestamp
- whether audit was run immediately after append

## No-human-relay rule

- Engineer 1 never waits for Engineer 3 to be manually notified.
- Engineer 3 never waits for Engineer 1 to send a direct message.
- Engineer 2 never waits to be asked whether the checkpoint changed; the checkpoint feed file is the source of truth.
- The user should not need to manually forward result filenames between engineers.

## Continuous utilization rule

- `GPU1` should stay occupied by first-host formal runs or first-host formal reruns.
- `GPU3` should stay occupied by second-host formal runs or second-host formal reruns.
- `GPU2` should stay occupied by recovery work whenever the audit feed shows a rejected universal result.
- `GPU0` should be treated as a fast-turnaround smoke / reproduction card for the recovery owner, not as an idle reserve.
- Sequential dependence is allowed only at the level of a single host queue; it is not allowed across hosts, audit, and recovery.
