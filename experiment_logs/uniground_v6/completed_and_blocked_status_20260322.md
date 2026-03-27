# UniGround V6 Completed / Blocked Status (2026-03-22)

## 1. What Is Already Completed

### 1.1 Route split and runner closure

- `UniGround` has been physically separated from the old internal pipeline.
- New universal runtime exists: `uniground_runtime.py`
- New universal eval entry exists: `run_uniground_eval.py`
- Universal methods now have an isolated output root: `logs/uniground_v6/`
- Supported method contract now includes:
  - `base`
  - `tlra_internal_zero`
  - `tlra_internal_calib`
  - `uniground`
  - `external_global_prior`
  - `uniground_no_gate`
  - `uniground_no_abstain`
  - `uniground_global_only`
  - `uniground_region_only`
- `qwen3-vl-4b` has now been wired into the universal route model registry and eval entry.

### 1.2 First smoke results already completed

The following first-host smoke results have already been reported as completed through the new UniGround entry on `Qwen3-VL-8B`:

- `base`
- `tlra_internal_zero`
- `tlra_internal_calib`

Observed reported outputs:

- primary hallucination metric
- `ITL / TPOT`
- `peak_vram`
- reserved universal audit slots:
  - `intervention_coverage`
  - `prefix_ambiguity_rate`
  - `span_collapse_errors`
  - `abstention_rate`
  - `abort_trigger_rate`

### 1.3 First-host formal mini-runs now recovered

The following `POPE random`, `mini_test=32`, `Qwen3-VL-8B` mini-runs are now recovered through the new UniGround entry:

- `base`
  - `f1=0.9333`
  - `ITL/TPOT=22.17 ms/token`
  - `peak_vram=17.637 GB`
- `tlra_internal_zero`
  - `f1=0.9333`
  - `ITL/TPOT=21.32 ms/token`
  - `peak_vram=17.637 GB`
- `tlra_internal_calib`
  - `f1=0.9333`
  - `ITL/TPOT=21.28 ms/token`
  - `peak_vram=17.637 GB`
- `external_global_prior`
  - `f1=0.9333`
  - `ITL/TPOT=71.9 ms/token`
  - `peak_vram=17.637 GB`
  - `intervention_coverage=1.0`
  - `prefix_ambiguity_rate=0.5574`
  - `span_collapse_errors=103.4062`
  - `abstention_rate=0.0`
  - `abort_trigger_rate=0.0`
  - `candidate_construction_ms=0.9772`
  - `sidecar_scoring_ms=46.7546`
  - `bridge_redistribution_ms=0.0546`
  - `jitter_ms=17.3336`

These runs mean the first-host internal controls and the first-host `External_Global_Prior` baseline are no longer merely planned; they now exist as recovered mini-run assets.

### 1.4 Universal-contract side is already closed

The following contract / validator assets are already completed:

- `PSI_UNIV_TRAINING_CONTRACT_20260322.md`
- `UNIGROUND_UNIVERSALITY_CHECKLIST_20260322.md`
- `UNIVERSAL_CLAIM_FAILURE_AUDIT_20260322.md`
- `validate_uniground_universality.py`
- `audit_uniground_batch.py`

Completed smoke assets for the validator side:

- synthetic config dump:
  - `logs/v3_contract/psi_univ_synthetic_smoke.config.json`
- synthetic result:
  - `logs/v3_contract/uniground_result_synthetic_smoke.json`

This means the universality contract, result schema, and `abstention -> abort` validator logic have already been exercised at least on a synthetic smoke path.

### 1.5 Additional engineering closures now completed

- The canonical external encoder directory has been uploaded in full to:
  - `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`
- The uploaded directory now includes the complete downloaded checkpoint family rather than only a minimal inference subset.
- `export_universal_coco_payload.py` now exists to export frozen external-space payloads containing:
  - `image_embeddings`
  - `candidate_embeddings`
  - `prefix_embeddings`
  - `labels`
  - `metadata`
- `train_universal_plugin.py` has been enhanced so checkpoint contracts, config dumps, checkpoint sha256, payload hash, and source hashes are emitted together.
- `qwen3-vl-4b` second-host base smoke has already passed on the remote side.
- Batch universality audit entrypoints now exist:
  - `audit_uniground_batch.py`
  - `experiment_logs/uniground_v6/universality_audit_latest.md`
  - `experiment_logs/uniground_v6/universality_audit_latest.json`

### 1.6 First real `Psi_univ` artifact is now available

The project is no longer blocked on checkpoint non-existence.

Recovered artifact:

- checkpoint:
  - `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- config dump:
  - `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json`
- checkpoint sha256:
  - `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`

Training payload:

- payload:
  - `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt`
- payload config:
  - `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.config.json`
- payload sha256:
  - `0722c23467469f3dfef7db1216ca0e216d839ea5e2503fb212f3f52828510e67`

Validation status:

- `load_universal_scorer(...)`: pass
- `validate_uniground_universality.py --checkpoint ...`: pass

This means the universal route is now blocked on evaluation closure, not on checkpoint absence.

## 2. What Is Truly Blocking

### Blocker A. Canonical external encoder weights

This blocker is now resolved for the first-host path.

Recovered asset:

- `openai/clip-vit-large-patch14`
- uploaded remote path:
  - `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`

Impact:

- `external_global_prior` can now be run honestly on the first host.
- The external encoder is no longer the primary blocking asset for `3.25`-oriented first-host evidence.

### Blocker B. Formal `TLRA_univ` result recovery has now begun, but cross-host acceptance is not yet closed

The following real-shared-checkpoint rows have now been recovered:

- first-host `uniground`
- first-host `uniground_no_gate`
- first-host `uniground_global_only`
- first-host `uniground_no_abstain`
- second-host `uniground`
- second-host `uniground_no_gate`
- second-host `uniground_global_only`

What is still missing is not execution existence, but validator-approved closure across both hosts.

### Blocker C. Reproducible train-to-eval handoff still needs validator-approved end-to-end closure

The feature-export, train, and checkpoint chain now exists and has produced a real artifact. What remains is operational closure:

- first-host batch audit pass on the resulting `TLRA_univ` JSONs
- second-host `uniground` audit pass with the same checkpoint
- second-host `uniground_global_only` audit pass with the same checkpoint

This is no longer a build blocker. It is now a run-and-verify blocker.

### Blocker D. Second-host portability is blocked by one concrete runtime semantics failure

The runner / registry gap is now resolved at the code level:

- `qwen3-vl-8b`
- `qwen3-vl-4b`
- `qwen3-vl-2b`

However, second-host **universal** portability is still not closed, because the latest real-shared-checkpoint second-host runs show the following rejection signature:

- `abort_trigger_rate > 0`
- `abort_backoff_verified_steps == 0`

This means the remaining blocker is a runtime semantics mismatch in the `abstention -> abort/back-off` path, not missing assets.

## 3. What Is NOT Currently Blocked By Download

The following pieces are not the main blockers right now:

- `POPE` mainline evaluation
- first-host internal controls on `Qwen3-VL-8B`
- first-host `External_Global_Prior`
- universality validator logic
- result schema and runtime audit fields

In other words, the project is **not** broadly blocked on benchmark datasets. It is blocked on a small number of critical universal-route assets.

## 4. What The User Can Download And Upload Immediately

At this point, the user does **not** need to spend time downloading the canonical external encoder for the first-host path; that asset has already been recovered.

Potential user-side download is now secondary:

1. `Qwen/Qwen3-VL-4B-Instruct`

This is only necessary if the remote does not already have a stable copy for the second-host portability run.

## 5. What Engineers Can Continue Running Right Now

### Can run now

- `base` on the new UniGround runner
- `tlra_internal_zero` on the new UniGround runner
- `tlra_internal_calib` on the new UniGround runner
- `external_global_prior` on the new UniGround runner
- validator checks on any future result JSONs
- first-host `uniground` / ablation recovery with the real checkpoint
- second-host base / plumbing verification on `qwen3-vl-4b`
- second-host `uniground` execution with the shared checkpoint
- batch validator / audit refresh on every new `TLRA_univ` result
- runtime recovery for `abort_backoff_verified_steps` semantics on rejected second-host rows

### Not yet recovered

- first-host `uniground`
- first-host `uniground_no_gate`
- first-host `uniground_global_only`
- first formal 5-way UniGround comparison table
- cross-host hot-swap verification with the shared real `Psi_univ`

## 6. Immediate Coordination Rule

The universal route should now be treated as blocked primarily on exactly two acceptance closures:

1. validator-approved second-host `TLRA_univ` result recovery with the real shared checkpoint
2. validator-approved second-host `TLRA_univ_global_only` result recovery with the same checkpoint

Everything else should either:

- keep running if it does not depend on these assets, or
- be labeled as implementation closure rather than benchmark download work.
