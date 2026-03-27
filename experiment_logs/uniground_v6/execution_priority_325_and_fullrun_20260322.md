# UniGround Execution Priority for 3.25 and Full-Run Ramp (2026-03-22)

## Core objective

This checklist separates the work into two layers:

1. the minimum evidence required before the `3.25` abstract deadline;
2. the engineering groundwork required to launch the later full-run phase without stalling again.

The guiding principle is simple:

- do not wait for the full universal route before collecting all non-blocked evidence;
- do not leave GPUs idle when internal controls, global-prior baselines, portability plumbing, or `Psi_univ` training closure can already be advanced in parallel.

## A. What must be closed before 3.25

### A1. First-host mini-run evidence must already exist and be preserved

Required status:

- `Qwen3-VL-8B`
- `POPE random`
- `mini_test=32`
- completed:
  - `base`
  - `tlra_internal_zero`
  - `tlra_internal_calib`
  - `external_global_prior`

Reason:

- this is the current minimum comparison substrate for the abstract;
- these runs establish that the new UniGround runner is real and that the first-host global-prior baseline is no longer hypothetical.

### A2. The first real `Psi_univ` checkpoint now exists

Recovered artifact:

- `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- sha256:
  - `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`

This means the `3.25` critical path has advanced to the next stage:

`run first-host uniground -> validate -> run second-host uniground -> validate -> summarize portability`

### A3. First-host formal comparison target for 3.25

Immediately rerun on the same host and same split:

- `base`
- `tlra_internal_zero`
- `tlra_internal_calib`
- `external_global_prior`
- `uniground`

These five rows form the minimum first-host evidence block for the abstract, and four of them are already recovered.

### A4. Second-host portability target for 3.25

Immediately using the same `Psi_univ` checkpoint:

- run `Qwen3-VL-2B` first if it is already stable;
- run `Qwen3-VL-4B` if the model registry and remote asset are stable;
- if `4B` is unavailable, report that explicitly and do not fake heterogeneity.

The output must at least establish:

- same checkpoint reused;
- no learned re-attach;
- same external encoder family;
- same validator contract.

## B. What should keep GPUs busy immediately

### GPU1

Purpose:

- first-host result recovery and formal mini-runs

Immediate queue:

1. preserve and mirror all first-host JSONs locally;
2. keep `base`, `tlra_internal_zero`, `tlra_internal_calib`, `external_global_prior` ready on `Qwen3-VL-8B`;
3. immediately run first-host `uniground`;
4. then run first-host `uniground_no_gate`;
5. then run first-host `uniground_global_only`.

### GPU2

Purpose:

- fallback training / rerun card and ablation overflow card

Immediate queue:

1. preserve the current checkpoint, config dump, and payload hashes as the canonical artifact set;
2. if first-host or second-host universal runs expose instability, use this card for fast rerun / threshold sweep / emergency checkpoint retrain;
3. otherwise prepare the next ablation-ready checkpoint or payload variant only after the first host and second host main runs are in motion.

### GPU3

Purpose:

- second-host portability plumbing

Immediate queue:

1. verify `qwen3-vl-4b` availability on the remote;
2. if available, keep it wired into `run_uniground_eval.py`;
3. otherwise stabilize `qwen3-vl-2b` as the first second-host path;
4. immediately run second-host `uniground`;
5. then run second-host `uniground_no_gate`;
6. then run second-host `uniground_global_only` if time allows.

### GPU0

Purpose:

- lightweight validation and short-smoke work only

Immediate queue:

1. validator checks on every new `TLRA_univ` result;
2. short checkpoint smoke if needed;
3. no long benchmark occupation unless the other cards are saturated.

## C. What is actually blocking, and what is not

### True blocker

- first-host `uniground` result has not yet been recovered with the real checkpoint
- second-host hot-swap result has not yet been recovered with the same checkpoint
- validator-approved real `TLRA_univ` batch is still missing

### No longer first-order blocker

- canonical `CLIP ViT-L/14` external encoder for first-host path
- first-host `External_Global_Prior`
- first-host internal controls
- benchmark dataset download in general

### Conditional blocker

- `Qwen3-VL-4B-Instruct` only matters if the remote lacks a stable copy and we need it for the second-host path

## D. What the user may still need to download manually

At this point, there is only one user-side download worth considering:

- `Qwen/Qwen3-VL-4B-Instruct`

And even that is needed only if the remote does not already contain a stable usable copy.

The user no longer needs to spend time on the first-host canonical external encoder.

## E. Minimum abstract-ready evidence package

Before the `3.25` abstract, the project should aim to have:

1. first-host mini-runs for:
   - `base`
   - `tlra_internal_zero`
   - `tlra_internal_calib`
   - `external_global_prior`
   - `uniground`
2. one real shared `Psi_univ` checkpoint
3. one validator-approved `TLRA_univ` result JSON
4. one second-host portability run with the same checkpoint
5. one short portability note stating whether the evidence is:
   - strong universal,
   - portable but weaker than internal,
   - or not yet sufficient

## F. Operational rule

The project should no longer be described as blocked on checkpoint creation or on downloads.

The project is now primarily blocked on **result recovery with the real shared checkpoint**, while both hosts and the validator can already be advanced in parallel.
