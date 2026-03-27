# Theory-Aligned Engineer Context Package (2026-03-23)

## 1. Mission

Your job is **not** to invent a new method.

Your job is to implement and verify the method already fixed in:

- `论文方法章节_完整版_VEB_UniGround_20260323.md`

The hard requirement is:

> engineering must satisfy the latest theory, instead of drifting back into a benchmark-specific POPE patcher or a host-internal prior manipulation method.

This package tells you:

1. what the current theory actually requires,
2. what the current code already satisfies,
3. what is still incomplete or only partially aligned,
4. what you should do next.

## 2. Non-Negotiable Theory Contract

Treat the following as fixed.

### 2.1 The method is an external evidence bottleneck

The method should be understood as:

- parse task query into a semantic object of verification,
- retrieve object-conditioned visual evidence,
- evaluate paired hypotheses through a universal scorer,
- constrain the final answer decision through external evidence,
- back off when evidence is weak.

It is **not**:

- internal prior extraction,
- internal attention rewriting,
- a host-specific answer head,
- a POPE-only classifier.

### 2.2 `Psi_univ` must remain universal

`Psi_univ` is:

- a universal external evidence scorer,
- defined in frozen external semantic space,
- responsible for support / contradiction / abstention semantics.

`Psi_univ` is **not**:

- a yes/no task head,
- a POPE classifier,
- a host-specific adapter.

Task-specific behavior must live in:

- deterministic task adapter,
- deterministic decision layer.

### 2.3 The verifier must control designated answer labels directly

For binary verification tasks, the method should not depend on:

- incidental exposure of `yes/no` in a generic frontier,
- a generic top-k patcher happening to touch the answer.

The control target must be:

- the designated binary answer interval itself.

### 2.4 Retrieval must be query-conditioned

Evidence retrieval must be tied to the task query / object query, not merely to arbitrary candidate frontier text.

### 2.5 Support-aware backoff is mandatory

Weak evidence must not force override.

Abstention and evidence confidence are part of the control rule, not just diagnostics.

## 3. Canonical References

Read these first.

### Theory / contract

- `论文方法章节_完整版_VEB_UniGround_20260323.md`
- `experiment_logs/veb_qa_methodology_alignment_20260323.md`
- `方法论纠偏文档_V5~V22.md`

### Previous handoff / background

- `experiment_logs/handoff_context_package_20260323_uniground_pope.md`

### Current implementation hotspots

- `uniground_v2/task_adapter.py`
- `uniground_v2/bridge.py`
- `uniground_v2/trigger.py`
- `uniground_v2/runtime.py`
- `uniground_v2/regions.py`
- `run_uniground_v2_eval.py`
- `export_uniground_v2_coco_features.py`
- `uniground_v2/training_dataset.py`
- `train_universal_plugin_v2.py`

### Validation assets

- `experiment_logs/experiment_registry_latest.md`
- `tests/test_pope_task_adapter.py`
- `tests/test_bridge_task_adapter_binary.py`
- `tests/test_pope_verifier_trigger.py`
- `tests/test_v2_runtime_batch_rows.py`
- `tests/test_region_retrieval_topr.py`
- `tests/test_training_dataset_region_shapes.py`
- `tests/test_training_payload_schema_v2.py`

## 4. What Has Already Been Implemented Correctly

The current code is **meaningfully closer** to the theory than the older hybrid version.

### 4.1 Task-protocol runtime context exists

`uniground_v2/task_adapter.py` now builds runtime context using capability-style fields:

- `task_name`
- `task_family`
- `task_query_text`
- `retrieval_query_text`
- `scorer_query_text`
- `answer_labels`
- `hypothesis_text_by_label`
- `decision_mode`
- `decision_scope`
- `retrieval_scope`

This is the correct direction and removes the old hard-coded `pope_verifier` identity from the core runtime.

### 4.2 Explicit answer-label control exists

`uniground_v2/runtime.py` now has a dedicated explicit answer-label path.

This is important:

- it no longer requires the verifier path to wait for generic frontier candidates,
- it can build answer-label candidates directly from runtime context,
- it applies bias to designated answer token IDs rather than relying on incidental frontier exposure.

This is a real theory alignment improvement.

### 4.3 Retrieval is generalized by capability, not benchmark name

`uniground_v2/regions.py` now uses:

- `retrieval_scope`
- `retrieval_query_text`

instead of POPE-specific mode names.

That is the correct architectural boundary.

### 4.4 Training/export semantics were moved toward query+hypothesis form

The feature export and training path were updated so that supervision is now described using:

- `query_text`
- `hypothesis_text`
- support / contradiction / abstain labels

This is much closer to the theory than the old prefix/candidate naming alone.

### 4.5 Fixed-slice validation has been run

The current implementation was not left at "it compiles".

It has already passed local and remote regression tests and produced fixed-slice results recorded in:

- `experiment_logs/experiment_registry_latest.md`

Notably, on the newest 100-sample theory-aligned LLaVA slice matrix:

- `random`: verifier > frontier > base
- `popular`: verifier > frontier > base
- `adversarial`: verifier > frontier > base

So the new direction is not only cleaner; it is also empirically moving in the right direction.

## 5. What Is Still NOT Fully Complete

This is the most important section.

The answer to "is engineering now fully adapted to the latest theory?" is:

> **No, not fully. It is substantially improved and largely correct for the POPE verifier runtime, but the overall theory contract is only partially completed.**

The remaining mismatch points are below.

### 5.1 The theory is broader than the currently verified implementation

The method chapter is task-general.

The current implementation is only strongly verified for:

- POPE-style binary verification.

What exists for other tasks is mostly:

- protocol scaffolding,
- legacy frontier path,
- limited adapter context.

So:

- the architecture is more general now,
- but the empirical and implementation proof is still mainly POPE-centered.

### 5.2 The explicit answer-label path is theory-faithful for binary tasks, not yet a complete cross-task decision framework

The current verifier kernel is a strong POPE/binary implementation.

But the broader theory decomposition says:

- binary verification is one instantiation,
- other tasks should also map into deterministic adapter + decision-layer structure.

That broader generalization is **not yet fully realized**.

Right now the codebase is best described as:

- theory-aligned for binary verification,
- not yet fully theory-complete for all task families claimed in the methodology.

### 5.3 `Psi_univ` training is more aligned, but still not the final strongest universal curriculum

The current training/export path is much better than before, but it is still mostly:

- COCO-derived object presence / absence supervision,
- templated universal hypotheses,
- abstention terms.

What is still missing relative to the strongest methodology notes:

- richer prior-conflict negative mining,
- broader universal hypothesis templating,
- harder support calibration for ambiguous / visually weak cases,
- a more explicit universal counterfactual curriculum for object presence reversal.

This means the training boundary is now cleaner, but not yet "finished" in the strongest sense.

### 5.4 The paper-level universality claim is stronger than the code-level verification status

The theory claims:

- universal scorer,
- deterministic task adapter,
- deterministic decision layer,
- reusable across hosts and tasks.

The code currently supports:

- multiple hosts better than before,
- binary verifier runtime in a host-agnostic external route,
- partial generalization in interfaces.

But it does **not yet prove**:

- robust multi-task realization beyond POPE-style control,
- final polished task-general evaluation path.

So be careful:

- the code is no longer obviously violating the theory in the old way,
- but it has not fully discharged every broad theoretical obligation in the chapter.

## 6. Practical Judgment

Use the following judgment exactly.

### 6.1 What has been completed

- Task-protocol runtime context: largely completed.
- Explicit answer-label decision control for binary verification: completed.
- Retrieval generalization by runtime capability fields: largely completed.
- Universal training/export contract shift toward query+hypothesis semantics: partially completed, directionally correct.
- Regression tests and fixed-slice POPE validation: completed.

### 6.2 What has not been fully completed

- Full cross-task realization of the theory beyond POPE/binary verification.
- Final strongest universal `Psi_univ` supervision curriculum implied by the methodology notes.
- A truly paper-level complete implementation proof for every claimed task boundary.

### 6.3 Bottom-line conclusion

The current system should be described as:

> a substantially theory-aligned POPE verifier implementation with improved universal boundaries, but not yet a fully complete realization of the full task-general method chapter.

## 7. Exact Instructions To The New Engineer

Follow these instructions in order.

### Step 1. Do not redesign the method

Do not invent:

- internal prior branches,
- new host-specific adapters,
- benchmark-specific `Psi_univ` heads,
- shortcut heuristic patches to force scores up.

If a change weakens the universality boundary, reject it.

### Step 2. Treat the current POPE verifier runtime as the reference kernel

Preserve and audit the following properties:

- explicit answer-label control,
- answer-step-only activation,
- object-conditioned retrieval,
- support-aware backoff,
- host-vs-evidence audit outputs.

Do not regress these.

### Step 3. Audit remaining theory mismatches honestly

You must explicitly judge whether each theory claim in the method chapter is:

- already implemented,
- partially implemented,
- not yet implemented.

Do not collapse these into a vague "mostly aligned" statement.

### Step 4. Extend the implementation only along theory-approved directions

The next valid engineering moves are:

1. strengthen cross-task protocol completeness without breaking the binary verifier path,
2. improve universal `Psi_univ` supervision in a task-general way,
3. expand verification without reintroducing frontier-conditioned verifier behavior,
4. keep evaluation and training language consistent with query+hypothesis universal semantics.

### Step 5. Preserve a strict boundary around `Psi_univ`

Allowed:

- stronger universal hypothesis families,
- better absence mining,
- improved abstention calibration,
- better object-conditioned retrieval supervision.

Not allowed:

- retraining `Psi_univ` directly as a POPE classifier,
- turning it into a host-coupled yes/no head,
- using host-native hidden states in the learned path.

### Step 6. Verify before scaling

Before any large full-volume rerun:

- run targeted unit/regression tests,
- run fixed slices,
- compare `base` vs `frontier` vs `verifier`,
- update registry only after verified outputs land.

## 8. Specific Open Questions For Engineering

These are the questions you should resolve next in code, not in theory.

1. How should the deterministic task adapter be extended so that the same protocol can cover non-POPE binary tasks cleanly?
2. What is the strongest universal training data expansion that still does not collapse `Psi_univ` into benchmark supervision?
3. Which pieces of the current frontier path should remain as generic fallback infrastructure, and which should be clearly separated from the verifier path?
4. How should evaluation runners expose theory-completeness status per task family instead of implicitly assuming POPE is the whole story?

## 9. Short Message To The Engineer

If you only remember one thing, remember this:

> The hard part is no longer "make POPE move a bit." The hard part is to preserve the external universal evidence bottleneck story while making the implementation genuinely worthy of the method chapter. The current code is much closer than before, but it is not yet the final complete realization of the full theory.
