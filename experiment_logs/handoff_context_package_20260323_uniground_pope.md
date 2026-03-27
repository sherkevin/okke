# UniGround / POPE Handoff Context Package (2026-03-23)

## 1. Current Mission

The immediate goal is **not** to keep patching the current code until it accidentally reaches a better `POPE` score.

The real goal is:

> Re-align the engineering implementation with the new theory while preserving the universality boundary of `Psi_univ`.

This means the next agent must treat **theory-implementation consistency** as a hard constraint, not as a "nice to have".

## 2. Non-Negotiable Theory Constraints

The following points are the current contract and should be treated as mandatory.

### 2.1 `Psi_univ` is still necessary

`Psi_univ` is still needed in the new method, but its role must be precisely defined.

It is **not**:

- a POPE-specific classifier
- a yes/no task head
- a dataset-specific adapter trained for one benchmark

It **is**:

- a universal external evidence scorer in a frozen shared embedding space
- a learned mapping from `(image / regions, query / hypothesis text)` to evidence signals
- the module that outputs generic evidence dimensions such as support / contradiction / abstention

Without `Psi_univ`, the method falls into one of two bad cases:

1. Hardcoded similarity only
   - easy to keep universal, but usually too weak and poorly calibrated

2. Host-coupled task head
   - may improve one benchmark, but destroys the universality story

So the correct answer is not "remove `Psi`", but:

> keep `Psi` as the universal evidence interpreter, and move task-specific behavior out of `Psi` into the task adapter and decision layer.

### 2.2 `Psi_univ` must remain task-general

`Psi_univ` should be useful for many downstream tasks and many host models.

Therefore:

- do **not** train `Psi_univ` directly on `POPE` labels as if it were a benchmark-specific classifier
- do **not** collapse `Psi_univ` into a binary yes/no verifier head
- do **not** define `Psi_univ` as "the module for POPE"

The correct decomposition is:

- `Psi_univ`: universal evidence scoring
- task adapter: map a downstream task into a universal evidence query
- decision layer: convert generic evidence into task-specific control

For `POPE`, task-specific logic should live in:

- object extraction from the question
- construction of `H_yes(o)` / `H_no(o)`
- answer-step-only binary control

not inside `Psi_univ` training itself.

### 2.3 The runtime should not depend on incidental frontier exposure

The new theory says the method should control the **final answer decision interval**.

That means the verifier path should not require:

- host frontier already surfacing `yes/no`
- generic top-k token-local candidates being present first

The correct implementation target is:

> explicit binary answer control over the relevant answer labels, using universal evidence, instead of generic frontier patching that happens to touch binary tokens.

## 3. Why `Psi_univ` Is Needed Under the New Theory

The new theory says:

- retrieve object-conditioned evidence `E = Retrieve(X, o)`
- compare object hypotheses `H_yes(o)` and `H_no(o)`
- steer the final yes/no answer only through evidence

This still requires a module that can answer:

- how much does visual evidence support the queried hypothesis?
- how much does it contradict it?
- how uncertain / ungrounded is the evidence?

That is the role of `Psi_univ`.

So the clean theoretical formulation is:

1. universal representation:
   - frozen image / region / text embeddings

2. universal scorer:
   - `Psi_univ(E, q_or_hypothesis) -> (e_sup, e_con, e_abs)`

3. task adapter:
   - POPE converts `(Q)` into `(o, H_yes(o), H_no(o))`

4. decision layer:
   - binary control maps universal evidence scores to yes/no steering

In other words:

> `Psi_univ` should remain universal; POPE should only define how to query and use it.

## 4. What The Current Code Gets Right

The current implementation already has several pieces that are directionally correct.

### 4.1 Explicit object metadata exists

`uniground_v2/task_adapter.py`

- extracts `object_label`
- builds `yes_hypothesis`
- builds `no_hypothesis`
- stores them in `runtime_context`

This is consistent with the new theory.

### 4.2 Retrieval is object-conditioned in verifier mode

`uniground_v2/regions.py`

- uses `verifier_query_text` for retrieval in `pope_verifier`

This is also consistent with the new theory.

### 4.3 Decision audit tracks host-vs-evidence flips

`run_uniground_v2_eval.py`

- logs `host_no_to_yes`
- logs `host_yes_to_no`
- logs changed-choice rates

This is necessary for theory-faithful evaluation.

### 4.4 Weak-evidence backoff exists

`uniground_v2/runtime.py`

- has abstention logic
- has evidence thresholds
- avoids forced control when evidence is weak

This is aligned with the "support-aware backoff" principle.

## 5. Where The Current Code Still Violates The Theory

These are the most important mismatch points.

### 5.1 The verifier path still depends on generic top-k frontier exposure

Current files:

- `uniground_v2/trigger.py`
- `uniground_v2/runtime.py`

Problem:

- verifier only fires if binary labels appear in current `top_ids`
- runtime only builds candidates from `scores.topk(...)`

So although it is called a verifier, it still behaves like a frontier-conditioned patcher.

This is not the final theoretical implementation.

### 5.2 `Psi_univ` training is still old-style universal semantic supervision

Current files:

- `export_uniground_v2_coco_features.py`
- `train_universal_plugin_v2.py`
- `uniground_v2/training_dataset.py`

Problem:

- data export creates generic triples:
  - positive category name
  - absent category name
  - abstain term
- training optimizes support / contradiction / abstain on these generic records

This means:

- the scorer is still a generic universal scorer
- but it has not been re-expressed in the new theoretical language carefully enough
- more importantly, the current runtime started behaving like a POPE verifier while the training objective stayed in the previous generic formulation

This is a **boundary mismatch**, not just a tuning issue.

### 5.3 The implementation drifted into an "intermediate bridge version"

The current code is best described as:

> a transitional implementation that partially overlays POPE-verifier logic on top of the old frontier-based universal runtime

That is why results improved relative to `frontier`, but still do not cleanly realize the new theory.

## 6. Why The Previous Implementation Drift Happened

This should be stated honestly for the next window.

The main reason is **not only** that the conversation got long.

The actual failure was:

1. a fast experimental recovery path was prioritized
2. the existing frontier runtime was incrementally modified instead of replacing the kernel boundary
3. the universal training pipeline was left untouched to preserve a working checkpoint
4. the resulting implementation became a hybrid

So the problem is:

- not just context length
- but a failure to freeze the architecture boundary strictly enough

Still, the long context now makes it easier to continue patching the hybrid path by inertia.

That is a valid reason to hand off to a fresh window with this package.

## 7. Exact Position On `Psi_univ`

The next agent should use the following wording exactly.

### Correct wording

`Psi_univ` is a **universal evidence scorer**, not a benchmark-specific classifier.

### Incorrect wording

- `Psi_univ` is the POPE verifier
- `Psi_univ` should be retrained just for POPE
- `Psi_univ` should become a task-specific yes/no head

### Allowed improvement direction

You may improve `Psi_univ` training in a way that remains universal, for example:

- stronger object-presence supervision from generic annotations
- harder absence / contradiction mining
- better abstention calibration
- broader hypothesis templating

But the improvement must remain:

- task-general
- host-agnostic
- external-space only

## 8. Required Next Implementation Plan

The next agent should implement in this order.

### Step 1. Rebuild the verifier runtime boundary

Goal:

- remove dependence on incidental top-k binary exposure
- make verifier operate on explicit answer labels

Interpretation:

- binary control should address the yes/no answer interval directly
- generic frontier candidates can remain for non-POPE modes, but should not define verifier behavior

### Step 2. Preserve `Psi_univ` as universal

Goal:

- do not turn `Psi_univ` into a POPE-trained head

Interpretation:

- keep support / contradiction / abstain as generic evidence semantics
- let POPE only query them through `H_yes(o)` / `H_no(o)`

### Step 3. Re-audit training data

Goal:

- ensure training still supports universal object evidence

Interpretation:

- if training data is expanded, expand it in a universal way
- do not use POPE benchmark supervision as the main training objective

### Step 4. Only then rerun small-sample POPE

Goal:

- verify theory-aligned runtime before any full run

Required comparison:

- base
- old frontier path
- rebuilt verifier path

across:

- `random`
- `popular`
- `adversarial`

## 9. Files That Matter Most

### Theory / contract

- `experiment_logs/veb_qa_methodology_alignment_20260323.md`

### Runtime / task logic

- `uniground_v2/task_adapter.py`
- `uniground_v2/trigger.py`
- `uniground_v2/bridge.py`
- `uniground_v2/runtime.py`
- `uniground_v2/regions.py`
- `run_uniground_v2_eval.py`

### Universal training path

- `export_uniground_v2_coco_features.py`
- `uniground_v2/training_dataset.py`
- `train_universal_plugin_v2.py`

### Experiment tracking

- `experiment_logs/experiment_registry_latest.md`
- `experiment_logs/0325_0401_新方法实验冲刺总表_20260323.md`

## 10. Latest Experimental Bottom Line

Latest fixed-slice round:

- `LLaVA-1.5-7B`
- `POPE`
- `200` samples per split
- `frontier` vs `verifier`

Observed:

- verifier beats frontier on all three splits
- but still does not achieve the target threshold
- this means the transitional verifier logic helps, but the theory is still not fully realized

Representative results:

- `random`: frontier `0.8783`, verifier `0.8842`
- `popular`: frontier `0.8557`, verifier `0.8705`
- `adversarial`: frontier `0.8019`, verifier `0.8276`

Interpretation:

- the direction is better
- the architecture boundary is still wrong

## 11. Handoff Summary

If starting a new window, the next agent should begin from this summary:

> The current code is a hybrid between the old frontier runtime and the new POPE verifier theory. `Psi_univ` must remain a universal evidence scorer, not a POPE-specific classifier. The next step is to rebuild the verifier runtime so it controls explicit yes/no answer labels without depending on frontier exposure, while preserving universal `Psi_univ` training and only using task-specific logic in the adapter and decision layer.
