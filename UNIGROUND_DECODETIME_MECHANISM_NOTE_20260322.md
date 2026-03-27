# UniGround Decode-Time Mechanism Note

## Scope

This note fixes the implementation boundary for the `TLRA_univ` / `uniground` route.
It is the reference for the new independent runner `run_uniground_eval.py`.

## What UniGround Reads

At decode step `t`, the host MLLM is already running autoregressive generation.
UniGround reads only:

- the current Top-`M` next-token candidate logits from the host model;
- decoded candidate token strings derived from those Top-`M` IDs;
- the decoded prefix tail;
- frozen external image embeddings;
- frozen external text embeddings for the prefix and candidate spans.

UniGround does **not** read:

- host hidden states as learned inputs;
- `lm_head.weight`;
- learned per-model adapters;
- model-native lexical projections.

## Where Top-M Is Read

`run_uniground_eval.py` installs a logits processor from `uniground_runtime.py`.
Inside the processor, the current score tensor is read at each generation step and
the active candidate window is built by:

1. taking the current Top-`M` token IDs;
2. decoding them through the host tokenizer;
3. canonicalizing them into short candidate spans;
4. collapsing repeated normalized spans for bridge auditing.

This is decode-time intervention because the candidate set is read from the live
generation loop, not from a post-hoc reranking pass.

## Where Bias Is Written Back

After scoring candidate spans, UniGround produces a bounded bias only for the
current Top-`M` candidates. The bias is then scattered back into the live logits
tensor for the same decode step.

This means:

- the plugin does not replace the decoder;
- the plugin does not emit a final answer string;
- the plugin does not reorder completed sequences after generation.

The final token is still sampled/selected by the original host MLLM from the
modified logits.

## Learned Module vs Parameter-Free Bridge

### Learned module

The only learned module in the universal route is `Psi_univ`, loaded through a
checkpoint passed as `--psi_checkpoint`. It consumes frozen external embeddings and
returns:

- support
- contradiction
- abstention

### Parameter-free bridge

The following pieces are deterministic and non-learned:

- Top-`M` candidate extraction
- token-to-string decoding
- span canonicalization / normalization
- span-collapse accounting
- prefix ambiguity audit
- score-to-logit redistribution over the current token prefixes
- abort / back-off decision once abstention or ambiguity crosses the configured rule

## Abort Rule

`abstention` is not just logged.

In the new runtime, intervention is aborted when:

- the current Top-`M` window has no groundable candidate spans;
- prefix ambiguity is severe enough to trigger the configured ambiguity back-off;
- the abstention score crosses the configured threshold.

When abort fires, the host logits are returned unchanged for that step.

## Method Relations

- `base`: no intervention.
- `tlra_internal_zero`: old internal control path; reads host-native geometry.
- `tlra_internal_calib`: old internal learned control; still a control, not universal.
- `external_global_prior`: frozen global visual prior baseline; no candidate-local learned scorer.
- `uniground`: universal decode-time sidecar; requires `Psi_univ`.

## Metrics That Must Land

The independent runner is designed to emit:

- hallucination main metric
- intervention coverage
- ITL / TPOT
- peak VRAM
- prefix ambiguity rate
- span collapse errors
- abstention rate
- abort trigger rate
- candidate construction latency
- sidecar scoring latency
- bridge redistribution latency
- jitter

## Result Isolation

All new outputs land in `logs/uniground_v6/`.

This prevents the universal route from being mixed into:

- old `logs/minitest/`
- old `phi_calib` directories
- old internal-only table assets
