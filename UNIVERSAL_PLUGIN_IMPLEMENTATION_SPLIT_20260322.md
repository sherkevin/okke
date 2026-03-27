# Universal Plugin Implementation Split

## Purpose

Prevent the project from mixing two fundamentally different routes:

- `TLRA_internal`: model-coupled hidden/logit path
- `TLRA_univ`: strict universal sidecar path

## Route A: Internal path

These files now belong explicitly to the internal route:

- `bra_logits_processor.py`
- `bra_operator_multi.py`
- `train_bra_calib.py`
- current `run_eval_pipeline.py`

Characteristics:

- consumes model-native hidden states
- depends on `lm_head.weight`
- may use vocab-side masking
- supports `Phi_calib`

Correct role:

- internal control
- ceiling estimator
- comparison point against universal route

## Route B: Universal path

These files now belong explicitly to the universal route:

- `bra_universal_plugin.py`
- `train_universal_plugin.py`

Characteristics:

- consumes frozen external image/text embeddings
- operates on decoded strings/spans
- uses parameter-free tokenizer bridge
- never depends on `lm_head.weight`

Correct role:

- strict train-once, run-anywhere plugin
- main paper novelty

## Interface Contract

### Universal path may read

- Top-`M` candidate logits
- decoded candidate strings
- decoded prefix text
- frozen public image/text embeddings

### Universal path may not read

- model-native hidden states as learned inputs
- model-family adapters
- learned per-model bridges

### Universal path may write

- bounded next-token logit bias on current Top-`M` candidates

### Universal path may not write

- model-specific lexical-space projections
- learned vocab-row alignment modules

## Migration Rule

Any future feature must answer this first:

1. does it rely on model-native geometry?
2. if yes, it belongs to `TLRA_internal`
3. if no, it may belong to `TLRA_univ`

This rule is now part of the benchmark discipline. It exists to stop future
iterations from accidentally reintroducing false universality.
