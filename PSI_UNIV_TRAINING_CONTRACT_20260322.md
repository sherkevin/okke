# Psi_univ Training Contract

## Scope

This contract freezes the only learned module used by the universal route:

- paper method: `UniGround`
- implementation route: `TLRA_univ`
- learned module: `Psi_univ`

All older learned routes such as `TLRA_calib`, `Phi_calib`, and `Base + LoRA` are controls only.

## Canonical External Semantic Space

- frozen vision encoder: `openai/clip-vit-large-patch14::image`
- frozen text encoder: `openai/clip-vit-large-patch14::text`
- embedding dim: recorded from the training payload and checkpoint metadata as `embedding_dim`
- region features: must be explicitly dumped as `region_features_enabled` in the checkpoint contract

This encoder pair is the only canonical configuration for strict universality validation in the current benchmark.

## Allowed Inputs

`Psi_univ` may train only on frozen external observations:

- `image_embeddings`
- `candidate_embeddings`
- `prefix_embeddings`
- optional frozen `region_embeddings`

It may not train on:

- model-native hidden states
- `lm_head.weight`
- per-model learned adapters

## Checkpoint Contract

Every `Psi_univ` checkpoint must dump:

- `checkpoint_format`
- `contract_version`
- `plugin_route`
- `learned_module`
- `frozen_vision_encoder_name`
- `frozen_text_encoder_name`
- `embedding_dim`
- `region_features_enabled`
- `uses_model_native_hidden_states`
- `uses_lm_head_geometry`
- `learned_per_model_adapter`
- `parameter_free_tokenizer_bridge`
- `source_hashes`

The current frozen checkpoint format is `psi_univ_checkpoint_v1`.

## Config Dump Contract

Every training run must also write a JSON config dump containing:

- checkpoint path
- checkpoint sha256
- contract block
- training args

This dump is part of the reproducibility surface and is required by the validator.

## Version / Hash Discipline

- contract version: `uniground_train_contract_v1`
- checkpoint sha256: must be computed after save
- source hashes: must include at least `train_universal_plugin.py` and `bra_universal_plugin.py`

## Required Runtime Audit Link

Runtime results that claim to be `TLRA_univ` must carry the same checkpoint identity and must expose:

- `prefix_ambiguity_rate`
- `span_collapse_errors`
- `suffix_stability_rate`
- `abstention_rate`
- `abort_trigger_rate`
- `abort_backoff_verified_steps`
- `latency_split`

Without these fields, the result is not a valid universality asset.
