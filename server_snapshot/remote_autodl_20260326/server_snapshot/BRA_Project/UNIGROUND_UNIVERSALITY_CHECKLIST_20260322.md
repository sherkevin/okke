# UniGround Universality Checklist

Use this checklist before accepting any `TLRA_univ` result.

## Checkpoint

- The run uses exactly one shared `Psi_univ` checkpoint.
- The checkpoint contract says `plugin_route = TLRA_univ`.
- The checkpoint contract says `learned_module = Psi_univ`.
- The checkpoint contract freezes `openai/clip-vit-large-patch14::image`.
- The checkpoint contract freezes `openai/clip-vit-large-patch14::text`.
- The checkpoint contract says `uses_model_native_hidden_states = false`.
- The checkpoint contract says `uses_lm_head_geometry = false`.
- The checkpoint contract says `learned_per_model_adapter = false`.

## Runtime identity

- The result JSON includes `universal_claim_manifest`.
- The result JSON carries the same `checkpoint_sha256` as the accepted `Psi_univ` checkpoint.
- The host-model side only contributes Top-`M` candidates and final decoding.
- The bridge is tokenizer-dependent if needed, but parameter-free.

## Required metrics

- `prefix_ambiguity_rate`
- `span_collapse_errors`
- `suffix_stability_rate`
- `abstention_rate`
- `abort_trigger_rate`
- `abort_backoff_verified_steps`
- `latency_split.candidate_construction_ms`
- `latency_split.sidecar_scoring_ms`
- `latency_split.bridge_redistribution_ms`
- `latency_split.jitter_ms`

## Abort-rule reality check

- `abstention` is not accepted as a passive score only.
- If `abort_trigger_rate > 0`, then `abort_backoff_verified_steps > 0`.
- If `abort_trigger_rate = 0`, then `abort_backoff_verified_steps = 0`.
- Any run failing this coupling check cannot be accepted as a structure-safe UniGround result.

## Mandatory comparison logic

- `TLRA_univ` must be compared against `External_Global_Prior`.
- `TLRA_univ_global_only` must remain available as an ablation.
- If `External_Global_Prior` matches or exceeds `TLRA_univ`, the token-local candidate-aware claim must contract.

## One-line acceptance rule

A result is a valid `UniGround` asset only if it uses the shared `Psi_univ` checkpoint, introduces no learned per-model adapter, exposes tokenizer-bridge and abort metrics, and passes the abort back-off validation.

## Batch audit sync

- Every new batch of `TLRA_univ` result JSONs must be rechecked with `validate_uniground_universality.py`.
- The synced batch conclusion should be refreshed in `experiment_logs/uniground_v6/universality_audit_latest.md`.
- Any rejected result remains ineligible for table use until a fresh JSON passes the validator.
