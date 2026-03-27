# Universal Claim Failure Audit

This note defines which behaviors immediately invalidate the strict universal claim for `UniGround`.

## Automatic universal-claim failures

- A different learned adapter is trained for each host model.
- `Psi_univ` consumes host-model hidden states as learned inputs.
- The universal route reads or depends on `lm_head.weight`.
- The bridge from span scores back to logits is learned per model.
- Runtime results do not preserve a shared `Psi_univ` checkpoint identity.
- `abstention` is logged but does not trigger a real abort / back-off behavior.
- `External_Global_Prior` is missing while the paper still claims candidate-aware necessity.

## Automatic evidence downgrades

- `prefix_ambiguity_rate` is not reported.
- `span_collapse_errors` are not reported.
- `suffix_stability_rate` is not reported.
- latency is reported only as total throughput without split.
- the result omits `abort_trigger_rate`.
- the result omits `abort_backoff_verified_steps`.

These failures may not prove the method is wrong, but they do make the run unusable as evidence for the strict universal claim.

## Claim-contraction triggers

- If `TLRA_univ` requires per-model learned attachment logic, the strict universality claim fails.
- If `TLRA_univ` is indistinguishable from `External_Global_Prior`, the candidate-aware local-control claim must contract.
- If `TLRA_univ` works only by forcing unsafe interventions without reliable abort behavior, the structure-safe claim must contract.
- If tokenizer bridge instability dominates outcomes, the paper must foreground portability limits rather than oversell hot-swap robustness.

## Reviewer-facing summary

`TLRA_univ` can claim strict universality only when the same `Psi_univ` checkpoint transfers across host models, the bridge remains parameter-free, no host-native geometry is used by the learned path, and abstention provably changes intervention behavior through real back-off.
