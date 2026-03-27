# UniGround Universality Audit Latest

- Generated at: `2026-03-22T16:03:04.158902`
- Scanned TLRA_univ results: `16`
- Accepted: `7`
- Rejected: `9`

## Acceptance Rule

- If abort_trigger_rate > 0 then abort_backoff_verified_steps must be > 0; otherwise reject the result.

## Required Fields

- `universal_claim_manifest`
- `psi_univ checkpoint sha`
- `prefix_ambiguity_rate`
- `span_collapse_errors`
- `suffix_stability_rate`
- `abstention_rate`
- `abort_trigger_rate`
- `abort_backoff_verified_steps`
- `latency_split`

## Latest Conclusion

- At least one TLRA_univ result failed universality audit; those results are not acceptable for table use or claim support.

## Rejected Results

- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_20260322_155316.json`
  missing: prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json`
  abort rule failure: abort_trigger_rate=1.0 abort_backoff_verified_steps=0.0
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_155509.json`
  missing: prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json`
  abort rule failure: abort_trigger_rate=1.0 abort_backoff_verified_steps=0.0
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_external_global_prior_20260322_142907.json`
  missing: psi_univ checkpoint sha, prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_external_global_prior_20260322_143006.json`
  missing: psi_univ checkpoint sha, prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_20260322_153703.json`
  missing: prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_global_only_20260322_153723.json`
  missing: prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_no_gate_20260322_153713.json`
  missing: prefix_ambiguity_rate, span_collapse_errors, suffix_stability_rate, abstention_rate, abort_trigger_rate, abort_backoff_verified_steps
  abort rule failure: abort_trigger_rate=None abort_backoff_verified_steps=None

## Accepted Results

- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_qwen4b/uniground_qwen3-vl-4b_pope_uniground_no_gate_20260322_155450.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_external_global_prior_20260322_143157.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_external_global_prior_20260322_143555.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_20260322_154331.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_global_only_20260322_155102.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_no_abstain_20260322_155750.json`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v6/uniground_qwen3-vl-8b_pope_uniground_no_gate_20260322_154721.json`
