# Audit Pass/Fail Feed (2026-03-22)

Append one block per audit run:

- timestamp
- batch source
- pass list
- fail list
- fail reasons
## 2026-03-22T16:02:05.863857
- batch source:
  - uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json
  - uniground_qwen3-vl-4b_pope_uniground_no_gate_20260322_155450.json
  - uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json
- pass list:
  - uniground_qwen3-vl-4b_pope_uniground_no_gate_20260322_155450.json
- fail list:
  - uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json
  - uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json
- fail reasons:
  - uniground_qwen3-vl-4b_pope_uniground_20260322_155943.json: abort_rule_failure:abort_trigger_rate=1.0,abort_backoff_verified_steps=0.0
  - uniground_qwen3-vl-4b_pope_uniground_global_only_20260322_160009.json: abort_rule_failure:abort_trigger_rate=1.0,abort_backoff_verified_steps=0.0

