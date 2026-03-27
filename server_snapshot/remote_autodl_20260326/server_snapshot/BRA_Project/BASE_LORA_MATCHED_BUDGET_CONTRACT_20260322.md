# Base + LoRA Matched-Budget Contract

## Scope

This note defines the formal `Base + LoRA` control to be matched against `Phi_calib` for the benchmark line. It is intentionally narrow: the goal is to lock the data budget, training budget, and target-function contract before formal seed runs.

## Data Budget

- Reference object: `Phi_calib`
- Formal target budget: `50,000` caption records
- Intended formal path: `/root/autodl-tmp/BRA_Project/train_data/phi_calib_matched_budget_50k.jsonl`
- Precheck path only: `/root/autodl-tmp/BRA_Project/smoke_data/lora_matched_budget_smoke.jsonl`

The precheck file is only for loader / checkpoint / reload validation. It is not the formal matched-budget training data.

## Training Budget

- Model: `Qwen3-VL-2B-Instruct`
- Adapter family: `LoRA`
- Rank: `256`
- Alpha: `512`
- Dropout: `0.0`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Optimizer: `AdamW`
- Learning rate: `5e-5`
- Weight decay: `0.0`
- Formal schedule: `50,000` steps
- Checkpoint cadence: every `5,000` steps
- Seeds: `1`, `2`, `3`

The seed3 precheck keeps the same high-rank topology but truncates to `2` steps and enables checkpoint reload validation.

## Objective Contract

- Matched reference: `Phi_calib`
- Contract objective: `VASM-masked next-token cross-entropy`
- Current script status: the training scaffold records this contract in config and `training_meta.json`, and the seed3 precheck verifies entrypoint health, logging, checkpoint writing, and adapter reload.

## Current Readiness

- `seed1/seed2/seed3` formal config package: ready
- `seed3_precheck` config: ready
- Formal 50k JSONL at `/root/autodl-tmp/BRA_Project/train_data/phi_calib_matched_budget_50k.jsonl`: not yet present on remote at the time of writing

Until that formal JSONL is available, only the precheck run can be executed honestly.
