# Remote AutoDL Snapshot

This directory contains a filtered local snapshot pulled from the remote AutoDL server on `2026-03-26`.

## Included roots

- `/root/autodl-tmp/A-OSP_Project`
- `/root/autodl-tmp/BRA_Project`
- `/root/autodl-tmp/CHIRO`
- `/root/autodl-tmp/CHORD`
- `/root/autodl-tmp/EKKO`
- `/root/autodl-tmp/OPERA_clean_official`

## Included content policy

The snapshot keeps project code, logs, JSON results, scripts, method documents, and experiment artifacts that are practical to version in Git.

The snapshot excludes heavyweight assets that are unsuitable for GitHub versioning, including:

- model weights
- datasets
- checkpoints
- caches
- `.git` directories from remote clones
- very large binary log artifacts

## Explicitly excluded large files

- `/root/autodl-tmp/A-OSP_Project/logs/flash_attn_prebuilt.whl`
- `/root/autodl-tmp/BRA_Project/logs/uniground_v2_fullrun/train2014_full_features.pt`
- `/root/autodl-tmp/BRA_Project/logs/v3_contract/lora_runs/base_lora_qwen3vl2b_data-phi_calib_seed3_precheck_obj-vasm_masked_ce_contract_steps-2_r-256_a-512_20260322_111355/adapter_model.safetensors`
- `/root/autodl-tmp/BRA_Project/logs/v3_contract/lora_runs/base_lora_qwen3vl2b_data-phi_calib_seed3_precheck_obj-vasm_masked_ce_contract_steps-2_r-256_a-512_20260322_111355/checkpoint-step-1/adapter_model.safetensors`
- `/root/autodl-tmp/BRA_Project/logs/v3_contract/lora_runs/base_lora_qwen3vl2b_data-phi_calib_seed3_precheck_obj-vasm_masked_ce_contract_steps-2_r-256_a-512_20260322_111355/checkpoint-step-2/adapter_model.safetensors`

## Layout

The pulled remote content is nested under `server_snapshot/` to avoid overwriting the existing local working tree while preserving the original remote project boundaries.
