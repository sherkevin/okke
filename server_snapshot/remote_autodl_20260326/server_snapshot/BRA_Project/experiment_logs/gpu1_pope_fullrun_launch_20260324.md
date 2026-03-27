# GPU1 POPE Full Run Launch (2026-03-24)

## Status

- Status: `running`
- Host: `llava-v1.5-7b`
- Dataset: `POPE`
- Splits: `random`, `popular`, `adversarial`
- Sample count: `3000` per split
- Controller: `verifier`
- GPU assignment: `GPU1 only`

## Runtime Config

- `psi_checkpoint`: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- `external_encoder`: `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`
- `observation_mode`: `detector_regions`
- `detector_model`: `/root/autodl-tmp/BRA_Project/models/grounding-dino-base`
- `retrieval_top_r`: `1`
- `eval_batch_size`: `1`

## Remote Log

- `/root/autodl-tmp/BRA_Project/logs/uniground_v2_queue/llava_gpu1_theoryalign_verifier_full_pope_20260323_164626.log`

## Expected Result Paths

- `random`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_random_full3000_*.json`
- `popular`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_popular_full3000_*.json`
- `adversarial`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_adversarial_full3000_*.json`

## Notes

- The launch reuses the same theory-aligned verifier configuration that passed the `500/split` fixed-slice validation.
- `GPU0` was intentionally not used.
- At launch-time health check, the run had already entered the `random 3000` sample loop on the server.
