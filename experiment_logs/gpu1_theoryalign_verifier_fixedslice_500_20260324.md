# GPU1 Theory-Aligned Verifier Fixed-Slice 500 (2026-03-24)

## Scope

- Host: `llava-v1.5-7b`
- Dataset: `POPE`
- Controller: `verifier`
- Constraint: `GPU1 only`, no use of `GPU0`
- Checkpoint: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- Observation mode: `detector_regions`
- External encoder: `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`
- Detector: `/root/autodl-tmp/BRA_Project/models/grounding-dino-base`
- Slice size: `500` per split

## Results

| Split | Accuracy | Precision | Recall | F1 | host_yes_to_no | host_no_to_yes |
| --- | --- | --- | --- | --- | --- | --- |
| `random` | `0.8840` | `0.9364` | `0.8240` | `0.8766` | `0` | `2` |
| `popular` | `0.8720` | `0.9115` | `0.8240` | `0.8655` | `0` | `0` |
| `adversarial` | `0.8320` | `0.8374` | `0.8240` | `0.8306` | `0` | `2` |

## Remote Result Paths

- `random`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_random_500_20260324_003546.json`
- `popular`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_popular_500_20260324_003831.json`
- `adversarial`: `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_gpu1_theoryalign_verifier_adversarial_500_20260324_004116.json`

## Remote Log

- Queue log: `/root/autodl-tmp/BRA_Project/logs/uniground_v2_queue/llava_gpu1_theoryalign_verifier500_20260323_163300.log`

## Notes

- All three runs kept the theory-aligned `verifier` path active.
- All three runs preserved `object_extraction_success_rate = 1.0` and `semantic_alias_hit_rate = 1.0`.
- No split produced `host_yes_to_no`, which is consistent with conservative support-aware backoff.
- This is a fixed-slice validation round, not a paper-main-table full run.
