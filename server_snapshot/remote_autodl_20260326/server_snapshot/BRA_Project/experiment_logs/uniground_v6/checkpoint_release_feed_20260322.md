# Checkpoint Release Feed (2026-03-22)

## Canonical Release

- checkpoint path: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- config path: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json`
- sha256: `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`
- whether replacement happened: `no`
- short reason: `first real canonical Psi_univ checkpoint produced under TLRA_univ contract`

## Payload Link

- payload path: `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt`
- payload sha256: `0722c23467469f3dfef7db1216ca0e216d839ea5e2503fb212f3f52828510e67`

## Release Notes

- `load_universal_scorer(...)`: `PASS`
- `validate_uniground_universality.py --checkpoint ...`: `PASS`
- frozen vision encoder: `openai/clip-vit-large-patch14::image`
- frozen text encoder: `openai/clip-vit-large-patch14::text`

Append future replacement notices below this block if the checkpoint changes.
