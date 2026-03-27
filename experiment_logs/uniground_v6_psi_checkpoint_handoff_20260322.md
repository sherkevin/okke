# UniGround V6 Psi Checkpoint Handoff

日期：`2026-03-22`

## Real Psi_univ Checkpoint

- checkpoint: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt`
- config dump: `/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json`
- checkpoint sha256: `daecaca88f0a8d70a539264788aeb45aa2d92c530367a0a34e3b2b70edf83f33`

## Training Payload

- payload: `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt`
- payload config dump: `/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.config.json`
- payload sha256: `0722c23467469f3dfef7db1216ca0e216d839ea5e2503fb212f3f52828510e67`

## Frozen External Space

- frozen vision encoder: `openai/clip-vit-large-patch14::image`
- frozen text encoder: `openai/clip-vit-large-patch14::text`
- encoder load path: `/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14`
- embedding dim: `768`

## Contract Status

- `plugin_route = TLRA_univ`
- `learned_module = Psi_univ`
- `uses_model_native_hidden_states = false`
- `uses_lm_head_geometry = false`
- `learned_per_model_adapter = false`
- `parameter_free_tokenizer_bridge = true`
- validator checkpoint status: `PASS`
- minimal smoke via `load_universal_scorer(...)`: `PASS`

## Training Source

- image_dir: `/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014`
- instances_json: `/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/instances_val2014.json`
- training_image_count: `4096`
- training_record_count: `12288`

## Source Hashes

- `train_universal_plugin.py`: `f03427f18df52cd1f22be710aa4e2d3679f709d94579bd00fcbc1f2ad3eebab0`
- `bra_universal_plugin.py`: `4c71ddedaf358297fd45f01b86ea5948bfcce7a8c4e4f7df0fb718e79cc8c470`
- `export_universal_coco_payload.py`: `108a42771fa5871eb7ae543447049cdf6ac97e649742c19c62f3cd3ab161ac15`
- `uniground_runtime.py`: `b797187a15d2427c722d22a2c212ab62d9111f84d91f30836d15d51b0308cd1a`
- `instances_json`: `e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47`
- `image_listing`: `f7fc383fcedfca5c03857ded8ec71eed9d725b37c70f646fa12c8a2524966d8b`

## Second-Host Runner Evidence

- `qwen3-vl-4b` is now wired into:
  - `run_eval_pipeline.py`
  - `run_uniground_eval.py`
- second-host smoke proof:
  - `/root/autodl-tmp/BRA_Project/logs/uniground_v6/second_host_smoke/uniground_qwen3-vl-4b_pope_base_20260322_141148.json`

## Immediate Reuse Target

- this checkpoint is the first real shared `Psi_univ` artifact for follow-up `uniground` / `external_global_prior` / portability runs
- handoff target: Engineer 1 and Engineer 3
