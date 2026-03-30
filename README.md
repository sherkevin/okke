# CHORD

CHORD is a runnable multimodal evaluation repository for decode-time hallucination mitigation. This tree keeps the patched generation stack, multimodal wrappers, offline anchor preprocessing, and benchmark entry scripts needed to reproduce the current CHORD evaluation flow without shipping logs, weights, notebooks, or unrelated training assets.

## What is included

- `transformers-4.29.2/`: vendored generation stack with CHORD beam-search integration.
- `minigpt4/`: model wrappers used by the supported evaluation backends.
- `chord/`: knowledge-kernel scoring, anchor cache, detector client/server, and oracle rollout logic.
- `pope_eval.py`: POPE evaluation entrypoint.
- `precompute_pope_anchor_cache.py`: offline anchor-cache builder for POPE.
- `chair_eval.py` and `chair.py`: CHAIR generation and scoring entrypoints.
- `eval_configs/`: model-specific evaluation configs.
- `pope_coco/`: POPE split definitions.

## Setup

```bash
conda env create -f environment.yml
conda activate chord
python -m pip install -e transformers-4.29.2
```

The patched decode core lives in `transformers-4.29.2/src/transformers/generation/utils.py`.

## Required assets

This repo does not ship checkpoints or datasets. Before running evaluation, point the configs to local copies of the required model weights and prepare MSCOCO 2014 locally.

- `eval_configs/llava-1.5_eval.yaml`: set the merged LLaVA-1.5 checkpoint path.
- `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`: set the Vicuna v1.1 path.
- `minigpt4/configs/models/minigpt4_vicuna0.yaml`: set the Vicuna v0 path.
- `eval_configs/minigpt4_eval.yaml`: set the MiniGPT-4 checkpoint path.
- `eval_configs/shikra_eval.yaml`: set the Shikra checkpoint path.

## CHORD workflow

1. Precompute anchors once for a POPE split.
2. Run POPE with or without CHORD enabled.
3. Run CHAIR generation, then score the saved captions.

### 1. Precompute offline anchors

```bash
python precompute_pope_anchor_cache.py \
  --model llava-1.5 \
  --pope-type random \
  --data-path /path/to/COCO_2014/val2014 \
  --output-jsonl /path/to/anchor_cache_random.jsonl
```

### 2. Run POPE

Baseline beam-search path:

```bash
python pope_eval.py \
  --model llava-1.5 \
  --pope-type random \
  --data_path /path/to/COCO_2014/val2014 \
  --gpu-id 0 \
  --beam 5 \
  --scale_factor 50 \
  --threshold 15 \
  --num_attn_candidates 5 \
  --penalty_weights 1
```

CHORD-enabled path:

```bash
python pope_eval.py \
  --model llava-1.5 \
  --pope-type random \
  --data_path /path/to/COCO_2014/val2014 \
  --gpu-id 0 \
  --beam 5 \
  --scale_factor 50 \
  --threshold 15 \
  --num_attn_candidates 5 \
  --penalty_weights 1 \
  --chord-enable \
  --anchor-cache-jsonl /path/to/anchor_cache_random.jsonl \
  --lambda-cur 1.0 \
  --lambda-fut 0.5 \
  --future-horizon 10 \
  --future-topk 5 \
  --tau-abort 0.1
```

Useful CHORD-specific arguments:

- `--anchor-cache-jsonl`: required when `--chord-enable` is set.
- `--lambda-cur`: current-step knowledge-kernel weight.
- `--lambda-fut`: oracle-rollout weight.
- `--lambda-txt`: text-dominance penalty inside rollout scoring.
- `--future-horizon`: rollout depth.
- `--future-topk`: number of branches to probe per step.
- `--tau-abort`: early-stop threshold for weak visual support.
- `--attention-last-n-layers`: number of trailing layers used for attention reduction.

### 3. Run CHAIR

Generate captions:

```bash
python chair_eval.py \
  --model llava-1.5 \
  --data_path /path/to/COCO_2014/val2014 \
  --gpu-id 0 \
  --beam 5 \
  --scale_factor 50 \
  --threshold 15 \
  --num_attn_candidates 5 \
  --penalty_weights 1
```

Score captions:

```bash
python chair.py \
  --cap_file /path/to/generated_captions.jsonl \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path /path/to/COCO/annotations_trainval2014/annotations \
  --save_path /path/to/chair_scored.jsonl
```

## Notes

- The current CHORD evaluation path expects `batch_size=1` when `--chord-enable` is active.
- The offline anchor cache is intentionally decoupled from autoregressive decoding; decode-time runs only do cache lookup and score fusion.
- The vendored `transformers` tree was trimmed to runtime-critical files only.

## Acknowledgement

This repository builds on the LAVIS and MiniGPT-4 codebases and uses the standalone CHAIR metric implementation. The current tree keeps only the components required for CHORD evaluation and benchmarking.
