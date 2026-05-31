#!/usr/bin/env bash
set -euo pipefail

BASE=/media/data3/dengkw/chord_rebuttal_20260529
cd "$BASE"

export XDG_CACHE_HOME="$BASE/cache/xdg"
export PIP_CACHE_DIR="$BASE/cache/pip"
export HF_HOME="$BASE/cache/hf"
export HUGGINGFACE_HUB_CACHE="$BASE/cache/hf/hub"
export TMPDIR="$BASE/cache/tmp"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

VENV="$BASE/venv/bin/python"
RUN_DIR="$BASE/runs/real_rebuttal_20260529"
mkdir -p "$RUN_DIR" assets/pope assets/coco2014/val2014 assets/models cache/tmp

{
  echo "[phase] start $(date -Iseconds)"
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

  "$VENV" - <<'PY'
import torch, transformers, huggingface_hub
print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print('transformers', transformers.__version__)
print('hf_hub', huggingface_hub.__version__)
PY

  echo "[phase] download POPE annotations"
  if [ ! -s assets/pope/coco_pope_adversarial.json ]; then
    for url in \
      https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json \
      https://gh-proxy.com/https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json \
      https://mirror.ghproxy.com/https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json; do
      echo "trying $url"
      if curl -L --connect-timeout 20 --max-time 120 -o assets/pope/coco_pope_adversarial.json.tmp "$url"; then
        if "$VENV" - <<'PY'; then
from pathlib import Path
import json
p = Path('assets/pope/coco_pope_adversarial.json.tmp')
lines = p.read_text(encoding='utf-8').splitlines()
assert len(lines) > 10
json.loads(lines[0])
print('valid_lines', len(lines))
PY
          mv assets/pope/coco_pope_adversarial.json.tmp assets/pope/coco_pope_adversarial.json
          break
        fi
      fi
    done
  fi
  test -s assets/pope/coco_pope_adversarial.json

  echo "[phase] download COCO val2014 images for first 64 POPE adversarial samples"
  "$VENV" - <<'PY'
from pathlib import Path
import json, urllib.request, sys, time

pope = Path('assets/pope/coco_pope_adversarial.json')
out = Path('assets/coco2014/val2014')
out.mkdir(parents=True, exist_ok=True)
records = [json.loads(line) for line in pope.read_text(encoding='utf-8').splitlines()[:64]]
missing = []
for record in records:
    name = record['image']
    path = out / name
    if path.exists() and path.stat().st_size > 1000:
        continue
    missing.append(('http://images.cocodataset.org/val2014/' + name, path))

print('records', len(records), 'missing', len(missing))
for idx, (url, path) in enumerate(missing, 1):
    tmp = path.with_suffix(path.suffix + '.tmp')
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size <= 1000:
                raise RuntimeError('tiny download')
            tmp.replace(path)
            print('downloaded', idx, path.name, path.stat().st_size)
            break
        except Exception as exc:
            print('retry', idx, attempt, url, exc, file=sys.stderr)
            time.sleep(2)
    else:
        raise RuntimeError(f'failed image {url}')
PY

  echo "[phase] download GroundingDINO model"
  if [ ! -s assets/models/grounding-dino-base/config.json ] || [ ! -s assets/models/grounding-dino-base/pytorch_model.bin ]; then
    "$VENV" -m huggingface_hub.commands.huggingface_cli download IDEA-Research/grounding-dino-base \
      --local-dir assets/models/grounding-dino-base --local-dir-use-symlinks False
  fi

  echo "[phase] download LLaVA model (large; may continue for a while)"
  if [ ! -s assets/models/llava-v1.5-7b/config.json ] || [ ! -s assets/models/llava-v1.5-7b/pytorch_model-00001-of-00002.bin ] || [ ! -s assets/models/llava-v1.5-7b/pytorch_model-00002-of-00002.bin ]; then
    "$VENV" -m huggingface_hub.commands.huggingface_cli download liuhaotian/llava-v1.5-7b \
      --local-dir assets/models/llava-v1.5-7b --local-dir-use-symlinks False
  fi

  echo "[phase] precompute anchors for 64-sample POPE adv smoke"
  PYTHONPATH="$BASE/EKKO" "$VENV" EKKO/precompute_pope_anchor_cache.py \
    --pope-path assets/pope/coco_pope_adversarial.json \
    --data-path assets/coco2014/val2014 \
    --output-jsonl "$RUN_DIR/anchors_pope_adv64.jsonl" \
    --grounding-dino-path assets/models/grounding-dino-base \
    --detector-python "$VENV" \
    --detector-device cuda:0 \
    --limit 64

  echo "[phase] run LLaVA POPE smoke: P+C and Full"
  cd "$BASE/EKKO"
  PYTHONPATH="$BASE/EKKO:$BASE/EKKO/transformers-4.29.2/src" "$VENV" pope_eval.py \
    --model llava-1.5 --pope-type adversarial --gpu-id 0 \
    --data_path "$BASE/assets/coco2014/val2014" \
    --pope-path "$BASE/assets/pope/coco_pope_adversarial.json" \
    --llava-ckpt "$BASE/assets/models/llava-v1.5-7b" \
    --llava-proc-path "$BASE/assets/models/llava-v1.5-7b" \
    --batch_size 1 --num_workers 0 --limit 16 --max-new-tokens 8 \
    --chord-enable --anchor-cache-jsonl "$RUN_DIR/anchors_pope_adv64.jsonl" \
    --lambda-cur 0.25 --lambda-fut 0 --future-horizon 0 --future-topk 0 \
    --attention-last-n-layers 4 \
    --output-json "$RUN_DIR/llava_pope_adv16_pc.json"

  PYTHONPATH="$BASE/EKKO:$BASE/EKKO/transformers-4.29.2/src" "$VENV" pope_eval.py \
    --model llava-1.5 --pope-type adversarial --gpu-id 0 \
    --data_path "$BASE/assets/coco2014/val2014" \
    --pope-path "$BASE/assets/pope/coco_pope_adversarial.json" \
    --llava-ckpt "$BASE/assets/models/llava-v1.5-7b" \
    --llava-proc-path "$BASE/assets/models/llava-v1.5-7b" \
    --batch_size 1 --num_workers 0 --limit 16 --max-new-tokens 8 \
    --chord-enable --anchor-cache-jsonl "$RUN_DIR/anchors_pope_adv64.jsonl" \
    --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 3 --future-topk 5 \
    --attention-last-n-layers 4 \
    --output-json "$RUN_DIR/llava_pope_adv16_full.json"

  echo "[phase] complete $(date -Iseconds)"
} > "$RUN_DIR/run.log" 2>&1
