#!/usr/bin/env bash
set -u

BASE=${BASE:-/media/data3/dengkw/chord_rebuttal_20260529}
LIMIT=${LIMIT:-64}
RUN_DIR=${RUN_DIR:-$BASE/runs/real_rebuttal_20260529_extended}
TAG=${TAG:-adv${LIMIT}}
LLAVA_GPUS=${LLAVA_GPUS:-0,2,3,6}
VENV="$BASE/venv/bin/python"

export XDG_CACHE_HOME="$BASE/cache/xdg"
export PIP_CACHE_DIR="$BASE/cache/pip"
export HF_HOME="$BASE/cache/hf"
export HUGGINGFACE_HUB_CACHE="$BASE/cache/hf/hub"
export TRANSFORMERS_CACHE="$BASE/cache/hf/transformers"
export TMPDIR="$BASE/cache/tmp"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "$RUN_DIR" "$BASE/logs" "$BASE/assets/pope" "$BASE/assets/coco2014/val2014" "$BASE/cache/tmp"
LOG="$RUN_DIR/run_extended.log"
exec > >(tee -a "$LOG") 2>&1

echo "[phase] extended start $(date -Iseconds)"
echo "[config] BASE=$BASE RUN_DIR=$RUN_DIR LIMIT=$LIMIT TAG=$TAG LLAVA_GPUS=$LLAVA_GPUS"
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

wait_for_llava() {
  echo "[phase] wait for LLaVA local files"
  while true; do
    if [ -s "$BASE/assets/models/llava-v1.5-7b/config.json" ] \
      && [ -s "$BASE/assets/models/llava-v1.5-7b/pytorch_model-00001-of-00002.bin" ] \
      && [ -s "$BASE/assets/models/llava-v1.5-7b/pytorch_model-00002-of-00002.bin" ]; then
      du -sh "$BASE/assets/models/llava-v1.5-7b"
      break
    fi
    date -Iseconds
    du -sh "$BASE/assets/models/llava-v1.5-7b" "$BASE/assets/models/llava-v1.5-7b/.cache" 2>/dev/null || true
    sleep 60
  done
}

stop_legacy_smoke() {
  local pid_file="$BASE/logs/run_real_rebuttal_experiments_20260529.pid"
  local pid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if ps -fp "$pid" | grep -q "run_real_rebuttal_experiments_20260529.sh"; then
      echo "[phase] stopping legacy smoke runner pid=$pid after model download"
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
      sleep 10
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

download_pope_and_images() {
  echo "[phase] ensure POPE and COCO images"
  if [ ! -s "$BASE/assets/pope/coco_pope_adversarial.json" ]; then
    curl -L --connect-timeout 20 --max-time 120 \
      -o "$BASE/assets/pope/coco_pope_adversarial.json.tmp" \
      https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json
    mv "$BASE/assets/pope/coco_pope_adversarial.json.tmp" "$BASE/assets/pope/coco_pope_adversarial.json"
  fi
  "$VENV" - <<PY
from pathlib import Path
import json, sys, time, urllib.request

limit = int("$LIMIT")
pope = Path("$BASE/assets/pope/coco_pope_adversarial.json")
out = Path("$BASE/assets/coco2014/val2014")
out.mkdir(parents=True, exist_ok=True)
records = [json.loads(line) for line in pope.read_text(encoding="utf-8").splitlines()[:limit]]
missing = []
for record in records:
    path = out / record["image"]
    if not (path.exists() and path.stat().st_size > 1000):
        missing.append(("http://images.cocodataset.org/val2014/" + record["image"], path))
print("records", len(records), "missing", len(missing))
for idx, (url, path) in enumerate(missing, 1):
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size <= 1000:
                raise RuntimeError("tiny download")
            tmp.replace(path)
            print("downloaded", idx, path.name, path.stat().st_size)
            break
        except Exception as exc:
            print("retry", idx, attempt, url, exc, file=sys.stderr)
            time.sleep(2)
    else:
        raise RuntimeError(f"failed image {url}")
PY
}

build_anchor_controls() {
  echo "[phase] precompute real anchors"
  local anchors="$RUN_DIR/anchors_pope_${TAG}.jsonl"
  local summary="$RUN_DIR/anchors_pope_${TAG}_summary.json"
  local start end status
  if [ ! -s "$anchors" ] || [ "$(wc -l < "$anchors" 2>/dev/null || echo 0)" -lt "$LIMIT" ]; then
    start=$(date +%s%3N)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$BASE/EKKO" "$VENV" "$BASE/EKKO/precompute_pope_anchor_cache.py" \
      --pope-path "$BASE/assets/pope/coco_pope_adversarial.json" \
      --data-path "$BASE/assets/coco2014/val2014" \
      --output-jsonl "$anchors" \
      --grounding-dino-path "$BASE/assets/models/grounding-dino-base" \
      --detector-python "$VENV" \
      --detector-device cuda:0 \
      --limit "$LIMIT"
    status=$?
    end=$(date +%s%3N)
    "$VENV" - <<PY
import json
limit = int("$LIMIT")
elapsed_ms = max(0, int("$end") - int("$start"))
payload = {
  "limit": limit,
  "total_detector_ms": elapsed_ms,
  "mean_detector_ms": elapsed_ms / limit if limit else None,
  "status": int("$status"),
  "gpu": 0,
}
open("$summary", "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\\n")
PY
  fi

  echo "[phase] create uniform/random anchor controls"
  "$VENV" - <<PY
from pathlib import Path
import json, random

src = Path("$anchors")
uniform = Path("$RUN_DIR/anchors_pope_${TAG}_uniform.jsonl")
random_path = Path("$RUN_DIR/anchors_pope_${TAG}_random_matched.jsonl")
rng = random.Random(20260529)

with src.open("r", encoding="utf-8") as handle, uniform.open("w", encoding="utf-8") as out:
    for line in handle:
        payload = json.loads(line)
        payload["relevance"] = [0.0 for _ in payload.get("relevance", [])]
        out.write(json.dumps(payload, ensure_ascii=False) + "\\n")

with src.open("r", encoding="utf-8") as handle, random_path.open("w", encoding="utf-8") as out:
    for line in handle:
        payload = json.loads(line)
        grid = payload.get("grid_size") or [24, 24]
        total = int(grid[0]) * int(grid[1])
        new_membership = []
        for row in payload.get("membership", []):
            area = sum(1 for value in row if float(value) > 0)
            cells = set(rng.sample(range(total), min(area, total))) if area > 0 else set()
            new_membership.append([1.0 if idx in cells else 0.0 for idx in range(total)])
        payload["membership"] = new_membership
        out.write(json.dumps(payload, ensure_ascii=False) + "\\n")
PY
}

run_one() {
  local name="$1"
  local gpu="$2"
  local anchor="$3"
  shift 3
  local output="$RUN_DIR/${name}.json"
  local run_log="$RUN_DIR/${name}.log"
  if [ -s "$output" ]; then
    echo "[skip] $name output exists"
    return 0
  fi
  echo "[run] name=$name gpu=$gpu anchor=$anchor args=$*"
  (
    cd "$BASE/EKKO" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$BASE/EKKO:$BASE/EKKO/transformers-4.29.2/src" "$VENV" pope_eval.py \
      --model llava-1.5 --pope-type adversarial --gpu-id 0 \
      --data_path "$BASE/assets/coco2014/val2014" \
      --pope-path "$BASE/assets/pope/coco_pope_adversarial.json" \
      --llava-ckpt "$BASE/assets/models/llava-v1.5-7b" \
      --batch_size 1 --num_workers 0 --limit "$LIMIT" --max-new-tokens 20 --beam 5 \
      $anchor \
      --attention-last-n-layers 4 \
      --output-json "$output" \
      "$@" \
      --options model.load_8bit=True model.device_map=balanced_low_0
  ) > "$run_log" 2>&1
  local status=$?
  echo "[done] name=$name status=$status log=$run_log output=$output"
  echo "$status" > "$RUN_DIR/${name}.status"
  return 0
}

wait_for_llava
stop_legacy_smoke
download_pope_and_images
build_anchor_controls

REAL_ANCHOR="--chord-enable --anchor-cache-jsonl $RUN_DIR/anchors_pope_${TAG}.jsonl"
UNIFORM_ANCHOR="--chord-enable --anchor-cache-jsonl $RUN_DIR/anchors_pope_${TAG}_uniform.jsonl"
RANDOM_ANCHOR="--chord-enable --anchor-cache-jsonl $RUN_DIR/anchors_pope_${TAG}_random_matched.jsonl"
NO_ANCHOR=""

echo "[phase] launch LLaVA sequentially on verified idle 2080Ti set $LLAVA_GPUS"
run_one llava_pope_${TAG}_opera "$LLAVA_GPUS" "$NO_ANCHOR"
run_one llava_pope_${TAG}_same_anchor_non_chord "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0 --lambda-fut 0 --future-horizon 0 --future-topk 0
run_one llava_pope_${TAG}_pc_real "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0 --future-horizon 0 --future-topk 0
run_one llava_pope_${TAG}_full_k5_m3 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 3 --future-topk 5
run_one llava_pope_${TAG}_full_k3_m2 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 2 --future-topk 3
run_one llava_pope_${TAG}_full_k5_m1 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 1 --future-topk 5
run_one llava_pope_${TAG}_full_k5_m2 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 2 --future-topk 5
run_one llava_pope_${TAG}_full_k5_m4 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 4 --future-topk 5
run_one llava_pope_${TAG}_full_k10_m3 "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0.25 --lambda-fut 0.05 --future-horizon 3 --future-topk 10
run_one llava_pope_${TAG}_pc_uniform "$LLAVA_GPUS" "$UNIFORM_ANCHOR" --lambda-cur 0.25 --lambda-fut 0 --future-horizon 0 --future-topk 0
run_one llava_pope_${TAG}_pc_random "$LLAVA_GPUS" "$RANDOM_ANCHOR" --lambda-cur 0.25 --lambda-fut 0 --future-horizon 0 --future-topk 0
run_one llava_pope_${TAG}_past_future "$LLAVA_GPUS" "$REAL_ANCHOR" --lambda-cur 0 --lambda-fut 0.05 --future-horizon 3 --future-topk 5

echo "[phase] summarize"
"$VENV" "$BASE/logs/summarize_real_rebuttal_metrics_20260529.py" \
  --run-dir "$RUN_DIR" \
  --tag "$TAG" \
  --output-json "$RUN_DIR/real_rebuttal_metrics_20260529.json" \
  --output-md "$RUN_DIR/real_rebuttal_metrics_20260529.md"

echo "[phase] extended complete $(date -Iseconds)"
find "$RUN_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
