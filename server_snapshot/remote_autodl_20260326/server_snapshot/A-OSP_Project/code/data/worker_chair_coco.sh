#!/usr/bin/env bash
# 下载 COCO instances_val2014.json 并建 chair.pkl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ANNO_DIR="$ROOT/data/chair/coco_annotations"
CHAIR_DIR="$ROOT/data/chair/CHAIR-metric-standalone"
ZIP="$ROOT/data/chair/annotations_trainval2014.zip"
TARGET="$ANNO_DIR/instances_val2014.json"

mkdir -p "$ANNO_DIR"

if [[ -f "$TARGET" ]] && [[ "$(wc -c < "$TARGET")" -gt 40000000 ]]; then
  echo "[COCO] instances_val2014.json already complete ($(du -h "$TARGET"|cut -f1))"
else
  echo "[COCO] Downloading COCO annotations zip (~252 MB) ..."
  rm -f "$ZIP"
  curl -L --progress-bar --retry 5 --retry-delay 3 \
       -o "$ZIP" "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
  echo "[COCO] Extracting instances_val2014.json ..."
  unzip -o -j "$ZIP" "annotations/instances_val2014.json" -d "$ANNO_DIR"
  rm -f "$ZIP"
  echo "[COCO] Extracted: $(du -h "$TARGET"|cut -f1)"
fi

# 符号链到 CHAIR repo 内
LINK="$CHAIR_DIR/coco_annotations/instances_val2014.json"
mkdir -p "$CHAIR_DIR/coco_annotations"
[[ -e "$LINK" ]] || ln -sf "$TARGET" "$LINK"

# 构建 chair.pkl（如未存在）
PKL="$CHAIR_DIR/chair.pkl"
if [[ ! -f "$PKL" ]] || [[ "$(wc -c < "$PKL")" -lt 100000 ]]; then
  echo "[CHAIR] Building chair.pkl ..."
  cd "$CHAIR_DIR"
  python3 chair.py --cache 2>&1 | tail -10
  echo "[CHAIR] chair.pkl done: $(du -h "$PKL"|cut -f1)"
fi

echo "[CHAIR] DONE"
