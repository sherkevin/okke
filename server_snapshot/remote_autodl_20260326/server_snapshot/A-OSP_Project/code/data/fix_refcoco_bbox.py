#!/usr/bin/env python3
"""
RefCOCO bbox normaliser — fix_refcoco_bbox.py
==============================================
The lmms-lab/RefCOCO dataset stores bounding boxes in COCO format:
    [x_min, y_min, width, height]  (absolute pixel coordinates)

Qwen2-VL / Qwen3-VL grounding evaluation expects either:
  • absolute [x1, y1, x2, y2]  (before normalisation)  ← stored in manifest
  • normalised 0-1000 scale    (at inference time, using actual image W/H)

This script:
  1. Converts every row's 'bbox' from [x, y, w, h] → [x1, y1, x2, y2] in-place.
  2. Adds 'bbox_format': "xyxy_abs" annotation to each row for downstream clarity.
  3. Reads image size from PIL and adds 'image_wh': [W, H] so the eval script can
     trivially normalise without re-opening images.
  4. Writes the updated manifest back (atomic rename).

Idempotent: if 'bbox_format' == 'xyxy_abs' already, it skips that row.

Usage:
    python code/data/fix_refcoco_bbox.py
    python code/data/fix_refcoco_bbox.py --dry-run   # preview only
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/benchmarks/refcoco/refcoco_manifest.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"[fix_refcoco_bbox] manifest not found: {MANIFEST}"); sys.exit(1)

    try:
        from PIL import Image as PilImage
        PIL_OK = True
    except ImportError:
        PIL_OK = False
        print("[fix_refcoco_bbox] WARNING: PIL unavailable – image_wh will not be added")

    rows, changed = [], 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("bbox_format") == "xyxy_abs":
                rows.append(row)
                continue
            bbox = row.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                row["bbox"] = [x, y, x + w, y + h]
                row["bbox_format"] = "xyxy_abs"
            # Add image W/H for easy normalisation at eval time
            if PIL_OK and "image_wh" not in row:
                ip = row.get("image_path")
                if ip:
                    abs_p = ROOT / ip
                    try:
                        with PilImage.open(abs_p) as img:
                            row["image_wh"] = list(img.size)  # [W, H]
                    except Exception:
                        pass
            rows.append(row)
            changed += 1

    print(f"[fix_refcoco_bbox] {changed}/{len(rows)} rows converted xywh→xyxy_abs")

    if args.dry_run:
        print("[fix_refcoco_bbox] --dry-run: no files written")
        print(json.dumps(rows[0], indent=2))
        return

    tmp = MANIFEST.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    shutil.move(str(tmp), str(MANIFEST))
    print(f"[fix_refcoco_bbox] ✓ manifest updated → {MANIFEST}")

if __name__ == "__main__":
    main()
