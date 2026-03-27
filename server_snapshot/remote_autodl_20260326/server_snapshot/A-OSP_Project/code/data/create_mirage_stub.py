#!/usr/bin/env python3
"""从本地 COCO val 图生成 MIRAGE 评测管线的占位 manifest（非官方 MIRAGE 全量数据）。"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
coco_dir = PROJECT_ROOT / "data" / "coco_val2014"
images = sorted(coco_dir.glob("*.jpg"))[:50]

mirage_dir = PROJECT_ROOT / "data" / "benchmarks" / "mirage"
mirage_dir.mkdir(parents=True, exist_ok=True)

manifest_path = mirage_dir / "mirage_manifest.jsonl"
with open(manifest_path, "w", encoding="utf-8") as f:
    for i, img_path in enumerate(images):
        record = {
            "question_id": f"mirage_{i}",
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "question": (
                "Does the main object appear to be spatially in front of or behind the "
                "background elements? Answer with 'A' for in front, or 'B' for behind."
            ),
            "A": "In front",
            "B": "Behind",
            "C": "Neither",
            "D": "Cannot tell",
            "answer": "A",
            "_stub": True,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {len(images)} MIRAGE stub samples at {manifest_path}")
