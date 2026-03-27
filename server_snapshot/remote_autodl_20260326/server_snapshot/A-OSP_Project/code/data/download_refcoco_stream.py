#!/usr/bin/env python3
"""RefCOCO 流式拉取；结束后 os._exit(0) 避免 pyarrow/datasets 析构崩溃。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int, help="max samples from val stream")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = args.out if args.out.is_absolute() else root / args.out
    n = max(1, args.n)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    try:
        stream = load_dataset("lmms-lab/RefCOCO", split="val", streaming=True)
    except Exception as e:
        print(f"[RefCOCO] FAIL: {e}", flush=True)
        os._exit(1)

    rows = []
    for i, row in enumerate(tqdm(stream, total=n, desc="[RefCOCO]")):
        if i >= n:
            break
        img = row.get("image")
        ip = img_dir / f"refcoco_{i:05d}.jpg"
        if img is not None and hasattr(img, "save"):
            img.convert("RGB").save(ip)
        rows.append({
            "_index": i,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "bbox": row.get("bbox"),
            "file_name": row.get("file_name"),
            "image_path": str(ip.relative_to(root)) if ip.exists() else None,
        })
    man = out / "refcoco_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[RefCOCO] manifest → {man}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e, file=sys.stderr, flush=True)
        os._exit(1)
