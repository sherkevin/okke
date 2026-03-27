#!/usr/bin/env python3
"""
lmms-lab/TextVQA 流式拉取前 n 条（避免整表 load_dataset 卡死）。
进程结束前 os._exit(0)，规避 datasets/pyarrow 析构时的已知 PyGILState_Release 崩溃。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int, help="number of validation samples")
    p.add_argument("--out", type=Path, required=True, help="output dir under data/benchmarks/textvqa")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    out: Path = args.out
    if not out.is_absolute():
        out = root / out
    n = max(1, args.n)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    stream = load_dataset("lmms-lab/TextVQA", split="validation", streaming=True)
    rows = []
    for i, row in enumerate(tqdm(stream, total=n, desc="[TextVQA] stream")):
        if i >= n:
            break
        entry = {
            "_index": i,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "answers": row.get("answers"),
        }
        img = row.get("image")
        if img is not None and hasattr(img, "save"):
            ip = img_dir / f"textvqa_{i:05d}.png"
            img.convert("RGB").save(ip)
            entry["image_path"] = str(ip.relative_to(root))
        rows.append(entry)

    man = out / "textvqa_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[TextVQA] manifest → {man} ({len(rows)} rows)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TextVQA] worker FAIL: {e}", file=sys.stderr, flush=True)
        os._exit(1)
