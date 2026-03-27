#!/usr/bin/env python3
"""COD10K 伪装目标图像 — chandrabhuma/animal_cod10k  (os._exit 避免崩溃)"""
import json, os, sys
from pathlib import Path

def main():
    from datasets import load_dataset
    from tqdm import tqdm
    root = Path(__file__).resolve().parents[2]
    out = root / "data/benchmarks/cod10k"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 目标 50 张 — 实验大纲 Sprint 1.3 / 4.3 / 4.5 各需 10-30 张
    N = 100
    try:
        ds = load_dataset("chandrabhuma/animal_cod10k", split="train", streaming=True)
    except Exception as e:
        print(f"[COD10K] train failed: {e}", flush=True)
        try:
            ds = load_dataset("chandrabhuma/animal_cod10k_train", split="train", streaming=True)
        except Exception as e2:
            print(f"[COD10K] FAIL: {e2}", flush=True)
            os._exit(1)

    rows = []
    for i, row in enumerate(tqdm(ds, total=N, desc="[COD10K]")):
        if i >= N: break
        img = row.get("image")
        ip = img_dir / f"cod10k_{i:05d}.jpg"
        if img is not None and hasattr(img, "save") and not ip.exists():
            img.convert("RGB").save(ip)
        rows.append({
            "_index": i,
            "id": row.get("id", f"cod10k_{i}"),
            "question": row.get("question",""),
            "answer": row.get("answer",""),
            "image_path": str(ip.relative_to(root)) if ip.exists() else None,
        })

    man = out / "cod10k_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[COD10K] manifest → {man} ({len(rows)} rows)", flush=True)
    os._exit(0)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"[COD10K] FAIL: {e}", flush=True)
        os._exit(1)
