#!/usr/bin/env python3
"""MMMU 全量 30 subjects validation → data/benchmarks/mmmu/  (os._exit 避免 PyArrow 崩溃)"""
import json, os, sys
from pathlib import Path

MMMU_CONFIGS = [
    "Accounting","Agriculture","Architecture_and_Engineering","Art","Art_Theory",
    "Basic_Medical_Science","Biology","Chemistry","Clinical_Medicine","Computer_Science",
    "Design","Diagnostics_and_Laboratory_Medicine","Economics","Electronics",
    "Energy_and_Power","Finance","Geography","History","Literature","Manage",
    "Marketing","Materials","Math","Mechanical_Engineering","Music","Pharmacy",
    "Physics","Psychology","Public_Health","Sociology",
]

def main():
    import ast
    from datasets import load_dataset, concatenate_datasets
    from PIL import Image
    from tqdm import tqdm

    root = Path(__file__).resolve().parents[2]
    out = root / "data/benchmarks/mmmu"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for cfg in MMMU_CONFIGS:
        try:
            p = load_dataset("MMMU/MMMU", cfg, split="validation")
            parts.append(p)
            print(f"[MMMU] loaded {cfg} ({len(p)} rows)", flush=True)
        except Exception as e:
            print(f"[MMMU] skip {cfg}: {e}", flush=True)

    if not parts:
        print("[MMMU] FAIL: no config loaded", flush=True)
        os._exit(1)

    ds = concatenate_datasets(parts)
    print(f"[MMMU] total rows: {len(ds)}", flush=True)

    rows = []
    for i, row in enumerate(tqdm(ds, desc="[MMMU]")):
        opts = row.get("options")
        if isinstance(opts, str):
            try: opts = ast.literal_eval(opts)
            except: opts = [opts]
        img = None
        for j in range(1, 8):
            v = row.get(f"image_{j}")
            if v is not None and hasattr(v, "save"):
                img = v; break
        ip = img_dir / f"mmmu_{i:05d}.png"
        if img and not ip.exists():
            img.convert("RGB").save(ip)
        rows.append({
            "_index": i,
            "id": row.get("id", f"mmmu_{i}"),
            "question": row.get("question", ""),
            "options": opts,
            "answer": row.get("answer"),
            "image_path": str(ip.relative_to(root)) if ip.exists() else None,
        })

    man = out / "mmmu_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[MMMU] manifest → {man} ({len(rows)} rows)", flush=True)
    os._exit(0)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"[MMMU] worker FAIL: {e}", flush=True)
        os._exit(1)
