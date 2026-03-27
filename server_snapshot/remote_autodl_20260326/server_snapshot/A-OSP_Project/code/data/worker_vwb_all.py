#!/usr/bin/env python3
"""VisualWebBench 全 7 tasks  (os._exit 避免 PyArrow 崩溃)"""
import json, os, sys
from pathlib import Path

VWB_TASKS = ["action_ground","action_prediction","element_ground","element_ocr",
              "heading_ocr","web_caption","webqa"]

def main():
    from datasets import load_dataset
    from tqdm import tqdm
    root = Path(__file__).resolve().parents[2]
    out = root / "data/benchmarks/visualwebbench"
    ok_any = False
    for task in VWB_TASKS:
        img_dir = out / "images" / task
        img_dir.mkdir(parents=True, exist_ok=True)
        man = out / f"vwb_{task}_manifest.jsonl"
        if man.exists() and man.stat().st_size > 500:
            count = sum(1 for _ in open(man))
            print(f"[VWB:{task}] already have {count} rows, skip", flush=True)
            ok_any = True
            continue
        try:
            ds = load_dataset("visualwebbench/VisualWebBench", name=task, split="test")
            rows = []
            for i, item in enumerate(tqdm(ds, desc=f"[VWB:{task}]")):
                img = item.get("image") or item.get("screenshot")
                ip = img_dir / f"{task}_{i:05d}.png"
                if img is not None and not ip.exists():
                    img.convert("RGB").save(ip)
                rec = {k: v for k, v in item.items() if k not in ("image","screenshot")}
                rec["_index"] = i
                rec["image_path"] = str(ip.relative_to(root)) if ip.exists() else None
                rows.append(rec)
            with open(man, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            print(f"[VWB:{task}] → {man} ({len(rows)} rows)", flush=True)
            ok_any = True
        except Exception as e:
            print(f"[VWB:{task}] ERR: {e}", flush=True)
    os._exit(0 if ok_any else 1)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"[VWB] FAIL: {e}", flush=True)
        os._exit(1)
