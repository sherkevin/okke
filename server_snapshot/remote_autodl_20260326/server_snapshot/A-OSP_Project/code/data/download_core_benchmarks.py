#!/usr/bin/env python3
"""
Task 1: General Capability & Modern Hallucination Suite
=======================================================
Targets:
  A) MMBench_EN   — general instruction-following capability proof
  B) AMBER        — modern generative hallucination stress test (2024+)

Usage:
  python code/data/download_core_benchmarks.py --mini_test          # 10 samples each (VERIFY FIRST)
  python code/data/download_core_benchmarks.py --full               # full download (after auth)

All assets land under  data/benchmarks/{mmbench,amber}/
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from io import BytesIO

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

MMBENCH_HF_CANDIDATES = [
    "opencompass/MMBench",
    "HuggingFaceM4/MMBench",
]

AMBER_GITHUB_BASE = (
    "https://raw.githubusercontent.com/junyangwang0410/AMBER/master"
)
AMBER_JSON_FILES = [
    "data/query/query_all.json",
    "data/query/query_generative.json",
    "data/query/query_discriminative.json",
    "data/annotations.json",
]
AMBER_IMAGE_GDRIVE_ID = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"


def download_mmbench(n_samples: int, out_dir: Path):
    """Download MMBench_EN via HuggingFace datasets library."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)

    ds = None
    split_candidates = ["validation", "dev", "test"]
    for repo_id in MMBENCH_HF_CANDIDATES:
        for split_name in split_candidates:
            try:
                print(f"[MMBench] Trying {repo_id} split={split_name} ...")
                ds = load_dataset(repo_id, split=split_name)
                print(f"[MMBench] Loaded from {repo_id}/{split_name}  ({len(ds)} rows)")
                break
            except Exception as e:
                print(f"[MMBench] {repo_id}/{split_name} failed: {e}")
        if ds is not None:
            break

    if ds is None:
        print("[MMBench] ERROR: Could not load from any candidate repo.")
        print("          Manually check: https://huggingface.co/datasets/opencompass/MMBench")
        return False

    subset = ds.select(range(min(n_samples, len(ds))))
    print(f"[MMBench] Selected {len(subset)} / {len(ds)} rows")

    print("[MMBench] Schema preview (first row keys):")
    row0 = subset[0]
    for k, v in row0.items():
        vtype = type(v).__name__
        vpreview = str(v)[:120] if not isinstance(v, bytes) else f"<bytes len={len(v)}>"
        print(f"  {k:30s}  ({vtype})  {vpreview}")

    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    manifest = []
    for i, row in enumerate(tqdm(subset, desc="[MMBench] Saving")):
        entry = {}
        for k, v in row.items():
            if k == "image" and v is not None:
                try:
                    from PIL import Image
                    if isinstance(v, Image.Image):
                        img_path = img_dir / f"mmbench_{i:05d}.png"
                        v.save(str(img_path))
                        entry["image_path"] = str(img_path.relative_to(PROJECT_ROOT))
                        continue
                    elif isinstance(v, str) and len(v) > 100:
                        import base64
                        img_bytes = base64.b64decode(v)
                        img = Image.open(BytesIO(img_bytes))
                        img_path = img_dir / f"mmbench_{i:05d}.png"
                        img.save(str(img_path))
                        entry["image_path"] = str(img_path.relative_to(PROJECT_ROOT))
                        continue
                except Exception as e:
                    entry["image_decode_error"] = str(e)[:100]
            if isinstance(v, (str, int, float, bool)) or v is None:
                if k == "image":
                    continue
                entry[k] = v
        manifest.append(entry)

    manifest_path = out_dir / "mmbench_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[MMBench] Saved manifest → {manifest_path}")
    print(f"[MMBench] Images dir     → {img_dir}  ({len(list(img_dir.glob('*')))} files)")
    return True


def download_amber_json(out_dir: Path):
    """Download AMBER annotation JSONs from GitHub."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for rel_path in AMBER_JSON_FILES:
        url = f"{AMBER_GITHUB_BASE}/{rel_path}"
        local_name = Path(rel_path).name
        dest = out_dir / local_name
        if dest.exists():
            print(f"[AMBER-JSON] Already exists: {local_name}")
            downloaded.append(dest)
            continue

        print(f"[AMBER-JSON] Downloading {local_name} from {rel_path} ...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded.append(dest)
            print(f"[AMBER-JSON] Saved → {dest}  ({len(resp.content):,} bytes)")
        except requests.RequestException as e:
            alt_url = url.replace("/master/", "/main/")
            print(f"[AMBER-JSON] master branch failed ({e}), trying main ...")
            try:
                resp = requests.get(alt_url, timeout=120)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                downloaded.append(dest)
                print(f"[AMBER-JSON] Saved → {dest}  ({len(resp.content):,} bytes)")
            except requests.RequestException as e2:
                print(f"[AMBER-JSON] FAILED to download {local_name}: {e2}")

    return downloaded


def download_amber_images_mini(json_path: Path, out_dir: Path, n_samples: int):
    """
    Parse AMBER JSON, extract image URLs/IDs for the first n_samples,
    and attempt to download them. AMBER images are hosted on Google Drive.
    For mini_test we just verify the JSON schema and save sample metadata.
    """
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        subset = data[:n_samples]
    elif isinstance(data, dict):
        first_key = next(iter(data))
        if isinstance(data[first_key], list):
            subset = data[first_key][:n_samples]
        else:
            subset = list(data.values())[:n_samples]
    else:
        print(f"[AMBER] Unexpected JSON root type: {type(data)}")
        return False

    print(f"\n[AMBER] JSON schema preview (first entry):")
    if len(subset) > 0:
        for k, v in (subset[0].items() if isinstance(subset[0], dict) else [("value", subset[0])]):
            vpreview = str(v)[:150]
            print(f"  {k:30s}  ({type(v).__name__})  {vpreview}")

    print(f"\n[AMBER] Total entries in JSON: {len(data) if isinstance(data, list) else 'dict'}")
    print(f"[AMBER] Mini-test subset: {len(subset)} entries")

    sample_path = out_dir / "amber_mini_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"[AMBER] Saved mini sample → {sample_path}")

    image_fields = []
    if isinstance(subset[0], dict):
        for k, v in subset[0].items():
            if "image" in k.lower() or "img" in k.lower() or "path" in k.lower():
                image_fields.append(k)

    if image_fields:
        print(f"[AMBER] Detected image-related fields: {image_fields}")
        print("[AMBER] NOTE: Full images are hosted on Google Drive (14GB+).")
        print(f"        GDrive file ID: {AMBER_IMAGE_GDRIVE_ID}")
        print("        Use `gdown {AMBER_IMAGE_GDRIVE_ID}` for full download after authorization.")
    else:
        print("[AMBER] No obvious image URL fields found. Images may use integer IDs.")
        print(f"        GDrive file ID for images: {AMBER_IMAGE_GDRIVE_ID}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download core benchmarks: MMBench + AMBER"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mini_test", action="store_true",
        help="Download only 10 samples per dataset for pipeline verification"
    )
    mode.add_argument(
        "--full", action="store_true",
        help="Download complete datasets (run --mini_test first!)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of samples for --mini_test (default: 10)"
    )
    args = parser.parse_args()

    n = args.n_samples if args.mini_test else None

    print("=" * 70)
    print("A-OSP Data Infrastructure — Task 1: Core Benchmarks")
    print(f"Mode: {'MINI TEST (' + str(n) + ' samples)' if args.mini_test else 'FULL'}")
    print("=" * 70)

    mmbench_dir = DATA_DIR / "mmbench"
    amber_dir = DATA_DIR / "amber"

    # --- Target A: MMBench ---
    print("\n" + "─" * 50)
    print("TARGET A: MMBench_EN")
    print("─" * 50)
    if args.mini_test:
        ok_mmb = download_mmbench(n, mmbench_dir)
    else:
        ok_mmb = download_mmbench(999_999, mmbench_dir)
    print(f"[MMBench] {'SUCCESS' if ok_mmb else 'FAILED'}")

    # --- Target B: AMBER ---
    print("\n" + "─" * 50)
    print("TARGET B: AMBER Hallucination Benchmark")
    print("─" * 50)
    amber_jsons = download_amber_json(amber_dir)
    ok_amber = len(amber_jsons) > 0

    if ok_amber:
        gen_json = amber_dir / "query_generative.json"
        all_json = amber_dir / "query_all.json"
        target_json = gen_json if gen_json.exists() else (all_json if all_json.exists() else amber_jsons[0])
        download_amber_images_mini(target_json, amber_dir, n if args.mini_test else 999_999)

    print(f"[AMBER]   {'SUCCESS' if ok_amber else 'FAILED'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  MMBench : {'OK' if ok_mmb else 'FAIL'}  → {mmbench_dir}")
    print(f"  AMBER   : {'OK' if ok_amber else 'FAIL'}  → {amber_dir}")
    if args.mini_test:
        print("\n  >>> Pipeline verified. Inspect outputs above, then run --full <<<")


if __name__ == "__main__":
    main()
