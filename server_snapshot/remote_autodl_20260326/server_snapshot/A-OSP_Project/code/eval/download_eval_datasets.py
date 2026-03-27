#!/usr/bin/env python3
"""
A-OSP Evaluation Pipeline — Dataset Downloader
================================================
Downloads and prepares all evaluation datasets:
  1. COCO val2014 images (only those needed by POPE)
  2. MMHal-Bench images + metadata (96 samples, long-text hallucination)

Usage:
    python download_eval_datasets.py --tasks all
    python download_eval_datasets.py --tasks coco_pope
    python download_eval_datasets.py --tasks mmhal
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COCO_IMAGE_DIR = PROJECT_ROOT / "data" / "coco_val2014"
POPE_DIR = PROJECT_ROOT / "data" / "pope"
MMHAL_DIR = PROJECT_ROOT / "data" / "mmhal_bench"


# ═══════════════════════════════════════════════════════════════════
# COCO val2014 — download only images referenced by POPE
# ═══════════════════════════════════════════════════════════════════

def collect_pope_image_ids() -> set[str]:
    """Scan all POPE JSONL files and collect unique image IDs."""
    ids = set()
    for jl in POPE_DIR.glob("pope_coco_*.jsonl"):
        if "mini" in jl.name:
            continue
        with open(jl) as f:
            for line in f:
                rec = json.loads(line.strip())
                ids.add(rec["image"])
    return ids


def download_single_coco(img_name: str) -> str:
    """Download one COCO val2014 image. Returns status string."""
    jpg = f"{img_name}.jpg"
    dest = COCO_IMAGE_DIR / jpg
    if dest.exists() and dest.stat().st_size > 1000:
        return f"[SKIP] {jpg}"

    url = f"http://images.cocodataset.org/val2014/{jpg}"
    try:
        urllib.request.urlretrieve(url, str(dest))
        return f"[OK]   {jpg} ({dest.stat().st_size // 1024}KB)"
    except Exception as e:
        return f"[FAIL] {jpg}: {e}"


def download_coco_for_pope(max_workers: int = 8):
    """Download all COCO val2014 images referenced by POPE splits."""
    COCO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_ids = collect_pope_image_ids()
    print(f"[COCO] Found {len(image_ids)} unique images across POPE splits")

    already = sum(
        1 for i in image_ids
        if (COCO_IMAGE_DIR / f"{i}.jpg").exists()
    )
    print(f"[COCO] Already downloaded: {already} / {len(image_ids)}")

    to_download = [i for i in image_ids if not (COCO_IMAGE_DIR / f"{i}.jpg").exists()]
    if not to_download:
        print("[COCO] All images present. Skipping.")
        return

    print(f"[COCO] Downloading {len(to_download)} images ({max_workers} threads) ...")
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_single_coco, img): img for img in to_download}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if "[OK]" in result:
                ok += 1
            elif "[FAIL]" in result:
                fail += 1
                print(f"  {result}")
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(to_download)} (ok={ok}, fail={fail})")

    elapsed = time.time() - t0
    print(f"[COCO] Done in {elapsed:.0f}s — ok={ok}, fail={fail}, skip={already}")


# ═══════════════════════════════════════════════════════════════════
# MMHal-Bench — download images + prepare JSONL
# ═══════════════════════════════════════════════════════════════════

def download_mmhal_bench(max_workers: int = 8):
    """Download MMHal-Bench images from HuggingFace repo + prepare JSONL."""
    MMHAL_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = MMHAL_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    # Step 1: Download images from HF repo
    from huggingface_hub import hf_hub_download, list_repo_tree

    print("[MMHal] Listing images from Shengcao1006/MMHal-Bench ...")
    repo_files = list(list_repo_tree(
        "Shengcao1006/MMHal-Bench", repo_type="dataset", recursive=True
    ))
    image_files = [
        f.rfilename for f in repo_files
        if hasattr(f, "rfilename") and f.rfilename.startswith("images/")
    ]
    print(f"[MMHal] Found {len(image_files)} images in repo")

    downloaded = 0
    for fname in image_files:
        basename = os.path.basename(fname)
        dest = img_dir / basename
        if dest.exists() and dest.stat().st_size > 100:
            continue
        try:
            local = hf_hub_download(
                "Shengcao1006/MMHal-Bench", fname,
                repo_type="dataset", local_dir=str(MMHAL_DIR / "_hf_cache"),
            )
            import shutil
            shutil.copy2(local, str(dest))
            downloaded += 1
        except Exception as e:
            print(f"  [FAIL] {fname}: {e}")

    print(f"[MMHal] Downloaded {downloaded} new images")

    # Step 2: Load response_template.json and map images
    template_path = PROJECT_ROOT / "data" / "mmhal_bench_raw" / "response_template.json"
    if not template_path.exists():
        print("[MMHal] Downloading response_template.json ...")
        hf_hub_download(
            "Shengcao1006/MMHal-Bench", "response_template.json",
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT / "data" / "mmhal_bench_raw"),
        )

    with open(template_path) as f:
        raw_data = json.load(f)

    # Step 3: Build image_id → local filename mapping
    hf_images = {os.path.splitext(os.path.basename(f))[0]: os.path.basename(f) for f in image_files}

    # Step 4: Convert to our JSONL format
    outpath = MMHAL_DIR / "mmhal_bench.jsonl"
    count = 0
    missing_images = 0
    with open(outpath, "w") as f:
        for i, item in enumerate(raw_data):
            image_id = item["image_id"]

            local_img = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_dir / f"{image_id}{ext}"
                if candidate.exists():
                    local_img = candidate.name
                    break
            if local_img is None:
                for fname in os.listdir(img_dir):
                    if image_id in fname:
                        local_img = fname
                        break

            if local_img is None:
                missing_images += 1

            record = {
                "question_id": i,
                "image_id": image_id,
                "image": local_img or f"{image_id}.jpg",
                "image_src": item.get("image_src", ""),
                "question": item["question"],
                "gt_answer": item["gt_answer"],
                "question_type": item["question_type"],
                "question_topic": item["question_topic"],
                "image_content": item.get("image_content", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[MMHal] Saved {count} samples → {outpath}")
    if missing_images:
        print(f"[MMHal] WARNING: {missing_images} images not found locally")

    # Step 5: Also try Flickr fallback for missing images
    if missing_images > 0:
        print("[MMHal] Attempting Flickr URL fallback for missing images ...")
        _download_mmhal_flickr_fallback(raw_data, img_dir, hf_images)


def _download_mmhal_flickr_fallback(raw_data, img_dir, hf_images):
    """Try downloading missing images from their Flickr source URLs."""
    downloaded = 0
    for item in raw_data:
        image_id = item["image_id"]
        exists = any(
            (img_dir / f"{image_id}{ext}").exists()
            for ext in [".jpg", ".png", ".jpeg"]
        ) or any(image_id in f for f in os.listdir(img_dir))

        if exists:
            continue

        src_url = item.get("image_src", "")
        if not src_url:
            continue

        ext = os.path.splitext(src_url)[1] or ".jpg"
        dest = img_dir / f"{image_id}{ext}"
        try:
            urllib.request.urlretrieve(src_url, str(dest))
            downloaded += 1
        except Exception:
            pass

    print(f"[MMHal] Flickr fallback: downloaded {downloaded} additional images")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="A-OSP Dataset Downloader")
    p.add_argument("--tasks", type=str, default="all",
                    choices=["all", "coco_pope", "mmhal"],
                    help="Which datasets to download")
    p.add_argument("--max_workers", type=int, default=8)
    args = p.parse_args()

    if args.tasks in ("all", "coco_pope"):
        download_coco_for_pope(max_workers=args.max_workers)

    if args.tasks in ("all", "mmhal"):
        download_mmhal_bench(max_workers=args.max_workers)

    print("\n[Done] Dataset download complete.")


if __name__ == "__main__":
    main()
