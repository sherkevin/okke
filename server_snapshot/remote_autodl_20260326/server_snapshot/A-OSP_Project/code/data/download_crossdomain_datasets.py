#!/usr/bin/env python3
"""
Task 2: Zero-Shot Cross-Domain Suite
=====================================
Targets:
  C) IU X-Ray   — medical domain, open-access chest radiographs + reports
  D) ChartQA    — dense numerical domain, chart images + QA pairs

Usage:
  python code/data/download_crossdomain_datasets.py --mini_test      # 5-10 samples each
  python code/data/download_crossdomain_datasets.py --full           # full download

All assets land under  data/benchmarks/{iu_xray,chartqa}/
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from io import BytesIO

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

IU_XRAY_REPO = "dz-osamu/IU-Xray"
IU_XRAY_HF_CANDIDATES = [
    (IU_XRAY_REPO, None),
    ("jkottu/iu-xray-dataset", None),
]

# JSON 中路径形如 /iu_xray/image/CXR2279_IM-0865/0.png，zip 内为 .../iu_xray/images/CXR2279_IM-0865/0.png
_IU_XRAY_PATH_RE = re.compile(r"(CXR\d+_IM-\d+/\d+\.png)")


def _iu_xray_find_zip_member(zip_names, pseudo_path):
    m = _IU_XRAY_PATH_RE.search(pseudo_path)
    if not m:
        return None
    suffix = m.group(1)
    for n in zip_names:
        if n.endswith(suffix) and "images" in n.replace("\\", "/"):
            return n
    return None


def _extract_iu_xray_images_from_hub_zip(
    n_samples: int,
    out_dir: Path,
    img_dir: Path,
) -> list[dict]:
    """
    从 dz-osamu/IU-Xray 的 image.zip 中按需解压前 n 条 val 样本对应的 PNG（约 1GB 缓存，仅首次下载）。
    """
    from huggingface_hub import hf_hub_download

    val_path = hf_hub_download(IU_XRAY_REPO, "val.jsonl", repo_type="dataset")
    zip_path = hf_hub_download(IU_XRAY_REPO, "image.zip", repo_type="dataset")

    lines = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            lines.append(json.loads(line))

    manifest: list[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for i, row in enumerate(tqdm(lines, desc="[IU-XRay] extract from zip")):
            entry = {"_index": i}
            local_paths: list[str] = []
            for k, v in row.items():
                if k != "images":
                    entry[k] = v
            for j, pseudo in enumerate(row.get("images", [])):
                member = _iu_xray_find_zip_member(names, pseudo)
                dest = img_dir / f"iu_xray_{i:05d}_{j}.png"
                if member:
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    local_paths.append(str(dest.relative_to(PROJECT_ROOT)))
                else:
                    local_paths.append("")
            entry["image_paths_local"] = local_paths
            entry["image_path"] = local_paths[0] if local_paths else None
            manifest.append(entry)

    return manifest

CHARTQA_HF_REPO = "HuggingFaceM4/ChartQA"


# ──────────────────────────────────────────────────────────────
#  Target C: IU X-Ray
# ──────────────────────────────────────────────────────────────

def download_iu_xray(n_samples: int, out_dir: Path):
    """
    Download Indiana University Chest X-Ray dataset via HuggingFace.
    Primary: dz-osamu/IU-Xray val.jsonl + image.zip (selective extract).
    Fallback: load_dataset when zip/jsonl unavailable.
    """
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print("[IU-XRay] Primary: val.jsonl + image.zip (dz-osamu/IU-Xray, cached after first fetch) ...")
    try:
        manifest = _extract_iu_xray_images_from_hub_zip(n_samples, out_dir, img_dir)
        manifest_path = out_dir / "iu_xray_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        n_imgs = len(list(img_dir.glob("*.png")))
        print(f"[IU-XRay] Manifest → {manifest_path}  ({len(manifest)} entries)")
        print(f"[IU-XRay] Images   → {img_dir}  ({n_imgs} png files)")
        if n_imgs > 0:
            from PIL import Image as PILImage
            sample_img = PILImage.open(next(img_dir.glob("*.png")))
            print(f"[IU-XRay] Sample image: {sample_img.size}, mode={sample_img.mode}")
        return True
    except Exception as e:
        print(f"[IU-XRay] Primary failed ({e}); trying load_dataset fallback ...")

    ds = None
    used_repo = None
    for repo_id, config in IU_XRAY_HF_CANDIDATES:
        try:
            print(f"[IU-XRay] Trying HuggingFace: {repo_id} (config={config}) ...")
            if config:
                ds = load_dataset(repo_id, config, split="train")
            else:
                ds = load_dataset(repo_id, split="train")
            used_repo = repo_id
            print(f"[IU-XRay] Loaded from {repo_id}  ({len(ds)} rows)")
            break
        except Exception as e:
            print(f"[IU-XRay] {repo_id} failed: {e}")

    if ds is None:
        print("[IU-XRay] ERROR: Could not load from HuggingFace.")
        print("          Fallback: download from Kaggle or OpenI manually.")
        print("          Kaggle:  https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university")
        print("          HF alt:  https://huggingface.co/datasets/dz-osamu/IU-Xray")
        return False

    subset = ds.select(range(min(n_samples, len(ds))))
    print(f"[IU-XRay] Selected {len(subset)} / {len(ds)} rows")

    print("\n[IU-XRay] Schema preview (first row):")
    row0 = subset[0]
    for k, v in row0.items():
        vtype = type(v).__name__
        if isinstance(v, bytes):
            vpreview = f"<bytes len={len(v)}>"
        elif isinstance(v, str) and len(v) > 200:
            vpreview = v[:200] + "..."
        elif isinstance(v, list) and len(v) > 5:
            vpreview = str(v[:3]) + f" ... ({len(v)} items)"
        else:
            vpreview = str(v)[:200]
        print(f"  {k:30s}  ({vtype})  {vpreview}")

    manifest = []
    saved_imgs = 0
    for i, row in enumerate(tqdm(subset, desc="[IU-XRay] Saving")):
        entry = {"_index": i}

        for k, v in row.items():
            if k == "image" and v is not None:
                try:
                    from PIL import Image as PILImage
                    if isinstance(v, PILImage.Image):
                        img_path = img_dir / f"iu_xray_{i:05d}.png"
                        v.save(str(img_path))
                        entry["image_path"] = str(img_path.relative_to(PROJECT_ROOT))
                        saved_imgs += 1
                        continue
                except Exception:
                    pass

            if isinstance(v, list):
                entry[k] = v
            elif isinstance(v, (str, int, float, bool)) or v is None:
                entry[k] = v

        manifest.append(entry)

    manifest_path = out_dir / "iu_xray_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_imgs = len(list(img_dir.glob("*")))
    print(f"[IU-XRay] Manifest → {manifest_path}  ({len(manifest)} entries)")
    print(f"[IU-XRay] Images   → {img_dir}  ({n_imgs} saved as files)")

    if n_imgs > 0:
        from PIL import Image as PILImage
        sample_img = PILImage.open(next(img_dir.iterdir()))
        print(f"[IU-XRay] Sample image: {sample_img.size}, mode={sample_img.mode}")

    report_fields = [k for k in row0.keys() if any(
        term in k.lower() for term in [
            "report", "finding", "impression", "text", "caption",
            "response", "query", "question", "answer"
        ]
    )]
    if report_fields:
        print(f"[IU-XRay] Report/QA fields detected: {report_fields}")
        for rf in report_fields:
            sample_text = row0[rf]
            if isinstance(sample_text, str):
                print(f"  {rf}: {sample_text[:300]}")
    else:
        print("[IU-XRay] No obvious report/text field detected — inspect manifest manually.")

    img_field = [k for k in row0.keys() if "image" in k.lower()]
    if img_field and n_imgs == 0:
        sample_val = row0[img_field[0]]
        if isinstance(sample_val, list) and all(isinstance(x, str) for x in sample_val):
            print(f"\n[IU-XRay] NOTE: Image field '{img_field[0]}' contains PATH strings (not embedded):")
            for p in sample_val[:3]:
                print(f"    {p}")
            print("[IU-XRay] Images must be downloaded separately or fetched from HF file storage.")
            print(f"[IU-XRay] HF repo: {used_repo}")

    return True


# ──────────────────────────────────────────────────────────────
#  Target D: ChartQA
# ──────────────────────────────────────────────────────────────

def download_chartqa(n_samples: int, out_dir: Path):
    """Download ChartQA dataset via HuggingFace. Saves chart images + QA pairs."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"[ChartQA] Loading from HuggingFace: {CHARTQA_HF_REPO} ...")
    try:
        ds = load_dataset(CHARTQA_HF_REPO, split="test")
        print(f"[ChartQA] Loaded test split ({len(ds)} rows)")
    except Exception as e:
        print(f"[ChartQA] test split failed ({e}), trying train ...")
        try:
            ds = load_dataset(CHARTQA_HF_REPO, split="train")
            print(f"[ChartQA] Loaded train split ({len(ds)} rows)")
        except Exception as e2:
            print(f"[ChartQA] ERROR: Could not load ChartQA: {e2}")
            return False

    subset = ds.select(range(min(n_samples, len(ds))))
    print(f"[ChartQA] Selected {len(subset)} / {len(ds)} rows")

    print("\n[ChartQA] Schema preview (first row):")
    row0 = subset[0]
    for k, v in row0.items():
        vtype = type(v).__name__
        if isinstance(v, bytes):
            vpreview = f"<bytes len={len(v)}>"
        else:
            vpreview = str(v)[:200]
        print(f"  {k:30s}  ({vtype})  {vpreview}")

    manifest = []
    for i, row in enumerate(tqdm(subset, desc="[ChartQA] Saving")):
        entry = {"_index": i}
        for k, v in row.items():
            if k == "image" and v is not None:
                try:
                    from PIL import Image
                    if isinstance(v, Image.Image):
                        img_path = img_dir / f"chart_{i:05d}.png"
                        v.save(str(img_path))
                        entry["image_path"] = str(img_path.relative_to(PROJECT_ROOT))
                        continue
                except Exception:
                    pass
            if isinstance(v, (str, int, float, bool, list)) or v is None:
                entry[k] = v
        manifest.append(entry)

    manifest_path = out_dir / "chartqa_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_imgs = len(list(img_dir.glob("*")))
    print(f"[ChartQA] Manifest → {manifest_path}  ({len(manifest)} entries)")
    print(f"[ChartQA] Images   → {img_dir}  ({n_imgs} files)")

    qa_fields = [k for k in row0.keys() if any(
        term in k.lower() for term in ["query", "question", "answer", "label"]
    )]
    if qa_fields:
        print(f"[ChartQA] QA fields: {qa_fields}")
        for qf in qa_fields:
            print(f"  {qf}: {str(row0[qf])[:200]}")

    return True


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download cross-domain datasets: IU X-Ray + ChartQA"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mini_test", action="store_true",
        help="Download only 5-10 samples per dataset for verification"
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

    n = args.n_samples if args.mini_test else 999_999

    print("=" * 70)
    print("A-OSP Data Infrastructure — Task 2: Cross-Domain Suite")
    print(f"Mode: {'MINI TEST (' + str(n) + ' samples)' if args.mini_test else 'FULL'}")
    print("=" * 70)

    iu_dir = DATA_DIR / "iu_xray"
    cqa_dir = DATA_DIR / "chartqa"

    # --- Target C: IU X-Ray ---
    print("\n" + "─" * 50)
    print("TARGET C: IU X-Ray (Medical Domain)")
    print("─" * 50)
    ok_iu = download_iu_xray(n, iu_dir)
    print(f"[IU-XRay] {'SUCCESS' if ok_iu else 'FAILED'}")

    # --- Target D: ChartQA ---
    print("\n" + "─" * 50)
    print("TARGET D: ChartQA (Dense Numerical Domain)")
    print("─" * 50)
    ok_cqa = download_chartqa(n, cqa_dir)
    print(f"[ChartQA] {'SUCCESS' if ok_cqa else 'FAILED'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  IU X-Ray : {'OK' if ok_iu else 'FAIL'}  → {iu_dir}")
    print(f"  ChartQA  : {'OK' if ok_cqa else 'FAIL'}  → {cqa_dir}")
    if args.mini_test:
        print("\n  >>> Pipeline verified. Inspect outputs, then run --full <<<")


if __name__ == "__main__":
    main()
