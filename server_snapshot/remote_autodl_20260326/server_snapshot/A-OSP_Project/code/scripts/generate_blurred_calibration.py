"""
A-OSP 校准弹药生成器
====================
从 MSCOCO val2014 中随机抽取 N 张图片，生成两类流形掩码：
  1. 强高斯模糊 (Strong Gaussian Blur)  → blurred_calibration/blur/
  2. 纯色均值图 (Solid Mean-RGB Image) → blurred_calibration/solid/

输出供 Agent 1 的 subspace_extractor.py 读取，用于 Distribution-Shift Guidance。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

# ────────────────────────────────────────────────────────────────
# 默认路径与超参数
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")
DEFAULT_COCO_DIR = PROJECT_ROOT / "data" / "coco_val2014"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "blurred_calibration"

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANN_URL = ("http://images.cocodataset.org/annotations/"
                "annotations_trainval2014.zip")

BLUR_KERNEL_SIZE = 151          # 强高斯模糊核大小（必须为奇数）
BLUR_SIGMA = 80                 # 极大 sigma，摧毁一切高频纹理
NUM_SAMPLES = 200
SEED = 42


def download_and_extract_coco(coco_dir: Path):
    """下载 COCO val2014 图片集并解压。"""
    coco_dir.mkdir(parents=True, exist_ok=True)
    zip_path = coco_dir / "val2014.zip"

    if (coco_dir / "val2014").exists() and len(list((coco_dir / "val2014").glob("*.jpg"))) > 1000:
        print(f"[INFO] COCO val2014 已存在于 {coco_dir / 'val2014'}，跳过下载。")
        return coco_dir / "val2014"

    if not zip_path.exists():
        print(f"[INFO] 正在下载 COCO val2014 ({COCO_VAL_URL})...")
        print("       这可能需要几分钟，文件大小约 6.2GB。")
        urlretrieve(COCO_VAL_URL, str(zip_path), _download_progress)
        print()

    print("[INFO] 正在解压...")
    import zipfile
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(coco_dir))
    print(f"[INFO] 解压完成 → {coco_dir / 'val2014'}")

    zip_path.unlink(missing_ok=True)
    return coco_dir / "val2014"


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def apply_strong_gaussian_blur(img_bgr: np.ndarray) -> np.ndarray:
    """强高斯模糊：保留低频色彩分布，摧毁所有高频纹理和细粒度视觉线索。"""
    return cv2.GaussianBlur(img_bgr, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)


def generate_solid_mean_image(img_bgr: np.ndarray) -> np.ndarray:
    """纯色均值图：用整张图的 RGB 均值填满，作为流形掩码纯粹性验证。"""
    mean_color = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
    return np.full_like(img_bgr, mean_color)


def main():
    parser = argparse.ArgumentParser(description="A-OSP 校准弹药生成器")
    parser.add_argument("--coco_dir", type=str, default=str(DEFAULT_COCO_DIR),
                        help="COCO val2014 存放路径")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="输出路径")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES,
                        help="采样图片数量")
    parser.add_argument("--blur_kernel", type=int, default=BLUR_KERNEL_SIZE,
                        help="高斯模糊核大小（奇数）")
    parser.add_argument("--blur_sigma", type=float, default=BLUR_SIGMA,
                        help="高斯模糊 sigma")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip_download", action="store_true",
                        help="跳过下载，假定图片已在 coco_dir/val2014/")
    parser.add_argument("--coco_image_dir", type=str, default=None,
                        help="直接指定 COCO 图片目录（跳过下载逻辑）")
    args = parser.parse_args()

    global BLUR_KERNEL_SIZE, BLUR_SIGMA
    BLUR_KERNEL_SIZE = args.blur_kernel
    BLUR_SIGMA = args.blur_sigma

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 定位 COCO 图片 ──────────────────────────────────────────
    if args.coco_image_dir:
        img_dir = Path(args.coco_image_dir)
    elif args.skip_download:
        img_dir = Path(args.coco_dir) / "val2014"
    else:
        img_dir = download_and_extract_coco(Path(args.coco_dir))

    all_images = sorted(img_dir.glob("*.jpg"))
    if not all_images:
        all_images = sorted(img_dir.glob("*.JPEG")) + sorted(img_dir.glob("*.png"))
    if len(all_images) == 0:
        print(f"[ERROR] 在 {img_dir} 中未找到任何图片！请检查路径。")
        sys.exit(1)

    print(f"[INFO] 在 {img_dir} 中找到 {len(all_images)} 张图片。")

    sampled = random.sample(all_images, min(args.num_samples, len(all_images)))
    print(f"[INFO] 随机采样 {len(sampled)} 张用于校准集。")

    # ── 输出目录 ─────────────────────────────────────────────────
    out_root = Path(args.output_dir)
    blur_dir = out_root / "blur"
    solid_dir = out_root / "solid"
    blur_dir.mkdir(parents=True, exist_ok=True)
    solid_dir.mkdir(parents=True, exist_ok=True)

    # ── 处理图片 ─────────────────────────────────────────────────
    manifest = []
    for i, img_path in enumerate(sampled):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] 无法读取 {img_path}，跳过。")
            continue

        blurred = apply_strong_gaussian_blur(img_bgr)
        solid = generate_solid_mean_image(img_bgr)

        stem = img_path.stem
        blur_out = blur_dir / f"{stem}_blur.jpg"
        solid_out = solid_dir / f"{stem}_solid.jpg"

        cv2.imwrite(str(blur_out), blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(solid_out), solid, [cv2.IMWRITE_JPEG_QUALITY, 95])

        manifest.append({
            "index": i,
            "original": str(img_path),
            "blurred": str(blur_out),
            "solid": str(solid_out),
            "original_shape": list(img_bgr.shape[:2]),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(sampled):
            print(f"  [{i + 1}/{len(sampled)}] 已处理。")

    # ── 保存 manifest ────────────────────────────────────────────
    manifest_path = out_root / "calibration_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "total": len(manifest),
            "blur_kernel": BLUR_KERNEL_SIZE,
            "blur_sigma": BLUR_SIGMA,
            "seed": args.seed,
            "items": manifest,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[DONE] 校准弹药生成完毕！")
    print(f"  模糊图片 → {blur_dir}  ({len(manifest)} 张)")
    print(f"  纯色图片 → {solid_dir}  ({len(manifest)} 张)")
    print(f"  清单文件 → {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
