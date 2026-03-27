"""
A-OSP 校准弹药生成器 — 增量式 COCO 直连版
==========================================
从 images.cocodataset.org 并发下载 val2014 图片，
施加强高斯模糊，摧毁全部高频纹理。

核心策略：
  - 使用 requests.Session 连接池 + 自动重试，大幅提升下载可靠性
  - 在已知有效 ID 附近密集探测（COCO ID 有聚簇特征）
  - 增量模式：已有图片不重复下载，只补足至目标数量

用法：
  python generate_blurred_coco.py
  python generate_blurred_coco.py --num_samples 200 --blur_radius 20
"""

import argparse
import io
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")
OUTPUT_DIR = PROJECT_ROOT / "data" / "blurred_calibration"
COCO_IMG_BASE = "http://images.cocodataset.org/val2014/COCO_val2014_"

NUM_SAMPLES = 200
BLUR_RADIUS = 20
SEED = 42
MAX_WORKERS = 24
TIMEOUT = 20


def make_session() -> requests.Session:
    """创建带重试和连接池的 Session。"""
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "A-OSP/1.0"})
    return session


def build_candidate_pool(existing_ids: set, num_needed: int, seed: int) -> list:
    """
    构建候选 ID 池。COCO val2014 ID 有聚簇特征：
    许多有效 ID 相邻或间隔很小。
    策略：在已知有效 ID 附近 ±50 范围密集探测 + 大范围随机探测。
    """
    rng = np.random.RandomState(seed)
    candidates = set()

    # 策略 1：已知有效 ID 附近密集探测
    for eid in existing_ids:
        for offset in range(-30, 31):
            cand = eid + offset
            if cand > 0 and cand not in existing_ids:
                candidates.add(cand)

    # 策略 2：在 COCO val2014 高密度区间随机探测
    high_density_ranges = [
        (1, 10000),
        (100000, 200000),
        (250000, 400000),
        (450000, 580000),
    ]
    for lo, hi in high_density_ranges:
        batch = rng.randint(lo, hi, size=num_needed * 3)
        for b in batch:
            if b not in existing_ids:
                candidates.add(int(b))

    # 策略 3：全范围随机
    full_random = rng.randint(1, 600000, size=num_needed * 5)
    for r in full_random:
        if r not in existing_ids:
            candidates.add(int(r))

    result = list(candidates)
    rng.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="A-OSP 校准弹药生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--blur_radius", type=int, default=BLUR_RADIUS)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    blur_dir = out_dir / "blur"
    solid_dir = out_dir / "solid"
    blur_dir.mkdir(parents=True, exist_ok=True)
    solid_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载已有的 manifest（增量模式）────────────────────────
    manifest_path = out_dir / "calibration_manifest.json"
    manifest = []
    existing_ids = set()
    start_seq = 0

    if manifest_path.exists():
        with open(manifest_path) as f:
            old = json.load(f)
        manifest = old.get("items", [])
        existing_ids = {item["coco_image_id"] for item in manifest}
        start_seq = len(manifest)
        print(f"[INFO] 增量模式：已有 {start_seq} 张图片，继续补足至 {args.num_samples} 张。")

    still_needed = args.num_samples - start_seq
    if still_needed <= 0:
        print(f"[INFO] 已有 {start_seq} 张，满足 {args.num_samples} 张的要求。无需额外下载。")
        return 0

    print("=" * 60)
    print("  A-OSP 校准弹药生成器 (增量模式)")
    print(f"  已有: {start_seq} 张 | 还需: {still_needed} 张")
    print(f"  模糊参数: radius={args.blur_radius} (核宽={2*args.blur_radius+1}px)")
    print(f"  并发线程: {args.max_workers}")
    print("=" * 60)

    candidates = build_candidate_pool(existing_ids, still_needed, args.seed + start_seq)
    print(f"[INFO] 候选 ID 池: {len(candidates)} 个")

    session = make_session()
    t_start = time.time()
    lock = threading.Lock()
    new_count = 0
    fail_count = 0
    stop_flag = threading.Event()

    def process_one(img_id):
        nonlocal new_count, fail_count
        if stop_flag.is_set():
            return None

        url = f"{COCO_IMG_BASE}{img_id:012d}.jpg"
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                with lock:
                    fail_count += 1
                return None
            data = resp.content
        except Exception:
            with lock:
                fail_count += 1
            return None

        if stop_flag.is_set():
            return None

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            with lock:
                fail_count += 1
            return None

        blurred = img.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))

        arr = np.array(img)
        mean_rgb = arr.mean(axis=(0, 1)).astype(np.uint8)
        solid = Image.new("RGB", img.size, tuple(mean_rgb))

        with lock:
            if new_count >= still_needed:
                stop_flag.set()
                return None

            new_count += 1
            seq = start_seq + new_count

        fname_blur = f"blur_{seq:03d}.jpg"
        fname_solid = f"solid_{seq:03d}.jpg"
        blurred.save(blur_dir / fname_blur, quality=95)
        solid.save(solid_dir / fname_solid, quality=95)

        entry = {
            "index": seq,
            "coco_image_id": int(img_id),
            "blur_file": f"blur/{fname_blur}",
            "solid_file": f"solid/{fname_solid}",
            "original_size": list(img.size),
        }

        with lock:
            manifest.append(entry)
            total = start_seq + new_count
            if new_count % 10 == 0 or new_count == still_needed:
                elapsed = time.time() - t_start
                rate = new_count / elapsed if elapsed > 0 else 0
                print(f"  [{total}/{args.num_samples}]  "
                      f"+{new_count} 新 | 404={fail_count} | "
                      f"{rate:.1f} img/s | {elapsed:.0f}s")

            if new_count >= still_needed:
                stop_flag.set()

        return entry

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        batch_size = 48
        for batch_start in range(0, len(candidates), batch_size):
            if stop_flag.is_set():
                break

            batch = candidates[batch_start:batch_start + batch_size]
            futures = [executor.submit(process_one, iid) for iid in batch]

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

            if stop_flag.is_set():
                break

    manifest.sort(key=lambda x: x["index"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(manifest),
            "blur_radius": args.blur_radius,
            "blur_kernel_width": 2 * args.blur_radius + 1,
            "seed": args.seed,
            "source": "COCO val2014 (images.cocodataset.org)",
            "items": manifest,
        }, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_start
    final_count = start_seq + new_count
    print(f"\n{'='*60}")
    print(f"  校准弹药生成完毕！")
    print(f"  最终: {final_count}/{args.num_samples} 张")
    print(f"  本次新增: {new_count} 张 | 探测失败: {fail_count}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  输出:")
    print(f"    模糊图 → {blur_dir}/")
    print(f"    纯色图 → {solid_dir}/")
    print(f"    清单   → {manifest_path}")
    print(f"{'='*60}")

    if final_count >= args.num_samples:
        print(f"\n  ✓ Agent 1 可直接读取: {blur_dir}")
        return 0
    else:
        print(f"\n  ⚠ 仅获取 {final_count} 张，请重新运行以继续增量下载。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
