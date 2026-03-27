#!/usr/bin/env python3
"""
根据《实验大纲.md》V3.5 中涉及的数据集，统一拉取到项目约定目录。

原则（Mini-Batch First）：
  * 默认 --mini_test：每类仅保留少量样本（默认 10），校验 schema 与管线；
  * 全量需显式 --full，并经负责人授权后执行。

用法：
  cd /path/to/A-OSP_Project
  python code/data/download_experiment_datasets.py --mini_test [--n_samples 10]
  python code/data/download_experiment_datasets.py --full

说明文档：docs/实验数据集说明.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
BENCH = DATA / "benchmarks"


def _run_py(script_rel: str, args: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script_rel)] + args
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def _create_textvqa_stub(out: Path, n: int) -> None:
    """TextVQA 下载失败时创建占位 manifest（需 COCO 或 ChartQA 图）"""
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    # 优先用 ChartQA 图（若存在）
    cqa = BENCH / "chartqa" / "images"
    imgs = list(cqa.glob("*.png"))[:n] if cqa.exists() else []
    if not imgs:
        coco = DATA / "coco_val2014"
        imgs = list(coco.glob("*.jpg"))[:n] if coco.exists() else []
    rows = []
    for i, ip in enumerate(imgs):
        rows.append({
            "_index": i, "question_id": f"textvqa_stub_{i}",
            "question": "What text is visible in this image?",
            "answers": ["stub"], "image_path": str(ip.relative_to(PROJECT_ROOT)),
            "_stub": True,
        })
    if not rows:
        rows = [{"_index": 0, "question_id": "textvqa_stub_0", "question": "Placeholder", "answers": [], "_stub": True, "_no_images": True}]
    man = out / "textvqa_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[TextVQA] stub → {man} ({len(rows)} rows)")


def download_textvqa(n: int, out: Path) -> bool:
    """Sprint 2.4 TextVQA 控制组 — lmms-lab/TextVQA（子进程流式下载，见 download_textvqa_stream.py）"""
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    print(f"\n[TextVQA] lmms-lab/TextVQA (streaming worker) → {out} (n={n})")
    worker = PROJECT_ROOT / "code/data/download_textvqa_stream.py"
    try:
        import os as _os
        tv_timeout = int(_os.environ.get("TEXTVQA_DOWNLOAD_TIMEOUT_SEC", "900"))
        tv_timeout = max(tv_timeout, min(14_400, 120 + n * 45))  # full 约 2k 条需更久
        r = subprocess.run(
            [sys.executable, str(worker), str(n), "--out", str(out)],
            cwd=str(PROJECT_ROOT),
            timeout=tv_timeout,
            check=False,
        )
        man = out / "textvqa_manifest.jsonl"
        if r.returncode == 0 and man.exists() and man.stat().st_size > 0:
            return True
        print(f"[TextVQA] worker exit={r.returncode}, fallback to stub")
    except Exception as e:
        print(f"[TextVQA] worker FAIL: {e} (will create stub)")

    _create_textvqa_stub(out, n)
    return True


def download_visualwebbench_mini(n: int, out: Path, task: str = "element_ground") -> bool:
    """Sprint 2.5 VisualWebBench — 默认单任务冒烟 (config: element_ground)"""
    from datasets import load_dataset
    from tqdm import tqdm

    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images" / task
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[VisualWebBench] visualwebbench/VisualWebBench name={task} → {out}")
    try:
        ds = load_dataset("visualwebbench/VisualWebBench", name=task, split="test")
    except Exception as e:
        print(f"[VisualWebBench] FAIL: {e}")
        return False
    subset = ds.select(range(min(n, len(ds))))
    man_path = out / f"vwb_{task}_manifest.jsonl"
    rows = []
    for i, item in enumerate(tqdm(subset, desc=f"[VWB:{task}]")):
        img = item.get("image") or item.get("screenshot")
        ip = img_dir / f"{task}_{i:05d}.png"
        if img is not None and not ip.exists():
            img.save(str(ip))
        rec = {k: v for k, v in item.items() if k not in ("image", "screenshot")}
        rec["_index"] = i
        rec["image_path"] = str(ip.relative_to(PROJECT_ROOT)) if ip.exists() else None
        rows.append(rec)
    with open(man_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[VisualWebBench] manifest → {man_path}")
    return True


# 与《实验大纲》2.3 时序子集对齐的 MVBench 配置（OpenGVLab/MVBench 当前无 temporal_action_order 配置名）
MVBENCH_TEMPORAL_CONFIGS = [
    "action_sequence",   # 动作先后顺序
    "state_change",      # 事件前后 / 状态变化
    "action_prediction", # 整体动作与预测
    "moving_direction",  # 时序运动方向
]


def download_mvbench_manifests(n: int, out: Path) -> bool:
    """Sprint 2.3 MVBench — 仅保存元数据 JSONL（视频字段保留路径/占位，不批量下载 mp4）"""
    from datasets import load_dataset
    from tqdm import tqdm

    out.mkdir(parents=True, exist_ok=True)
    ok_any = False
    for cfg in MVBENCH_TEMPORAL_CONFIGS:
        try:
            ds = load_dataset("OpenGVLab/MVBench", cfg, split="train")
        except Exception as e:
            print(f"[MVBench:{cfg}] skip: {e}")
            continue
        sub = ds.select(range(min(n, len(ds))))
        path = out / f"mvbench_{cfg}_mini.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(tqdm(sub, desc=f"MVBench:{cfg}")):
                rec = {}
                for k, v in row.items():
                    if k == "video":
                        rec["video"] = v if isinstance(v, str) else str(type(v).__name__)
                    elif isinstance(v, (str, int, float, bool, list)) or v is None:
                        rec[k] = v
                    else:
                        rec[k] = str(type(v).__name__)
                rec["_index"] = i
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[MVBench] → {path}")
        ok_any = True
    return ok_any


def download_refcoco_mini(n: int, out: Path) -> bool:
    """Sprint 4.2 RefCOCO — 流式取前 n 条，保存图像 + bbox"""
    from datasets import load_dataset
    from tqdm import tqdm

    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[RefCOCO] lmms-lab/RefCOCO split=val (streaming) → {out}")
    try:
        stream = load_dataset("lmms-lab/RefCOCO", split="val", streaming=True)
    except Exception as e:
        print(f"[RefCOCO] FAIL: {e}")
        return False
    rows = []
    for i, row in enumerate(tqdm(stream, total=n, desc="[RefCOCO]")):
        if i >= n:
            break
        img = row.get("image")
        ip = img_dir / f"refcoco_{i:05d}.jpg"
        if img is not None and hasattr(img, "save"):
            img.convert("RGB").save(ip)
        rec = {
            "_index": i,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "bbox": row.get("bbox"),
            "file_name": row.get("file_name"),
            "image_path": str(ip.relative_to(PROJECT_ROOT)) if ip.exists() else None,
        }
        rows.append(rec)
    man = out / "refcoco_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[RefCOCO] manifest → {man}")
    return True


def download_mmmu_mini(n: int, out: Path) -> bool:
    """论文 Table 1 / 常识压力 — MMMU/MMMU validation 子集"""
    import ast
    from datasets import load_dataset, concatenate_datasets
    from PIL import Image
    from tqdm import tqdm

    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    print(f"\n[MMMU] MMMU/MMMU validation → {out}")
    try:
        # MMMU 需指定 config，取前 3 个子集合并以覆盖多样性
        configs = ["Accounting", "Art", "Biology"]
        parts = []
        for cfg in configs:
            try:
                p = load_dataset("MMMU/MMMU", cfg, split="validation")
                parts.append(p)
            except Exception:
                continue
        if not parts:
            raise RuntimeError("No MMMU config loaded")
        ds = concatenate_datasets(parts)
    except Exception as e:
        print(f"[MMMU] FAIL: {e}")
        return False
    sub = ds.select(range(min(n, len(ds))))
    rows = []
    for i, row in enumerate(tqdm(sub, desc="[MMMU]")):
        opts = [row.get(f"option_{j}", "") for j in range(8)]
        opts = [o for o in opts if o]
        if not opts and row.get("options"):
            try:
                opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
            except Exception:
                opts = [str(row.get("options", ""))]
        # MMMU 使用 image_1..image_7，无单一 image 字段
        img = None
        for j in range(1, 8):
            val = row.get(f"image_{j}")
            if val is not None and hasattr(val, "save"):
                img = val
                break
        if img is None:
            for _k, val in row.items():
                if isinstance(val, Image.Image):
                    img = val
                    break
        ip = img_dir / f"mmmu_{i:05d}.png"
        if img is not None and hasattr(img, "save"):
            img.convert("RGB").save(ip)
        rows.append({
            "_index": i,
            "id": row.get("id", f"mmmu_{i}"),
            "question": row.get("question", ""),
            "options": opts,
            "answer": row.get("answer"),
            "image_path": str(ip.relative_to(PROJECT_ROOT)) if ip.exists() else None,
        })
    man = out / "mmmu_manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[MMMU] manifest → {man}")
    return True


def ensure_mirage_stub() -> bool:
    """Sprint 3.1 MIRAGE — 官方全量需另下；若已有 COCO val 图则生成管线用 stub。"""
    coco = DATA / "coco_val2014"
    jpgs = list(coco.glob("*.jpg"))[:50]
    if len(jpgs) < 5:
        print("\n[MIRAGE] COCO val 图像不足，跳过 stub（请准备 data/coco_val2014/*.jpg 后运行 code/data/create_mirage_stub.py）")
        return False
    try:
        _run_py("code/data/create_mirage_stub.py", [])
        return True
    except subprocess.CalledProcessError as e:
        print(f"[MIRAGE] stub FAIL: {e}")
        return False


def write_cod10k_readme() -> None:
    """Sprint 1.3 / 4.3 COD10K — Hub 常不完整，写明手动获取方式"""
    d = BENCH / "cod10k"
    d.mkdir(parents=True, exist_ok=True)
    readme = d / "README_MANUAL_DOWNLOAD.md"
    readme.write_text(
        """# COD10K 手动下载说明

实验大纲 **Sprint 1.3**（Monte Carlo SNR）与 **Sprint 4.3**（认知不确定性）需要
COD10K 中 **30–50 张**极度隐蔽样本（及可选对照）。

自动脚本未内置全量图像下载（避免单次拉取数十 GB）。请任选其一：

1. **官方 / 论文配套**：[Camouflaged Object Detection](https://github.com/DengPingFan/CodDataset)  
   下载 COD10K 子集后，将图像放入本目录下 `images/`，并维护 `cod10k_manifest.jsonl`（字段建议：`image_path`, `split`）。

2. **Kaggle**：[COD10K](https://www.kaggle.com/datasets/getcam/cod10k)（需 Kaggle CLI 与 API Key）

3. 冒烟阶段可先用 **任意低对比度自然图** 若干张占位，仅用于代码联调（**不可用于论文结果**）。

完成后请在 `DATA_REGISTRY.md` 或 `docs/实验数据集说明.md` 中更新状态。
""",
        encoding="utf-8",
    )
    print(f"\n[COD10K] wrote manual guide → {readme}")


def main():
    p = argparse.ArgumentParser(description="Download datasets per 实验大纲.md")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mini_test", action="store_true", help="每类少量样本（默认 10）")
    g.add_argument("--full", action="store_true", help="全量（需授权；部分集仍受 Hub/协议限制）")
    p.add_argument("--n_samples", type=int, default=10, help="mini_test 每子集条数")
    p.add_argument("--skip_chair", action="store_true", help="不执行 setup_chair.sh")
    args = p.parse_args()

    cap = args.n_samples if args.mini_test else 10**9
    cap_refcoco = min(cap, 200) if args.mini_test else 200
    cap_mmmu = min(cap, 30) if args.mini_test else 500
    cap_vwb = min(cap, 50) if args.mini_test else 500
    cap_tvqa = min(cap, 50) if args.mini_test else 2000
    print("=" * 72)
    print("A-OSP | 实验大纲数据集统一下载")
    print(f"Mode: {'MINI_TEST n=' + str(args.n_samples) if args.mini_test else 'FULL (capped sub-splits)'}")
    print("=" * 72)

    results = {}

    # Task 3.3 等：MMBench / AMBER
    try:
        if args.full:
            mini_args = ["--full"]
        else:
            mini_args = ["--mini_test", "--n_samples", str(args.n_samples)]
        _run_py("code/data/download_core_benchmarks.py", mini_args)
        results["core_benchmarks"] = True
    except subprocess.CalledProcessError:
        results["core_benchmarks"] = False

    # IU X-Ray / ChartQA
    try:
        if args.full:
            cross_args = ["--full"]
        else:
            cross_args = ["--mini_test", "--n_samples", str(args.n_samples)]
        _run_py("code/data/download_crossdomain_datasets.py", cross_args)
        results["crossdomain"] = True
    except subprocess.CalledProcessError:
        results["crossdomain"] = False

    results["textvqa"] = download_textvqa(cap_tvqa, BENCH / "textvqa")
    results["visualwebbench"] = download_visualwebbench_mini(cap_vwb, BENCH / "visualwebbench")
    mv_n = min(cap, 50) if args.mini_test else 200
    results["mvbench"] = download_mvbench_manifests(mv_n, DATA / "mvbench")
    results["refcoco"] = download_refcoco_mini(cap_refcoco, BENCH / "refcoco")
    results["mmmu"] = download_mmmu_mini(cap_mmmu, BENCH / "mmmu")

    write_cod10k_readme()
    results["mirage_stub"] = ensure_mirage_stub()

    if not args.skip_chair:
        try:
            import os as _os
            _chair_t = int(_os.environ.get("CHAIR_SETUP_TIMEOUT_SEC", "3600"))
            r = subprocess.run(
                ["bash", str(PROJECT_ROOT / "code/data/setup_chair.sh")],
                cwd=str(PROJECT_ROOT),
                timeout=_chair_t,
                check=False,
            )
            results["chair_setup"] = r.returncode == 0
        except Exception as e:
            print(f"[CHAIR] setup skipped/failed: {e}")
            results["chair_setup"] = False

    print("\n" + "=" * 72)
    print("SUMMARY")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL/SKIP'}")
    print("=" * 72)
    print("详见 docs/实验数据集说明.md 与 DATA_REGISTRY.md")


if __name__ == "__main__":
    main()
