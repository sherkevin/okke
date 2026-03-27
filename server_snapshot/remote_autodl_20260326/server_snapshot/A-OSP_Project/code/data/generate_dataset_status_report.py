#!/usr/bin/env python3
"""生成数据集本地落盘状态 Markdown 报告（供 VPN 全量下载后归档）。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
BENCH = DATA / "benchmarks"


def _du_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    n = 0
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        n += len(list(d.rglob(pat)))
    return n


def _jsonl_lines(f: Path) -> int | str:
    if not f.is_file():
        return "missing"
    try:
        with open(f, encoding="utf-8") as fp:
            return sum(1 for _ in fp)
    except OSError:
        return "error"


def main() -> None:
    out = PROJECT / "logs" / "dataset_download_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    checks = [
        ("MMBench", BENCH / "mmbench", ["mmbench_manifest.jsonl"], BENCH / "mmbench" / "images"),
        ("AMBER JSON", BENCH / "amber", ["query_all.json"], None),
        ("AMBER images", BENCH / "amber" / "images", [], BENCH / "amber" / "images"),
        ("ChartQA", BENCH / "chartqa", ["chartqa_manifest.jsonl"], BENCH / "chartqa" / "images"),
        ("IU X-Ray", BENCH / "iu_xray", ["iu_xray_manifest.jsonl"], BENCH / "iu_xray" / "images"),
        ("TextVQA", BENCH / "textvqa", ["textvqa_manifest.jsonl"], BENCH / "textvqa" / "images"),
        ("VisualWebBench", BENCH / "visualwebbench", ["vwb_element_ground_manifest.jsonl"], BENCH / "visualwebbench"),
        ("RefCOCO", BENCH / "refcoco", ["refcoco_manifest.jsonl"], BENCH / "refcoco" / "images"),
        ("MMMU", BENCH / "mmmu", ["mmmu_manifest.jsonl"], BENCH / "mmmu" / "images"),
        ("MIRAGE stub", BENCH / "mirage", ["mirage_manifest.jsonl"], None),
        ("COD10K", BENCH / "cod10k", ["README_MANUAL_DOWNLOAD.md"], BENCH / "cod10k"),
        ("CHAIR", DATA / "chair", [], DATA / "chair"),
        ("MVBench meta", DATA / "mvbench", [], DATA / "mvbench"),
    ]

    for name, base, manifests, img_root in checks:
        mb = _du_mb(base) if base.exists() else 0.0
        man_info = []
        for m in manifests:
            man_info.append(f"`{m}`: {_jsonl_lines(base / m)}")
        imgs = _count_images(img_root) if img_root and img_root.exists() else 0
        rows.append((name, f"{mb:.1f}", ", ".join(man_info) if man_info else "-", imgs))

    lines = [
        "# 数据集下载状态报告",
        "",
        f"- **生成时间（UTC）**: {datetime.now(timezone.utc).isoformat()}",
        f"- **项目根**: `{PROJECT}`",
        "",
        "| 数据集 | 目录约 MB | manifest / 说明 | 图像文件数（递归 png/jpg） |",
        "|--------|------------|-----------------|---------------------------|",
    ]
    for name, mb, man, imgs in rows:
        lines.append(f"| {name} | {mb} | {man} | {imgs} |")

    lines.extend([
        "",
        "## 说明",
        "",
        "- AMBER 图像若为 0，请用 `gdown` 从 Google Drive 拉取（见 `docs/实验数据集说明.md`）。",
        "- COD10K 需按 `data/benchmarks/cod10k/README_MANUAL_DOWNLOAD.md` 手动放置图像。",
        "- MVBench 脚本主要产出 JSONL；视频需按 OpenGVLab 协议另下。",
        "",
    ])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
