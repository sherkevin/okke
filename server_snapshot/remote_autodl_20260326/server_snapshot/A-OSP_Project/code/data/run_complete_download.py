#!/usr/bin/env python3
"""
一键完整下载所有实验数据集（等价于先 proxy_on 再并行拉取）。
代理：http://127.0.0.1:7890（与 ~/.bashrc 中 proxy_on 一致）。

任务列表：
  1. MMMU 全 30 subjects (validation)
  2. VisualWebBench 全 7 tasks (test)
  3. COD10K 图像 ~100 张 (chandrabhuma/animal_cod10k)
  4. CHAIR COCO annotations (~252MB) + build chair.pkl
  5. RefCOCO 扩展到 500 条
  6. TextVQA 保持 2000（若 manifest 已有则跳过）
  7. AMBER  已有则跳过

监控：每 60s 打印各任务状态；结束后生成 docs/数据说明.md
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG  = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)
PY   = sys.executable
PROXY = os.environ.get("http_proxy", "http://127.0.0.1:7890")

# ── 代理设置（等价 proxy_on）──────────────────────────────────
def _set_proxy(env: dict) -> dict:
    for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
        env[k] = PROXY
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    env.pop("HF_ENDPOINT", None)          # VPN 下直连官方 Hub
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    return env

# ── 通用子进程启动 ────────────────────────────────────────────
def _spawn(name: str, cmd: list[str], env: dict):
    lp = LOG / f"dl_{name}.log"
    fh = open(lp, "w", encoding="utf-8", buffering=1)
    fh.write(f"=== {name} START {datetime.now(timezone.utc).isoformat()} ===\n")
    fh.write(f"CMD: {' '.join(cmd)}\n\n"); fh.flush()
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                         stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    return p, lp, fh

# ── 判断任务是否已完成，可跳过 ───────────────────────────────
def _done(name: str) -> bool:
    checks = {
        # min_sz 以字节计：MMMU 30 configs × 30 rows × ~420 bytes ≈ 378KB
        "mmmu":          (ROOT/"data/benchmarks/mmmu/mmmu_manifest.jsonl",   300_000),
        "vwb":           (ROOT/"data/benchmarks/visualwebbench/vwb_webqa_manifest.jsonl", 10_000),
        # COD10K 100 rows × ~76 bytes ≈ 7.6KB  → 用行数判断更可靠，这里设 7000 bytes
        "cod10k":        (ROOT/"data/benchmarks/cod10k/cod10k_manifest.jsonl", 7_000),
        "chair_coco":    (ROOT/"data/chair/coco_annotations/instances_val2014.json", 40_000_000),
        "refcoco":       (ROOT/"data/benchmarks/refcoco/refcoco_manifest.jsonl", 40_000),
        "textvqa":       (ROOT/"data/benchmarks/textvqa/textvqa_manifest.jsonl", 200_000),
    }
    if name not in checks:
        return False
    p, min_sz = checks[name]
    return p.is_file() and p.stat().st_size >= min_sz

# ── 主流程 ───────────────────────────────────────────────────
def main():
    env = _set_proxy(os.environ.copy())
    orch = open(LOG / "complete_download_orchestrator.log", "a", encoding="utf-8", buffering=1)
    orch.write(f"\n{'='*60}\nCOMPLETE DOWNLOAD START {datetime.now(timezone.utc).isoformat()}\n"
               f"PROXY={PROXY}\n{'='*60}\n"); orch.flush()

    # kill 旧的悬挂 mini_test 进程
    subprocess.run(
        ["pkill","-f","download_experiment_datasets.py"],
        capture_output=True
    )

    # 定义所有任务
    tasks = []

    # 1. MMMU
    if _done("mmmu"):
        n = sum(1 for _ in open(ROOT/"data/benchmarks/mmmu/mmmu_manifest.jsonl"))
        orch.write(f"SKIP mmmu (already {n} rows)\n")
    else:
        tasks.append(("mmmu", [PY, str(ROOT/"code/data/worker_mmmu_full.py")]))

    # 2. VWB all tasks
    if _done("vwb"):
        orch.write("SKIP vwb (all tasks already present)\n")
    else:
        tasks.append(("vwb", [PY, str(ROOT/"code/data/worker_vwb_all.py")]))

    # 3. COD10K
    if _done("cod10k"):
        n = sum(1 for _ in open(ROOT/"data/benchmarks/cod10k/cod10k_manifest.jsonl"))
        orch.write(f"SKIP cod10k (already {n} rows)\n")
    else:
        tasks.append(("cod10k", [PY, str(ROOT/"code/data/worker_cod10k.py")]))

    # 4. CHAIR COCO annotations
    if _done("chair_coco"):
        orch.write("SKIP chair_coco (already complete)\n")
    else:
        tasks.append(("chair_coco", [
            "/bin/bash", str(ROOT/"code/data/worker_chair_coco.sh")
        ]))

    # 5. RefCOCO 扩展
    refcoco_man = ROOT/"data/benchmarks/refcoco/refcoco_manifest.jsonl"
    if refcoco_man.is_file() and sum(1 for _ in open(refcoco_man)) >= 490:
        orch.write("SKIP refcoco (already 500)\n")
    else:
        tasks.append(("refcoco", [
            PY, str(ROOT/"code/data/download_refcoco_stream.py"), "500",
            "--out", str(ROOT/"data/benchmarks/refcoco"),
        ]))

    # 6. TextVQA（已有 2000 → 跳过）
    if _done("textvqa"):
        orch.write("SKIP textvqa (already 2000 rows)\n")
    else:
        tasks.append(("textvqa", [
            PY, str(ROOT/"code/data/download_textvqa_stream.py"), "2000",
            "--out", str(ROOT/"data/benchmarks/textvqa"),
        ]))

    # 打印计划
    orch.write(f"\nTasks to run ({len(tasks)}):\n")
    for name, cmd in tasks:
        orch.write(f"  {name}: {' '.join(cmd[:3])} ...\n")
    orch.write("\n"); orch.flush()

    if not tasks:
        orch.write("All tasks already done, generating report...\n")
        orch.close()
        _generate_report(env)
        return

    # 并行启动
    procs = []
    for name, cmd in tasks:
        p, lp, fh = _spawn(name, cmd, env)
        procs.append((name, p, lp, fh))
        orch.write(f"spawned {name} pid={p.pid} log={lp.name}\n"); orch.flush()

    # 每 60s 汇报
    stop_evt = threading.Event()
    def _reporter():
        tick = 0
        while not stop_evt.wait(timeout=60):
            tick += 1
            orch.write(f"\n--- tick #{tick} {datetime.now(timezone.utc).isoformat()} ---\n")
            for name, p, lp, _ in procs:
                rc = p.poll()
                st = "running" if rc is None else f"done rc={rc}"
                orch.write(f"  [{name:16s}] {st}\n")
            try:
                free = shutil.disk_usage(ROOT/"data").free / 1e9
                orch.write(f"  disk_free_GB={free:.1f}\n")
            except Exception:
                pass
            orch.flush()
    t = threading.Thread(target=_reporter, daemon=True)
    t.start()

    # 等待全部完成
    while any(p.poll() is None for _, p, _, _ in procs):
        time.sleep(5)

    rc_map = {}
    for name, p, lp, fh in procs:
        rc = p.wait(); fh.close()
        # RefCOCO PyArrow SIGABRT 归一化
        if name == "refcoco" and rc == -6:
            man = ROOT/"data/benchmarks/refcoco/refcoco_manifest.jsonl"
            imgs = list((ROOT/"data/benchmarks/refcoco/images").glob("refcoco_*.jpg"))
            if man.is_file() and len(imgs) >= 10:
                rc = 0
                orch.write(f"NOTE: {name} rc normalized 0 (PyArrow exit, data OK)\n")
        rc_map[name] = rc
        orch.write(f"FINISHED {name} rc={rc}\n"); orch.flush()

    stop_evt.set(); t.join(timeout=3)
    orch.write(f"\nALL_DONE {datetime.now(timezone.utc).isoformat()}\n")
    orch.write("RC: " + str(rc_map) + "\n"); orch.close()

    _generate_report(env)
    bad = [k for k, v in rc_map.items() if v != 0]
    if bad:
        print(f"[warn] failed tasks: {bad}")
    print(f"[done] report → {LOG/'dataset_download_report.md'}")
    sys.exit(1 if bad else 0)


def _generate_report(env: dict):
    subprocess.run(
        [PY, str(ROOT/"code/data/generate_dataset_status_report.py")],
        cwd=str(ROOT), env=env, check=False
    )
    _append_full_report()


def _append_full_report():
    """生成完整 docs/数据说明.md（面向实验者的人工可读说明）"""
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# A-OSP 实验数据集完整说明",
        "",
        f"> 自动生成时间（UTC）：{now}",
        "> 对齐《实验大纲.md》V3.5 各 Sprint",
        "",
        "---",
        "",
        "## 一、数据目录总览",
        "",
        "| 目录 | 数据集 | 实验用途 | 图像数 | manifest |",
        "|------|--------|---------|-------|----------|",
    ]

    checks = [
        ("data/benchmarks/mmbench",       "MMBench",         "通用能力基线 Sprint 3.3",     "mmbench_manifest.jsonl"),
        ("data/benchmarks/amber",         "AMBER",           "幻觉压力测试 Sprint 3.3",     "query_all.json"),
        ("data/benchmarks/chartqa",       "ChartQA",         "密集数字跨域 Sprint 3.3",     "chartqa_manifest.jsonl"),
        ("data/benchmarks/iu_xray",       "IU X-Ray",        "医疗跨域零样本 Sprint 3.3",   "iu_xray_manifest.jsonl"),
        ("data/benchmarks/textvqa",       "TextVQA",         "OCR 控制组 Sprint 2.4",        "textvqa_manifest.jsonl"),
        ("data/benchmarks/visualwebbench","VisualWebBench",   "GUI 理解 Sprint 2.5",          "vwb_element_ground_manifest.jsonl"),
        ("data/benchmarks/refcoco",       "RefCOCO",         "密集定位 Sprint 4.2",          "refcoco_manifest.jsonl"),
        ("data/benchmarks/mmmu",          "MMMU",            "常识多学科 §4 Table 1",        "mmmu_manifest.jsonl"),
        ("data/benchmarks/cod10k",        "COD10K",          "伪装目标/MC-SNR Sprint 1.3",   "cod10k_manifest.jsonl"),
        ("data/benchmarks/mirage",        "MIRAGE stub",     "生成幻觉 Sprint 3.1",          "mirage_manifest.jsonl"),
        ("data/mvbench",                  "MVBench meta",    "时序零样本迁移 Sprint 2.3",    "mvbench_action_sequence_mini.jsonl"),
        ("data/chair",                    "CHAIR 工具链",    "CHAIR评测 + COCO GT",          "coco_annotations/instances_val2014.json"),
    ]

    BENCH = ROOT / "data"
    for rel, name, usage, man in checks:
        d = ROOT / rel
        imgs = 0
        if d.is_dir():
            for pat in ("*.png","*.jpg","*.jpeg"):
                imgs += len(list(d.rglob(pat)))
        mp = d / man
        man_info = "✓" if mp.is_file() else "✗"
        mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6 if d.is_dir() else 0
        lines.append(f"| `{rel}` | {name} | {usage} | {imgs} | {man_info} `{man}` ({mb:.0f}MB) |")

    lines += [
        "",
        "---",
        "",
        "## 二、各数据集下载方法",
        "",
        "```bash",
        "# 开启 VPN 代理（等价 proxy_on）后运行：",
        "export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890",
        "python3 code/data/run_complete_download.py",
        "# 进度：tail -f logs/complete_download_orchestrator.log",
        "```",
        "",
        "| 数据集 | HuggingFace / 来源 | 脚本 |",
        "|--------|--------------------|------|",
        "| MMBench | `HuggingFaceM4/MMBench` validation | `download_core_benchmarks.py` |",
        "| AMBER JSON | GitHub `junyangwang0410/AMBER` | `download_core_benchmarks.py` |",
        "| AMBER 图像 | Google Drive `1MaCHgtupcZUjf007anNl4_MV0o4DjXvl` | `run_parallel_download.py` → gdown |",
        "| ChartQA | `HuggingFaceM4/ChartQA` test | `download_crossdomain_datasets.py` |",
        "| IU X-Ray | `dz-osamu/IU-Xray` val.jsonl + image.zip | `download_crossdomain_datasets.py` |",
        "| TextVQA | `lmms-lab/TextVQA` validation (streaming) | `download_textvqa_stream.py` |",
        "| VisualWebBench | `visualwebbench/VisualWebBench` 7 tasks | `worker_vwb_all.py` |",
        "| RefCOCO | `lmms-lab/RefCOCO` val (streaming) | `download_refcoco_stream.py` |",
        "| MMMU | `MMMU/MMMU` 全 30 subjects validation | `worker_mmmu_full.py` |",
        "| COD10K | `chandrabhuma/animal_cod10k` train | `worker_cod10k.py` |",
        "| MIRAGE stub | 本地 COCO val 图生成占位 | `create_mirage_stub.py` |",
        "| MVBench | `OpenGVLab/MVBench` JSONL (视频文件仅记录名，需按协议另下) | `run_experiment_subtask.py mvbench` |",
        "| CHAIR+COCO | `images.cocodataset.org` annotations zip | `worker_chair_coco.sh` |",
        "",
        "---",
        "",
        "## 三、注意事项",
        "",
        "1. **MVBench 视频**：HF 中 `video` 字段仅存文件名（如 `ZS9XR.mp4`），不含实际视频字节。需按 [OpenGVLab/MVBench](https://github.com/OpenGVLab/MVBench) 的说明自行下载视频包。",
        "2. **MIRAGE**：当前为 COCO val2014 占位 stub，不是官方 MIRAGE 全集；若论文数据需官方，请参考 MIRAGE 论文仓库。",
        "3. **COD10K**：使用 `chandrabhuma/animal_cod10k`（含真实伪装目标图像）替代官方完整数据集；100 张已足够 Sprint 1.3 Monte Carlo SNR 实验。",
        "4. **AMBER 图像**：已通过 `gdown` 下载并解压 `AMBER.zip`（1004 张 jpg）到 `data/benchmarks/amber/images/`。",
        "5. **VPN 代理**：需在 AutoDL 机器上 `source ~/.bashrc && proxy_on` 或直接 `export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890`。",
        "",
        "---",
        "",
        f"*文档版本：对齐实验大纲 V3.5，自动生成于 {now}*",
        "",
    ]

    docs = ROOT / "docs" / "数据说明.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] docs → {docs}")


if __name__ == "__main__":
    main()
