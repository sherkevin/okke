from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

REMOTE_ROOT = "/root/autodl-tmp/BRA_Project"
REMOTE_NIGHT_LOG_DIR = f"{REMOTE_ROOT}/logs/night_v2"


@dataclass
class ProbeResult:
    generated_path: Path
    remote_reachable: bool
    remote_error: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def extract_version(path: Path) -> int | None:
    match = re.search(r"_V(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def next_checklist_version(workspace: Path) -> int:
    versions = []
    for path in workspace.glob("今夜4GPU任务清单_V*.md"):
        version = extract_version(path)
        if version is not None:
            versions.append(version)
    return (max(versions) + 1) if versions else 1


def latest_matching(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def safe_rel(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace)).replace("\\", "/")
    except ValueError:
        return str(path)


def summarize_stage0(remote_mirror: Path) -> dict[str, Any]:
    path = remote_mirror / "gpu0_stage0_tlra_zero_8b.json"
    if not path.exists():
        return {"status": "missing", "summary": "缺少 `Stage 0` 结果文件。"}
    data = read_json(path)
    patch_top10 = data.get("patch_top10_overlap")
    image_top1 = data.get("top1_overlap")
    candidate_overlap = data.get("candidate_window_overlap")
    verdict = "partial"
    note = "保留 `TLRA_zero` 为待定分支。"
    if isinstance(patch_top10, (int, float)) and patch_top10 < 0.10:
        verdict = "risky"
        note = "patch-level overlap 偏低，`TLRA_zero` 更像 viability probe，不应直接当主表核心。"
    return {
        "status": verdict,
        "summary": (
            f"`Stage 0` 已有镜像结果：image top1={image_top1:.4f}, "
            f"patch top10={patch_top10:.4f}, candidate overlap={candidate_overlap:.3f}。{note}"
        ),
        "path": path,
    }


def summarize_chain_b(remote_mirror: Path) -> dict[str, Any]:
    full_mmbench = remote_mirror / "gpu2_mmbench_tlra_full.json"
    no_vasm_mmbench = remote_mirror / "gpu2_mmbench_tlra_no_vasm.json"
    full_mme = remote_mirror / "gpu2_mme_tlra_full.json"
    full_mmmu = remote_mirror / "gpu2_mmmu_tlra_full.json"
    if not all(p.exists() for p in [full_mmbench, no_vasm_mmbench, full_mme, full_mmmu]):
        return {"status": "partial", "summary": "Chain B 仅部分落地，仍缺完整镜像或对照文件。"}

    mmbench_full = read_json(full_mmbench)["qwen3vl2b"][0]
    mmbench_no_vasm = read_json(no_vasm_mmbench)["qwen3vl2b"][0]
    mme_full = read_json(full_mme)["qwen3vl2b"][0]
    mmmu_full = read_json(full_mmmu)["qwen3vl2b"][0]
    return {
        "status": "partial",
        "summary": (
            "Chain B pilot 已有镜像："
            f"MMBench full={mmbench_full['bra']['accuracy']:.3f} vs no_vasm={mmbench_no_vasm['bra']['accuracy']:.3f}，"
            f"MME full acc={mme_full['bra']['accuracy']:.3f}，"
            f"MMMU pilot acc={mmmu_full['bra']['accuracy']:.3f} (n={mmmu_full['sample_count']})。"
            " 但当前仍是 pilot，不是 `MMMU Hard` 正式主表。"
        ),
        "paths": [full_mmbench, no_vasm_mmbench, full_mme, full_mmmu],
    }


def summarize_chain_c(remote_mirror: Path) -> dict[str, Any]:
    freak_mean = remote_mirror / "gpu3_freak_tlra_meanpool.json"
    freak_topk = remote_mirror / "gpu3_freak_tlra_adaptivetopk.json"
    docvqa = remote_mirror / "docvqa_bra_zero_matrix.json"
    vidhalluc = remote_mirror / "gpu3_vidhalluc_tlra_adaptivetopk.json"
    parts: list[str] = []
    status = "partial"

    if freak_mean.exists() and freak_topk.exists():
        mean_data = read_json(freak_mean)["qwen3vl2b"][0]
        topk_data = read_json(freak_topk)["qwen3vl2b"][0]
        parts.append(
            "FREAK pilot 已有镜像："
            f"MeanPool={mean_data['bra']['accuracy']:.3f}, "
            f"AdaptiveTopK={topk_data['bra']['accuracy']:.3f}。"
        )
        if topk_data["bra"]["accuracy"] <= mean_data["bra"]["accuracy"]:
            parts.append("当前 `AdaptiveTopK` 尚未证明优于 `MeanPool`，这条主张仍未闭合。")
            status = "risky"
    else:
        parts.append("FREAK pilot 尚未形成完整镜像。")

    if docvqa.exists():
        docvqa_data = read_json(docvqa)
        if docvqa_data.get("qwen3vl2b"):
            parts.append("DocVQA 已有可读结果。")
        else:
            parts.append("DocVQA 镜像文件存在但结果为空，说明数据虽然补齐，当前 loader / 布局仍未真正跑通。")
            status = "risky"
    else:
        parts.append("DocVQA 仍没有本地镜像结果。")

    if vidhalluc.exists():
        vidhalluc_data = read_json(vidhalluc)
        if vidhalluc_data.get("qwen3vl2b"):
            parts.append("VidHalluc 已有可读 pilot 结果。")
        else:
            parts.append("VidHalluc 镜像文件为空，bounded video pilot 仍需重新验证。")
            status = "risky"
    else:
        parts.append("VidHalluc 尚无镜像结果。")

    return {"status": status, "summary": " ".join(parts), "paths": [p for p in [freak_mean, freak_topk, docvqa, vidhalluc] if p.exists()]}


def summarize_chain_a(remote_mirror: Path) -> dict[str, Any]:
    required = [
        latest_matching(remote_mirror, "base_pope_*.json"),
        latest_matching(remote_mirror, "vcd_pope_*.json"),
        latest_matching(remote_mirror, "tlra_zero_pope_*.json"),
        latest_matching(remote_mirror, "base_chair_*.json"),
    ]
    present = [p for p in required if p is not None]
    if len(present) < 4:
        return {"status": "partial", "summary": "Chain A 有历史镜像，但缺少完整可对照组合或最新统一批次。"}
    return {
        "status": "partial",
        "summary": (
            "Chain A 已有历史镜像（`base/vcd/tlra_zero` on `POPE`，`base` on `CHAIR`），"
            "但仍需要在同一批次、同一参数上重跑主表，尤其要确保 `CHAIR` 的 `AGL` 不再被 cap 掩盖。"
        ),
        "paths": present,
    }


def summarize_download_log(workspace: Path) -> dict[str, str]:
    path = workspace / "experiment_logs" / "download_docvqa_videomme_20260321.md"
    if not path.exists():
        return {
            "docvqa": "unknown",
            "videomme": "unknown",
        }
    text = read_text(path)
    docvqa = "complete" if "DocVQA`：**已完成**" in text else "unknown"
    videomme = "in_progress" if "Video-MME`：**已启动后台下载**" in text else "unknown"
    return {"docvqa": docvqa, "videomme": videomme}


def probe_remote() -> dict[str, Any]:
    remote_script = r"""
import json
import os
from pathlib import Path

root = Path("/root/autodl-tmp/BRA_Project")
night = root / "logs" / "night_v2"
datasets = root / "datasets"
video = datasets / "video"
targets = {
    "docvqa": datasets / "DocVQA",
    "docvqa_hf": datasets / "DocVQA_hf",
    "vidhalluc": video / "chaoyuli_VidHalluc",
    "videomme": video / "Video-MME",
    "videomme_hf": video / "Video-MME_hf",
    "night_log_dir": night,
    "projector_v_matrix": root / "models" / "V_matrix.pt",
    "projector_v_matrix_q3": root / "models" / "V_matrix_q3.pt",
    "projector_v_matrix_q3_mini": root / "models" / "V_matrix_q3_mini.pt",
}

def summarize(path: Path):
    info = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info["type"] = "dir" if path.is_dir() else "file"
    if path.is_dir():
        entries = sorted(os.listdir(path))
        info["entry_count"] = len(entries)
        info["sample_entries"] = entries[:12]
        info["file_count"] = sum(len(files) for _, _, files in os.walk(path))
    else:
        info["size_bytes"] = path.stat().st_size
    return info

night_files = []
if night.exists():
    for item in sorted(night.iterdir()):
        stat = item.stat()
        night_files.append(
            {
                "name": item.name,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

print(json.dumps({"targets": {k: summarize(v) for k, v in targets.items()}, "night_files": night_files}, ensure_ascii=False))
"""

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
        stdin, stdout, stderr = client.exec_command(
            f"export PATH=\"/root/miniconda3/bin:$PATH\"; python - <<'PY'\n{remote_script}\nPY",
            timeout=120,
        )
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        if err:
            return {"reachable": False, "error": err}
        return {"reachable": True, "data": json.loads(out)}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}
    finally:
        client.close()


def build_completion_lines(workspace: Path, remote: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    remote_mirror = workspace / "experiment_logs" / "remote_mirror"
    stage0 = summarize_stage0(remote_mirror)
    chain_a = summarize_chain_a(remote_mirror)
    chain_b = summarize_chain_b(remote_mirror)
    chain_c = summarize_chain_c(remote_mirror)
    downloads = summarize_download_log(workspace)

    lines = [
        f"- `Stage 0`：`{stage0['status']}`。{stage0['summary']}",
        f"- `Chain A`：`{chain_a['status']}`。{chain_a['summary']}",
        f"- `Chain B`：`{chain_b['status']}`。{chain_b['summary']}",
        f"- `Chain C / video`：`{chain_c['status']}`。{chain_c['summary']}",
        f"- `DocVQA` 资源：`{downloads['docvqa']}`。"
        + (" 远程已复核。" if remote.get("reachable") and remote.get("data", {}).get("targets", {}).get("docvqa", {}).get("exists") else " 当前仅能依据本地下载日志与镜像判断。"),
        f"- `Video-MME` 资源：`{downloads['videomme']}`。"
        + (" 远程已复核。" if remote.get("reachable") and remote.get("data", {}).get("targets", {}).get("videomme", {}).get("exists") else " 当前远程未复核完成状态。"),
        "- `TLRA_calib` 正式 checkpoint 身份：`unresolved`。",
        "- `MMMU Hard` manifest：`unresolved`。",
        "- `Base + 5k LoRA`：`unresolved`。",
    ]
    return lines, {
        "stage0": stage0,
        "chain_a": chain_a,
        "chain_b": chain_b,
        "chain_c": chain_c,
        "downloads": downloads,
    }


def build_remote_lines(remote: dict[str, Any]) -> list[str]:
    if not remote.get("reachable"):
        return [
            "- 当前远程探查：`unreachable`",
            f"- 错误：`{remote.get('error', 'unknown')}`",
            "- 结论：今夜任务的第一优先级必须加入“控制面恢复/SSH 可达性复核”，否则不要盲目假设 GPU 在稳定执行。",
        ]

    data = remote["data"]
    targets = data["targets"]
    night_files = data.get("night_files", [])
    lines = [
        "- 当前远程探查：`reachable`",
        f"- `DocVQA`：exists=`{targets['docvqa']['exists']}` / file_count=`{targets['docvqa'].get('file_count', 0)}`",
        f"- `VidHalluc`：exists=`{targets['vidhalluc']['exists']}` / file_count=`{targets['vidhalluc'].get('file_count', 0)}`",
        f"- `Video-MME`：exists=`{targets['videomme']['exists']}` / file_count=`{targets['videomme'].get('file_count', 0)}`",
        f"- `night_v2` 文件数：`{len(night_files)}`",
    ]
    if night_files:
        preview = ", ".join(item["name"] for item in night_files[:8])
        lines.append(f"- `night_v2` 文件预览：`{preview}`")
    return lines


def build_gpu_plan(remote: dict[str, Any], summaries: dict[str, Any]) -> str:
    remote_blocked = not remote.get("reachable")
    gpu0 = [
        "先做控制面复核：确认 SSH 恢复、`logs/night_v2/` 可列出、`V_matrix*.pt` 可见。",
        "若主机恢复可达，重跑 `Stage 0` 并补 `TLRA_calib` 分支；若 `patch_top10_overlap` 仍低于 `0.10`，将 `TLRA_zero` 降为 appendix-only probe。",
    ]
    gpu1 = [
        "优先重跑 `POPE + CHAIR` 主表，统一使用同一批次配置和高 `chair_max_new_tokens`。",
        "如果 `projector_checkpoint` 今夜冻结，追加 `tlra_calib`；否则保留 `base/vcd/dola/tlra_zero` 四方法可比主表。",
    ]
    gpu2 = [
        "继续 `MMBench + MME + MMMU` 路线，但把重点放在 `TLRA_full` vs `TLRA_no_VASM` 的正式对照。",
        "并行冻结 `MMMU Hard` manifest；在它冻结前，所有 `MMMU` 结果都只记作 pilot，不进最终主表。",
    ]
    gpu3 = [
        "第一优先级从“下载 DocVQA”改成“修通 DocVQA loader / layout”，先做 20-sample smoke。",
        "若 DocVQA 冒烟通过，再跑 `DocVQA + FREAK`；若仍 `no_samples`，立刻回切 `FREAK` 复核与 `VidHalluc` 重试，不要空转。",
    ]

    prefix = "在主机恢复可达后，"
    if not remote_blocked:
        prefix = ""

    return "\n".join(
        [
            "## 4. 今夜 4 GPU 新任务清单",
            "",
            "### GPU0：控制面 + Stage 0 判定",
            "",
            *(f"- {prefix}{item}" for item in gpu0),
            "",
            "```bash",
            "ssh -p 47559 root@connect.westc.seetacloud.com",
            "cd /root/autodl-tmp/BRA_Project && mkdir -p logs/night_v3",
            "ls -lah models/V_matrix*.pt",
            "python tlra_semantic_validity_pilot.py --model qwen3-vl-8b --method tlra_zero --n_samples 64 --topk 10 --candidate_window 50 --output logs/night_v3/gpu0_stage0_tlra_zero_8b.json",
            "python tlra_semantic_validity_pilot.py --model qwen3-vl-8b --method tlra_calib --projector_checkpoint /root/autodl-tmp/BRA_Project/models/REPLACE_ME.pt --n_samples 64 --topk 10 --candidate_window 50 --output logs/night_v3/gpu0_stage0_tlra_calib_8b.json",
            "```",
            "",
            "### GPU1：Chain A 正式主表",
            "",
            *(f"- {prefix}{item}" for item in gpu1),
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=1 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH=\"/root/miniconda3/bin:$PATH\" && \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method base      --mini_test 200 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method vcd       --mini_test 200 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method dola      --mini_test 200 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method tlra_zero --mini_test 200 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base      --mini_test 150 --chair_max_new_tokens 384 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd       --mini_test 150 --chair_max_new_tokens 384 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola      --mini_test 150 --chair_max_new_tokens 384 ; \\",
            "python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method tlra_zero --mini_test 150 --chair_max_new_tokens 384' > logs/night_v3/gpu1_chainA.log 2>&1 &",
            "```",
            "",
            "### GPU2：Chain B 正式化",
            "",
            *(f"- {prefix}{item}" for item in gpu2),
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=2 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH=\"/root/miniconda3/bin:$PATH\" && \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_full    --output logs/night_v3/gpu2_mmbench_tlra_full.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mmbench_tlra_no_vasm.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mme     --n_samples 300 --bra_method tlra_full    --output logs/night_v3/gpu2_mme_tlra_full.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mme     --n_samples 300 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mme_tlra_no_vasm.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu    --n_samples 200 --bra_method tlra_full    --output logs/night_v3/gpu2_mmmu_tlra_full.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu    --n_samples 200 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mmmu_tlra_no_vasm.json' > logs/night_v3/gpu2_chainB.log 2>&1 &",
            "```",
            "",
            "### GPU3：DocVQA 修通优先，其次 Chain C / video",
            "",
            *(f"- {prefix}{item}" for item in gpu3),
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=3 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH=\"/root/miniconda3/bin:$PATH\" && \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset docvqa --n_samples 20  --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_docvqa_smoke.json || exit 17 ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset docvqa --n_samples 200 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_docvqa_tlra_adaptivetopk.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset freak  --n_samples 200 --bra_method tlra_meanpool    --output logs/night_v3/gpu3_freak_tlra_meanpool.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset freak  --n_samples 200 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_freak_tlra_adaptivetopk.json ; \\",
            "python bra_eval_matrix.py --model qwen3vl2b --dataset vidhalluc --n_samples 80 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_vidhalluc_tlra_adaptivetopk.json' > logs/night_v3/gpu3_chainC.log 2>&1 &",
            "```",
        ]
    )


def build_markdown(workspace: Path, version: int, remote: dict[str, Any]) -> str:
    completion_lines, summaries = build_completion_lines(workspace, remote)
    remote_lines = build_remote_lines(remote)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 今夜4GPU任务清单_V{version}",
        "",
        f"> 生成时间：`{generated_at}`",
        "> 生成方式：本地镜像结果 + 远程自动巡检（带超时降级）",
        "",
        "## 1. 当前完成情况",
        "",
        *completion_lines,
        "",
        "## 2. 当前数据/远端支撑情况",
        "",
        *remote_lines,
        "",
        "## 3. 本轮调度结论",
        "",
    ]

    if not remote.get("reachable"):
        lines.extend(
            [
                "- 由于远端当前不可达，本轮任务单采用保守调度：先恢复控制面，再启动或续跑 GPU 任务。",
                "- 但从本地镜像可确认：`Stage 0`、`Chain B pilot`、`FREAK pilot` 已有支撑数据，因此不是“完全没活可做”，而是“需要先确认远端状态后再接续”。",
                "- `DocVQA` 已不再被视为下载阻塞项；现在真正的阻塞点变成：`DocVQA loader/layout`、`TLRA_calib checkpoint` 身份、`MMMU Hard` manifest、远端连通性。",
            ]
        )
    else:
        lines.extend(
            [
                "- 远端当前可达，因此本轮任务单以“继续推进正式主表 + 修复空结果分支”为主。",
                "- 当前最需要优先解决的不是下载，而是把 `DocVQA` 和 `VidHalluc` 从空结果/不可复现状态推进到可读结果。",
                "- `AdaptiveTopK` 目前尚未在 `FREAK` 镜像里稳定优于 `MeanPool`，因此 Chain C 是今夜必须复核的核心风险。 ",
            ]
        )

    lines.extend(
        [
            "",
            build_gpu_plan(remote, summaries),
            "",
            "## 5. 明早前必须回收的最小产物",
            "",
            "1. 一份新的 `Stage 0` 判定：`TLRA_zero` 是保留为主表、附录，还是仅作 viability probe。",
            "2. 一套统一批次的 `POPE + CHAIR` 可比结果，包含 `AGL` 和至少一个效率字段。",
            "3. 一套 `MMBench + MME + MMMU pilot/Hard-prep` 的 `TLRA_full` vs `TLRA_no_VASM` 对照。",
            "4. 一份可读的 `DocVQA` 结果，或者一份清晰的 `DocVQA loader/layout` 失败诊断。",
            "5. 一份 `FREAK` 新结果，用于重新判断 `AdaptiveTopK` 是否真的强于 `MeanPool`。",
            "6. 一份 `VidHalluc` 可读结果，若仍为空则明确降级为 appendix-only pilot。",
            "",
            "## 6. 定时巡检规则",
            "",
            "- 本巡检器每次运行都会生成一个新的 `今夜4GPU任务清单_Vx.md`。",
            "- 若远端不可达，下一轮自动把“控制面恢复”置顶，不会误判为实验全失败。",
            "- 若远端恢复可达，下一轮会自动把优先级切回结果回收与空结果修复。",
            "- 若你希望停止循环巡检，只需结束本地后台运行的 `night_4gpu_patrol.py` 进程。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_once(workspace: Path) -> ProbeResult:
    version = next_checklist_version(workspace)
    remote = probe_remote()
    content = build_markdown(workspace, version, remote)
    path = workspace / f"今夜4GPU任务清单_V{version}.md"
    path.write_text(content, encoding="utf-8")
    return ProbeResult(
        generated_path=path,
        remote_reachable=remote.get("reachable", False),
        remote_error=remote.get("error", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodic 4-GPU night task patrol.")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--interval-seconds", type=int, default=3600, help="Sleep interval between patrol runs")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    while True:
        result = run_once(workspace)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{stamp}] generated={result.generated_path.name} "
            f"remote_reachable={result.remote_reachable} "
            f"remote_error={result.remote_error}"
        )
        if not args.loop:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
