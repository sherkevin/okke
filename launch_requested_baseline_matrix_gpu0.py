#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/requested_baseline_matrix"
REMOTE_MINITEST_DIR = f"{REMOTE_PROJECT}/logs/minitest"
MMBENCH_REMOTE_FILE = f"{REMOTE_PROJECT}/datasets/MMBench_EN_hf/data/dev-00000-of-00001-75b6649fb044d38b.parquet"

POPE_COUNT = 3000
CHAIR_COUNT = 5000
CHAIR_MAX_NEW_TOKENS = 384
POPE_SPLITS = ("random", "popular", "adversarial")
DATASET_ORDER = ("pope", "mmbench", "chair")
METHODS = ("base", "opera", "vcd", "dola")

MODEL_REMOTE_DIRS = {
    "qwen2-vl-7b": "Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    "llava-v1.5-7b": "llava-1.5-7b-hf",
    "instructblip-7b": "instructblip-vicuna-7b",
}
DEFAULT_MODELS = ("qwen3-vl-8b", "qwen2.5-vl-7b", "instructblip-7b", "llava-v1.5-7b")


@dataclass(frozen=True)
class EvalJob:
    model: str
    dataset: str
    method: str
    split: str
    mini_test: int
    extra_args: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _remote_state_script() -> str:
    return f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
import json

try:
    import pandas as pd
except Exception:
    pd = None

project = Path("{REMOTE_PROJECT}")
models_dir = project / "models"
minitest_dir = project / "logs" / "minitest"
mmbench_file = Path("{MMBENCH_REMOTE_FILE}")

model_dirs = sorted([p.name for p in models_dir.iterdir()]) if models_dir.exists() else []
entries = []
if minitest_dir.exists():
    for path in sorted(minitest_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries.append({{
            "file": path.name,
            "status": payload.get("status"),
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "method": payload.get("method"),
            "pope_split": payload.get("pope_split"),
            "sample_count": payload.get("sample_count", payload.get("n_samples")),
            "n_errors": payload.get("n_errors", 0),
        }})

mmbench_count = None
if pd is not None and mmbench_file.exists():
    mmbench_count = int(len(pd.read_parquet(mmbench_file)))

summary = {{
    "model_dirs": model_dirs,
    "entries": entries,
    "mmbench_count": mmbench_count,
    "has_pope_random": (project / "datasets/POPE/output/coco/coco_pope_random.json").exists(),
    "has_pope_popular": (project / "datasets/POPE/output/coco/coco_pope_popular.json").exists(),
    "has_pope_adversarial": (project / "datasets/POPE/output/coco/coco_pope_adversarial.json").exists(),
    "has_chair_instances": (project / "datasets/coco2014/annotations/instances_val2014.json").exists(),
    "has_chair_captions": (project / "datasets/coco2014/annotations/captions_val2014.json").exists(),
    "has_mmbench": mmbench_file.exists(),
}}
print(json.dumps(summary, ensure_ascii=False))
PY
"""


def fetch_remote_state() -> dict:
    proc = run_remote_bash(_remote_state_script(), timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch remote state:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Remote state was not valid JSON:\n{proc.stdout}") from exc


def build_jobs(models: tuple[str, ...], mmbench_count: int) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    for model in models:
        for split in POPE_SPLITS:
            for method in METHODS:
                jobs.append(
                    EvalJob(
                        model=model,
                        dataset="pope",
                        method=method,
                        split=split,
                        mini_test=POPE_COUNT,
                        extra_args=("--pope_split", split),
                    )
                )
        for method in METHODS:
            jobs.append(
                EvalJob(
                    model=model,
                    dataset="mmbench",
                    method=method,
                    split="default",
                    mini_test=mmbench_count,
                )
            )
        for method in METHODS:
            jobs.append(
                EvalJob(
                    model=model,
                    dataset="chair",
                    method=method,
                    split="default",
                    mini_test=CHAIR_COUNT,
                    extra_args=("--chair_max_new_tokens", str(CHAIR_MAX_NEW_TOKENS)),
                )
            )
    return jobs


def eval_command(job: EvalJob) -> str:
    parts = [
        REMOTE_PYTHON,
        f"{REMOTE_PROJECT}/run_eval_pipeline.py",
        "--model",
        job.model,
        "--dataset",
        job.dataset,
        "--method",
        job.method,
        "--mini_test",
        str(job.mini_test),
        *job.extra_args,
    ]
    return _shell_join(parts)


def _matches_job(entry: dict, job: EvalJob) -> bool:
    if entry.get("model") != job.model:
        return False
    if entry.get("dataset") != job.dataset:
        return False
    if entry.get("method") != job.method:
        return False
    if job.dataset == "pope" and entry.get("pope_split") != job.split:
        return False
    return True


def _is_completed(entry: dict, job: EvalJob) -> bool:
    if not _matches_job(entry, job):
        return False
    sample_count = int(entry.get("sample_count", -1) or -1)
    n_errors = int(entry.get("n_errors", 0) or 0)
    status = entry.get("status")
    return sample_count == job.mini_test and n_errors == 0 and status in (None, "final")


def build_matrix(remote_state: dict, jobs: list[EvalJob]) -> list[dict]:
    model_dirs = set(remote_state.get("model_dirs", []))
    entries = list(remote_state.get("entries", []))
    rows: list[dict] = []
    for job in jobs:
        expected_model_dir = MODEL_REMOTE_DIRS[job.model]
        relevant = [entry for entry in entries if _matches_job(entry, job)]
        relevant.sort(key=lambda item: item.get("file", ""))
        completed = [entry for entry in relevant if _is_completed(entry, job)]
        latest = relevant[-1] if relevant else None
        if expected_model_dir not in model_dirs:
            status = "blocked_missing_remote_model"
        elif completed:
            status = "completed"
        else:
            status = "missing"
        rows.append(
            {
                "model": job.model,
                "dataset": job.dataset,
                "split": job.split,
                "method": job.method,
                "target_samples": job.mini_test,
                "remote_model_dir": expected_model_dir,
                "status": status,
                "completed_json": completed[-1]["file"] if completed else None,
                "latest_json": latest.get("file") if latest else None,
                "latest_status": latest.get("status") if latest else None,
                "latest_sample_count": latest.get("sample_count") if latest else None,
                "latest_n_errors": latest.get("n_errors") if latest else None,
            }
        )
    return rows


def build_runner_body(*, matrix_log: str, manifest_tsv: str, jobs: list[EvalJob]) -> str:
    lines: list[str] = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {REMOTE_PROJECT}",
        "source /etc/network_turbo >/dev/null 2>&1 || true",
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "export PYTHONUNBUFFERED=1",
        "export CUDA_VISIBLE_DEVICES=0",
        f'MATRIX_LOG="{matrix_log}"',
        f'MANIFEST="{manifest_tsv}"',
        f'MINITEST_DIR="{REMOTE_MINITEST_DIR}"',
        'echo "[MATRIX] requested_baseline_matrix_gpu0 start $(date -Iseconds)"',
        'printf "iso_time\\tmodel\\tdataset\\tsplit\\tmethod\\ttarget_samples\\taction\\texit_code\\tresult_json\\tmatrix_parent_log\\n" > "$MANIFEST"',
        "",
        'find_completed_json() {',
        '  local model="$1"',
        '  local dataset="$2"',
        '  local split="$3"',
        '  local method="$4"',
        '  local target="$5"',
        f'  "{REMOTE_PYTHON}" - "$model" "$dataset" "$split" "$method" "$target" "$MINITEST_DIR" <<\'PYEOF\'',
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "model, dataset, split, method, target, log_dir = sys.argv[1:]",
        "target = int(target)",
        "root = Path(log_dir)",
        "candidates = sorted(root.glob('*.json'), key=lambda p: p.name)",
        "matched = []",
        "for path in candidates:",
        "    try:",
        "        payload = json.loads(path.read_text(encoding='utf-8'))",
        "    except Exception:",
        "        continue",
        "    if payload.get('model') != model or payload.get('dataset') != dataset or payload.get('method') != method:",
        "        continue",
        "    if dataset == 'pope' and payload.get('pope_split') != split:",
        "        continue",
        "    sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)",
        "    n_errors = int(payload.get('n_errors', 0) or 0)",
        "    status = payload.get('status')",
        "    if sample_count == target and n_errors == 0 and status in (None, 'final'):",
        "        matched.append(str(path))",
        "if matched:",
        "    print(matched[-1])",
        "PYEOF",
        '}',
        "",
    ]

    for job in jobs:
        cmd = eval_command(job)
        lines.append(f'echo "[CHECK] {job.tag} $(date -Iseconds)"')
        lines.append(
            f'existing=$(find_completed_json "{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "{job.mini_test}")'
        )
        lines.append('if [ -n "$existing" ]; then')
        lines.append(f'  echo "[SKIP] {job.tag} existing=$existing $(date -Iseconds)"')
        lines.append(
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "{job.mini_test}" '
            '"skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append('else')
        lines.append(f'  echo "[START] {job.tag} $(date -Iseconds)"')
        lines.append(f'  {cmd}')
        lines.append('  rc=$?')
        lines.append(
            f'  result=$(find_completed_json "{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "{job.mini_test}")'
        )
        lines.append(
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "{job.mini_test}" '
            '"executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'  echo "[DONE] {job.tag} rc=$rc result=$result $(date -Iseconds)"')
        lines.append('  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
        lines.append('fi')
        lines.append("")

    lines.append('echo "[MATRIX] requested_baseline_matrix_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def summarize_matrix(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "total": len(rows),
        "counts": counts,
        "missing_rows": [row for row in rows if row["status"] == "missing"],
        "blocked_rows": [row for row in rows if row["status"].startswith("blocked_")],
        "completed_rows": [row for row in rows if row["status"] == "completed"],
    }


def write_local_matrix_snapshot(rows: list[dict], *, mmbench_count: int, remote_state: dict, ts: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "experiment_logs" / "matrix_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"requested_baseline_matrix_gpu0_{ts}.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mmbench_count": mmbench_count,
        "remote_model_dirs": remote_state.get("model_dirs", []),
        "matrix": rows,
        "summary": summarize_matrix(rows),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_launch_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and launch the requested baseline matrix on remote GPU0 without rerunning completed cells."
    )
    parser.add_argument("--mode", choices=["summary", "render", "launch"], default="summary")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REMOTE_DIRS),
        default=list(DEFAULT_MODELS),
    )
    args = parser.parse_args()

    remote_state = fetch_remote_state()
    mmbench_count = remote_state.get("mmbench_count")
    if not isinstance(mmbench_count, int) or mmbench_count <= 0:
        raise RuntimeError(f"Remote MMBench row count is invalid: {mmbench_count!r}")

    jobs = build_jobs(tuple(args.models), mmbench_count)
    rows = build_matrix(remote_state, jobs)
    summary = summarize_matrix(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_snapshot = write_local_matrix_snapshot(rows, mmbench_count=mmbench_count, remote_state=remote_state, ts=ts)

    if args.mode == "summary":
        print(
            json.dumps(
                {
                    "models": args.models,
                    "mmbench_count": mmbench_count,
                    "matrix_snapshot": str(matrix_snapshot),
                    **summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    runnable_missing_jobs = [
        EvalJob(
            model=row["model"],
            dataset=row["dataset"],
            split=row["split"],
            method=row["method"],
            mini_test=row["target_samples"],
            extra_args=(
                ("--pope_split", row["split"])
                if row["dataset"] == "pope"
                else ("--chair_max_new_tokens", str(CHAIR_MAX_NEW_TOKENS))
                if row["dataset"] == "chair"
                else ()
            ),
        )
        for row in rows
        if row["status"] == "missing"
    ]

    matrix_log = f"{REMOTE_LOG_DIR}/requested_baseline_matrix_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/requested_baseline_matrix_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/requested_baseline_matrix_gpu0_{ts}.manifest.tsv"
    body = build_runner_body(matrix_log=matrix_log, manifest_tsv=manifest_tsv, jobs=runnable_missing_jobs)

    local_runner_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_runner_dir.mkdir(parents=True, exist_ok=True)
    local_runner = local_runner_dir / f"requested_baseline_matrix_gpu0_{ts}.sh"
    local_runner.write_text(body, encoding="utf-8")

    if args.mode == "render":
        print(body)
        return 0

    if not runnable_missing_jobs:
        print(
            json.dumps(
                {
                    "message": "No runnable missing jobs found.",
                    "matrix_snapshot": str(matrix_snapshot),
                    "blocked_rows": summary["blocked_rows"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    proc = run_remote_bash(
        remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log),
        timeout=300,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    launch_meta = parse_launch_output(proc.stdout)
    print("\n--- Launch Summary ---\n")
    print(
        json.dumps(
            {
                "matrix_snapshot": str(matrix_snapshot),
                "local_runner": str(local_runner),
                "job_count": len(runnable_missing_jobs),
                "blocked_count": len(summary["blocked_rows"]),
                "remote": {
                    "log": launch_meta.get("LOG", matrix_log),
                    "runner": launch_meta.get("RUNNER", runner_sh),
                    "launch_pid": launch_meta.get("LAUNCH_PID"),
                    "manifest": manifest_tsv,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
