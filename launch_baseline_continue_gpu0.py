#!/usr/bin/env python3
"""
Queue the remaining requested baseline runs on GPU0 after the current queue finishes.

Requested scope:
  - models: llava-v1.5-7b, instructblip-7b
  - methods: base, dola, opera
  - datasets: pope, chair, mmbench

Rules:
  - do not rerun full cells that already have a completed JSON
  - optionally wait for another remote PID before launching
  - keep a manifest of skipped vs executed jobs
"""
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/baseline_continue"
REMOTE_MINITEST = f"{REMOTE_PROJECT}/logs/minitest"
MMBENCH_REMOTE_FILE = f"{REMOTE_PROJECT}/datasets/MMBench_EN_hf/data/dev-00000-of-00001-75b6649fb044d38b.parquet"

POPE_COUNT = 3000
CHAIR_COUNT = 5000
CHAIR_MAX_NEW_TOKENS = 384
DEFAULT_WAIT_PID = 65635


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


def _remote_mmbench_count() -> int:
    script = f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
import pandas as pd
df = pd.read_parquet("{MMBENCH_REMOTE_FILE}")
print(len(df))
PY
"""
    proc = run_remote_bash(script, timeout=180, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to count remote MMBench rows:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return int(proc.stdout.strip().splitlines()[-1])


def build_jobs(mmbench_count: int, datasets: tuple[str, ...]) -> list[EvalJob]:
    jobs: list[EvalJob] = []

    if "pope" in datasets:
        for model in ("llava-v1.5-7b", "instructblip-7b"):
            for split in ("random", "popular", "adversarial"):
                for method in ("base", "dola", "opera"):
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

    if "chair" in datasets:
        for model in ("llava-v1.5-7b", "instructblip-7b"):
            for method in ("base", "dola", "opera"):
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

    if "mmbench" in datasets:
        for model in ("llava-v1.5-7b", "instructblip-7b"):
            for method in ("base", "dola", "opera"):
                jobs.append(
                    EvalJob(
                        model=model,
                        dataset="mmbench",
                        method=method,
                        split="default",
                        mini_test=mmbench_count,
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
    return " ".join(shlex.quote(p) for p in parts)


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
        f'MINITEST_DIR="{REMOTE_MINITEST}"',
        'echo "[MATRIX] baseline_continue_gpu0 start $(date -Iseconds)"',
        'printf "iso_time\\tmodel\\tdataset\\tsplit\\tmethod\\taction\\texit_code\\tresult_json\\tmatrix_parent_log\\n" > "$MANIFEST"',
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
        "candidates = sorted(root.glob(f\"{method}_{dataset}_*.json\"), key=lambda p: p.stat().st_mtime, reverse=True)",
        "for path in candidates:",
        "    try:",
        "        payload = json.loads(path.read_text(encoding='utf-8'))",
        "    except Exception:",
        "        continue",
        "    if payload.get('model') != model:",
        "        continue",
        "    if payload.get('dataset') != dataset:",
        "        continue",
        "    if payload.get('method') != method:",
        "        continue",
        "    if dataset == 'pope' and payload.get('pope_split') != split:",
        "        continue",
        "    sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)",
        "    n_errors = int(payload.get('n_errors', 0) or 0)",
        "    status = payload.get('status')",
        "    if sample_count != target or n_errors != 0:",
        "        continue",
        "    if status not in (None, 'final'):",
        "        continue",
        "    print(str(path))",
        "    break",
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
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" '
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
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" '
            '"executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'  echo "[DONE] {job.tag} rc=$rc result=$result $(date -Iseconds)"')
        lines.append('  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
        lines.append('fi')
        lines.append("")

    lines.append('echo "[MATRIX] baseline_continue_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str, wait_for_pid: int | None) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")

    watch_sh = runner_sh.replace(".sh", ".watch.sh")
    watch_log = matrix_log.replace(".log", ".watch.log")

    launcher = (
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
    )
    if wait_for_pid is not None:
        launcher = (
            f"cat > \"{watch_sh}\" <<'EOF'\n"
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"WAIT_PID={wait_for_pid}\n"
            f"RUNNER={runner_sh}\n"
            f"MATRIX_LOG={matrix_log}\n"
            f"WATCH_LOG={watch_log}\n"
            "echo \"[WATCH] installed $(date -Iseconds)\" >> \"$WATCH_LOG\"\n"
            "while kill -0 \"$WAIT_PID\" 2>/dev/null; do sleep 20; done\n"
            "echo \"[WATCH] dependency cleared $(date -Iseconds); launching continuation queue\" | tee -a \"$WATCH_LOG\" >> \"$MATRIX_LOG\"\n"
            "nohup /bin/bash \"$RUNNER\" >> \"$MATRIX_LOG\" 2>&1 < /dev/null &\n"
            "echo \"[WATCH] launch_pid=$! $(date -Iseconds)\" | tee -a \"$WATCH_LOG\" >> \"$MATRIX_LOG\"\n"
            "EOF\n"
            f"chmod +x \"{watch_sh}\"\n"
            f"nohup /bin/bash \"{watch_sh}\" > /dev/null 2>&1 < /dev/null &\n"
            "echo \"WATCH_PID=$!\"\n"
            f"echo \"WATCHER={watch_sh}\"\n"
            f"echo \"WATCH_LOG={watch_log}\"\n"
        )

    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"{launcher}"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue remaining requested baselines on GPU0 without rerunning completed full cells.")
    parser.add_argument("--mode", choices=["dry-run", "render", "launch"], default="dry-run")
    parser.add_argument("--wait-for-pid", type=int, default=DEFAULT_WAIT_PID)
    parser.add_argument("--mmbench-count", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=["pope", "chair", "mmbench"], default=["pope", "chair", "mmbench"])
    args = parser.parse_args()

    wait_for_pid = None if args.wait_for_pid is not None and args.wait_for_pid <= 0 else args.wait_for_pid
    mmbench_count = args.mmbench_count if args.mmbench_count is not None else _remote_mmbench_count()
    jobs = build_jobs(mmbench_count=mmbench_count, datasets=tuple(args.datasets))
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/baseline_continue_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/baseline_continue_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/baseline_continue_gpu0_{ts}.manifest.tsv"

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "wait_for_pid": args.wait_for_pid,
                    "datasets": args.datasets,
                    "mmbench_count": mmbench_count,
                    "job_count": len(jobs),
                    "first": [job.tag for job in jobs[:5]],
                    "last": [job.tag for job in jobs[-5:]],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print("\nRemote paths (preview):")
        print(matrix_log)
        print(runner_sh)
        print(manifest_tsv)
        return 0

    body = build_runner_body(matrix_log=matrix_log, manifest_tsv=manifest_tsv, jobs=jobs)
    local_backup_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_backup_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_backup_dir / f"baseline_continue_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")
    print(f"Wrote local runner copy: {local_copy}")

    if args.mode == "render":
        print(body)
        return 0

    proc = run_remote_bash(
        remote_launch_script(
            runner_body=body,
            runner_sh=runner_sh,
            matrix_log=matrix_log,
            wait_for_pid=wait_for_pid,
        ),
        timeout=300,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\n--- Suggested registry row ---\n")
    parent = f"baseline_continue_gpu0_{ts}"
    state = "scheduled-after-pid" if wait_for_pid is not None else "running"
    print(
        f"| `{parent}` | `llava-v1.5-7b + instructblip-7b` | `POPE+CHAIR+MMBench` | "
        f"`mixed` | `{state}` | `{matrix_log}` | `see manifest {manifest_tsv}` | "
        "Continuation queue on GPU0 for requested `base+dola+opera`; runner skips already completed full cells at runtime. |"
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
