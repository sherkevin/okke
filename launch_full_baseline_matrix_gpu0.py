#!/usr/bin/env python3
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

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/baseline_full_matrix"
REMOTE_RESULTS_DIR = f"{REMOTE_LOG_DIR}/results"
MMBENCH_REMOTE_FILE = f"{REMOTE_PROJECT}/datasets/MMBench_EN_hf/data/dev-00000-of-00001-75b6649fb044d38b.parquet"

MODELS = (
    "qwen3-vl-8b",
    "qwen2-vl-7b",
    "qwen2.5-vl-7b",
    "instructblip-7b",
    "llava-v1.5-7b",
)
METHODS = ("base", "opera", "vcd", "dola")
POPE_SPLITS = ("random", "popular", "adversarial")
POPE_COUNT = 3000
CHAIR_COUNT = 5000
CHAIR_MAX_NEW_TOKENS = 384
MODEL_REMOTE_DIRS = {
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    "qwen2-vl-7b": "Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B-Instruct",
    "instructblip-7b": "instructblip-vicuna-7b",
    "llava-v1.5-7b": "llava-1.5-7b-hf",
}


@dataclass(frozen=True)
class EvalJob:
    model: str
    dataset: str
    method: str
    split: str
    mini_test: int
    extra_args: tuple[str, ...]
    output_json: str

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _remote_mmbench_count() -> int:
    script = f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
import pandas as pd
print(len(pd.read_parquet("{MMBENCH_REMOTE_FILE}")))
PY
"""
    proc = run_remote_bash(script, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to count remote MMBench rows:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return int(proc.stdout.strip().splitlines()[-1])


def build_jobs(mmbench_count: int) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    for model in MODELS:
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
                        output_json=f"{REMOTE_RESULTS_DIR}/{model}/pope/{split}__{method}.json",
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
                    extra_args=(),
                    output_json=f"{REMOTE_RESULTS_DIR}/{model}/mmbench/{method}.json",
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
                    output_json=f"{REMOTE_RESULTS_DIR}/{model}/chair/{method}.json",
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
        "--output_json",
        job.output_json,
        *job.extra_args,
    ]
    return _shell_join(parts)


def build_runner_body(*, jobs: list[EvalJob], matrix_log: str, manifest_tsv: str) -> str:
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
        f'RESULT_ROOT="{REMOTE_RESULTS_DIR}"',
        'echo "[MATRIX] baseline_full_matrix_gpu0 start $(date -Iseconds)"',
        'printf "iso_time\\tmodel\\tdataset\\tsplit\\tmethod\\texit_code\\tresult_json\\tmatrix_parent_log\\n" > "$MANIFEST"',
        'mkdir -p "$RESULT_ROOT"',
        "",
    ]
    for job in jobs:
        cmd = eval_command(job)
        lines.append(f'echo "[START] {job.tag} $(date -Iseconds)"')
        lines.append(f'mkdir -p "{Path(job.output_json).parent.as_posix()}"')
        lines.append(f"{cmd}")
        lines.append("rc=$?")
        lines.append(
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "$rc" "{job.output_json}" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'echo "[DONE] {job.tag} rc=$rc output={job.output_json} $(date -Iseconds)"')
        lines.append('if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
        lines.append("")
    lines.append('echo "[MATRIX] baseline_full_matrix_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR} {REMOTE_RESULTS_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def preflight_script() -> str:
    return f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
import json
project = Path("{REMOTE_PROJECT}")
summary = {{
    "models_present": {{
        key: (project / "models" / value).exists()
        for key, value in {MODEL_REMOTE_DIRS!r}.items()
    }},
    "has_pope_random": (project / "datasets/POPE/output/coco/coco_pope_random.json").exists(),
    "has_pope_popular": (project / "datasets/POPE/output/coco/coco_pope_popular.json").exists(),
    "has_pope_adversarial": (project / "datasets/POPE/output/coco/coco_pope_adversarial.json").exists(),
    "has_mmbench": Path("{MMBENCH_REMOTE_FILE}").exists(),
    "has_instances": (project / "datasets/coco2014/annotations/instances_val2014.json").exists(),
    "has_captions": (project / "datasets/coco2014/annotations/captions_val2014.json").exists(),
}}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the full baseline matrix on remote GPU0 from scratch.")
    parser.add_argument("--mode", choices=["preflight", "render", "launch"], default="preflight")
    args = parser.parse_args()

    if args.mode == "preflight":
        proc = run_remote_bash(preflight_script(), timeout=300, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    mmbench_count = _remote_mmbench_count()
    jobs = build_jobs(mmbench_count)
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/baseline_full_matrix_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/baseline_full_matrix_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/baseline_full_matrix_gpu0_{ts}.manifest.tsv"
    body = build_runner_body(jobs=jobs, matrix_log=matrix_log, manifest_tsv=manifest_tsv)

    local_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_dir / f"baseline_full_matrix_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")

    if args.mode == "render":
        print(body)
        return 0

    proc = run_remote_bash(
        remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log),
        timeout=300,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\n--- Launch Summary ---\n")
    print(
        json.dumps(
            {
                "mmbench_count": mmbench_count,
                "job_count": len(jobs),
                "local_runner": str(local_copy),
                "remote_log": matrix_log,
                "remote_runner": runner_sh,
                "remote_manifest": manifest_tsv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
