from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/BRA_Project")
LOG_DIR = ROOT / "logs" / "v3_engineer_a"
PYTHON = "/root/miniconda3/bin/python"


def job_spec(name: str) -> tuple[str, str]:
    if name == "stage0":
        log_path = LOG_DIR / "stage0_tlra_zero_8b.log"
        command = (
            f"cd {ROOT} && "
            f"CUDA_VISIBLE_DEVICES=0 {PYTHON} tlra_semantic_validity_pilot.py "
            "--model qwen3-vl-8b --method tlra_zero --n_samples 64 "
            "--topk 10 --candidate_window 50 "
            "--output logs/v3_engineer_a/stage0_tlra_zero_8b.json"
        )
        return command, str(log_path)

    if name == "chainA":
        log_path = LOG_DIR / "chainA_main.log"
        commands = [
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 200",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 200",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 200",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method tlra_zero --mini_test 200",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base --mini_test 150 --chair_max_new_tokens 384",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd --mini_test 150 --chair_max_new_tokens 384",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola --mini_test 150 --chair_max_new_tokens 384",
            "CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method tlra_zero --mini_test 150 --chair_max_new_tokens 384",
        ]
        command = f"cd {ROOT} && " + "; ".join(commands)
        return command, str(log_path)

    if name == "chainB":
        log_path = LOG_DIR / "chainB_main.log"
        commands = [
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_full --output logs/v3_engineer_a/mmbench_tlra_full.json",
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_no_vasm --output logs/v3_engineer_a/mmbench_tlra_no_vasm.json",
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mme --n_samples 300 --bra_method tlra_full --output logs/v3_engineer_a/mme_tlra_full.json",
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mme --n_samples 300 --bra_method tlra_no_vasm --output logs/v3_engineer_a/mme_tlra_no_vasm.json",
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu --n_samples 200 --bra_method tlra_full --output logs/v3_engineer_a/mmmu_tlra_full.json",
            "CUDA_VISIBLE_DEVICES=2 /root/miniconda3/bin/python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu --n_samples 200 --bra_method tlra_no_vasm --output logs/v3_engineer_a/mmmu_tlra_no_vasm.json",
        ]
        command = f"cd {ROOT} && " + "; ".join(commands)
        return command, str(log_path)

    raise ValueError(f"Unsupported job: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, choices=["stage0", "chainA", "chainB"])
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    command, log_path = job_spec(args.job)

    env = os.environ.copy()
    env["PATH"] = f"/root/miniconda3/bin:{env.get('PATH', '')}"

    with open(log_path, "ab", buffering=0) as log_file:
        log_file.write(f"=== launch {args.job} ===\n".encode("utf-8"))
        log_file.write(f"{command}\n".encode("utf-8"))
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    print(process.pid)
    print(log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
