#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/BRA_Project")
MODEL_DIRS = {
    "qwen3-vl-4b": PROJECT / "models" / "Qwen3-VL-4B-Instruct",
    "qwen3-vl-2b": PROJECT / "models" / "Qwen3-VL-2B-Instruct",
}
DEFAULT_EXTERNAL_ENCODER = PROJECT / "models" / "clip-vit-large-patch14"
OUTPUT_DIRS = {
    "qwen3-vl-4b": PROJECT / "logs" / "uniground_v6" / "second_host_qwen4b",
    "qwen3-vl-2b": PROJECT / "logs" / "uniground_v6" / "second_host_qwen2b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed second-host UniGround entry.")
    parser.add_argument("--dataset", default="pope", choices=["pope", "chair"])
    parser.add_argument(
        "--method",
        default="base",
        choices=[
            "base",
            "uniground",
            "uniground_no_gate",
            "uniground_global_only",
            "external_global_prior",
            "uniground_no_abstain",
            "uniground_region_only",
            "tlra_internal_zero",
            "tlra_internal_calib",
        ],
    )
    parser.add_argument("--mini_test", type=int, default=4)
    parser.add_argument("--pope_split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--chair_max_new_tokens", type=int, default=384)
    parser.add_argument("--psi_checkpoint", default=None)
    parser.add_argument("--external_encoder", default=str(DEFAULT_EXTERNAL_ENCODER))
    parser.add_argument("--external_device", default="cpu")
    parser.add_argument("--projector_checkpoint", default=None)
    parser.add_argument("--force-model", default=None, choices=["qwen3-vl-4b", "qwen3-vl-2b"])
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def choose_model(force_model: str | None) -> str:
    if force_model:
        if not MODEL_DIRS[force_model].exists():
            raise FileNotFoundError(f"Requested second-host model is missing: {MODEL_DIRS[force_model]}")
        return force_model
    for model_key in ("qwen3-vl-4b", "qwen3-vl-2b"):
        if MODEL_DIRS[model_key].exists():
            return model_key
    raise FileNotFoundError("Neither second-host model directory exists; expected Qwen3-VL-4B-Instruct or Qwen3-VL-2B-Instruct.")


def main() -> int:
    args = parse_args()
    model_key = choose_model(args.force_model)
    output_dir = args.output_dir or str(OUTPUT_DIRS[model_key])

    cmd = [
        sys.executable,
        str(PROJECT / "run_uniground_eval.py"),
        "--model",
        model_key,
        "--dataset",
        args.dataset,
        "--method",
        args.method,
        "--mini_test",
        str(args.mini_test),
        "--pope_split",
        args.pope_split,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--chair_max_new_tokens",
        str(args.chair_max_new_tokens),
        "--external_encoder",
        args.external_encoder,
        "--external_device",
        args.external_device,
        "--output_dir",
        output_dir,
    ]
    if args.psi_checkpoint:
        cmd.extend(["--psi_checkpoint", args.psi_checkpoint])
    if args.projector_checkpoint:
        cmd.extend(["--projector_checkpoint", args.projector_checkpoint])

    print("Selected second host:", model_key)
    print("Output dir:", output_dir)
    print("Command:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT))


if __name__ == "__main__":
    raise SystemExit(main())
