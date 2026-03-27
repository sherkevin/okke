#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/BRA_Project")
PIPELINE = PROJECT / "run_eval_pipeline.py"
MATRIX = PROJECT / "bra_eval_matrix.py"

EVIDENCE_CHAINS = {
    "a": ["pope", "chair"],
    "b": ["mmbench", "mme", "mmmu"],
    "c": ["freak", "docvqa"],
    "video": ["vidhalluc", "video_mme"],
}

MAIN_METHODS = [
    "base", "vcd", "opera", "dola",
    "bra_zero", "bra_calib",
    "tlra_zero", "tlra_calib", "tlra_full", "tlra_adaptivetopk",
]
ABLATION_METHODS = [
    "bra_meanpool", "bra_max", "bra_no_vasm", "bra_v1_like",
    "tlra_meanpool", "tlra_no_vasm", "tlra_randomk",
]
PIPELINE_METHODS = {
    "base", "vcd", "opera", "dola",
    "bra_zero", "bra_calib", "bra_meanpool", "bra_max", "bra_no_vasm", "bra_v1_like",
    "tlra_zero", "tlra_calib", "tlra_full", "tlra_adaptivetopk",
    "tlra_meanpool", "tlra_no_vasm", "tlra_randomk",
}
MATRIX_METHODS = {
    "bra_zero", "bra_calib", "bra_meanpool", "bra_max", "bra_no_vasm", "bra_v1_like",
    "tlra_zero", "tlra_calib", "tlra_full", "tlra_adaptivetopk",
    "tlra_meanpool", "tlra_no_vasm", "tlra_randomk",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl-2b")
    parser.add_argument("--mini_test", type=int, default=20)
    parser.add_argument("--chains", nargs="+", default=["a"])
    parser.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    parser.add_argument("--pope_split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--vasm_artifact", default=None)
    parser.add_argument("--projector_checkpoint", default=None)
    parser.add_argument("--output_json", default="logs/minitest/benchmark_runner.json")
    parser.add_argument("--output_csv", default="logs/minitest/benchmark_runner.csv")
    return parser.parse_args()


def datasets_for_chains(chains):
    out = []
    for chain in chains:
        out.extend(EVIDENCE_CHAINS[chain])
    return out


def normalize_method_name(method: str) -> str:
    aliases = {
        "bra_maxpool": "bra_max",
        "ablation_meanpool": "bra_meanpool",
        "ablation_maxpool": "bra_max",
        "no_mask": "bra_no_vasm",
        "bra": "bra_zero",
        "tlra": "tlra_zero",
        "tlra_v2": "tlra_full",
        "tlra_max": "bra_max",
        "tlra_v1_like": "bra_v1_like",
    }
    return aliases.get(method.lower(), method.lower())


def run_pipeline_method(method: str, dataset: str, args):
    cmd = [
        sys.executable, str(PIPELINE),
        "--method", method,
        "--dataset", dataset,
        "--model", args.model,
        "--mini_test", str(args.mini_test),
        "--pope_split", args.pope_split,
    ]
    if args.vasm_artifact:
        cmd.extend(["--vasm_artifact", args.vasm_artifact])
    if args.projector_checkpoint:
        cmd.extend(["--projector_checkpoint", args.projector_checkpoint])
    subprocess.run(cmd, check=True, cwd=str(PROJECT))

    log_dir = PROJECT / "logs" / "minitest"
    candidates = sorted(log_dir.glob(f"{method}_{dataset}_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No result file found for {method} / {dataset}")
    with open(candidates[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def run_matrix_method(method: str, dataset: str, args):
    output_name = f"logs/minitest/{dataset}_{method}_matrix.json"
    cmd = [
        sys.executable, str(MATRIX),
        "--model", "qwen3vl2b",
        "--dataset", dataset,
        "--n_samples", str(args.mini_test),
        "--bra_method", method,
        "--output", output_name,
    ]
    if args.vasm_artifact:
        cmd.extend(["--vasm_artifact", args.vasm_artifact])
    if args.projector_checkpoint:
        cmd.extend(["--projector_checkpoint", args.projector_checkpoint])
    subprocess.run(cmd, check=True, cwd=str(PROJECT))

    out_path = PROJECT / output_name
    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    first_model = next(iter(payload.values()))
    return first_model[0] if first_model else None


def run_one(method: str, dataset: str, args):
    method = normalize_method_name(method)
    if dataset in {"pope", "chair"}:
        if method in PIPELINE_METHODS:
            return run_pipeline_method(method, dataset, args)
        raise ValueError(f"Unsupported method for pipeline dataset: {method}")

    if method == "base":
        result = run_matrix_method("bra_zero", dataset, args)
        if result is None:
            return {"dataset": dataset, "method": "base", "notes": ["no_samples"]}
        return {
            "dataset": dataset,
            "method": "base",
            "model_family": result.get("model_family", "qwen3_vl"),
            "sample_count": result.get("sample_count", result.get("n_samples", 0)),
            "notes": result.get("notes", []),
            **result["baseline"],
        }

    if method in MATRIX_METHODS:
        result = run_matrix_method(method, dataset, args)
        if result is None:
            return {"dataset": dataset, "method": method, "notes": ["no_samples"]}
        return {
            "dataset": dataset,
            "method": method,
            "model_family": result.get("model_family", "qwen3_vl"),
            "sample_count": result.get("sample_count", result.get("n_samples", 0)),
            "notes": result.get("notes", []),
            **result["bra"],
        }

    return {
        "dataset": dataset,
        "method": method,
        "sample_count": 0,
        "notes": ["unsupported_for_dataset"],
    }


def flatten_for_csv(results):
    rows = []
    for dataset, per_method in results.items():
        for method, item in per_method.items():
            row = {
                "dataset": dataset,
                "method": method,
                "model_family": item.get("model_family"),
                "sample_count": item.get("sample_count"),
                "accuracy": item.get("accuracy"),
                "f1": item.get("f1"),
                "chair_s": item.get("chair_s"),
                "chair_i": item.get("chair_i"),
                "agl": item.get("agl"),
                "itl_ms_per_token": item.get("itl_ms_per_token"),
                "tokens_per_second": item.get("tokens_per_second"),
                "peak_vram_gb": item.get("peak_vram_gb"),
                "intervention_rate": item.get("intervention_rate"),
                "notes": "|".join(item.get("notes", [])) if isinstance(item.get("notes"), list) else item.get("notes"),
            }
            rows.append(row)
    return rows


def summarize(results):
    print("\n" + "=" * 80)
    print("BRA BENCHMARK RUNNER")
    print("=" * 80)
    for dataset, per_method in results.items():
        print(f"\n[{dataset}]")
        for method, item in per_method.items():
            notes = item.get("notes", [])
            note_text = f" notes={notes}" if notes else ""
            if "chair_s" in item:
                print(
                    f"  {method:<14s} chair_s={item.get('chair_s', 0):.4f} "
                    f"chair_i={item.get('chair_i', 0):.4f} agl={item.get('agl', 0):.2f} "
                    f"itl={item.get('itl_ms_per_token', 0):.2f}{note_text}"
                )
            else:
                print(
                    f"  {method:<14s} acc={item.get('accuracy', 0):.4f} "
                    f"f1={item.get('f1', 0):.4f} agl={item.get('agl', 0):.2f} "
                    f"itl={item.get('itl_ms_per_token', 0):.2f}{note_text}"
                )


def write_outputs(results, args):
    out_json = PROJECT / args.output_json
    out_csv = PROJECT / args.output_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = flatten_for_csv(results)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "dataset", "method", "model_family", "sample_count", "accuracy", "f1",
            "chair_s", "chair_i", "agl", "itl_ms_per_token", "tokens_per_second",
            "peak_vram_gb", "intervention_rate", "notes",
        ])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    print(f"\nSaved JSON to {out_json}")
    print(f"Saved CSV to {out_csv}")


def main():
    args = parse_args()
    datasets = datasets_for_chains(args.chains)
    results = {}
    for dataset in datasets:
        per_method = {}
        for method in args.methods:
            print(f"\n=== RUN {dataset} / {method} ===")
            try:
                per_method[method] = run_one(method, dataset, args)
            except Exception as exc:
                per_method[method] = {
                    "dataset": dataset,
                    "method": method,
                    "sample_count": 0,
                    "notes": [f"error:{exc}"],
                }
        results[dataset] = per_method
    summarize(results)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
