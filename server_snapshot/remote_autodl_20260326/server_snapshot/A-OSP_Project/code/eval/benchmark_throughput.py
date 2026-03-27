#!/usr/bin/env python3
"""
A-OSP Evaluation Pipeline — Throughput Benchmark (512 Tokens)
==============================================================
Measures Tokens/s, peak VRAM, and per-token latency for Table X / Pareto bubble.

CRITICAL — Lock GPU frequency BEFORE running:
    sudo nvidia-smi -lgc 2000,2000

This counters thermal throttling on consumer GPUs (RTX 5090 / 4090 / etc.)
to ensure reproducible, fair cross-method comparisons.

Protocol:
  1. Warm-up: 3 forward passes (discarded).
  2. Benchmark: 5 timed passes generating exactly 512 tokens each.
  3. Report: median throughput, peak memory, per-token latency.

Usage:
    python benchmark_throughput.py \
        --model_path /root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct \
        --output_dir /root/autodl-tmp/A-OSP_Project/logs/eval_results \
        --tag        base_qwen2vl7b_throughput
"""

import argparse
import gc
import json
import os
import sys
import time
import statistics
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_utils import load_qwen2vl, log_gpu_memory, save_csv_summary


GEN_LENGTH = 512
WARMUP_RUNS = 3
BENCH_RUNS = 5

FREQ_LOCK_CMD = "sudo nvidia-smi -lgc 2000,2000"
FREQ_RESET_CMD = "sudo nvidia-smi -rgc"


def check_gpu_freq_lock():
    """Print a loud warning if frequency lock is not confirmed."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║  ⚠  GPU FREQUENCY LOCK REMINDER                            ║
║                                                              ║
║  Before running this benchmark, execute:                     ║
║      sudo nvidia-smi -lgc 2000,2000                         ║
║                                                              ║
║  After benchmarking, reset with:                             ║
║      sudo nvidia-smi -rgc                                   ║
║                                                              ║
║  Failure to lock frequency will produce unreliable results   ║
║  due to thermal throttling on consumer GPUs.                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def build_dummy_input(processor, device) -> tuple:
    """Create a deterministic dummy input (grey image + generic prompt)."""
    image = Image.new("RGB", (448, 448), color=(128, 128, 128))
    prompt = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)
    return inputs, inputs["input_ids"].shape[1]


def run_single_generation(
    model, inputs, input_len: int, aosp_handle=None,
) -> tuple[int, float, int]:
    """
    Generate exactly GEN_LENGTH tokens.
    Returns (actual_gen_len, latency_seconds, peak_memory_bytes).
    """
    torch.cuda.reset_peak_memory_stats()

    if aosp_handle is not None:
        aosp_handle.reset()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GEN_LENGTH,
            min_new_tokens=GEN_LENGTH,
            do_sample=False,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    gen_len = output_ids.shape[1] - input_len
    peak_mem = torch.cuda.max_memory_allocated()

    del output_ids
    gc.collect()
    return int(gen_len), latency, peak_mem


def run_benchmark(args):
    check_gpu_freq_lock()

    model, processor = load_qwen2vl(args.model_path)
    inputs, input_len = build_dummy_input(processor, model.device)

    # ── Optional A-OSP hook ──
    aosp_handle = None
    if args.enable_aosp:
        from aosp_hook import apply_aosp_hook
        aosp_handle = apply_aosp_hook(
            model, args.v_matrix,
            alpha=args.alpha, mu=args.mu, beta=args.beta,
        )
        print(f"[Bench] A-OSP hook enabled (alpha={args.alpha}, mu={args.mu})")

    # ── Warm-up ──
    print(f"[Warmup] Running {WARMUP_RUNS} passes (discarded) ...")
    for i in range(WARMUP_RUNS):
        gen_len, lat, _ = run_single_generation(model, inputs, input_len, aosp_handle)
        print(f"  warmup {i+1}: gen_len={gen_len}, latency={lat:.2f}s")

    # ── Benchmark ──
    print(f"\n[Bench] Running {BENCH_RUNS} timed passes @ {GEN_LENGTH} tokens ...")
    latencies = []
    throughputs = []
    peak_mems = []

    for i in range(BENCH_RUNS):
        gen_len, lat, peak_mem = run_single_generation(
            model, inputs, input_len, aosp_handle
        )
        tps = gen_len / lat
        latencies.append(lat)
        throughputs.append(tps)
        peak_mems.append(peak_mem)
        print(
            f"  run {i+1}: gen_len={gen_len} | "
            f"latency={lat:.3f}s | "
            f"throughput={tps:.1f} tok/s | "
            f"peak_mem={peak_mem / 1024**3:.2f} GB"
        )

    # ── Report ──
    med_tps = statistics.median(throughputs)
    med_lat = statistics.median(latencies)
    max_peak_gb = max(peak_mems) / 1024**3
    per_token_ms = (med_lat / GEN_LENGTH) * 1000

    report = {
        "method": args.method_tag,
        "model": "Qwen2-VL-7B-Instruct",
        "gen_length": GEN_LENGTH,
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "median_throughput_tps": round(med_tps, 2),
        "median_latency_s": round(med_lat, 3),
        "per_token_latency_ms": round(per_token_ms, 3),
        "peak_memory_gb": round(max_peak_gb, 2),
        "all_throughputs": [round(t, 2) for t in throughputs],
        "all_latencies": [round(l, 3) for l in latencies],
    }

    result_file = os.path.join(args.output_dir, f"{args.tag}_throughput.json")
    csv_file = os.path.join(args.output_dir, f"{args.tag}_throughput.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(result_file, "w") as f:
        json.dump(report, f, indent=2)

    save_csv_summary(csv_file, {
        k: v for k, v in report.items()
        if not isinstance(v, list)
    })

    print("\n" + "=" * 60)
    print("  THROUGHPUT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Median Throughput : {med_tps:.2f} Tokens/s")
    print(f"  Median Latency    : {med_lat:.3f} s  (for {GEN_LENGTH} tokens)")
    print(f"  Per-token Latency : {per_token_ms:.3f} ms")
    print(f"  Peak VRAM         : {max_peak_gb:.2f} GB")
    print("=" * 60)
    print(f"  JSON → {result_file}")
    print(f"  CSV  → {csv_file}")

    if aosp_handle is not None:
        aosp_handle.remove()

    log_gpu_memory("final")


def parse_args():
    p = argparse.ArgumentParser(description="A-OSP Throughput Benchmark")
    p.add_argument(
        "--model_path", type=str,
        default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="/root/autodl-tmp/A-OSP_Project/logs/eval_results",
    )
    p.add_argument("--tag", type=str, default="base_qwen2vl7b")
    p.add_argument(
        "--method_tag", type=str, default="base",
        help="Method label for comparison table (base / aosp / vcd / ...)",
    )
    # A-OSP options
    p.add_argument("--enable_aosp", action="store_true",
                    help="Enable A-OSP intervention for throughput comparison")
    p.add_argument("--v_matrix", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
