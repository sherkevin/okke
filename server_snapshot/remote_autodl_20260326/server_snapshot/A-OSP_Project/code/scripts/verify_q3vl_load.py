#!/usr/bin/env python3
"""
Quick verification that Qwen3-VL-8B loads correctly and can do inference.
Run this immediately after download completes to verify before running evals.

Usage:
    python verify_q3vl_load.py
    python verify_q3vl_load.py --model_path /path/to/model
"""
import argparse, sys, time, gc
from pathlib import Path

import torch

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
DEFAULT_MODEL = str(PROJECT / "models" / "Qwen3-VL-8B-Instruct")


def check_shards(model_path: str) -> bool:
    shards = sorted(Path(model_path).glob("model-*.safetensors"))
    print(f"[Check] Found {len(shards)} safetensor shards:")
    for s in shards:
        size_gb = s.stat().st_size / 1e9
        print(f"  {s.name}: {size_gb:.2f} GB")
    incomplete = list(Path(model_path).glob(".cache/**/*.incomplete"))
    if incomplete:
        for f in incomplete:
            size_mb = f.stat().st_size / 1e6
            print(f"  [INCOMPLETE] {f.name}: {size_mb:.1f} MB")
        print("[WARN] Incomplete files found - download not complete!")
        return False
    if len(shards) < 4:
        print(f"[WARN] Only {len(shards)}/4 shards found")
        return False
    print("[OK] All shards present, no incomplete files")
    return True


def run_inference_test(model, processor) -> dict:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Create a simple test image
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What color is this image? Answer in one word."},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=20, do_sample=False)
    latency = time.time() - t0

    gen_ids = output_ids[0, input_len:]
    response = processor.decode(gen_ids, skip_special_tokens=True).strip()
    tps = gen_ids.shape[0] / latency

    del inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "response": response,
        "n_tokens": gen_ids.shape[0],
        "latency_s": round(latency, 2),
        "tokens_per_sec": round(tps, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--skip_inference", action="store_true")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Qwen3-VL-8B Load Verification")
    print(f"  model: {args.model_path}")
    print(f"{'='*60}\n")

    # Step 1: Check shards
    ok = check_shards(args.model_path)
    if not ok:
        print("\n[FAIL] Shard check failed. Please wait for download to complete.")
        sys.exit(1)

    # Step 2: Load model
    sys.path.insert(0, str(PROJECT / "code" / "eval"))
    from eval_utils import load_qwen2vl, log_gpu_memory

    print("\n[Loading] model + processor...")
    t0 = time.time()
    model, processor = load_qwen2vl(args.model_path)
    load_time = time.time() - t0
    print(f"[OK] Loaded in {load_time:.1f}s")
    log_gpu_memory("after_load")

    # Verify architecture
    if hasattr(model.model, 'language_model'):
        n_layers = len(model.model.language_model.layers)
        hidden = model.model.language_model.layers[0].input_layernorm.weight.shape[0]
    else:
        n_layers = len(model.model.layers)
        hidden = model.config.hidden_size
    print(f"[Architecture] n_layers={n_layers}, hidden_size={hidden}")
    assert n_layers == 36, f"Expected 36 layers, got {n_layers}"
    assert hidden == 4096, f"Expected hidden_size=4096, got {hidden}"
    print("[OK] Architecture verified: 36 layers × 4096 dims")

    # Step 3: Inference test
    if not args.skip_inference:
        print("\n[Inference] Running test inference...")
        result = run_inference_test(model, processor)
        print(f"[OK] Response: '{result['response']}'")
        print(f"     Tokens: {result['n_tokens']} | "
              f"Latency: {result['latency_s']}s | "
              f"Speed: {result['tokens_per_sec']} tok/s")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("✓ Qwen3-VL-8B-Instruct VERIFIED READY")
    print(f"  Next step: python wait_and_extract.sh (or run directly)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
