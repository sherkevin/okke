#!/usr/bin/env python3
"""
Download MVBench Temporal Action Order subset for 10-sample mini-batch.
MVBench is hosted on HuggingFace: OpenGVLab/MVBench
We focus on the "Temporal Action Order" task type.
"""

import os, json, random

MVBENCH_DIR = "/root/autodl-tmp/A-OSP_Project/data/mvbench"
os.makedirs(MVBENCH_DIR, exist_ok=True)

print("=== Downloading MVBench (Temporal Action Order subset) ===")

try:
    from datasets import load_dataset
    print("[1/3] Attempting HuggingFace datasets load...")
    ds = load_dataset(
        "OpenGVLab/MVBench",
        name="temporal_action_order",
        split="test",
        cache_dir=MVBENCH_DIR,
        trust_remote_code=True,
    )
    print(f"  Loaded {len(ds)} samples from OpenGVLab/MVBench (temporal_action_order)")

    # Save first 50 as JSONL
    out_file = os.path.join(MVBENCH_DIR, "temporal_action_order_50.jsonl")
    with open(out_file, "w") as f:
        for i, sample in enumerate(ds):
            if i >= 50:
                break
            row = {k: (v if not hasattr(v, 'save') else None) for k,v in sample.items() if k != 'video'}
            f.write(json.dumps(row) + "\n")
    print(f"  Saved 50-sample subset → {out_file}")

except Exception as e:
    print(f"  HF load failed: {e}")
    print("[2/3] Trying snapshot_download...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="OpenGVLab/MVBench",
            repo_type="dataset",
            local_dir=MVBENCH_DIR,
            allow_patterns=["temporal_action_order*", "*.json", "*.jsonl"],
            ignore_patterns=["*.mp4", "*.avi", "*.mkv"],  # skip videos for now
        )
        print(f"  Downloaded metadata to {MVBENCH_DIR}")
    except Exception as e2:
        print(f"  snapshot_download also failed: {e2}")
        print("[3/3] Creating synthetic 10-sample MVBench Temporal Action Order data for pipeline test...")

        # Create synthetic test data that matches MVBench format
        CATEGORIES = [
            "Temporal Action Order",
        ]

        ACTION_PAIRS = [
            ("person picks up a cup", "person drinks from the cup"),
            ("person opens a door", "person walks through the door"),
            ("chef chops vegetables", "chef puts vegetables in pan"),
            ("athlete starts running", "athlete jumps over hurdle"),
            ("person writes on paper", "person folds the paper"),
            ("worker hammers nail", "worker sands the surface"),
            ("baker puts bread in oven", "baker takes bread out"),
            ("person turns on faucet", "person washes hands"),
            ("student opens textbook", "student takes notes"),
            ("mechanic lifts hood", "mechanic checks engine"),
        ]

        WRONG_ORDERS = [
            ("person drinks from the cup", "person picks up a cup"),
            ("person walks through the door", "person opens a door"),
            ("chef puts vegetables in pan", "chef chops vegetables"),
            ("athlete jumps over hurdle", "athlete starts running"),
            ("person folds the paper", "person writes on paper"),
            ("worker sands the surface", "worker hammers nail"),
            ("baker takes bread out", "baker puts bread in oven"),
            ("person washes hands", "person turns on faucet"),
            ("student takes notes", "student opens textbook"),
            ("mechanic checks engine", "mechanic lifts hood"),
        ]

        # Each sample: question asks which sequence is correct
        samples = []
        for i in range(10):
            a1, a2 = ACTION_PAIRS[i]
            w1, w2 = WRONG_ORDERS[i]
            # Randomly decide correct answer position
            if random.random() > 0.5:
                choices = [f"First {a1}, then {a2}.", f"First {w1}, then {w2}."]
                answer = "A"
            else:
                choices = [f"First {w1}, then {w2}.", f"First {a1}, then {a2}."]
                answer = "B"

            sample = {
                "id": f"mvbench_tao_{i:04d}",
                "task_type": "Temporal Action Order",
                "question": f"In what order do the actions occur in the video?",
                "choices": choices,
                "answer": answer,
                "video_path": None,  # No actual video - will use synthetic test
                "synthetic": True,
            }
            samples.append(sample)

        out_file = os.path.join(MVBENCH_DIR, "temporal_action_order_10_synthetic.jsonl")
        with open(out_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Created 10 synthetic samples → {out_file}")

print("\nDone.")
