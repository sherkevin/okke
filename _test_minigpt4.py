"""Test MiniGPT-4 model loading and basic inference."""
import sys
import os
import torch

sys.path.insert(0, "/root/autodl-tmp/BRA_Project/MiniGPT-4")
os.chdir("/root/autodl-tmp/BRA_Project/MiniGPT-4")

print("Testing MiniGPT-4 imports...")
try:
    from minigpt4.common.config import Config
    print("  Config import OK")
except Exception as e:
    print(f"  Config import FAIL: {e}")

try:
    from minigpt4.models.mini_gpt4 import MiniGPT4
    print("  MiniGPT4 model import OK")
except Exception as e:
    print(f"  MiniGPT4 import FAIL: {e}")

# Check if LLaMA/Vicuna weights are available
llama_paths = [
    "/root/autodl-tmp/BRA_Project/models/MiniGPT-4-LLaMA-7B",
    "/root/autodl-tmp/BRA_Project/models/vicuna-7b",
    "/root/autodl-tmp/BRA_Project/models/llava-1.5-7b-hf",
]
print("\nChecking LLaMA/Vicuna paths:")
for p in llama_paths:
    exists = os.path.isdir(p)
    if exists:
        files = os.listdir(p)
        print(f"  {p}: EXISTS ({len(files)} files)")
    else:
        print(f"  {p}: NOT FOUND")

# Check checkpoint
ckpt = "/root/autodl-tmp/BRA_Project/checkpoints/pretrained_minigpt4.pth"
if os.path.exists(ckpt):
    size = os.path.getsize(ckpt) / 1e6
    print(f"\nCheckpoint: {ckpt} ({size:.1f} MB)")
    data = torch.load(ckpt, map_location="cpu")
    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    if isinstance(data, dict) and "model" in data:
        model_keys = list(data["model"].keys())
        print(f"  Model params: {len(model_keys)}")
        print(f"  First 5: {model_keys[:5]}")
else:
    print(f"\nCheckpoint NOT FOUND: {ckpt}")

print("\nMiniGPT-4 setup check complete")
