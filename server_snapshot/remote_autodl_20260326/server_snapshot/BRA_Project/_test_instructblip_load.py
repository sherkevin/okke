"""Quick smoke test: load InstructBLIP-Vicuna-7B in bfloat16 and run one inference."""
import torch
import os

MODEL = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"

print("=== InstructBLIP Load Test ===")
print(f"Model path: {MODEL}")

# Check all shards present
shards = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
for s in shards:
    p = os.path.join(MODEL, s)
    sz = os.path.getsize(p) / 1e9 if os.path.exists(p) else 0
    status = "OK" if sz > 1.0 else "MISSING"
    print(f"  [{status}] {s}: {sz:.2f} GB")

print("\nLoading processor...")
from transformers import InstructBlipProcessor
proc = InstructBlipProcessor.from_pretrained(MODEL)
print(f"  OK: processor loaded")

print("\nLoading model (bfloat16)...")
from transformers import InstructBlipForConditionalGeneration
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
)
model = model.cuda().eval()

vram = torch.cuda.max_memory_allocated() / 1e9
print(f"  OK: model loaded, VRAM used: {vram:.2f} GB")

# Simple inference with a dummy image
print("\nRunning dummy inference...")
from PIL import Image
import requests
from io import BytesIO

# Create a simple solid-color test image
img = Image.new("RGB", (224, 224), color=(128, 128, 128))
prompt = "What color is this image?"

inputs = proc(images=img, text=prompt, return_tensors="pt").to("cuda", torch.bfloat16)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

answer = proc.decode(output[0], skip_special_tokens=True).strip()
print(f"  Question: {prompt}")
print(f"  Answer  : {answer}")

peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\n  Peak VRAM: {peak_vram:.2f} GB")
print("\n=== LOAD TEST PASSED ===")
