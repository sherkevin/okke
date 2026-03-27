import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

f = hf_hub_download(
    "wangrongsheng/MiniGPT-4-LLaMA-7B",
    "pretrained_minigpt4_7b.pth",
    local_dir="/root/autodl-tmp/BRA_Project/checkpoints",
    resume_download=True,
)
print(f"Downloaded: {f}")
print(f"Size: {os.path.getsize(f) / 1e6:.1f} MB")
