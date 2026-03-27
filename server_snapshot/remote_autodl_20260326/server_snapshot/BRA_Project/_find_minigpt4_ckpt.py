import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import list_repo_files, hf_hub_download

# Check official repo
try:
    files = list_repo_files("Vision-CAIR/MiniGPT-4")
    print("Vision-CAIR/MiniGPT-4 files:", files)
except Exception as e:
    print(f"Vision-CAIR/MiniGPT-4 error: {e}")

# Try downloading from official
for fname in ["pretrained_minigpt4_7b.pth", "pretrained_minigpt4.pth"]:
    try:
        f = hf_hub_download(
            "Vision-CAIR/MiniGPT-4", fname,
            local_dir="/root/autodl-tmp/BRA_Project/checkpoints")
        print(f"Downloaded: {f}, size: {os.path.getsize(f)/1e6:.1f}MB")
        break
    except Exception as e:
        print(f"{fname}: {e}")
