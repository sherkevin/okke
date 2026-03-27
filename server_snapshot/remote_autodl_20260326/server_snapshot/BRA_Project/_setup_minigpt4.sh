#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh

echo "=== Setting up MiniGPT-4 environment ==="

# Create separate env to avoid transformers version conflict
if conda env list | grep -q minigpt4; then
    echo "minigpt4 env already exists, activating..."
else
    echo "Creating minigpt4 conda env..."
    conda create -n minigpt4 python=3.10 -y
fi

conda activate minigpt4

# Install dependencies
echo "Installing dependencies..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.35.2
pip install salesforce-lavis
pip install accelerate bitsandbytes

# Configure MiniGPT-4
cd /root/autodl-tmp/BRA_Project/MiniGPT-4

# Update the model config to point to the local LLaMA path
echo "Configuring model paths..."
cat > /root/autodl-tmp/BRA_Project/MiniGPT-4/eval_configs/minigpt4_local.yaml << 'YAML'
model:
  arch: minigpt4
  model_type: pretrain_vicuna0
  max_txt_len: 160
  end_sym: "###"
  low_resource: True
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: '/root/autodl-tmp/BRA_Project/checkpoints/pretrained_minigpt4_7b.pth'

datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"

run:
  task: image_text_pretrain
YAML

# Update the llama_model path in the model config
sed -i 's|"please set this value to the path of vicuna model"|"/root/autodl-tmp/BRA_Project/models/MiniGPT-4-LLaMA-7B"|g' \
    minigpt4/configs/models/minigpt4_vicuna0.yaml

echo "Updated model config:"
cat minigpt4/configs/models/minigpt4_vicuna0.yaml

# Download pretrained checkpoint (projection layer, ~25MB)
echo ""
echo "=== Downloading pretrained checkpoint ==="
mkdir -p /root/autodl-tmp/BRA_Project/checkpoints

CKPT_PATH="/root/autodl-tmp/BRA_Project/checkpoints/pretrained_minigpt4_7b.pth"
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already exists"
else
    # Try multiple download sources
    echo "Trying HuggingFace mirror..."
    wget -q "https://hf-mirror.com/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/pretrained_minigpt4_7b.pth" \
        -O "$CKPT_PATH" 2>/dev/null && echo "Downloaded from HF mirror" || {
        echo "HF mirror failed, trying direct..."
        pip install gdown 2>/dev/null
        gdown "1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R" -O "$CKPT_PATH" 2>/dev/null && echo "Downloaded via gdown" || {
            echo "FAILED to download checkpoint. Manual download needed."
        }
    }
fi

if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint size: $(du -h $CKPT_PATH | cut -f1)"
fi

# Test import
echo ""
echo "=== Testing MiniGPT-4 import ==="
cd /root/autodl-tmp/BRA_Project/MiniGPT-4
python3 -c "
import sys
sys.path.insert(0, '.')
from minigpt4.common.config import Config
print('Config import OK')
from minigpt4.models import MiniGPT4
print('MiniGPT4 model import OK')
" 2>&1 || echo "Import test had issues (may need further debugging)"

echo ""
echo "=== MiniGPT-4 setup DONE ==="
