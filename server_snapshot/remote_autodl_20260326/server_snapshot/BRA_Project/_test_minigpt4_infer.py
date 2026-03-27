"""Test MiniGPT-4 model loading and basic inference."""
import sys
import os
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/BRA_Project/MiniGPT-4")
os.chdir("/root/autodl-tmp/BRA_Project/MiniGPT-4")
os.environ["WANDB_MODE"] = "disabled"

from omegaconf import OmegaConf
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# Build config from YAML
cfg_path = "/root/autodl-tmp/BRA_Project/MiniGPT-4/eval_configs/minigpt4_local.yaml"
cfg = OmegaConf.load(cfg_path)

model_config = cfg.model
model_cls = registry.get_model_class(model_config.arch)
print(f"Model class: {model_cls}")
print(f"Config: {OmegaConf.to_yaml(model_config)}")

# Update llama_model path
model_config.llama_model = "/root/autodl-tmp/BRA_Project/models/MiniGPT-4-LLaMA-7B"

print("Loading model (low_resource=True -> 8bit)...")
try:
    model = model_cls.from_config(model_config)
    model = model.eval()
    print(f"Model loaded! Type: {type(model)}")

    # Check model structure
    if hasattr(model, "llama_model"):
        print(f"  LLaMA model: {type(model.llama_model)}")
    if hasattr(model, "Qformer"):
        print(f"  Q-Former: {type(model.Qformer)}")
    if hasattr(model, "llama_proj"):
        print(f"  Projection layer: {type(model.llama_proj)}")

    # Try inference
    image_path = sorted(
        (p for p in
         ("/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014").split()),
        key=str)
    coco_dir = "/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014"
    imgs = sorted(os.listdir(coco_dir))[:1]
    if imgs:
        img = Image.open(os.path.join(coco_dir, imgs[0])).convert("RGB")
        print(f"\nTest image: {imgs[0]}")

        from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
        vis_processor = Blip2ImageEvalProcessor(image_size=224)
        image_tensor = vis_processor(img).unsqueeze(0).to(model.device)

        print("Running inference...")
        with torch.no_grad():
            output = model.generate({"image": image_tensor, "prompt": "Describe this image."})
        print(f"Output: {output}")
    else:
        print("No test images found")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nMiniGPT-4 test complete")
