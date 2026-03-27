import torch
from transformers import AutoProcessor
from PIL import Image
import sys
from pathlib import Path

# Load model via our util
sys.path.insert(0, str(Path('/root/autodl-tmp/A-OSP_Project/code').resolve()))
from eval.eval_utils import load_qwen2vl

model, processor = load_qwen2vl('/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct')

img_clear = Image.open('/root/autodl-tmp/A-OSP_Project/data/coco_val2014/COCO_val2014_000000000164.jpg').convert('RGB')
img_blur = Image.open('/root/autodl-tmp/A-OSP_Project/data/blurred_calibration/blur/COCO_val2014_000000000164.jpg').convert('RGB')

def get_variance(img):
    prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True).to(model.device)
    
    variances = []
    
    def hook(m, i, o):
        hidden = o[0]
        # Just grab the variance of the first token sequence (the prefill)
        var = hidden[0].var(dim=0).mean().item()
        variances.append(var)
        
    h = model.model.layers[24].register_forward_hook(hook)
    
    with torch.inference_mode():
        model(**inputs)
        
    h.remove()
    return sum(variances)/len(variances) if variances else 0

print("Clear image variance:", get_variance(img_clear))
print("Blurred image variance:", get_variance(img_blur))

