import json
import argparse
import time
import torch
import gc
import sys
import os
import re
sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from eval_utils import load_qwen2vl

def get_niah_json_prompt(processor, image_path, context_length=8000):
    needle = "The secret code for the vault is 42."
    filler = "The quick brown fox jumps over the lazy dog. "
    num_fillers = context_length // 10
    half_fillers = num_fillers // 2
    
    text = "Please read the following text carefully and answer the question at the end.\n\n"
    text += filler * half_fillers
    text += needle + " "
    text += filler * half_fillers
    text += "\nQuestion: What is the secret code for the vault? You MUST output your answer as a valid JSON object in exactly this format: {\"secret_code\": \"<your_answer>\"}."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    return inputs

def extract_json(text):
    try:
        # Try parsing the whole text
        return json.loads(text)
    except:
        # Try to find a JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/V_text_only.pt")
    args = parser.parse_args()
    
    model, processor = load_qwen2vl(args.model_path)
    
    coco_dir = Path("/root/autodl-tmp/A-OSP_Project/data/coco_val2014")
    image_paths = list(coco_dir.glob("*.jpg"))[:5]
    
    results = []
    base_success = 0
    aosp_success = 0
    
    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing Sample {i+1}/5 ---")
        inputs = get_niah_json_prompt(processor, str(img_path), context_length=8000)
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run BASE
        print("Running BASE...")
        with torch.no_grad():
            base_out_ids = model.generate(**inputs, max_new_tokens=50)
        base_ans = processor.decode(base_out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Base Answer: {base_ans}")
        base_json = extract_json(base_ans)
        if base_json and "secret_code" in base_json:
            base_success += 1
            
        # Run A-OSP
        print("Running A-OSP...")
        from aosp_hook import apply_aosp_hook
        handle = apply_aosp_hook(model, args.v_matrix)
        handle.reset()
        
        with torch.no_grad():
            aosp_out_ids = model.generate(**inputs, max_new_tokens=50)
        aosp_ans = processor.decode(aosp_out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"A-OSP Answer: {aosp_ans}")
        aosp_json = extract_json(aosp_ans)
        if aosp_json and "secret_code" in aosp_json:
            aosp_success += 1
            
        print(f"A-OSP Interventions: {handle.intervention_count}")
        handle.remove()
        
        results.append({
            "image": str(img_path.name),
            "base_ans": base_ans,
            "base_valid": bool(base_json and "secret_code" in base_json),
            "aosp_ans": aosp_ans,
            "aosp_valid": bool(aosp_json and "secret_code" in aosp_json),
            "interventions": handle.intervention_count
        })
        
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\n=== FINAL SUMMARY ===")
    print(f"Base JSON Parse Success Rate: {base_success}/5 ({(base_success/5)*100}%)")
    print(f"A-OSP JSON Parse Success Rate: {aosp_success}/5 ({(aosp_success/5)*100}%)")
    
    with open("/root/autodl-tmp/A-OSP_Project/logs/eval_results/niah_json_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved logs/eval_results/niah_json_results.json")

if __name__ == "__main__":
    main()
