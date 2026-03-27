import json
import argparse
import time
import torch
import gc
import sys
import os
sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qwen_vl_utils import process_vision_info
from eval_utils import load_qwen2vl

def run_majority_voting(model, processor, inputs, n_votes=5):
    """
    Runs Majority Voting TTC Baseline with Temperature=0.7, Top_p=0.9
    Avoids the 'Strawman Fallacy' of T=1.0.
    """
    torch.cuda.reset_peak_memory_stats()
    
    responses = []
    for i in range(n_votes):
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        ans = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(ans)
        
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    return responses, peak_vram

def run_self_correction(model, processor, image_path, original_prompt):
    """
    Runs Self-Correction TTC Baseline.
    Generates an initial answer, then prompts the model to correct it.
    """
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Base generation
    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": original_prompt},
            ],
        }
    ]
    text_prompt_1 = processor.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    image_inputs_1, video_inputs_1 = process_vision_info(messages_1)
    
    inputs_1 = processor(
        text=[text_prompt_1], images=image_inputs_1, videos=video_inputs_1, padding=True, return_tensors="pt"
    )
    inputs_1 = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs_1.items()}
    
    with torch.no_grad():
        out_ids_1 = model.generate(**inputs_1, max_new_tokens=100, do_sample=False)
    base_ans = processor.decode(out_ids_1[0][inputs_1["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # 2. Correction step
    messages_2 = messages_1 + [
        {"role": "assistant", "content": [{"type": "text", "text": base_ans}]},
        {"role": "user", "content": [{"type": "text", "text": "Review your previous answer and correct any hallucinations or logical errors. Provide the final corrected answer."}]}
    ]
    text_prompt_2 = processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    image_inputs_2, video_inputs_2 = process_vision_info(messages_2)
    
    inputs_2 = processor(
        text=[text_prompt_2], images=image_inputs_2, videos=video_inputs_2, padding=True, return_tensors="pt"
    )
    inputs_2 = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs_2.items()}
    
    with torch.no_grad():
        out_ids_2 = model.generate(**inputs_2, max_new_tokens=100, do_sample=False)
    corrected_ans = processor.decode(out_ids_2[0][inputs_2["input_ids"].shape[1]:], skip_special_tokens=True)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    return base_ans, corrected_ans, peak_vram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image_path", type=str, default="/root/autodl-tmp/A-OSP_Project/data/coco_val2014/COCO_val2014_000000000042.jpg")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        # fallback to any coco image
        from pathlib import Path
        coco_dir = Path("/root/autodl-tmp/A-OSP_Project/data/coco_val2014")
        args.image_path = str(list(coco_dir.glob("*.jpg"))[0])
        
    model, processor = load_qwen2vl(args.model_path)
    
    print("\n--- Testing Majority Voting (T=0.7, Top_p=0.9) ---")
    # Prepare inputs
    messages = [
        {"role": "user", "content": [{"type": "image", "image": args.image_path}, {"type": "text", "text": args.prompt}]}
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    votes, peak_vram_mv = run_majority_voting(model, processor, inputs, n_votes=3)
    for i, v in enumerate(votes):
        print(f"Vote {i+1}: {v}")
    print(f"Peak VRAM (Majority Voting): {peak_vram_mv:.2f} GB")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n--- Testing Self-Correction ---")
    base_ans, corrected_ans, peak_vram_sc = run_self_correction(model, processor, args.image_path, args.prompt)
    print(f"Base Answer: {base_ans}")
    print(f"Corrected Answer: {corrected_ans}")
    print(f"Peak VRAM (Self-Correction): {peak_vram_sc:.2f} GB")
    
    print("\n[SUCCESS] Iso-Compute TTC Pipeline Prepared.")

if __name__ == "__main__":
    main()
