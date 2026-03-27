import json
import argparse
import time
import math
import torch
import gc
import sys
import os
sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from eval_utils import load_qwen2vl

def get_niah_prompt(processor, image_path, context_length=8000):
    # Needle in the middle
    needle = "The secret code for the vault is 42."
    # Filler text
    filler = "The quick brown fox jumps over the lazy dog. "
    
    # Estimate tokens: 1 filler sentence is about 10 tokens.
    num_fillers = context_length // 10
    half_fillers = num_fillers // 2
    
    text = "Please read the following text carefully and answer the question at the end.\n\n"
    text += filler * half_fillers
    text += needle + " "
    text += filler * half_fillers
    text += "\nQuestion: What is the secret code for the vault?"
    
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

def run_niah_autoregressive(model, processor, inputs, aosp_handle=None, start_idx=None):
    """
    Prefills up to start_idx, then auto-regressively teacher-forces the rest of the sequence
    token by token, computing the PPL curve.
    """
    input_ids = inputs["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]
    
    # We will prefill the image and the first few text tokens.
    # The image tokens are at the beginning.
    if start_idx is None:
        # Find the end of image tokens. For Qwen2-VL, vision tokens are replaced in the sequence.
        # We can just prefill the first 1000 tokens (which covers the image) to be safe.
        start_idx = min(1000, seq_len // 4)
        
    prefill_input_ids = input_ids[:, :start_idx]
    
    kwargs = {
        "input_ids": prefill_input_ids,
        "use_cache": True,
    }
    if "pixel_values" in inputs:
        kwargs["pixel_values"] = inputs["pixel_values"].to(model.device)
    if "image_grid_thw" in inputs:
        kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(model.device)
        
    with torch.no_grad():
        outputs = model(**kwargs)
        
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :] # [batch, vocab]
    
    ppl_curve = []
    
    # Auto-regressive teacher forcing
    for i in tqdm(range(start_idx, seq_len), desc="Teacher Forcing"):
        target_id = input_ids[:, i]
        
        # Calculate loss (negative log likelihood)
        loss = torch.nn.functional.cross_entropy(next_logits, target_id)
        ppl_curve.append(loss.item())
        
        # Forward the next token
        with torch.no_grad():
            outputs = model(
                input_ids=target_id.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        
        # Memory management periodically
        if i % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    # Now, generate the answer
    answer_ids = []
    for _ in range(20):
        target_id = next_logits.argmax(dim=-1)
        answer_ids.append(target_id.item())
        with torch.no_grad():
            outputs = model(
                input_ids=target_id.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        if target_id.item() in [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]:
            break
            
    answer_text = processor.decode(answer_ids, skip_special_tokens=True)
    
    # Calculate sliding window PPL (e.g. window size 100)
    window_size = 100
    smoothed_ppl = []
    for i in range(len(ppl_curve) - window_size + 1):
        window_loss = sum(ppl_curve[i:i+window_size]) / window_size
        smoothed_ppl.append(math.exp(window_loss))
        
    final_ppl = math.exp(sum(ppl_curve[-window_size:]) / window_size) if len(ppl_curve) >= window_size else math.exp(sum(ppl_curve)/len(ppl_curve))
        
    return smoothed_ppl, final_ppl, answer_text, len(ppl_curve)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/V_text_only.pt")
    args = parser.parse_args()
    
    # 1. Load model
    model, processor = load_qwen2vl(args.model_path)
    
    # 2. Get 5 COCO images
    coco_dir = Path("/root/autodl-tmp/A-OSP_Project/data/coco_val2014")
    image_paths = list(coco_dir.glob("*.jpg"))[:5]
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing Sample {i+1}/5 ---")
        inputs = get_niah_prompt(processor, str(img_path), context_length=8000)
        print(f"Total prompt tokens: {inputs['input_ids'].shape[1]}")
        
        # Run BASE
        print("Running BASE...")
        base_curve, base_final_ppl, base_ans, num_steps = run_niah_autoregressive(model, processor, inputs)
        print(f"Base Final PPL: {base_final_ppl:.2f} | Answer: {base_ans}")
        
        # Run A-OSP
        print("Running A-OSP...")
        from aosp_hook import apply_aosp_hook
        handle = apply_aosp_hook(model, args.v_matrix)
        handle.reset()
        
        aosp_curve, aosp_final_ppl, aosp_ans, _ = run_niah_autoregressive(model, processor, inputs)
        print(f"A-OSP Final PPL: {aosp_final_ppl:.2f} | Answer: {aosp_ans}")
        print(f"A-OSP Interventions: {handle.intervention_count}")
        
        handle.remove()
        
        results.append({
            "image": str(img_path.name),
            "base_final_ppl": base_final_ppl,
            "aosp_final_ppl": aosp_final_ppl,
            "ppl_delta": aosp_final_ppl - base_final_ppl,
            "base_ans": base_ans,
            "aosp_ans": aosp_ans,
            "interventions": handle.intervention_count,
            "num_steps": num_steps,
            # store sampled curve points to save space (every 100th point)
            "base_curve_sampled": base_curve[::100],
            "aosp_curve_sampled": aosp_curve[::100]
        })
        
        gc.collect()
        torch.cuda.empty_cache()
        
    with open("niah_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== FINAL SUMMARY ===")
    avg_delta = sum(r["ppl_delta"] for r in results) / len(results)
    print(f"Average PPL Delta at 8k tokens: {avg_delta:.4f}")
    print("Saved niah_results.json")

if __name__ == "__main__":
    main()
