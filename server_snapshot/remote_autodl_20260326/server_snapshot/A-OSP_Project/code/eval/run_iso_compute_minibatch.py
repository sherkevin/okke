import json
import argparse
import time
import torch
import gc
import sys
import os
import random
from collections import Counter
sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from eval_utils import load_qwen2vl, compute_pope_metrics
from aosp_hook import apply_aosp_hook

def extract_answer(text):
    text = text.lower()
    if "yes" in text: return "yes"
    if "no" in text: return "no"
    return "yes"

def prepare_inputs(processor, image_path, question):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs, messages

def run_base(model, inputs):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return gen_ids, latency, peak_vram

def run_majority_voting(model, inputs, n_votes=3):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    responses = []
    all_gen_ids = []
    for _ in range(n_votes):
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9)
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        all_gen_ids.append(gen_ids)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    # return the longest sequence for length counting (conservative), but for answer we'll decode later
    return all_gen_ids, latency, peak_vram

def run_self_correction(model, processor, image_path, question, messages_1, inputs_1):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out_ids_1 = model.generate(**inputs_1, max_new_tokens=20, do_sample=False)
    base_ans = processor.decode(out_ids_1[0][inputs_1["input_ids"].shape[1]:], skip_special_tokens=True)
    
    messages_2 = messages_1 + [
        {"role": "assistant", "content": [{"type": "text", "text": base_ans}]},
        {"role": "user", "content": [{"type": "text", "text": "Review your previous answer and correct any hallucinations or logical errors. Provide the final corrected answer as Yes or No."}]}
    ]
    text_prompt_2 = processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    image_inputs_2, video_inputs_2 = process_vision_info(messages_2)
    inputs_2 = processor(text=[text_prompt_2], images=image_inputs_2, videos=video_inputs_2, padding=True, return_tensors="pt")
    inputs_2 = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs_2.items()}
    
    with torch.no_grad():
        out_ids_2 = model.generate(**inputs_2, max_new_tokens=20, do_sample=False)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    gen_ids = out_ids_2[0][inputs_2["input_ids"].shape[1]:]
    return gen_ids, latency, peak_vram

def run_vcd(model, processor, inputs_base, image_path, question):
    """
    Mock/Simplified VCD using a distorted image logic via standard generate to simulate latency and memory footprint.
    In a real implementation, we contrast logits. Here we run generate on distorted and use it as a proxy for compute cost.
    We will just output the base answer for metrics since it's a stand-in for VCD latency/memory.
    """
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # Dummy VCD: generate twice and take one. Simulates 2x compute.
    with torch.no_grad():
        out_ids_1 = model.generate(**inputs_base, max_new_tokens=20, do_sample=False)
        out_ids_2 = model.generate(**inputs_base, max_new_tokens=20, do_sample=False)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    gen_ids = out_ids_1[0][inputs_base["input_ids"].shape[1]:]
    return gen_ids, latency, peak_vram

def run_aosp(model, inputs, v_matrix_path):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    handle = apply_aosp_hook(model, v_matrix_path)
    handle.reset()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    handle.remove()
    return gen_ids, latency, peak_vram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/qwen3vl/V_text_only.pt")
    args = parser.parse_args()
    
    model, processor = load_qwen2vl(args.model_path)
    
    data_path = "/root/autodl-tmp/A-OSP_Project/data/pope/pope_coco_popular.jsonl"
    with open(data_path, "r") as f:
        lines = f.readlines()[:10]
        
    results = {"Base": [], "MajorityVoting": [], "SelfCorrection": [], "VCD": [], "A-OSP": []}
    
    image_dir = "/root/autodl-tmp/A-OSP_Project/data/coco_val2014"
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        img_path = os.path.join(image_dir, data["image"] + ".jpg")
        question = data["question"]
        gt = data["ground_truth"]
        
        print(f"\n--- Sample {i+1}/10: {data['image']} ---")
        inputs, messages = prepare_inputs(processor, img_path, question)
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Base
        gen_ids, lat, vram = run_base(model, inputs)
        ans = processor.decode(gen_ids, skip_special_tokens=True)
        results["Base"].append({"pred": extract_answer(ans), "gt": gt, "lat": lat, "vram": vram, "agl": len(ans.split())})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # MV
        all_gen_ids, lat, vram = run_majority_voting(model, inputs, n_votes=3)
        answers = [extract_answer(processor.decode(ids, skip_special_tokens=True)) for ids in all_gen_ids]
        final_ans = Counter(answers).most_common(1)[0][0]
        avg_len = sum(len(processor.decode(ids, skip_special_tokens=True).split()) for ids in all_gen_ids) / 3
        results["MajorityVoting"].append({"pred": final_ans, "gt": gt, "lat": lat, "vram": vram, "agl": avg_len})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # SC
        gen_ids, lat, vram = run_self_correction(model, processor, img_path, question, messages, inputs)
        ans = processor.decode(gen_ids, skip_special_tokens=True)
        results["SelfCorrection"].append({"pred": extract_answer(ans), "gt": gt, "lat": lat, "vram": vram, "agl": len(ans.split())})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # VCD
        gen_ids, lat, vram = run_vcd(model, processor, inputs, img_path, question)
        ans = processor.decode(gen_ids, skip_special_tokens=True)
        results["VCD"].append({"pred": extract_answer(ans), "gt": gt, "lat": lat, "vram": vram, "agl": len(ans.split())})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # A-OSP
        gen_ids, lat, vram = run_aosp(model, inputs, args.v_matrix)
        ans = processor.decode(gen_ids, skip_special_tokens=True)
        results["A-OSP"].append({"pred": extract_answer(ans), "gt": gt, "lat": lat, "vram": vram, "agl": len(ans.split())})
        
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\n=== FINAL TTC METRICS ===")
    summary = {}
    for cfg, data_list in results.items():
        correct = sum(1 for d in data_list if d["pred"] == d["gt"])
        acc = correct / len(data_list)
        avg_lat = sum(d["lat"] for d in data_list) / len(data_list)
        avg_vram = sum(d["vram"] for d in data_list) / len(data_list)
        avg_agl = sum(d["agl"] for d in data_list) / len(data_list)
        summary[cfg] = {
            "Accuracy": acc,
            "Latency_ms": avg_lat,
            "Peak_VRAM_GB": avg_vram,
            "AGL": avg_agl
        }
        print(f"{cfg}: Acc={acc:.2f}, Lat={avg_lat:.0f}ms, VRAM={avg_vram:.2f}GB, AGL={avg_agl:.1f}")
        
    with open("/root/autodl-tmp/A-OSP_Project/logs/eval_results/iso_compute_minibatch.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved logs/eval_results/iso_compute_minibatch.json")

if __name__ == "__main__":
    main()
