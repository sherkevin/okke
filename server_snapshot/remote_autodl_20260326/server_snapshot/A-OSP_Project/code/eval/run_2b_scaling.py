#!/usr/bin/env python3
"""
Scaling Matrix — Qwen3-VL-2B (Task: Sec 4.3, Table 1)
=======================================================
Fills the 2B column in Table 1 and Scaling Figure X.

Pipeline (single model load):
  1. Extract S_text_only_2b inline (100 text-only prompts, EVR check)
  2. MMHal-Bench (48 samples with images): Base vs A-OSP
  3. POPE adversarial (100 samples): Base vs A-OSP

Output: logs/eval_results/scaling_qwen3vl_2b.json
        Supports Table 1 (Scaling Matrix) and Figure X (Scaling plot).
"""

import os, sys, gc, json, time, re, torch, argparse
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # handle truncated/corrupted images gracefully

# ===========================================================================
# CONSTANTS
# ===========================================================================
PROJECT_ROOT = "/root/autodl-tmp/A-OSP_Project"
MODEL_PATH   = f"{PROJECT_ROOT}/models/Qwen3-VL-2B-Instruct"
LOG_DIR      = f"{PROJECT_ROOT}/logs/eval_results"
MODEL_DIR    = f"{PROJECT_ROOT}/models/qwen3vl"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

V_TEXT_ONLY_2B = f"{MODEL_DIR}/V_text_only_2b.pt"
OUTPUT_PATH    = f"{LOG_DIR}/scaling_qwen3vl_2b.json"

# 2B-specific hook layer: ~70% depth (layer 20 of 28 ≈ equivalent to 8B layer 29/36)
HOOK_LAYER_2B = 20
HIDDEN_DIM_2B = 2048
K_SUBSPACE    = 20     # number of principal directions

MMHAL_DIR  = f"{PROJECT_ROOT}/data/mmhal_bench"
POPE_PATH  = f"{PROJECT_ROOT}/data/pope/pope_coco_adversarial.jsonl"
COCO_DIR   = f"{PROJECT_ROOT}/data/coco_val2014"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip S_text_only extraction (use existing V_text_only_2b.pt)")
    p.add_argument("--n_extract",  type=int, default=100,
                   help="Number of prompts for S_text_only extraction")
    p.add_argument("--n_mmhal",    type=int, default=48,
                   help="MMHal-Bench samples (max 48 with images)")
    p.add_argument("--n_pope",     type=int, default=100,
                   help="POPE adversarial samples")
    p.add_argument("--alpha",      type=float, default=1.0)
    return p.parse_args()


# ===========================================================================
# MODEL LOADING
# ===========================================================================
def load_model():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print("[Model] Loading Qwen3-VL-2B-Instruct...")
    # Force full model onto GPU-0 (2B BF16 ≈ 5.4 GB; verified 14+ GB free alongside 8B run)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[Model] Loaded on cuda:0. GPU alloc: {used:.2f} GB")
    return model, processor


# ===========================================================================
# S_TEXT_ONLY EXTRACTION FOR 2B
# ===========================================================================
TEXT_ONLY_PROMPTS_2B = [
    # Visual hallucination anchors — prime language prior
    "The main object visible in the image is",
    "In this picture, I can clearly see",
    "The image shows a",
    "Looking at this image, the object is",
    "The item depicted here is",
    "This photograph contains",
    "The scene in the image features",
    "What is shown in this image is",
    "The picture displays",
    "In the foreground of the image is",
    "The central object in this image is",
    "The background of the image contains",
    "The color of the object in the image is",
    "The shape of the object visible is",
    "The animal in the image is",
    "The vehicle shown in the image is",
    "The food item in the image is",
    "The person in the image is",
    "The building visible in the image is",
    "The landscape in the image shows",
    # Temporal/sequence priors
    "After looking at the image, the action being performed is",
    "The activity shown in the image is",
    "The person in the image is currently",
    "The scene appears to depict",
    "Based on the image, the event is",
    # Object attribute priors
    "The material of the object in the image is",
    "The texture visible in the image is",
    "The size of the object in the image appears to be",
    "The position of the object in the image is",
    "The number of objects visible in the image is",
    # Common hallucination triggers
    "Yes, there is a cat in the image.",
    "Yes, there is a dog in the image.",
    "Yes, there is a car in the image.",
    "Yes, there is a person in the image.",
    "Yes, there is a bicycle in the image.",
    "No, there is no cat in the image.",
    "No, there is no dog in the image.",
    "No, there is no car in the image.",
    "The answer to the question about the image is",
    "Based on visual inspection, the correct answer is",
    # Factual anchoring
    "The famous landmark in the image is",
    "The celebrity visible in the image is",
    "The brand shown in the image is",
    "The location depicted in this image is",
    "The historical period shown in this image is",
    # Statistical co-occurrence
    "When I see a kitchen, I expect to also see",
    "When I see a beach, I expect to also see",
    "When I see a forest, I expect to also see",
    "When I see a city, I expect to also see",
    "When I see a sports field, I expect to also see",
    # Pattern completion
    "The image clearly shows that the answer is",
    "Based on common patterns, the object must be",
    "Given the context, the correct identification is",
    "The most likely object in this type of image is",
    "The typical response for this visual question is",
    # Multi-modal fusion errors
    "Combining the visual evidence and common knowledge, the answer is",
    "The visual features strongly suggest",
    "My visual assessment indicates",
    "The image content confirms that",
    "Visual reasoning supports the conclusion that",
    # Additional hallucination-prone contexts
    "The object's function in the image is",
    "The relationship between objects in the image is",
    "The emotion expressed in the image is",
    "The story depicted in the image is",
    "The message conveyed by the image is",
    # Answer-letter priors for MCQ
    "The answer is (A)",
    "The answer is (B)",
    "The answer is (C)",
    "The answer is (D)",
    "The correct option is A",
    "The correct option is B",
    "Looking at the options provided, option A",
    "Looking at the options provided, option B",
    "The most appropriate answer choice is",
    "Selecting from the given options, the answer is",
    # Medical/scientific hallucination priors
    "The medical condition visible in the image is",
    "The scientific phenomenon shown is",
    "The chemical structure visible is",
    "The biological specimen in the image is",
    "The geological formation shown is",
    # Counting/quantity priors
    "There are exactly two objects in the image",
    "There are exactly three objects in the image",
    "There is only one object visible in the image",
    "There are many objects visible in this image",
    "The count of items in the image is",
    # Confidence inflation
    "I am certain that the image shows",
    "Without any doubt, the image contains",
    "It is obvious from the image that",
    "Clearly, the main subject is",
    "The image unambiguously shows",
    # Text reading priors (to contrast with S_ocr)
    "The text in the image probably says",
    "Reading the text in the image:",
    "The words visible in the image include",
    "The number displayed in the image is",
    "The label shown in the image reads",
    # Spatial relationship priors
    "The object is located in the upper left corner",
    "The object is positioned in the center of the image",
    "To the left of the main object is",
    "Behind the main object in the image is",
    "The object appears in the background of the image",
    # Scale/comparison priors
    "Compared to a typical object, this one is",
    "The object in the image is approximately the size of",
    "Relative to other objects shown, this object is",
]


def extract_s_text_only(model, processor, n_prompts: int = 100) -> dict:
    """
    Extract S_text_only_2b via PCA on residual stream activations at Layer HOOK_LAYER_2B.
    Uses text-only prompts that maximally activate language-prior hallucination bias.
    Returns dict with V_bias, singular_values, evr, L_prior, etc.
    """
    from torch import nn

    print(f"\n[Extract S_text_only_2b] {n_prompts} prompts → Layer {HOOK_LAYER_2B}")

    prompts = TEXT_ONLY_PROMPTS_2B[:n_prompts]
    collected = []

    # Find decoder layers
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        layers = model.model.language_model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.layers
    else:
        layers = model.model.layers

    capture_buf = []

    def capture_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Take last token of text sequence (language prior probe)
        capture_buf.append(h[0, -1].detach().float().cpu())

    handle = layers[HOOK_LAYER_2B].register_forward_hook(capture_hook)

    for i, prompt_text in enumerate(prompts):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_prompts}] collecting activations...", flush=True)

        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt_text}]}]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat_text], return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            model(**inputs, output_hidden_states=False)

        del inputs
        gc.collect()

    handle.remove()

    # Build activation matrix [N, D]
    H = torch.stack(capture_buf)          # [N, D=2048]
    print(f"  Activation matrix: {H.shape}")

    # Mean-center
    H_mean = H.mean(dim=0)                # [D]
    H_c    = H - H_mean                   # [N, D]

    # SVD → principal directions
    U, S, Vt = torch.linalg.svd(H_c, full_matrices=False)  # Vt: [min(N,D), D]

    # EVR: cumulative energy of top-K
    energies  = (S ** 2)
    total_energy = energies.sum().item()
    evr_k = energies[:K_SUBSPACE].sum().item() / total_energy
    print(f"  EVR (top-{K_SUBSPACE}): {evr_k:.4f}")

    # L_prior: mean projection energy of activations onto V_bias
    V_bias = Vt[:K_SUBSPACE]              # [K, D]
    H_c_norm = H_c / (H_c.norm(dim=-1, keepdim=True) + 1e-8)
    projs = H_c_norm @ V_bias.T           # [N, K]
    L_prior = projs.norm(dim=-1).mean().item()
    print(f"  L_prior: {L_prior:.4f}")

    result = {
        "V_bias":          V_bias,
        "H_mean":          H_mean,
        "singular_values": S[:K_SUBSPACE].clone(),
        "evr":             float(evr_k),
        "L_prior":         float(L_prior),
        "K":               K_SUBSPACE,
        "num_samples":     n_prompts,
        "layer_idx":       HOOK_LAYER_2B,
        "model_id":        "Qwen3-VL-2B-Instruct",
        "tag":             "S_text_only_2b_zero_vision",
    }
    torch.save(result, V_TEXT_ONLY_2B)
    print(f"  Saved → {V_TEXT_ONLY_2B}")
    return result


# ===========================================================================
# A-OSP HOOK (2B)
# ===========================================================================
class AOSPHook2B:
    def __init__(self, V_bias, L_prior, K=20, alpha=1.0, mu=1.5, beta=0.9, layer_idx=20):
        self.V_bias  = V_bias.float()
        self.L_prior = L_prior
        self.K = K; self.alpha = alpha; self.mu = mu; self.beta = beta
        self.layer_idx = layer_idx
        self.reset()
        self.total_interventions = 0
        self.handle = None

    def reset(self):
        self.L_bar = self.L_prior
        self.t = 0

    def hook_fn(self, module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        rest   = out[1:] if isinstance(out, tuple) else None
        batch, seq_len, D = hidden.shape
        self.t += 1

        h = hidden[0].mean(dim=0) if seq_len > 1 else hidden[0, 0]
        V = self.V_bias.to(hidden.device, dtype=hidden.dtype)
        h_norm = h / (h.norm() + 1e-8)
        L_t = (h_norm @ V.T).norm().item()

        if L_t <= self.mu * self.L_bar:
            self.L_bar = self.beta * self.L_bar + (1 - self.beta) * L_t

        if L_t > self.mu * self.L_bar:
            self.total_interventions += 1
            H_flat  = hidden[0]
            proj_all = H_flat @ V.T
            H_proj   = H_flat - self.alpha * (proj_all @ V)
            scale    = H_flat.norm(dim=-1, keepdim=True) / (H_proj.norm(dim=-1, keepdim=True) + 1e-8)
            H_new    = hidden.clone(); H_new[0] = H_proj * scale
            return (H_new,) + rest if rest is not None else H_new
        return out

    def register(self, model):
        if (hasattr(model,"model") and hasattr(model.model,"language_model")
                and hasattr(model.model.language_model,"layers")):
            layers = model.model.language_model.layers
        elif hasattr(model,"model") and hasattr(model.model,"layers"):
            layers = model.model.layers
        else:
            layers = model.language_model.layers
        actual = self.layer_idx if self.layer_idx < len(layers) else len(layers)-4
        self.handle = layers[actual].register_forward_hook(self.hook_fn)
        print(f"[A-OSP-2B] Hook @ Layer {actual}/{len(layers)-1}, alpha={self.alpha}")

    def remove(self):
        if self.handle: self.handle.remove(); self.handle = None


# ===========================================================================
# INFERENCE HELPERS
# ===========================================================================
def run_vlm(model, processor, image_path, text_prompt,
            max_new_tokens=64) -> tuple[str, float]:
    """Single VLM inference with image. Returns ('', 0.0) on image-load error."""
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": 1280*720},
        {"type": "text",  "text": text_prompt},
    ]}]
    try:
        chat    = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        inp     = processor(text=[chat], images=imgs, return_tensors="pt", padding=True).to("cuda:0")
    except Exception as e:
        print(f"\n  [WARN] Image load failed ({os.path.basename(image_path)}): {e}")
        return "[IMAGE_ERROR]", 0.0

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    elapsed = time.time() - t0
    new_tok = out[0][inp.input_ids.shape[1]:]
    raw = processor.decode(new_tok, skip_special_tokens=True).strip()
    del inp; gc.collect(); torch.cuda.empty_cache()
    return raw, elapsed


# ===========================================================================
# MMHAL-BENCH EVALUATION
# ===========================================================================
HALLUCINATION_KEYWORDS = [
    r"\bno\b", r"\bnot\b", r"\bnone\b", r"\bwithout\b", r"\babsent\b",
    r"\bmissing\b", r"\bcannot see\b", r"\bcan'?t see\b", r"\bdo not see\b",
    r"\bdon'?t see\b", r"\bunable to\b", r"\bincorrect\b",
]

def score_mmhal_response(raw: str, gt: str) -> float:
    """
    Simplified MMHal-Bench scoring (0.0–1.0).
    Full eval requires GPT-4 judge; we use a lexical proxy:
      1.0  = response likely not hallucinating (key words from GT present)
      0.5  = ambiguous
      0.0  = response likely hallucinating (negation/absence language)
    """
    raw_l = raw.lower()
    gt_l  = gt.lower()

    # Check for hallucination-indicating negations
    for kw in HALLUCINATION_KEYWORDS:
        if re.search(kw, raw_l):
            # If GT itself mentions negation, this is correct
            if re.search(kw, gt_l):
                return 1.0
            return 0.0

    # Check keyword overlap with GT
    gt_words  = set(re.findall(r'\b\w{4,}\b', gt_l))
    raw_words = set(re.findall(r'\b\w{4,}\b', raw_l))
    if not gt_words:
        return 0.5
    overlap = len(gt_words & raw_words) / len(gt_words)
    return float(min(overlap, 1.0))


def eval_mmhal(model, processor, n_samples: int,
               hook: AOSPHook2B | None, mode_label: str) -> tuple[list, dict]:
    with open(f"{MMHAL_DIR}/mmhal_bench.jsonl") as f:
        samples = [json.loads(l) for l in f]

    img_dir = f"{MMHAL_DIR}/images/"
    valid   = [s for s in samples if os.path.exists(img_dir + s["image"])][:n_samples]
    N = len(valid)
    print(f"\n{'='*60}")
    print(f"  MMHal-Bench — {mode_label} | N={N}")
    print(f"{'='*60}")

    results = []
    total_score = 0.0

    for i, s in enumerate(valid):
        if hook: hook.reset()
        img_path = img_dir + s["image"]
        raw, elapsed = run_vlm(model, processor, img_path, s["question"],
                                max_new_tokens=128)
        score = score_mmhal_response(raw, s["gt_answer"])
        total_score += score
        interv = hook.total_interventions if hook else 0
        mark   = "✓" if score > 0.5 else ("~" if score == 0.5 else "✗")
        print(f"  [{i+1:2d}/{N}] {mark} score={score:.2f} interv={interv}"
              f"  Q: {s['question'][:45]}", flush=True)
        results.append({
            "id":           s["question_id"],
            "question":     s["question"],
            "gt_answer":    s["gt_answer"][:100],
            "raw_output":   raw[:200],
            "score":        score,
            "correct":      score > 0.5,
            "interventions": interv,
            "elapsed_s":    round(elapsed, 3),
        })

    avg_score = total_score / N if N > 0 else 0
    accuracy  = sum(1 for r in results if r["correct"]) / N if N > 0 else 0
    total_interv = sum(r["interventions"] for r in results)
    print(f"\n  Avg score: {avg_score:.3f}  Accuracy: {accuracy:.1%}  Interventions: {total_interv}")

    summary = {
        "dataset": "MMHal-Bench", "mode": mode_label, "n_samples": N,
        "avg_score": round(avg_score, 4), "accuracy": round(accuracy, 4),
        "total_interventions": total_interv,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results)/N, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return results, summary


# ===========================================================================
# POPE EVALUATION
# ===========================================================================
def eval_pope(model, processor, n_samples: int,
              hook: AOSPHook2B | None, mode_label: str) -> tuple[list, dict]:
    with open(POPE_PATH) as f:
        all_pope = [json.loads(l) for l in f]

    # Filter to available images
    valid = []
    for s in all_pope:
        img_path = f"{COCO_DIR}/{s['image']}.jpg"
        if os.path.exists(img_path):
            s["_img_path"] = img_path
            valid.append(s)
        if len(valid) >= n_samples:
            break

    N = len(valid)
    print(f"\n{'='*60}")
    print(f"  POPE adversarial — {mode_label} | N={N}")
    print(f"{'='*60}")

    yes_prompt = (
        f"{s['question']} "
        "Please answer with 'yes' or 'no' only."
    )

    results = []
    n_correct = 0

    for i, s in enumerate(valid):
        if hook: hook.reset()
        prompt = s["question"] + " Please answer with 'yes' or 'no' only."
        raw, elapsed = run_vlm(model, processor, s["_img_path"], prompt,
                                max_new_tokens=10)
        # Parse yes/no
        raw_l = raw.strip().lower()
        if re.search(r'\byes\b', raw_l):
            pred = "yes"
        elif re.search(r'\bno\b', raw_l):
            pred = "no"
        else:
            pred = "yes" if raw_l.startswith("y") else "no"

        gt      = s["ground_truth"].strip().lower()
        correct = (pred == gt)
        if correct: n_correct += 1
        interv  = hook.total_interventions if hook else 0
        mark    = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{N}] {mark} pred={pred:3s} gt={gt:3s} interv={interv}",
              flush=True)
        results.append({
            "id":           s["question_id"],
            "question":     s["question"],
            "gt":           gt,
            "pred":         pred,
            "correct":      correct,
            "interventions": interv,
            "elapsed_s":    round(elapsed, 3),
        })

    acc = n_correct / N if N > 0 else 0
    total_interv = sum(r["interventions"] for r in results)
    print(f"\n  Accuracy: {acc:.1%}  ({n_correct}/{N})  Interventions: {total_interv}")

    # Compute precision/recall/F1
    tp = sum(1 for r in results if r["pred"]=="yes" and r["gt"]=="yes")
    fp = sum(1 for r in results if r["pred"]=="yes" and r["gt"]=="no")
    fn = sum(1 for r in results if r["pred"]=="no"  and r["gt"]=="yes")
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    summary = {
        "dataset": "POPE-adversarial", "mode": mode_label, "n_samples": N,
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "total_interventions": total_interv,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results)/N, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return results, summary


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    args = parse_args()

    # ---- Load model (ONCE) ----
    model, processor = load_model()

    # ---- Extract or load S_text_only_2b ----
    if args.skip_extract and os.path.exists(V_TEXT_ONLY_2B):
        print(f"[Extract] Loading existing {V_TEXT_ONLY_2B}")
        v_data = torch.load(V_TEXT_ONLY_2B, map_location="cpu", weights_only=False)
        print(f"[Extract] EVR={v_data['evr']:.4f}, L_prior={v_data['L_prior']:.4f}")
    else:
        v_data = extract_s_text_only(model, processor, n_prompts=args.n_extract)

    V_bias  = v_data["V_bias"]
    L_prior = v_data["L_prior"]

    evr = v_data["evr"]
    if evr < 0.70:
        print(f"[WARN] EVR={evr:.4f} < 0.70 redline. Proceeding anyway for scaling baseline.")

    # ---- Run evaluations ----
    all_results   = {}
    all_summaries = {}

    for mode in ["base", "aosp"]:
        hook = None
        if mode == "aosp":
            hook = AOSPHook2B(V_bias=V_bias, L_prior=L_prior,
                              K=K_SUBSPACE, alpha=args.alpha,
                              mu=1.5, beta=0.9, layer_idx=HOOK_LAYER_2B)
            hook.register(model)

        label = "BASE" if mode == "base" else "A-OSP (S_text_only_2b)"

        # MMHal
        mmhal_results, mmhal_summary = eval_mmhal(
            model, processor, args.n_mmhal, hook, label
        )
        # POPE
        pope_results, pope_summary = eval_pope(
            model, processor, args.n_pope, hook, label
        )

        if hook: hook.remove()

        all_results[mode] = {
            "mmhal": mmhal_results,
            "pope":  pope_results,
        }
        all_summaries[mode] = {
            "mmhal": mmhal_summary,
            "pope":  pope_summary,
        }

    # ---- Comparison ----
    comparison = {}
    if "base" in all_summaries and "aosp" in all_summaries:
        b_mmhal = all_summaries["base"]["mmhal"]["avg_score"]
        a_mmhal = all_summaries["aosp"]["mmhal"]["avg_score"]
        b_pope  = all_summaries["base"]["pope"]["accuracy"]
        a_pope  = all_summaries["aosp"]["pope"]["accuracy"]

        comparison = {
            "mmhal_base_score":  b_mmhal,
            "mmhal_aosp_score":  a_mmhal,
            "mmhal_delta":       round(a_mmhal - b_mmhal, 4),
            "pope_base_acc":     b_pope,
            "pope_aosp_acc":     a_pope,
            "pope_delta":        round(a_pope - b_pope, 4),
            "aosp_interventions_mmhal": all_summaries["aosp"]["mmhal"]["total_interventions"],
            "aosp_interventions_pope":  all_summaries["aosp"]["pope"]["total_interventions"],
            "paper_note": "Fills 2B column in Table 1 (Scaling Matrix) and Scaling Figure X.",
        }
        print(f"\n{'='*60}")
        print(f"  SCALING MATRIX — 2B RESULTS")
        print(f"{'='*60}")
        print(f"  MMHal-Bench score:  Base={b_mmhal:.3f}  A-OSP={a_mmhal:.3f}  Δ={a_mmhal-b_mmhal:+.3f}")
        print(f"  POPE adversarial:   Base={b_pope:.1%}  A-OSP={a_pope:.1%}  Δ={a_pope-b_pope:+.1%}")

    # ---- Save ----
    output = {
        "task":         "Scaling Matrix — Qwen3-VL-2B",
        "paper_figure": "Table 1 (Scaling Matrix) / Scaling Figure X",
        "model":        "Qwen3-VL-2B-Instruct",
        "hook_layer":   HOOK_LAYER_2B,
        "v_tensor":     V_TEXT_ONLY_2B,
        "v_evr":        v_data["evr"],
        "v_L_prior":    v_data["L_prior"],
        "n_extract_prompts": args.n_extract,
        "results":      all_summaries,
        "comparison":   comparison,
        "per_sample":   all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Output → {OUTPUT_PATH}")
    print("SCALING EVAL COMPLETE")


if __name__ == "__main__":
    main()
