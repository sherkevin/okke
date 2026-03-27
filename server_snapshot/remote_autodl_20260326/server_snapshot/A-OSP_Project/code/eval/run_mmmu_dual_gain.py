#!/usr/bin/env python3
"""
MMMU Hard Subset — Dual-Gain Evaluation (Task 3.5b)
=====================================================
V3.5 Sprint 4: Evaluate A-OSP + Self-Correction synergy on the MMMU hard subset.

Scientific goal (Dual-Gain Hypothesis):
  A-OSP suppresses language-prior hallucination in Pass 1 (removing anchoring
  bias). Self-Correction (Pass 2) then refines the already-debiased hypothesis.
  Combined A-OSP + SC should outperform either alone, proving the interventions
  act in complementary dimensions of the reasoning pipeline.

Three evaluation modes:
  base             : Single-pass MCQ inference (Qwen3-VL-8B)
  self_correction  : 2-pass: initial answer → review & correct prompt
  aosp_sc          : A-OSP active during both passes + Self-Correction

Hard subset definition (STEM-heavy subjects with high language-prior risk):
  Math, Physics, Chemistry, Computer_Science, Electronics, Energy_and_Power,
  Mechanical_Engineering, Materials, Architecture_and_Engineering, Biology,
  Basic_Medical_Science, Diagnostics_and_Laboratory_Medicine

Output: logs/eval_results/mmmu_hard_dual_gain.json
        Supports Table 3 / Figure 4 (Dual-Gain ablation) in the paper.

Usage:
  python run_mmmu_dual_gain.py --mode base --n_samples 30
  python run_mmmu_dual_gain.py --mode self_correction --n_samples 30
  python run_mmmu_dual_gain.py --mode aosp_sc --n_samples 30
  python run_mmmu_dual_gain.py --mode all --n_samples 30   # run all three sequentially
"""

import os, sys, gc, json, time, argparse, re, torch
import numpy as np

# ===========================================================================
# PATH CONSTANTS
# ===========================================================================
PROJECT_ROOT   = "/root/autodl-tmp/A-OSP_Project"
MODEL_PATH     = f"{PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
V_TEXT_ONLY    = f"{PROJECT_ROOT}/models/qwen3vl/V_text_only.pt"
V_MATRIX_Q3    = f"{PROJECT_ROOT}/models/V_matrix_q3.pt"
MANIFEST_PATH  = f"{PROJECT_ROOT}/data/benchmarks/mmmu/mmmu_manifest.jsonl"
LOG_DIR        = f"{PROJECT_ROOT}/logs/eval_results"
OUTPUT_PATH    = f"{LOG_DIR}/mmmu_hard_dual_gain.json"
os.makedirs(LOG_DIR, exist_ok=True)

HOOK_LAYER = 29

# STEM hard subjects (high language-prior risk → A-OSP most beneficial)
HARD_SUBJECTS = {
    "Math", "Physics", "Chemistry", "Computer_Science",
    "Electronics", "Energy_and_Power", "Mechanical_Engineering",
    "Materials", "Architecture_and_Engineering", "Biology",
    "Basic_Medical_Science", "Diagnostics_and_Laboratory_Medicine",
}


# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MMMU Hard Subset Dual-Gain evaluation (Task 3.5b)"
    )
    p.add_argument("--mode",
                   choices=["base", "self_correction", "aosp_sc", "all"],
                   default="all",
                   help="Evaluation mode (all = run base→sc→aosp_sc sequentially)")
    p.add_argument("--n_samples", type=int, default=30,
                   help="Number of hard-subset samples to evaluate")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="A-OSP projection strength")
    p.add_argument("--v_matrix", type=str, default=None,
                   help="Override V_bias .pt path")
    p.add_argument("--max_new_tokens_pass1", type=int, default=64,
                   help="Max tokens for pass-1 answer generation")
    p.add_argument("--max_new_tokens_pass2", type=int, default=96,
                   help="Max tokens for pass-2 correction generation")
    return p.parse_args()


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_hard_samples(n: int) -> list[dict]:
    """
    Load n MMMU hard-subset samples.
    Hard = STEM-heavy subjects (see HARD_SUBJECTS).
    Sampling strategy: interleave across subjects for diversity.
    """
    import re as re_

    with open(MANIFEST_PATH) as f:
        all_samples = [json.loads(l) for l in f]

    # Filter to hard subjects
    hard = []
    for s in all_samples:
        m = re_.match(r'(?:validation|test)_(.+)_\d+', s['id'])
        if m and m.group(1) in HARD_SUBJECTS:
            # Resolve image path
            img = s.get('image_path', '')
            abs_img = os.path.join(PROJECT_ROOT, img) if not os.path.isabs(img) else img
            if os.path.exists(abs_img):
                s['_abs_image'] = abs_img
                s['_subject'] = m.group(1)
                hard.append(s)

    # Interleave subjects for diversity (round-robin)
    from collections import defaultdict
    by_subject = defaultdict(list)
    for s in hard:
        by_subject[s['_subject']].append(s)

    interleaved = []
    while len(interleaved) < len(hard):
        added = False
        for subj in sorted(by_subject.keys()):
            if by_subject[subj]:
                interleaved.append(by_subject[subj].pop(0))
                added = True
        if not added:
            break

    selected = interleaved[:n]
    print(f"[Data] {len(hard)} hard samples available; selected {len(selected)}")

    subject_counts = {}
    for s in selected:
        subject_counts[s['_subject']] = subject_counts.get(s['_subject'], 0) + 1
    print(f"[Data] Subject distribution: { {k: v for k, v in sorted(subject_counts.items())} }")
    return selected


# ===========================================================================
# MODEL / V-MATRIX LOADING
# ===========================================================================
def load_model():
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    D = (getattr(cfg, "hidden_size", None)
         or getattr(getattr(cfg, "text_config", cfg), "hidden_size", 4096))
    n_layers = (getattr(cfg, "num_hidden_layers", None)
                or getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 36))
    print(f"[Model] Qwen3-VL-8B: hidden={D}, layers={n_layers}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor, D, n_layers


def load_v_matrix(override: str | None = None) -> dict:
    for p in [override, V_TEXT_ONLY, V_MATRIX_Q3]:
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            evr = d.get("evr", float("nan"))
            lp  = d.get("L_prior", float("nan"))
            tag = d.get("tag", os.path.basename(p))
            print(f"[V_matrix] {os.path.basename(p)}: tag={tag}, "
                  f"shape={list(d['V_bias'].shape)}, EVR={evr:.4f}, L_prior={lp:.4f}")
            if p == V_TEXT_ONLY and evr < 0.70:
                print(f"[V_matrix] EVR={evr:.4f} < 0.70 redline — trying fallback")
                continue
            return d
    raise FileNotFoundError(f"No valid V_bias matrix found. Checked: {V_TEXT_ONLY}, {V_MATRIX_Q3}")


# ===========================================================================
# A-OSP HOOK
# ===========================================================================
class MMMAOSPHook:
    """
    A-OSP hook for MMMU evaluation.
    Identical architecture to VWB and MVBench hooks.
    Placed at Layer 29; MeanPool(t,m) for spatial inputs.
    """

    def __init__(self, V_bias: torch.Tensor, L_prior: float,
                 K: int = 20, alpha: float = 1.0,
                 mu: float = 1.5, beta: float = 0.9,
                 layer_idx: int = 29):
        self.V_bias   = V_bias.float()
        self.L_prior  = L_prior
        self.K        = K
        self.alpha    = alpha
        self.mu       = mu
        self.beta     = beta
        self.layer_idx = layer_idx
        self.reset()
        self.total_interventions = 0
        self.handle = None

    def reset(self):
        self.L_bar = self.L_prior
        self.t     = 0

    def hook_fn(self, module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            rest   = out[1:]
        else:
            hidden = out
            rest   = None

        batch, seq_len, D = hidden.shape
        self.t += 1

        # MeanPool for prefill (image+text tokens); scalar for decode
        h = hidden[0].mean(dim=0) if seq_len > 1 else hidden[0, 0]

        V      = self.V_bias.to(hidden.device, dtype=hidden.dtype)
        h_norm = h / (h.norm() + 1e-8)
        proj   = h_norm @ V.T
        L_t    = proj.norm().item()

        if L_t <= self.mu * self.L_bar:
            self.L_bar = self.beta * self.L_bar + (1.0 - self.beta) * L_t

        if L_t > self.mu * self.L_bar:
            self.total_interventions += 1
            V_dev   = self.V_bias.to(hidden.device, dtype=hidden.dtype)
            H_flat  = hidden[0]
            proj_all = H_flat @ V_dev.T
            H_proj   = H_flat - self.alpha * (proj_all @ V_dev)
            orig_norm = H_flat.norm(dim=-1, keepdim=True)
            proj_norm = H_proj.norm(dim=-1, keepdim=True) + 1e-8
            H_corrected = (H_proj / proj_norm) * orig_norm
            hidden_new = hidden.clone()
            hidden_new[0] = H_corrected
            return (hidden_new,) + rest if isinstance(out, tuple) else hidden_new

        return out

    def register(self, model):
        if (hasattr(model, "model") and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "layers")):
            layers = model.model.language_model.layers
        elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            raise AttributeError(f"Cannot find decoder layers in {type(model).__name__}")

        total      = len(layers)
        actual_idx = self.layer_idx if self.layer_idx < total else total - 4
        self.handle = layers[actual_idx].register_forward_hook(self.hook_fn)
        print(f"[A-OSP] Hook @ Layer {actual_idx}/{total-1}, alpha={self.alpha}")

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===========================================================================
# PROMPT CONSTRUCTION
# ===========================================================================
def build_pass1_prompt(sample: dict) -> str:
    """
    Pass-1 prompt: standard MCQ with image.
    Replaces <image N> placeholders with a generic reference.
    """
    question = re.sub(r'<image\s*\d*>', '[image]', sample['question']).strip()
    options  = sample['options']
    opts_str = "\n".join(f"({ltr}) {opt}" for ltr, opt in zip("ABCD", options))
    return (
        f"{question}\n\n"
        f"Options:\n{opts_str}\n\n"
        "Answer with the option letter only (e.g. A or B)."
    )


def build_pass2_prompt(sample: dict, pass1_answer: str) -> str:
    """
    Pass-2 prompt: self-correction.
    Shows the model its initial answer, asks it to review.
    Designed to leverage the debiased representation from A-OSP pass 1.
    """
    question = re.sub(r'<image\s*\d*>', '[image]', sample['question']).strip()
    options  = sample['options']
    opts_str = "\n".join(f"({ltr}) {opt}" for ltr, opt in zip("ABCD", options))
    return (
        f"{question}\n\n"
        f"Options:\n{opts_str}\n\n"
        f"You initially answered: ({pass1_answer})\n\n"
        "Please review the image and question carefully. "
        "If your initial answer is correct, confirm it. "
        "If you see an error, provide the corrected answer. "
        "Respond with only the option letter (e.g. A or B)."
    )


# ===========================================================================
# ANSWER PARSER
# ===========================================================================
def parse_answer(raw: str, n_opts: int = 4) -> str:
    """Extract A/B/C/D from raw LLM output."""
    text = raw.strip()
    valid = "ABCD"[:n_opts]

    # 1. Parenthesized: (A)
    m = re.search(r'\(([A-Da-d])\)', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 2. Explicit answer marker
    m = re.search(r'(?:answer(?:\s+is)?|option)[:\s]+\(?([A-Da-d])\)?', text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3. Bare letter at start
    m = re.match(r'^([A-Da-d])[.):\s\n]', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 4. Any standalone valid letter
    m = re.search(r'\b([A-Da-d])\b', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 5. Single letter entire response
    if text.upper() in valid:
        return text.upper()

    return "?"


# ===========================================================================
# SINGLE INFERENCE CALL
# ===========================================================================
def run_inference(model, processor, image_path: str, prompt_text: str,
                  max_new_tokens: int = 64) -> tuple[str, float]:
    """
    Run one forward pass through Qwen3-VL with a single image.
    Returns (raw_output_text, elapsed_seconds).
    """
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path,
             "max_pixels": 1280 * 720},
            {"type": "text", "text": prompt_text},
        ],
    }]

    chat_text     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    elapsed = time.time() - t0

    n_in    = inputs["input_ids"].shape[1]
    new_tok = out_ids[0][n_in:]
    raw     = processor.decode(new_tok, skip_special_tokens=True).strip()

    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return raw, elapsed


# ===========================================================================
# EVALUATE ONE MODE
# ===========================================================================
def evaluate_mode(
    mode: str,
    samples: list[dict],
    model,
    processor,
    args,
) -> tuple[list[dict], dict]:
    """
    Evaluate all samples for a given mode.

    mode='base'            → single pass
    mode='self_correction' → pass1 → pass2 (hook OFF both passes)
    mode='aosp_sc'         → pass1 + pass2 with A-OSP hook active
    """
    use_aosp = (mode == "aosp_sc")
    two_pass = (mode in {"self_correction", "aosp_sc"})

    hook = None
    if use_aosp:
        v_data  = load_v_matrix(args.v_matrix)
        V_bias  = v_data["V_bias"]
        L_prior = v_data["L_prior"]
        K       = v_data.get("K", 20)
        tag     = v_data.get("tag", "unknown")
        print(f"[A-OSP] tag={tag}, EVR={v_data.get('evr',0):.4f}, L_prior={L_prior:.2f}")
        hook = MMMAOSPHook(
            V_bias=V_bias, L_prior=L_prior, K=K,
            alpha=args.alpha, mu=1.5, beta=0.9, layer_idx=HOOK_LAYER,
        )
        hook.register(model)

    mode_labels = {
        "base":            "BASE (single-pass)",
        "self_correction": "SELF-CORRECTION (2-pass, no A-OSP)",
        "aosp_sc":         "A-OSP + SELF-CORRECTION (Dual-Gain)",
    }
    N = len(samples)

    print(f"\n{'='*65}")
    print(f"  {mode_labels[mode]}")
    print(f"  N={N} | Layer={HOOK_LAYER} | alpha={args.alpha}")
    print(f"{'='*65}\n")

    results    = []
    n_correct  = 0
    n_changed  = 0   # pass2 changed answer from pass1
    n_correct_after_change = 0

    for i, sample in enumerate(samples):
        img_path = sample["_abs_image"]
        gt       = sample["answer"].strip().upper()
        subj     = sample.get("_subject", "?")

        prefix = f"  [{i+1:2d}/{N}] {sample['id'][:35]:35s} "
        print(prefix, end="", flush=True)

        # ---- PASS 1 ----
        if hook:
            hook.reset()
        p1_prompt  = build_pass1_prompt(sample)
        p1_raw, t1 = run_inference(model, processor, img_path, p1_prompt,
                                   args.max_new_tokens_pass1)
        p1_ans = parse_answer(p1_raw, len(sample["options"]))

        p1_interv = hook.total_interventions if hook else 0

        # ---- PASS 2 (if applicable) ----
        p2_raw, p2_ans, t2 = None, None, 0.0
        p2_interv = 0

        if two_pass:
            if hook:
                hook.reset()
            p2_prompt  = build_pass2_prompt(sample, p1_ans)
            p2_raw, t2 = run_inference(model, processor, img_path, p2_prompt,
                                       args.max_new_tokens_pass2)
            p2_ans   = parse_answer(p2_raw, len(sample["options"]))
            p2_interv = (hook.total_interventions - p1_interv) if hook else 0

        # ---- Final answer ----
        final_ans = p2_ans if two_pass else p1_ans
        correct   = (final_ans == gt)
        changed   = two_pass and (p2_ans != p1_ans) and p2_ans != "?"

        if correct:
            n_correct += 1
        if changed:
            n_changed += 1
            if correct:
                n_correct_after_change += 1

        # Print per-sample result
        mark = "✓" if correct else "✗"
        if two_pass:
            change_str = f" [{p1_ans}→{p2_ans}]" if changed else f" [{p1_ans}=]"
        else:
            change_str = ""
        interv_total = p1_interv + p2_interv
        interv_str = f" [{interv_total}✗]" if interv_total > 0 else ""
        print(f"{mark} pred={final_ans} gt={gt}{change_str}{interv_str} {subj[:20]}")

        results.append({
            "id":             sample["id"],
            "subject":        subj,
            "question":       sample["question"][:120],
            "options":        sample["options"],
            "gt_answer":      gt,
            "pass1_answer":   p1_ans,
            "pass1_raw":      p1_raw[:200],
            "pass1_interventions": p1_interv,
            "pass2_answer":   p2_ans,
            "pass2_raw":      p2_raw[:200] if p2_raw else None,
            "pass2_interventions": p2_interv,
            "final_answer":   final_ans,
            "correct":        correct,
            "answer_changed": changed,
            "elapsed_pass1":  round(t1, 3),
            "elapsed_pass2":  round(t2, 3),
        })

    if hook:
        hook.remove()

    acc = n_correct / N if N > 0 else float("nan")
    total_interv = sum(r["pass1_interventions"] + (r["pass2_interventions"] or 0)
                       for r in results)

    print(f"\n  {'─'*55}")
    print(f"  Accuracy:         {acc:.1%}  ({n_correct}/{N})")
    if two_pass:
        print(f"  Answers changed:  {n_changed}  (correct after change: {n_correct_after_change})")
    if use_aosp:
        print(f"  Total interventions: {total_interv}")

    summary = {
        "mode":                       mode,
        "mode_label":                 mode_labels[mode],
        "n_samples":                  N,
        "n_correct":                  n_correct,
        "accuracy":                   round(acc, 4),
        "n_answers_changed":          n_changed,
        "n_correct_after_change":     n_correct_after_change,
        "total_interventions":        total_interv,
        "avg_interventions":          round(total_interv / N, 3),
        "avg_elapsed_s":              round(
            sum(r["elapsed_pass1"] + r["elapsed_pass2"] for r in results) / N, 3
        ),
        "timestamp":                  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hook_layer":                 HOOK_LAYER,
        "alpha":                      args.alpha,
    }
    return results, summary


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    args = parse_args()

    # ---- Load samples (CPU-only) ----
    samples = load_hard_samples(args.n_samples)
    if not samples:
        print("ERROR: No hard-subset samples found. Check manifest path.")
        sys.exit(1)

    # ---- Load model ----
    model, processor, D, n_layers = load_model()

    # ---- Run modes ----
    modes_to_run = (
        ["base", "self_correction", "aosp_sc"]
        if args.mode == "all"
        else [args.mode]
    )

    all_results  = {}
    all_summaries = {}

    for mode in modes_to_run:
        results, summary = evaluate_mode(mode, samples, model, processor, args)
        all_results[mode]    = results
        all_summaries[mode]  = summary

    # ---- Build comparison table ----
    comparison = {}
    if len(all_summaries) > 1:
        base_acc = all_summaries.get("base", {}).get("accuracy", float("nan"))
        sc_acc   = all_summaries.get("self_correction", {}).get("accuracy", base_acc)
        aosp_sc_acc = all_summaries.get("aosp_sc", {}).get("accuracy", base_acc)

        comparison = {
            "base_accuracy":    base_acc,
            "sc_accuracy":      sc_acc,
            "aosp_sc_accuracy": aosp_sc_acc,
            "delta_sc_vs_base":      round(sc_acc   - base_acc, 4),
            "delta_aosp_sc_vs_base": round(aosp_sc_acc - base_acc, 4),
            "delta_aosp_sc_vs_sc":   round(aosp_sc_acc - sc_acc, 4),
            "dual_gain_confirmed":   aosp_sc_acc > sc_acc and aosp_sc_acc > base_acc,
            "sc_gain_confirmed":     sc_acc > base_acc,
            "paper_claim": (
                "A-OSP + Self-Correction achieves Dual-Gain: A-OSP removes anchoring "
                "bias in the initial hypothesis (Pass 1), enabling more accurate "
                "self-review in Pass 2. Net accuracy gain > either intervention alone. "
                "Supports Orthogonal Direct Sum Decomposition (§4.5) and "
                "Dual-Gain ablation (Table 3 / Figure 4)."
            ),
        }

        print(f"\n{'='*65}")
        print(f"  DUAL-GAIN COMPARISON")
        print(f"{'='*65}")
        print(f"  Base           : {base_acc:.1%}")
        print(f"  + SC (2-pass)  : {sc_acc:.1%}  (Δ={comparison['delta_sc_vs_base']:+.1%})")
        print(f"  + A-OSP + SC   : {aosp_sc_acc:.1%}  (Δ={comparison['delta_aosp_sc_vs_base']:+.1%})")
        print(f"  Dual-gain confirmed: {comparison['dual_gain_confirmed']}")

    # ---- Save combined output ----
    output = {
        "task":         "MMMU Hard Subset — Dual-Gain (Task 3.5b)",
        "paper_figure": "Table 3 / Figure 4 (Dual-Gain ablation)",
        "hard_subjects": sorted(HARD_SUBJECTS),
        "n_samples":    len(samples),
        "hook_layer":   HOOK_LAYER,
        "v_tensor":     V_TEXT_ONLY,
        "alpha":        args.alpha,
        "results":      all_summaries,
        "comparison":   comparison,
        "per_sample":   all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Output → {OUTPUT_PATH}")
    print("EVAL COMPLETE")


if __name__ == "__main__":
    main()
