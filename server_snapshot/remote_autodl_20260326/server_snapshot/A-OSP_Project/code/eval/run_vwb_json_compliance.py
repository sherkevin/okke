#!/usr/bin/env python3
"""
Task 2.5 — VisualWebBench JSON Parse Success Rate
==================================================
Proves AgentOS formatting robustness of A-OSP under the
Markovian Language Attractor framework.

Scientific goal: When forced to produce structured JSON output,
does A-OSP's orthogonal projection of the language-inertia subspace
IMPROVE compliance by reducing the model's drift toward verbose,
unstructured "explanation-mode" responses?

Hypothesis (Orthogonal Direct Sum Decomposition §4.5):
  The hidden-state space H ≅ S_bias ⊕ S_task.
  S_bias drives free-form explanatory language (hallucination attractor).
  After projection: H' = H - alpha*(H @ V^T)V stays in S_task,
  which is more compatible with format-constrained generation.
  Prediction: AOSP JSON parse rate ≥ Base JSON parse rate.

Prompt design:
  Each screenshot → "Output your action sequence strictly as a JSON object:
  {\"action\": \"click\", \"element_id\": <N>}"
  where N ∈ {0..len(options)-1} indexes the candidate bounding box / link.

Metrics:
  - json_parse_success: json.loads() succeeds (primary metric)
  - schema_valid: parsed JSON has "action" and "element_id" keys
  - answer_correct: element_id matches ground-truth option index

Output: logs/eval_results/visualwebbench_json_compliance.json

Usage:
  python run_vwb_json_compliance.py --mode base  --n_samples 50
  python run_vwb_json_compliance.py --mode aosp  --n_samples 50
  python run_vwb_json_compliance.py --mode both  --n_samples 50
"""

import os, sys, gc, json, time, re, argparse, torch
from pathlib import Path

PROJECT_ROOT   = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_PATH     = str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")
V_TEXT_ONLY_Q3 = str(PROJECT_ROOT / "models" / "V_text_only_q3.pt")
V_MATRIX_Q3    = str(PROJECT_ROOT / "models" / "V_matrix_q3.pt")
MANIFEST_PATH  = str(PROJECT_ROOT / "data/benchmarks/visualwebbench/vwb_action_prediction_50.jsonl")
LOG_DIR        = str(PROJECT_ROOT / "logs" / "eval_results")
OUT_PATH       = str(PROJECT_ROOT / "logs/eval_results/visualwebbench_json_compliance.json")
os.makedirs(LOG_DIR, exist_ok=True)

HOOK_LAYER = 29   # consistent with V3.5 spec


# ===========================================================================
# ARGS
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "aosp", "both"], default="both")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--v_matrix", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_samples(n: int) -> list[dict]:
    samples = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            if len(samples) >= n:
                break
            rec = json.loads(line)
            img_path = rec.get("image_path", "")
            if not os.path.exists(img_path):
                continue
            # Options: list of 4–8 strings (page link titles for action_prediction)
            options = rec.get("options", [])
            answer  = rec.get("answer")   # integer index into options
            samples.append({
                "sample_id":  rec.get("id", f"vwb_{len(samples):04d}"),
                "image_path": img_path,
                "options":    options,
                "answer":     answer,      # correct element_id (0-based)
                "website":    rec.get("website", ""),
                "elem_desc":  rec.get("elem_desc", ""),
            })
    print(f"[Data] Loaded {len(samples)} samples with valid images")
    return samples


# ===========================================================================
# PROMPT
# ===========================================================================
def build_json_prompt(sample: dict) -> str:
    """
    Force strict JSON output encoding the predicted click action.

    Format: {"action": "click", "element_id": <N>}
    where N ∈ {0, 1, ..., len(options)-1} is the index of the element
    the agent should click on to complete the navigation/interaction task.

    The JSON constraint directly probes whether A-OSP reduces the model's
    tendency to drift into free-form explanation (language attractor) when
    a strict format is required.
    """
    options = sample["options"]
    n_opts  = len(options)

    # Build numbered option list
    opt_lines = "\n".join(
        f"  element_id {i}: {opt}" for i, opt in enumerate(options)
    )

    prompt = (
        "You are a web agent. Look at the screenshot and identify which "
        "element the user should click to complete the task.\n\n"
        f"Task: Click on the element that matches the highlighted target.\n\n"
        f"Available elements ({n_opts} total):\n{opt_lines}\n\n"
        "Output your action sequence STRICTLY as a JSON object with no "
        "additional text, explanation, or markdown:\n"
        '{"action": "click", "element_id": <integer>}'
    )
    return prompt


# ===========================================================================
# JSON VALIDATION
# ===========================================================================
def validate_json_response(raw: str, n_options: int) -> dict:
    """
    Parse and validate the model's JSON response.

    Returns:
      json_parse_success: True if json.loads() succeeds on the raw output
                          (or on the first JSON-like substring extracted)
      schema_valid:       True if parsed obj has "action" and "element_id"
      action_correct:     True if action == "click"
      element_id:         The predicted element_id (int or None)
      element_id_valid:   True if element_id ∈ [0, n_options)
    """
    result = {
        "json_parse_success": False,
        "schema_valid":       False,
        "action_correct":     False,
        "element_id":         None,
        "element_id_valid":   False,
        "parse_error":        None,
    }

    text = raw.strip()

    # Strip markdown code fences if present: ```json {...} ```
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Attempt 1: parse the entire response as JSON
    parsed = None
    try:
        parsed = json.loads(text)
        result["json_parse_success"] = True
    except json.JSONDecodeError as e1:
        result["parse_error"] = str(e1)

        # Attempt 2: extract first {...} block via regex
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                result["json_parse_success"] = True
                result["parse_error"] = f"extracted_block (original error: {e1})"
            except json.JSONDecodeError as e2:
                result["parse_error"] = f"both_failed: {e1} | {e2}"

    if parsed is None:
        return result

    # Schema validation
    if isinstance(parsed, dict) and "action" in parsed and "element_id" in parsed:
        result["schema_valid"] = True
        result["action_correct"] = (str(parsed.get("action", "")).lower() == "click")
        try:
            eid = int(parsed["element_id"])
            result["element_id"] = eid
            result["element_id_valid"] = (0 <= eid < n_options)
        except (ValueError, TypeError):
            result["element_id"] = parsed.get("element_id")

    return result


# ===========================================================================
# A-OSP HOOK (same as run_mvbench_action_sequence.py)
# ===========================================================================
class VWBAOSPHook:
    def __init__(self, V_bias, L_prior, K=20, alpha=0.5,
                 mu=1.5, beta=0.9, T_max=15, layer_idx=29):
        self.V_bias  = V_bias.float()
        self.L_prior = L_prior
        self.K, self.alpha, self.mu, self.beta = K, alpha, mu, beta
        self.T_max, self.layer_idx = T_max, layer_idx
        self.reset()
        self.total_interventions = 0
        self.total_steps         = 0
        self.handle              = None

    def reset(self):
        self.L_bar      = self.L_prior
        self.t          = 0
        self.N_adaptive = None

    def hook_fn(self, module, inp, out):
        if isinstance(out, tuple):
            hidden, rest = out[0], out[1:]
        else:
            hidden, rest = out, None

        _, seq_len, _ = hidden.shape
        self.t          += 1
        self.total_steps += 1

        h = hidden[0].mean(dim=0) if seq_len > 1 else hidden[0, 0]

        # Burn-in bypass (MCQ / structured format → near-zero entropy)
        if self.N_adaptive is None:
            self.N_adaptive = self.t

        V      = self.V_bias.to(hidden.device, dtype=hidden.dtype)
        h_norm = h / (h.norm() + 1e-8)
        L_t    = (h_norm @ V.T).norm().item()

        if L_t <= self.mu * self.L_bar:
            self.L_bar = self.beta * self.L_bar + (1.0 - self.beta) * L_t

        if L_t > self.mu * self.L_bar:
            self.total_interventions += 1
            V_dev    = V
            H_flat   = hidden[0]
            proj_all = H_flat @ V_dev.T
            H_proj   = H_flat - self.alpha * (proj_all @ V_dev)
            orig_norm = H_flat.norm(dim=-1, keepdim=True)
            proj_norm = H_proj.norm(dim=-1, keepdim=True) + 1e-8
            H_corr   = (H_proj / proj_norm) * orig_norm
            hidden_new = hidden.clone()
            hidden_new[0] = H_corr
            return (hidden_new,) + rest if rest is not None else hidden_new

        return out

    def register(self, model):
        for attr_chain in [
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
        ]:
            obj = model
            try:
                for attr in attr_chain:
                    obj = getattr(obj, attr)
                layers = obj
                break
            except AttributeError:
                continue
        else:
            raise AttributeError("Cannot find decoder layers")

        total      = len(layers)
        actual_idx = self.layer_idx if self.layer_idx < total else total - 4
        self.handle = layers[actual_idx].register_forward_hook(self.hook_fn)
        print(f"[A-OSP] Hook @ Layer {actual_idx}/{total-1}, alpha={self.alpha}")

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===========================================================================
# MODEL & V_MATRIX LOADING
# ===========================================================================
def load_model():
    from transformers import AutoProcessor, AutoConfig
    from transformers import Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    D   = (getattr(cfg, "hidden_size", None)
           or getattr(getattr(cfg, "text_config", cfg), "hidden_size", 4096))
    n_l = (getattr(cfg, "num_hidden_layers", None)
           or getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 36))
    print(f"[Model] Qwen3-VL-8B: hidden={D}, layers={n_l}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor


def load_v_matrix(override=None):
    for p in [override, V_TEXT_ONLY_Q3, V_MATRIX_Q3]:
        if p and os.path.exists(p):
            d   = torch.load(p, map_location="cpu", weights_only=True)
            evr = d.get("evr", float("nan"))
            tag = d.get("tag", os.path.basename(p))
            lp  = d.get("L_prior", float("nan"))
            print(f"[V_matrix] {os.path.basename(p)}: tag={tag}, "
                  f"shape={list(d['V_bias'].shape)}, EVR={evr:.4f}, L_prior={lp:.4f}")
            if p == V_TEXT_ONLY_Q3 and evr < 0.70:
                print(f"  ⚠ EVR={evr:.4f} < 0.70 redline — trying fallback")
                continue
            return d
    raise FileNotFoundError("No valid V_bias found")


# ===========================================================================
# SINGLE SAMPLE INFERENCE
# ===========================================================================
def infer_sample(model, processor, sample: dict, hook) -> dict:
    from qwen_vl_utils import process_vision_info

    prompt     = build_json_prompt(sample)
    image_path = sample["image_path"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ],
    }]

    try:
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        return {
            "sample_id":          sample["sample_id"],
            "error":              str(e),
            "json_parse_success": False,
            "schema_valid":       False,
            "action_correct":     False,
            "element_id":         None,
            "element_id_valid":   False,
            "answer_correct":     False,
            "raw_output":         "",
            "n_interventions":    0,
            "elapsed_s":          0.0,
        }

    if hook:
        hook.reset()

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=64,   # JSON is short; cap to avoid rambling
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    elapsed = time.time() - t0

    new_toks  = out_ids[0][inputs["input_ids"].shape[1]:]
    raw       = processor.decode(new_toks, skip_special_tokens=True).strip()
    n_options = len(sample["options"])
    vj        = validate_json_response(raw, n_options)

    answer_correct = (
        vj["element_id"] is not None
        and sample["answer"] is not None
        and int(vj["element_id"]) == int(sample["answer"])
    )

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sample_id":          sample["sample_id"],
        "website":            sample["website"],
        "n_options":          n_options,
        "answer_gt":          sample["answer"],
        "raw_output":         raw,
        "json_parse_success": vj["json_parse_success"],
        "schema_valid":       vj["schema_valid"],
        "action_correct":     vj["action_correct"],
        "element_id":         vj["element_id"],
        "element_id_valid":   vj["element_id_valid"],
        "answer_correct":     answer_correct,
        "parse_error":        vj.get("parse_error"),
        "n_interventions":    hook.total_interventions if hook else 0,
        "elapsed_s":          round(elapsed, 3),
    }


# ===========================================================================
# EVALUATE ONE MODE
# ===========================================================================
def evaluate_mode(mode: str, samples: list[dict],
                  model, processor, v_matrix_override=None,
                  alpha=0.5) -> dict:
    print(f"\n{'='*60}")
    print(f"  VisualWebBench JSON Compliance — {mode.upper()}")
    print(f"  N={len(samples)} | Prompt: forced JSON {{'action','element_id'}}")
    print(f"{'='*60}")

    hook = None
    if mode == "aosp":
        v_data  = load_v_matrix(v_matrix_override)
        hook    = VWBAOSPHook(
            V_bias=v_data["V_bias"], L_prior=v_data["L_prior"],
            K=v_data.get("K", 20), alpha=alpha, layer_idx=HOOK_LAYER,
        )
        hook.register(model)
        v_tag = v_data.get("tag", "unknown")
        print(f"[A-OSP] subspace={v_tag}, alpha={alpha}")

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1:2d}/{len(samples)}] {sample['sample_id']:20s} ", end="", flush=True)
        res = infer_sample(model, processor, sample, hook)
        results.append(res)

        parse_mark = "✓" if res["json_parse_success"] else "✗"
        schema_mark = "✓" if res["schema_valid"] else "✗"
        corr_mark   = "✓" if res["answer_correct"] else "✗"
        interv_str  = f" [{res['n_interventions']}x]" if res["n_interventions"] else ""
        print(f"parse:{parse_mark} schema:{schema_mark} ans:{corr_mark}{interv_str} "
              f"→ {repr(res['raw_output'][:50])}")

    if hook:
        hook.remove()

    # --- Aggregate metrics ---
    N = len(results)
    n_parse  = sum(1 for r in results if r["json_parse_success"])
    n_schema = sum(1 for r in results if r["schema_valid"])
    n_action = sum(1 for r in results if r["action_correct"])
    n_valid  = sum(1 for r in results if r["element_id_valid"])
    n_corr   = sum(1 for r in results if r["answer_correct"])
    n_interv = sum(r["n_interventions"] for r in results)

    summary = {
        "mode":                   mode,
        "n_samples":              N,
        "json_parse_success_rate": round(n_parse  / N, 4),
        "schema_valid_rate":       round(n_schema / N, 4),
        "action_correct_rate":     round(n_action / N, 4),
        "element_id_in_range_rate":round(n_valid  / N, 4),
        "answer_accuracy":         round(n_corr   / N, 4),
        "n_json_parsed":          n_parse,
        "n_schema_valid":         n_schema,
        "n_answer_correct":       n_corr,
        "total_interventions":    n_interv,
        "avg_interventions":      round(n_interv / N, 3),
        "avg_elapsed_s":          round(sum(r["elapsed_s"] for r in results) / N, 3),
        "v_tag":                  (hook.V_bias is not None if hook else None),
    }

    print(f"\n  JSON Parse Rate:  {summary['json_parse_success_rate']:.1%}  "
          f"({n_parse}/{N})")
    print(f"  Schema Valid:     {summary['schema_valid_rate']:.1%}  "
          f"({n_schema}/{N})")
    print(f"  Answer Accuracy:  {summary['answer_accuracy']:.1%}  "
          f"({n_corr}/{N})")
    if mode == "aosp":
        print(f"  Interventions:    {n_interv} total, "
              f"{summary['avg_interventions']:.2f}/sample")

    return {"summary": summary, "results": results}


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    args    = parse_args()
    samples = load_samples(args.n_samples)

    if not samples:
        print("ERROR: No valid samples found. Run data download first.")
        sys.exit(1)

    modes = ["base", "aosp"] if args.mode == "both" else [args.mode]

    print(f"\n[Task 2.5] VisualWebBench JSON Compliance — modes: {modes}")
    print(f"Hypothesis (§4.5 Direct Sum): A-OSP projects out S_bias → "
          f"model stays in S_task → higher JSON compliance")

    # Load model once for all modes
    model, processor = load_model()

    all_results = {}
    for mode in modes:
        res = evaluate_mode(
            mode, samples, model, processor,
            v_matrix_override=args.v_matrix,
            alpha=args.alpha,
        )
        all_results[mode] = res

    # --- Save combined output ---
    output = {
        "task":        "Task 2.5 — VisualWebBench JSON Parse Success Rate",
        "paper_claim": ("Supports Orthogonal Direct Sum Decomposition §4.5: "
                        "A-OSP projection reduces language-inertia drift, "
                        "improving structured JSON compliance in GUI agent tasks."),
        "dataset":     "visualwebbench/VisualWebBench action_prediction split",
        "n_samples":   len(samples),
        "hook_layer":  HOOK_LAYER,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results":     {m: v["summary"] for m, v in all_results.items()},
        "per_sample":  {m: v["results"] for m, v in all_results.items()},
    }

    # Print comparison table
    if "base" in all_results and "aosp" in all_results:
        bs = all_results["base"]["summary"]
        as_ = all_results["aosp"]["summary"]
        delta_parse  = as_["json_parse_success_rate"] - bs["json_parse_success_rate"]
        delta_schema = as_["schema_valid_rate"]        - bs["schema_valid_rate"]
        delta_acc    = as_["answer_accuracy"]          - bs["answer_accuracy"]

        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY (§4.5 Direct Sum Evidence)")
        print(f"{'='*60}")
        print(f"  {'Metric':<30} {'Base':>8} {'AOSP':>8} {'Delta':>8}")
        print(f"  {'-'*54}")
        print(f"  {'JSON Parse Rate':<30} "
              f"{bs['json_parse_success_rate']:>8.1%} "
              f"{as_['json_parse_success_rate']:>8.1%} "
              f"{'▲' if delta_parse>0 else '▼'}{abs(delta_parse):>6.1%}")
        print(f"  {'Schema Valid Rate':<30} "
              f"{bs['schema_valid_rate']:>8.1%} "
              f"{as_['schema_valid_rate']:>8.1%} "
              f"{'▲' if delta_schema>0 else '▼'}{abs(delta_schema):>6.1%}")
        print(f"  {'Answer Accuracy':<30} "
              f"{bs['answer_accuracy']:>8.1%} "
              f"{as_['answer_accuracy']:>8.1%} "
              f"{'▲' if delta_acc>0 else '▼'}{abs(delta_acc):>6.1%}")
        print(f"  {'AOSP Interventions':<30} {'—':>8} "
              f"{as_['total_interventions']:>8}  ({as_['avg_interventions']:.2f}/sample)")
        print(f"{'='*60}")

        output["comparison"] = {
            "delta_json_parse_rate": round(delta_parse,  4),
            "delta_schema_valid":    round(delta_schema, 4),
            "delta_answer_accuracy": round(delta_acc,    4),
            "hypothesis_supported":  delta_parse >= 0,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {OUT_PATH}")
    print("EVAL COMPLETE")


if __name__ == "__main__":
    main()
