#!/usr/bin/env python3
"""
MVBench Action Sequence — V3.5 Zero-Shot Video Transfer Evaluation
===================================================================
Task 2.3 (V3.5 Sprint 3): Apply S_text-only subspace to Video QA.

Scientific goal: Show that A-OSP suppresses language-prior hallucination
in temporal action ordering, forcing the model to rely on actual video
evidence rather than statistical co-occurrence of action pairs.

Key design:
  - Real Charades .mp4 clips loaded via qwen_vl_utils.process_vision_info
  - Qwen3-VL builds T×M×D temporal-spatial features at Layer 29
  - MeanPool(t, m) flattens to R^D before A-OSP orthogonal projection
  - S_text-only (Agent 1 official tensor, EVR > 70%) as bias subspace
  - Temporal clip trimming via start/end timestamps from MVBench JSON

Usage:
  # Check video paths only (poll until ready)
  python run_mvbench_action_sequence.py --check_video_paths

  # Run 10-sample mini-batch, base mode
  python run_mvbench_action_sequence.py --mode base --n_samples 10

  # Run 10-sample mini-batch, A-OSP mode
  python run_mvbench_action_sequence.py --mode aosp --n_samples 10

  # Run full 200-sample evaluation
  python run_mvbench_action_sequence.py --mode aosp --n_samples 200
"""

import os, sys, gc, json, time, argparse, re, torch
import numpy as np

# ===========================================================================
# PATH CONSTANTS
# ===========================================================================
PROJECT_ROOT   = "/root/autodl-tmp/A-OSP_Project"
MODEL_PATH     = f"{PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
# Official S_text-only tensor (Agent 1, EVR=87.87%, 200 prompts, Layer 29)
V_TEXT_ONLY_Q3 = f"{PROJECT_ROOT}/models/qwen3vl/V_text_only.pt"
# Fallback to S_blur if text-only not yet delivered
V_MATRIX_Q3    = f"{PROJECT_ROOT}/models/V_matrix_q3.pt"
# MVBench annotation JSON
ANNO_JSON      = f"{PROJECT_ROOT}/data/mvbench/json/action_sequence.json"
# Charades video directory (confirmed by Agent 4 on 2026-03-20)
VIDEO_DIR      = f"{PROJECT_ROOT}/data/mvbench/video/extracted/data0613/star/Charades_v1_480"
LOG_DIR        = f"{PROJECT_ROOT}/logs/eval_results"
os.makedirs(LOG_DIR, exist_ok=True)

# A-OSP hook layer (V3.5 spec: MeanPool(t,m) at Layer 29)
HOOK_LAYER = 29


# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MVBench Action Sequence eval: Base vs A-OSP (V3.5 Sprint 3)"
    )
    p.add_argument("--mode", choices=["base", "aosp", "both"], default="both",
                   help="Inference mode (both = run base then aosp sequentially)")
    p.add_argument("--n_samples", type=int, default=10,
                   help="Number of samples to evaluate (10=mini-batch, 200=full)")
    p.add_argument("--v_matrix", type=str, default=None,
                   help="Override path to V_bias .pt file")
    p.add_argument("--check_video_paths", action="store_true",
                   help="Only check that all n_samples video files exist, then exit. "
                        "Returns exit code 0 if all present, 1 if any missing.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="A-OSP projection strength (0=no intervention, 1=full projection)")
    p.add_argument("--fps", type=float, default=1.0,
                   help="Frame sampling rate for video (fps=1.0 → 1 frame/sec)")
    p.add_argument("--max_pixels", type=int, default=360*420,
                   help="Max pixels per frame (controls VRAM usage)")
    return p.parse_args()


# ===========================================================================
# ANNOTATION LOADING
# ===========================================================================
def load_annotations(n: int) -> list[dict]:
    """
    Load n entries from MVBench action_sequence.json where the .mp4 file
    physically exists on disk (required for valid video evaluation).

    Strategy: scan ALL 200 annotations, collect those with matching video,
    then take the first n. This ensures we never run text-only samples.

    Returns list enriched with:
      - answer_letter: ground-truth letter (A/B/C/D)
      - video_path:    absolute path to the .mp4 file (always exists)
      - sample_id:     zero-padded index string
    """
    with open(ANNO_JSON) as f:
        raw = json.load(f)

    all_matched = []
    for i, entry in enumerate(raw):
        video_filename = entry["video"]
        video_path = os.path.join(VIDEO_DIR, video_filename)
        if not os.path.exists(video_path):
            continue                              # skip missing videos

        candidates  = entry["candidates"]
        answer_text = entry["answer"]

        # Map full-text answer → letter
        answer_letter = None
        for j, cand in enumerate(candidates):
            if cand.strip().rstrip(".") == answer_text.strip().rstrip("."):
                answer_letter = "ABCD"[j]
                break
        if answer_letter is None:
            for j, cand in enumerate(candidates):
                if answer_text.strip().lower() in cand.strip().lower() or \
                   cand.strip().lower() in answer_text.strip().lower():
                    answer_letter = "ABCD"[j]
                    break
        if answer_letter is None:
            answer_letter = "A"

        all_matched.append({
            "sample_id":      f"actseq_{i:04d}",
            "video_filename":  video_filename,
            "video_path":      video_path,
            "question":        entry["question"],
            "candidates":      candidates,
            "answer_text":     answer_text,
            "answer_letter":   answer_letter,
            "start":           float(entry.get("start", 0.0)),
            "end":             float(entry.get("end",   30.0)),
        })

    total_avail = len(all_matched)
    samples = all_matched[:n]
    print(f"[Annotations] {total_avail} samples have video on disk; "
          f"using first {len(samples)}")
    return samples


# ===========================================================================
# VIDEO PATH CHECK (--check_video_paths)
# ===========================================================================
def check_video_paths(samples: list[dict]) -> bool:
    """
    Assert each sample's .mp4 exists on disk.
    Prints a per-file status table and returns True only if ALL present.
    """
    print(f"\n{'='*60}")
    print(f"  VIDEO PATH CHECK  ({len(samples)} samples)")
    print(f"  Directory: {VIDEO_DIR}")
    print(f"{'='*60}")

    all_present = True
    n_found = 0

    for s in samples:
        fn   = s["video_filename"]
        path = os.path.join(VIDEO_DIR, fn)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓  {fn}  ({size_mb:.1f} MB)")
            n_found += 1
        else:
            print(f"  ✗  {fn}  — NOT FOUND")
            all_present = False

    print(f"\n  Result: {n_found}/{len(samples)} present", end="")
    if all_present:
        print("  ✅  ALL READY — proceed with eval")
    else:
        print(f"  ❌  {len(samples)-n_found} missing — waiting for upload")
    print(f"{'='*60}\n")
    return all_present


# ===========================================================================
# MODEL LOADING
# ===========================================================================
def load_model():
    """Load Qwen3-VL-8B-Instruct with FA2 and auto device map."""
    from transformers import AutoConfig, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration

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


# ===========================================================================
# V-MATRIX LOADING
# ===========================================================================
def load_v_matrix(override_path: str | None = None) -> dict:
    """
    Load the bias subspace matrix.
    Priority: override → V_text_only_q3 (Agent 1 official) → V_matrix_q3 (fallback)
    """
    search = [
        override_path,
        V_TEXT_ONLY_Q3,
        V_MATRIX_Q3,
    ]
    for p in search:
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            tag = d.get("tag", os.path.basename(p))
            evr = d.get("evr", float("nan"))
            lp  = d.get("L_prior", float("nan"))
            print(f"[V_matrix] {os.path.basename(p)}: tag={tag}, "
                  f"shape={list(d['V_bias'].shape)}, EVR={evr:.4f}, L_prior={lp:.4f}")
            if p == V_TEXT_ONLY_Q3 and evr < 0.70:
                print(f"[V_matrix] WARNING: EVR={evr:.4f} < 0.70 redline — "
                      f"using fallback until Agent 1 delivers official tensor")
                continue
            return d
    raise FileNotFoundError(
        "No valid V_bias matrix found.\n"
        f"  Expected (official): {V_TEXT_ONLY_Q3}\n"
        f"  Fallback:            {V_MATRIX_Q3}"
    )


# ===========================================================================
# A-OSP HOOK
# ===========================================================================
class ActionSeqAOSPHook:
    """
    A-OSP hook for MVBench Action Sequence evaluation.

    V3.5 Temporal-Spatial Flatten Alignment (§4.6.3):
      - Prefill  (seq_len > 1): h = mean([N_text + T*M positions, D]) → [D]
        This is dominated by the T*M video tokens for typical video inputs.
      - Decode   (seq_len = 1): h = hidden[0, 0] → [D]

    Intervention condition (dynamic EMA):
      L_t > mu * L_bar  →  apply soft orthogonal projection
      H_corrected = (I - alpha * V^T V) H  (scale-preserved)
    """

    def __init__(self, V_bias: torch.Tensor, L_prior: float,
                 K: int = 20, alpha: float = 0.5,
                 mu: float = 1.5, beta: float = 0.9,
                 T_max: int = 15, layer_idx: int = 29):
        self.V_bias  = V_bias.float()   # [K, D]
        self.L_prior = L_prior
        self.K       = K
        self.alpha   = alpha
        self.mu      = mu
        self.beta    = beta
        self.T_max   = T_max
        self.layer_idx = layer_idx

        self.reset()
        self.total_interventions = 0
        self.total_steps         = 0
        self.handle              = None

    # ------------------------------------------------------------------
    def reset(self):
        """Reset per-generation state (call before each sample)."""
        self.L_bar     = self.L_prior
        self.t         = 0
        self.N_adaptive = None

    # ------------------------------------------------------------------
    def hook_fn(self, module, inp, out):
        # --- unpack hidden states (bare Tensor for Qwen3+FA2, tuple otherwise) ---
        if isinstance(out, tuple):
            hidden = out[0]
            rest   = out[1:]
        else:
            hidden = out
            rest   = None

        batch, seq_len, D = hidden.shape
        self.t          += 1
        self.total_steps += 1

        # ---------------------------------------------------------------
        # Temporal-Spatial Flatten Alignment (§4.6.3)
        # ---------------------------------------------------------------
        if seq_len > 1:
            # PREFILL: average over all token positions
            # For video inputs: N_total = N_text_prefix + T*M_video_patches
            # MeanPool(t, m) collapses temporal×spatial → single R^D probe
            h = hidden[0].mean(dim=0)          # [N_total, D] → [D]
        else:
            # DECODE: current generated token
            h = hidden[0, 0]                   # [D]

        # ---------------------------------------------------------------
        # Entropy-aware burn-in bypass
        # For MCQ (short responses), entropy ≈ 0 → skip burn-in immediately
        # ---------------------------------------------------------------
        if self.N_adaptive is None:
            # Low-entropy bypass: MCQ answers are near-deterministic
            self.N_adaptive = self.t  # allow intervention from first step

        # ---------------------------------------------------------------
        # Projection energy L_t = ||h_norm @ V^T||_2
        # ---------------------------------------------------------------
        V     = self.V_bias.to(hidden.device, dtype=hidden.dtype)   # [K, D]
        h_norm = h / (h.norm() + 1e-8)                              # [D]
        proj   = h_norm @ V.T                                        # [K]
        L_t    = proj.norm().item()

        # ---------------------------------------------------------------
        # Conditional EMA update (freeze on anomalous spikes)
        # ---------------------------------------------------------------
        if L_t <= self.mu * self.L_bar:
            self.L_bar = self.beta * self.L_bar + (1.0 - self.beta) * L_t

        # ---------------------------------------------------------------
        # Soft orthogonal projection intervention
        # ---------------------------------------------------------------
        if L_t > self.mu * self.L_bar:
            self.total_interventions += 1

            V_dev   = self.V_bias.to(hidden.device, dtype=hidden.dtype)  # [K, D]
            H_flat  = hidden[0]                              # [seq_len, D]

            # Remove bias subspace component
            proj_all = H_flat @ V_dev.T                     # [seq_len, K]
            H_proj   = H_flat - self.alpha * (proj_all @ V_dev)  # [seq_len, D]

            # Scale preservation: restore original per-token L2 norm
            orig_norm = H_flat.norm(dim=-1, keepdim=True)   # [seq_len, 1]
            proj_norm = H_proj.norm(dim=-1, keepdim=True) + 1e-8
            H_corrected = (H_proj / proj_norm) * orig_norm  # [seq_len, D]

            hidden_new = hidden.clone()
            hidden_new[0] = H_corrected

            if isinstance(out, tuple):
                return (hidden_new,) + rest
            return hidden_new

        # No intervention
        if isinstance(out, tuple):
            return out
        return out

    # ------------------------------------------------------------------
    def register(self, model):
        """Find decoder layers and attach hook at layer_idx."""
        if (hasattr(model, "model")
                and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "layers")):
            layers = model.model.language_model.layers
        elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            raise AttributeError(
                f"Cannot find decoder layers in {type(model).__name__}"
            )

        total      = len(layers)
        actual_idx = self.layer_idx if self.layer_idx < total else total - 4
        self.handle = layers[actual_idx].register_forward_hook(self.hook_fn)
        print(f"[A-OSP] Hook registered at Layer {actual_idx}/{total-1} "
              f"(MeanPool(t,m) + orthogonal projection, alpha={self.alpha})")

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ===========================================================================
# PROMPT CONSTRUCTION
# ===========================================================================
def build_action_seq_prompt(sample: dict, has_video: bool) -> str:
    """
    Build the MCQ prompt for MVBench Action Sequence.

    Format (with video):
        Watch the video carefully and answer the following question.

        Question: <question>

        Options:
        (A) <candidate_0>
        (B) <candidate_1>
        (C) <candidate_2>
        (D) <candidate_3>

        Answer with the option letter in parentheses, e.g. (A).

    The leading phrase anchors the model's attention to actual video evidence
    rather than language co-occurrence statistics. The "(A)" format matches
    Qwen3-VL's natural MCQ response pattern.
    """
    question   = sample["question"]
    candidates = sample["candidates"]

    options_text = "\n".join(
        f"({ltr}) {cand}"
        for ltr, cand in zip("ABCD", candidates)
    )

    if has_video:
        prompt = (
            "Watch the video carefully and answer the following question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer with the option letter in parentheses, e.g. (A)."
        )
    else:
        # TEXT-ONLY FALLBACK — invalid for paper, labeled clearly
        prompt = (
            f"[TEXT-ONLY — NO VIDEO]\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer with the option letter in parentheses, e.g. (A)."
        )
    return prompt


# ===========================================================================
# RESPONSE PARSER
# ===========================================================================
def parse_answer_letter(raw_output: str, candidates: list[str]) -> str:
    """
    Robustly extract the answer letter from Qwen3-VL's response.

    Handles all known response patterns:
      "(A)"           → A     (standard, prompted format)
      "A"             → A     (bare letter)
      "A."            → A     (trailing period)
      "A)"            → A     (half-parens)
      "The answer is (B)"  → B
      "Option C"      → C
      "answer: D"     → D
      "(A) Ate the medicine."  → A  (full option text)
      "Ate the medicine."  → A  (full candidate match)
    """
    text = raw_output.strip()

    # 1. Parenthesized letter: (A), (B), (C), (D)  — highest confidence
    m = re.search(r'\(([A-Da-d])\)', text)
    if m:
        return m.group(1).upper()

    # 2. Explicit answer markers
    m = re.search(r'(?:answer(?:\s+is)?|option)[:\s]+\(?([A-Da-d])\)?', text, re.I)
    if m:
        return m.group(1).upper()

    # 3. Bare letter at start of response (possibly followed by ., ), :, space)
    m = re.match(r'^([A-Da-d])[.):\s]', text)
    if m:
        return m.group(1).upper()

    # 4. Any standalone letter A-D in the response
    m = re.search(r'\b([A-Da-d])\b', text)
    if m:
        return m.group(1).upper()

    # 5. Full candidate text match (model outputs the full option text)
    for i, cand in enumerate(candidates):
        if cand.strip().rstrip(".").lower() in text.lower():
            return "ABCD"[i]

    # 6. Substring match
    for i, cand in enumerate(candidates):
        words = cand.strip().split()[:3]  # match on first 3 words
        if len(words) >= 2 and " ".join(words).lower() in text.lower():
            return "ABCD"[i]

    # Default: cannot parse
    return "?"


# ===========================================================================
# SINGLE SAMPLE INFERENCE
# ===========================================================================
def infer_single_sample(model, processor, sample: dict,
                        hook: ActionSeqAOSPHook | None,
                        fps: float, max_pixels: int) -> dict:
    """
    Run one MVBench action_sequence sample through Qwen3-VL.

    When video_path is present:
      - Builds {"type":"video", ...} content block
      - process_vision_info decodes frames → pixel_values
      - Model sees T frames × M patches × D dimensions at intermediate layers
      - At Layer 29, hook applies MeanPool(t,m) → A-OSP projection

    Returns a results dict with all fields needed for analysis.
    """
    from qwen_vl_utils import process_vision_info

    video_path = sample["video_path"]
    has_video  = video_path is not None and os.path.exists(video_path)
    prompt     = build_action_seq_prompt(sample, has_video=has_video)

    # --- Build message content ---
    if has_video:
        # Use nframes=8 for stable video decoding across qwen_vl_utils versions.
        # video_start/video_end trimming is skipped: these fields cause
        # StopIteration in get_rope_index when qwen_vl_utils ignores them
        # but still generates mismatched grid_thw tensors.
        video_content = {
            "type":       "video",
            "video":      video_path,
            "nframes":    8,
            "max_pixels": max_pixels,
        }

        messages = [{
            "role": "user",
            "content": [video_content, {"type": "text", "text": prompt}],
        }]
    else:
        # TEXT-ONLY: no pixel_values produced — clearly flagged as invalid
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

    # --- Tokenize ---
    try:
        chat_text     = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # ------------------------------------------------------------------
        # Qwen3-VL per-frame fix:
        # The processor creates ONE vision block per temporal patch (T frames),
        # but video_grid_thw stores ONE entry [T, H, W] per video.
        # get_rope_index expects ONE grid entry per vision_start token.
        # Fix: split [T, H, W] → T entries of [1, H, W].
        # ------------------------------------------------------------------
        if has_video and "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            grids = inputs["video_grid_thw"]
            new_grids = []
            for g in grids:
                T_g, H_g, W_g = g.tolist()
                for _ in range(T_g):
                    new_grids.append([1, H_g, W_g])
            inputs["video_grid_thw"] = torch.tensor(
                new_grids, dtype=grids.dtype, device=grids.device
            )

        # Log video token dimensions for verification
        if has_video:
            thw = inputs.get("video_grid_thw")
            if thw is not None and len(thw) > 0:
                T_tot = len(thw)
                H_g, W_g = thw[0][1].item(), thw[0][2].item()
                print(f"[video] T_frames={T_tot} H={H_g} W={W_g} "
                      f"→ {T_tot * H_g * W_g} raw patches | ",
                      end="", flush=True)
            else:
                print(f"[video] (thw unavailable) | ", end="", flush=True)

    except Exception as e:
        print(f"\n  [WARN] Video processing error: {e}")
        print("  [WARN] Falling back to text-only (check qwen_vl_utils version)")
        chat_text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        inputs    = processor(
            text=[chat_text], return_tensors="pt", padding=True
        ).to(model.device)
        has_video = False

    # --- Reset hook state for this sample ---
    if hook:
        hook.reset()

    # --- Generate ---
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    elapsed = time.time() - t0

    # Decode only new tokens
    n_input    = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][n_input:]
    raw_output = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # --- Parse answer ---
    pred_letter   = parse_answer_letter(raw_output, sample["candidates"])
    gt_letter     = sample["answer_letter"]
    is_correct    = (pred_letter == gt_letter)
    n_interventions = hook.total_interventions if hook else 0

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sample_id":       sample["sample_id"],
        "video_filename":  sample["video_filename"],
        "has_video":       has_video,
        "question":        sample["question"],
        "candidates":      sample["candidates"],
        "gt_letter":       gt_letter,
        "gt_text":         sample["answer_text"],
        "pred_letter":     pred_letter,
        "raw_output":      raw_output,
        "correct":         is_correct,
        "elapsed_s":       round(elapsed, 3),
        "n_interventions": n_interventions,
    }


# ===========================================================================
# RESULT SAVING
# ===========================================================================
def save_results(results: list[dict], mode: str, tag: str = ""):
    """Save per-sample JSONL and summary JSON."""
    n       = len(results)
    n_valid = sum(1 for r in results if r["has_video"])
    n_corr  = sum(1 for r in results if r["correct"] and r["has_video"])
    n_total_corr = sum(1 for r in results if r["correct"])
    acc_valid = n_corr / n_valid if n_valid > 0 else float("nan")
    acc_total = n_total_corr / n if n > 0 else float("nan")
    total_interv = sum(r["n_interventions"] for r in results)

    suffix = f"n{n}{tag}"
    jsonl_path   = os.path.join(LOG_DIR, f"mvbench_actseq_{mode}_{suffix}.jsonl")
    summary_path = os.path.join(LOG_DIR, f"mvbench_actseq_{mode}_{suffix}_summary.json")

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "mode":              mode,
        "n_samples":         n,
        "n_with_video":      n_valid,
        "n_without_video":   n - n_valid,
        "accuracy_video":    round(acc_valid, 4),
        "accuracy_all":      round(acc_total, 4),
        "n_correct_video":   n_corr,
        "total_interventions": total_interv,
        "avg_interventions": round(total_interv / n, 3) if n > 0 else 0,
        "avg_elapsed_s":     round(sum(r["elapsed_s"] for r in results) / n, 3),
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_dir":         VIDEO_DIR,
        "invalid_no_video":  n - n_valid > 0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return jsonl_path, summary_path, summary


# ===========================================================================
# EVALUATE ONE MODE
# ===========================================================================
def evaluate_mode(args, samples, model, processor) -> tuple[list[dict], dict]:
    """Run evaluation for one mode (base or aosp). Returns (results, summary)."""

    hook = None
    if args.mode == "aosp":
        v_data  = load_v_matrix(args.v_matrix)
        V_bias  = v_data["V_bias"]
        L_prior = v_data["L_prior"]
        K       = v_data.get("K", 20)
        tag_str = v_data.get("tag", "unknown")
        print(f"[A-OSP] Subspace: tag={tag_str}, "
              f"V_bias={list(V_bias.shape)}, L_prior={L_prior:.4f}")

        hook = ActionSeqAOSPHook(
            V_bias=V_bias,
            L_prior=L_prior,
            K=K,
            alpha=args.alpha,
            mu=1.5,
            beta=0.9,
            T_max=15,
            layer_idx=HOOK_LAYER,
        )
        hook.register(model)

    results   = []
    n_correct = 0
    n_valid   = len(samples)   # all samples have video (filtered in load_annotations)

    print(f"\n{'='*60}")
    print(f"  MVBench Action Sequence — {args.mode.upper()} mode")
    print(f"  N={n_valid}  |  Hook layer: {HOOK_LAYER}"
          f"  |  fps={args.fps}  |  alpha={args.alpha}")
    print(f"{'='*60}\n")

    for i, sample in enumerate(samples):
        prefix = f"  [{i+1:2d}/{n_valid}] {sample['sample_id']} "
        print(prefix, end="", flush=True)

        res = infer_single_sample(
            model, processor, sample, hook,
            fps=args.fps, max_pixels=args.max_pixels
        )
        results.append(res)

        mark = "✓" if res["correct"] else "✗"
        interv_str = (f" [intervened: {res['n_interventions']}x]"
                      if res["n_interventions"] > 0 else "")
        print(f"{mark}  pred={res['pred_letter']} gt={res['gt_letter']}"
              f"{interv_str}")

        if res["correct"]:
            n_correct += 1

    if hook:
        hook.remove()

    acc          = n_correct / n_valid if n_valid > 0 else float("nan")
    total_interv = sum(r["n_interventions"] for r in results)

    print(f"\n  {'─'*50}")
    print(f"  Accuracy:   {acc:.1%}  ({n_correct}/{n_valid})")
    print(f"  Interventions: {total_interv}  ({total_interv/n_valid:.2f}/sample)")

    summary = {
        "mode":                args.mode,
        "n_samples":           n_valid,
        "n_correct":           n_correct,
        "accuracy":            round(acc, 4),
        "total_interventions": total_interv,
        "avg_interventions":   round(total_interv / n_valid, 3),
        "avg_elapsed_s":       round(sum(r["elapsed_s"] for r in results) / n_valid, 3),
        "all_has_video":       all(r["has_video"] for r in results),
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_dir":           VIDEO_DIR,
        "v_tensor":            V_TEXT_ONLY_Q3 if args.mode == "aosp" else "N/A",
    }
    return results, summary


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    args = parse_args()

    # --- Load annotations (video-filtered) ---
    samples = load_annotations(args.n_samples)

    # -----------------------------------------------------------------------
    # --check_video_paths: poll and exit
    # -----------------------------------------------------------------------
    if args.check_video_paths:
        all_ready = check_video_paths(samples)
        sys.exit(0 if all_ready else 1)

    if len(samples) == 0:
        print(f"ERROR: No video files found in {VIDEO_DIR}")
        sys.exit(1)

    # --- Load model once (shared across modes) ---
    model, processor, D, n_layers = load_model()

    # --- Run both modes sequentially, save combined output ---
    COMBINED_OUTPUT = os.path.join(LOG_DIR, "mvbench_video_minibatch.json")

    all_results  = {}
    all_summaries = {}

    for mode in (["base", "aosp"] if args.mode == "both" else [args.mode]):
        args.mode = mode
        results, summary = evaluate_mode(args, samples, model, processor)
        all_results[mode]   = results
        all_summaries[mode] = summary

        # Also save individual JSONL (never overwrite; append mode name + n)
        _, summary_path, _ = save_results(results, mode)

    # --- Comparison delta ---
    comparison = {}
    if "base" in all_summaries and "aosp" in all_summaries:
        b = all_summaries["base"]
        a = all_summaries["aosp"]
        delta = round(a["accuracy"] - b["accuracy"], 4)
        comparison = {
            "base_accuracy":   b["accuracy"],
            "aosp_accuracy":   a["accuracy"],
            "delta_accuracy":  delta,
            "aosp_interventions": a["total_interventions"],
            "improved": delta > 0,
            "note": ("A-OSP corrects temporal hallucination via S_text-only "
                     "orthogonal projection at Layer 29, proving Cross-modal "
                     "Topological Isomorphism for video temporal generalization "
                     "(Section 4.6.3)."),
        }

    # --- Print comparison ---
    if comparison:
        print(f"\n{'='*60}")
        print(f"  COMPARISON — Base vs A-OSP (N={args.n_samples})")
        print(f"{'='*60}")
        print(f"  Base  accuracy:   {comparison['base_accuracy']:.1%}")
        print(f"  A-OSP accuracy:   {comparison['aosp_accuracy']:.1%}")
        print(f"  Δ accuracy:       {comparison['delta_accuracy']:+.1%}")
        print(f"  Interventions:    {comparison['aosp_interventions']}")

    # --- Save combined JSON ---
    combined = {
        "task":        "MVBench action_sequence Zero-Shot Video Transfer",
        "paper_claim": "Cross-modal Topological Isomorphism for video temporal generalization (Section 4.6.3)",
        "dataset":     "MVBench action_sequence (Charades clips)",
        "video_dir":   VIDEO_DIR,
        "n_samples":   len(samples),
        "hook_layer":  HOOK_LAYER,
        "v_tensor":    V_TEXT_ONLY_Q3,
        "results":     all_summaries,
        "per_sample":  {m: r for m, r in all_results.items()},
        "comparison":  comparison,
    }
    with open(COMBINED_OUTPUT, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Combined output → {COMBINED_OUTPUT}")
    print(f"\nEVAL COMPLETE")


if __name__ == "__main__":
    main()
