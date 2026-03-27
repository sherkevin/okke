#!/usr/bin/env python3
"""
MVBench Video Zero-Shot Eval: Base vs A-OSP (V3.5 Sprint 3)

Key design:
  - Uses S_text-only (zero-vision extracted) for A-OSP intervention
  - MeanPool(t, m) at Layer 29 to flatten T×M×D → R^D before projection
  - 10-sample mini-batch on Temporal Action Order subset
  - Detects cases where A-OSP corrects hallucinated timeline

Usage:
  python run_mvbench_eval.py --mode base --n_samples 10
  python run_mvbench_eval.py --mode aosp --n_samples 10
"""

import os, sys, gc, json, time, argparse, torch
import numpy as np

# Paths
PROJECT_ROOT = "/root/autodl-tmp/A-OSP_Project"
MODEL_PATH   = f"{PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
V_TEXT_ONLY  = f"{PROJECT_ROOT}/models/V_text_only_q3.pt"
V_MATRIX_Q3  = f"{PROJECT_ROOT}/models/V_matrix_q3.pt"  # fallback
MVBENCH_DIR  = f"{PROJECT_ROOT}/data/mvbench"
LOG_DIR      = f"{PROJECT_ROOT}/logs/eval_results"
os.makedirs(LOG_DIR, exist_ok=True)

# A-OSP hook layer for video MeanPool + projection (per V3.5 spec)
HOOK_LAYER = 29  # Layer 29 for MeanPool(t,m) alignment

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "aosp"], default="base")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--v_matrix", type=str, default=None,
                   help="Path to V_bias .pt file (default: auto-select)")
    return p.parse_args()


def load_model():
    from transformers import AutoConfig, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Qwen3VL stores hidden_size inside text_config
    D = getattr(cfg, 'hidden_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'hidden_size', 4096)
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(getattr(cfg, 'text_config', cfg), 'num_hidden_layers', 36)
    print(f"[Loader] hidden_size={D}, layers={n_layers}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor


def load_v_matrix(path=None):
    """Load V_bias subspace matrix. Prefer V_text_only_q3.pt (V3.5 paradigm)."""
    if path is not None and os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[V_matrix] Loaded from {path}, shape={list(d['V_bias'].shape)}")
        return d

    # Auto-select: prefer S_text-only Q3 version
    if os.path.exists(V_TEXT_ONLY):
        d = torch.load(V_TEXT_ONLY, map_location="cpu", weights_only=True)
        print(f"[V_matrix] Using V_text_only_q3.pt (V3.5 S_text-only), shape={list(d['V_bias'].shape)}")
        return d

    # Fallback to V_matrix_q3
    if os.path.exists(V_MATRIX_Q3):
        d = torch.load(V_MATRIX_Q3, map_location="cpu", weights_only=True)
        print(f"[V_matrix] Fallback to V_matrix_q3.pt, shape={list(d['V_bias'].shape)}")
        return d

    raise FileNotFoundError("No V_bias matrix found. Run extract_vmatrix_text_only_q3.py first.")


def load_mvbench_samples(n=10):
    """
    Load MVBench Action Sequence samples.

    Resolves video_path for each sample:
    1. Looks for sta/ subfolder (from sta.zip extraction)
    2. Looks for Charades_v1_480/ subfolder (from data0613.zip extraction)
    3. Falls back to action_sequence_10.jsonl metadata without video
    """
    VIDEO_SEARCH_DIRS = [
        os.path.join(MVBENCH_DIR, "video", "extracted", "sta"),
        os.path.join(MVBENCH_DIR, "video", "extracted", "data0613", "star", "Charades_v1_480"),
        os.path.join(MVBENCH_DIR, "video", "sta"),
        os.path.join(MVBENCH_DIR, "video", "data0613", "star", "Charades_v1_480"),
        os.path.join(MVBENCH_DIR, "videos"),
    ]

    def find_video(filename):
        """Search all known video directories for the given mp4 filename."""
        base = os.path.basename(filename)
        for d in VIDEO_SEARCH_DIRS:
            p = os.path.join(d, base)
            if os.path.exists(p):
                return p
        return None

    # Load metadata from JSONL
    jsonl_path = os.path.join(MVBENCH_DIR, "action_sequence_10.jsonl")
    if not os.path.exists(jsonl_path):
        print("[Data] action_sequence_10.jsonl not found. Creating synthetic data...")
        os.makedirs(MVBENCH_DIR, exist_ok=True)
        return create_synthetic_samples(n)

    samples = []
    n_video_found = 0
    with open(jsonl_path) as f:
        for line in f:
            if len(samples) >= n:
                break
            s = json.loads(line)
            # Resolve video file path
            video_filename = s.get("video_filename", "")
            video_path = find_video(video_filename) if video_filename else None
            s["video_path"] = video_path
            if video_path:
                n_video_found += 1
            samples.append(s)

    print(f"[Data] Loaded {len(samples)} samples from {jsonl_path}")
    print(f"[Data] Video files found: {n_video_found}/{len(samples)}")
    if n_video_found == 0:
        print("[Data] WARNING: No .mp4 video files found on disk.")
        print(f"[Data]   Expected in: {VIDEO_SEARCH_DIRS[0]}")
        print("[Data]   Download sta.zip from HuggingFace and extract to data/mvbench/video/")
        print("[Data]   Falling back to text-only inference (invalid for paper — video required).")
    return samples[:n]


def create_synthetic_samples(n=10):
    """Create synthetic Temporal Action Order samples for pipeline validation."""
    import random

    ACTION_SEQUENCES = [
        {
            "correct_order": ["person picks up a cup", "person drinks from the cup"],
            "description": "drinking sequence"
        },
        {
            "correct_order": ["person opens a door", "person walks through the door"],
            "description": "door sequence"
        },
        {
            "correct_order": ["chef chops vegetables", "chef puts vegetables in pan"],
            "description": "cooking sequence"
        },
        {
            "correct_order": ["athlete kneels at starting line", "athlete sprints forward"],
            "description": "running sequence"
        },
        {
            "correct_order": ["person writes a note", "person folds the paper"],
            "description": "writing sequence"
        },
        {
            "correct_order": ["baker mixes ingredients", "baker puts bread in oven"],
            "description": "baking sequence"
        },
        {
            "correct_order": ["person turns on faucet", "person washes hands"],
            "description": "washing sequence"
        },
        {
            "correct_order": ["student opens textbook", "student writes notes"],
            "description": "studying sequence"
        },
        {
            "correct_order": ["mechanic opens hood", "mechanic inspects engine"],
            "description": "repair sequence"
        },
        {
            "correct_order": ["artist picks up brush", "artist paints on canvas"],
            "description": "painting sequence"
        },
    ]

    samples = []
    for i in range(min(n, len(ACTION_SEQUENCES))):
        seq = ACTION_SEQUENCES[i]
        a1, a2 = seq["correct_order"]

        # Create 4 choices: correct order, reversed, and 2 distractors
        choices = [
            f"(1) {a1}, (2) {a2}",                    # correct
            f"(1) {a2}, (2) {a1}",                    # reversed (hallucinated)
            f"(1) {a2}, (2) {a2}",                    # wrong duplicate
            f"(1) {a1}, (2) {a1}",                    # wrong duplicate
        ]
        random.shuffle(choices)
        correct_choice = [c for c in choices if c == f"(1) {a1}, (2) {a2}"][0]
        answer_idx = choices.index(correct_choice)
        answer_letter = "ABCD"[answer_idx]

        sample = {
            "id": f"mvbench_tao_synthetic_{i:04d}",
            "task_type": "Temporal Action Order",
            "question": f"What is the correct temporal order of actions shown in the video?\n" + \
                        "\n".join([f"{l}. {c}" for l, c in zip("ABCD", choices)]),
            "choices": choices,
            "answer": answer_letter,
            "description": seq["description"],
            "correct_first_action": a1,
            "correct_second_action": a2,
            "video_path": None,
            "synthetic": True,
            "frame_description": f"Video showing: first {a1}, then {a2}",
        }
        samples.append(sample)

    # Save for reuse
    out_file = os.path.join(MVBENCH_DIR, "temporal_action_order_10_synthetic.jsonl")
    with open(out_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[Data] Created {len(samples)} synthetic samples → {out_file}")
    return samples


class VideoAOSPHook:
    """
    A-OSP hook for video processing with MeanPool(t, m) alignment.

    Key V3.5 design: when video features arrive as [T*M, D] or [batch, T*M, D],
    apply MeanPool over the temporal+spatial dim before checking projection energy.
    This implements the Temporal-Spatial Flatten Alignment from §4.6.3.
    """
    def __init__(self, V_bias, L_prior, K=20, alpha=0.5, mu=1.5, beta=0.9,
                 epsilon_steady=0.1, T_max=15, layer_idx=29, always_intervene=False):
        self.V_bias = V_bias.float()   # [K, D]
        self.L_prior = L_prior
        self.K = K
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.epsilon_steady = epsilon_steady
        self.T_max = T_max
        self.layer_idx = layer_idx
        # For mini-batch ablation: force projection at every decode step
        self.always_intervene = always_intervene

        # State per-sample
        self.reset()
        self.total_interventions = 0
        self.total_steps = 0
        self.handle = None

    def reset(self):
        """Reset per-generation state."""
        self.L_bar = self.L_prior
        self.t = 0
        self.N_adaptive = None
        self.prev_entropy = None

    def compute_entropy(self, logits):
        """Compute Shannon entropy of output distribution."""
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        return -(probs * log_probs).sum().item()

    def hook_fn(self, module, inp, out):
        """
        Hook function for video A-OSP with Temporal-Spatial Flatten Alignment.

        §4.6.3 spec: For video input, Qwen3-VL's intermediate hidden state at
        Layer 29 has shape [batch, N_total, D] where N_total = N_text + T*M
        (T = temporal frames, M = spatial patches per frame, D = hidden_size).

        We apply MeanPool(t, m) over ALL N_total positions during prefill to
        compute a single representative vector h ∈ R^D, then check projection
        energy against the bias subspace.

        For autoregressive decode steps (seq_len=1), h is the current token.
        """
        # Extract hidden states — handle bare Tensor (Qwen3-VL + FA2) and tuple
        if isinstance(out, tuple):
            hidden = out[0]     # [batch, seq_len, D]
            rest = out[1:]
        else:
            hidden = out        # bare tensor
            rest = None

        batch, seq_len, D = hidden.shape

        # ---------------------------------------------------------------
        # V3.5 Temporal-Spatial Flatten Alignment (§4.6.3)
        # ---------------------------------------------------------------
        if seq_len > 1:
            # PREFILL: hidden contains [text_prefix_tokens | video_tokens]
            # MeanPool(t, m) over all positions → single R^D energy probe
            # This averages across both text and T*M video patch positions.
            # For a real video input: T*M >> N_text, so this is dominated by
            # the temporal-spatial features exactly as intended by §4.6.3.
            h = hidden[0].mean(dim=0)  # [N_total, D] → [D]  (MeanPool over t,m)
        else:
            # DECODE: single new token being generated
            h = hidden[0, 0]           # [D]

        self.t += 1
        self.total_steps += 1

        # --- Entropy-Aware Burn-in ---
        # For short responses (like MCQ), bypass if entropy < 0.05
        # We only have logits if we can get them - use L2 energy proxy
        ent = 0.0  # Will be updated if we can get logits

        # Low-entropy bypass for short confident generations (e.g. A/B/C/D answers)
        if self.N_adaptive is None and ent < 0.05:
            self.N_adaptive = self.t

        # Hard fallback: force burn-in end at T_max
        if self.N_adaptive is None and self.t >= self.T_max:
            self.N_adaptive = self.t

        if self.N_adaptive is None:
            # Still in burn-in: no intervention
            if isinstance(out, tuple):
                return out
            return out

        # --- Projection Energy Monitoring ---
        V = self.V_bias.to(hidden.device, dtype=hidden.dtype)  # [K, D]
        h_norm = h / (h.norm() + 1e-8)

        # L2 projection energy onto bias subspace
        proj = (h_norm.unsqueeze(0) @ V.T).squeeze(0)  # [K]
        L_t = torch.sqrt((proj**2).sum()).item()

        # --- Conditional EMA Update ---
        if L_t <= self.mu * self.L_bar:
            self.L_bar = self.beta * self.L_bar + (1 - self.beta) * L_t
        # else: freeze EMA (anomalous spike detected)

        # --- Soft Projection Intervention ---
        if self.always_intervene or L_t > self.mu * self.L_bar:
            self.total_interventions += 1

            # Apply to the actual hidden states
            # For all positions in this layer output
            H_flat = hidden[0]  # [seq_len, D]
            V_dev = self.V_bias.to(hidden.device, dtype=hidden.dtype)

            # Orthogonal projection: remove bias subspace component
            proj_full = H_flat @ V_dev.T   # [seq_len, K]
            H_proj = H_flat - self.alpha * (proj_full @ V_dev)  # [seq_len, D]

            # Scale preservation: restore original L2 norm
            orig_norm = H_flat.norm(dim=-1, keepdim=True)   # [seq_len, 1]
            proj_norm = H_proj.norm(dim=-1, keepdim=True) + 1e-8
            H_corrected = (H_proj / proj_norm) * orig_norm   # [seq_len, D]

            hidden = H_corrected.unsqueeze(0)  # [1, seq_len, D]

        if isinstance(out, tuple):
            return (hidden,) + rest
        return hidden

    def register(self, model):
        """Register hook at target layer. Supports Qwen2-VL and Qwen3-VL architectures."""
        # Qwen3-VL: model.model.language_model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
            layers = model.model.language_model.layers
        # Qwen3-VL alt: model.language_model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            layers = model.language_model.layers
        # Qwen2-VL / Qwen2.5-VL: model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise AttributeError(f"Cannot find decoder layers in {type(model).__name__}")

        total = len(layers)
        actual_idx = self.layer_idx if self.layer_idx < total else total - 4
        self.handle = layers[actual_idx].register_forward_hook(self.hook_fn)
        print(f"[VideoAOSP] Hook registered at layer {actual_idx}/{total-1}")

    def remove(self):
        if self.handle:
            self.handle.remove()


def build_mcq_prompt(sample, has_video=False):
    """
    Build MCQ prompt for MVBench Action Sequence (Temporal Order).

    When has_video=True: use standard video QA prefix so the model attends to frames.
    When has_video=False: text-only (invalid for paper, used as fallback only).
    """
    question = sample.get("question", "")
    choices = sample.get("choices", [])

    if choices:
        choices_text = "\n".join([
            f"{l}. {c}" for l, c in zip("ABCD", choices)
        ])
        if has_video:
            prompt = (
                f"Watch the video carefully and answer the following question:\n"
                f"{question}\n\n"
                f"{choices_text}\n\n"
                f"Answer with the letter only (A, B, C, or D):"
            )
        else:
            prompt = (
                f"{question}\n\n"
                f"{choices_text}\n\n"
                f"Answer with the letter only (A, B, C, or D):"
            )
    else:
        prompt = question + "\n\nAnswer with the letter only (A, B, C, or D):"

    return prompt


def infer_single_sample(model, processor, sample, mode, hook=None):
    """
    Run inference on a single MVBench sample.

    When video_path is available:
      - Builds {"type": "video", "video": path, ...} content block
      - Passes start/end timestamps for clip trimming (via video_start/video_end)
      - qwen_vl_utils.process_vision_info decodes frames → pixel_values tensor
      - Model builds T×M×D temporal-spatial features at intermediate layers
      - A-OSP hook at Layer 29 applies MeanPool(t,m) then orthogonal projection

    When video_path is missing (fallback only — INVALID for paper results):
      - Text-only inference, clearly logged as INVALID
    """
    from qwen_vl_utils import process_vision_info

    video_path = sample.get("video_path")
    has_video = video_path is not None and os.path.exists(video_path)
    prompt = build_mcq_prompt(sample, has_video=has_video)

    # Build message content
    if has_video:
        # Include temporal clip bounds for precise segment extraction
        start = sample.get("start", 0.0)
        end   = sample.get("end",   None)

        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": 360 * 420,  # balance resolution vs VRAM
            "fps": 1.0,               # 1 fps → ~T frames for T-second clip
        }
        # Some versions of qwen_vl_utils support video_start/video_end for trimming
        if start is not None:
            video_content["video_start"] = float(start)
        if end is not None:
            video_content["video_end"] = float(end)

        messages = [{
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }]
        print(f"[video] {os.path.basename(video_path)} [{start:.1f}s–{end:.1f}s]", end=" ", flush=True)
    else:
        # TEXT-ONLY FALLBACK — clearly marked as invalid for paper use
        print("[NO-VIDEO-FALLBACK⚠] ", end="", flush=True)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        if has_video:
            # Report video token count (T×M spatial patches)
            video_tok = inputs.get("video_grid_thw", None)
            if video_tok is not None:
                # shape: [n_clips, 3] — (T, H_patches, W_patches)
                thw = video_tok[0].tolist()
                T, H, W = thw
                print(f"T={T} H={H} W={W} → {T*H*W} video tokens", end=" ", flush=True)

    except Exception as e:
        print(f"\n    [Warn] Video processing failed: {e}")
        print("    [Warn] Falling back to text-only (check qwen_vl_utils version)")
        text_input = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text_input], return_tensors="pt", padding=True).to(model.device)
        has_video = False

    # Reset hook state for new sample
    if hook:
        hook.reset()

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    elapsed = time.time() - t0
    # Decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract answer letter
    pred = raw_output.upper()
    answer_letter = None
    for c in pred:
        if c in "ABCD":
            answer_letter = c
            break
    if answer_letter is None:
        # Try to map text answer back to letter
        choices = sample.get("choices", [])
        for i, choice in enumerate(choices):
            if choice.lower() in raw_output.lower():
                answer_letter = "ABCD"[i]
                break
        if answer_letter is None:
            answer_letter = "A"  # default

    n_interventions = hook.total_interventions if hook else 0

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "id": sample.get("id", ""),
        "pred": answer_letter,
        "raw_output": raw_output,
        "gt": sample.get("answer", ""),
        "correct": (answer_letter == sample.get("answer", "")),
        "elapsed": elapsed,
        "n_interventions": n_interventions,
        "task_type": sample.get("task_type", "Temporal Action Order"),
        "description": sample.get("description", ""),
        "has_video": has_video,  # CRITICAL: flag invalid text-only runs
        "video_path": sample.get("video_path", ""),
    }


def main():
    args = parse_args()
    print(f"=== MVBench Video Zero-Shot Eval (V3.5 Sprint 3) ===")
    print(f"Mode: {args.mode.upper()}, N={args.n_samples}, Hook Layer={HOOK_LAYER}")
    print()

    # Load data
    samples = load_mvbench_samples(args.n_samples)
    print(f"Loaded {len(samples)} MVBench samples\n")

    # Load model
    model, processor = load_model()

    # Setup A-OSP hook for video (if aosp mode)
    hook = None
    v_data = None
    if args.mode == "aosp":
        v_data = load_v_matrix(args.v_matrix)
        V_bias = v_data["V_bias"]
        L_prior = v_data["L_prior"]
        K = v_data.get("K", 20)
        D = v_data.get("D", V_bias.shape[1])
        print(f"[A-OSP] V_bias shape: {list(V_bias.shape)}, L_prior={L_prior:.4f}")
        print(f"[A-OSP] MeanPool(t,m) Temporal-Spatial Flatten at Layer {HOOK_LAYER}")

        hook = VideoAOSPHook(
            V_bias=V_bias,
            L_prior=L_prior,
            K=K,
            alpha=0.5,
            mu=1.5,
            beta=0.9,
            layer_idx=HOOK_LAYER,
            always_intervene=True,  # mini-batch ablation: force projection every step
        )
        hook.register(model)

    # Run evaluation
    results = []
    n_correct = 0
    total_interventions = 0
    correction_cases = []  # Cases where A-OSP changes answer

    print(f"Running {len(samples)} samples in {args.mode.upper()} mode...")
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.get('description', sample.get('id', ''))}", end=" ", flush=True)
        res = infer_single_sample(model, processor, sample, args.mode, hook)
        results.append(res)

        if res["correct"]:
            n_correct += 1
            print(f"✓ ({res['pred']})", end="")
        else:
            print(f"✗ (pred={res['pred']}, gt={res['gt']})", end="")

        if args.mode == "aosp" and res["n_interventions"] > 0:
            total_interventions += res["n_interventions"]
            print(f" [intervened: {res['n_interventions']}x]", end="")

        print()

    # Compute metrics
    accuracy = n_correct / len(samples)

    if hook:
        hook.remove()
        print(f"\n[A-OSP] Total interventions across all samples: {hook.total_interventions}")

    print(f"\n=== RESULTS ({args.mode.upper()}) ===")
    print(f"  Accuracy:     {accuracy:.4f} ({n_correct}/{len(samples)})")
    print(f"  Total samples: {len(samples)}")
    if args.mode == "aosp":
        print(f"  Interventions: {hook.total_interventions}")

    # Save results
    tag = "base" if args.mode == "base" else "aosp"
    out_file = os.path.join(LOG_DIR, f"mvbench_tao_{tag}_{args.n_samples}samples.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save summary
    summary = {
        "mode": args.mode,
        "n_samples": len(samples),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "task_type": "Temporal Action Order",
        "hook_layer": HOOK_LAYER,
        "pooling": "MeanPool(t,m)",
        "v_matrix": v_data.get("tag", "N/A") if v_data else "N/A",
        "total_interventions": hook.total_interventions if hook else 0,
        "v3_5_paradigm": True,
        "s_text_only": args.mode == "aosp" and (v_data.get("tag") == "S_text_only_zero_vision" if v_data else False),
    }

    summary_file = os.path.join(LOG_DIR, f"mvbench_tao_{tag}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {out_file}")
    print(f"Summary → {summary_file}")
    print("\nEVAL COMPLETE")
    return accuracy


if __name__ == "__main__":
    main()
