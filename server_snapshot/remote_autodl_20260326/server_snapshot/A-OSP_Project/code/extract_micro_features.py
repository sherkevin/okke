"""
A-OSP Micro Feature Extractor — Figure 3 (UMAP) & Figure 4 (Energy Trajectory)
================================================================================
For each sample image:
  1. BASE mode: free-generate, capture hidden states at layers {5, 15, 24}
  2. A-OSP mode: intervened generate, capture:
     - Per-step energy trajectory (L_t, L_bar_t, triggered) → CSV for Figure 4
     - Hidden states at layers {5, 15, 24} → .pt for Figure 3 UMAP

Usage:
  python extract_micro_features.py                      # 1-sample mini-batch
  python extract_micro_features.py --num_samples 50     # expand to 50 samples
"""

import os
import sys
import csv
import json
import argparse
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ── Project paths ──────────────────────────────────────────────
PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_PATH = PROJECT / "models" / "Qwen2-VL-7B-Instruct"
V_MATRIX_PATH = PROJECT / "models" / "V_matrix.pt"
MANIFEST_PATH = PROJECT / "data" / "blurred_calibration" / "calibration_manifest.json"
OUTPUT_DIR = PROJECT / "data" / "micro_features"
COCO_BASE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

UMAP_LAYERS = [5, 15, 24]     # shallow → mid → deep (semantic decoupling zone)
INTERVENTION_LAYER = 24        # layer[-4] for Qwen2-VL-7B (28 layers)
MAX_NEW_TOKENS = 128           # longer generation to see energy dynamics
PROMPT = "Describe the image in detail:"


# ═══════════════════════════════════════════════════════════════
#  Multi-Layer Hidden State Capture
# ═══════════════════════════════════════════════════════════════
class MultiLayerCapture:
    """
    Registers hooks on multiple decoder layers simultaneously.
    Captures the last text token's hidden state at each generation step.
    Skips the prefill pass automatically.
    """

    def __init__(self, decoder_layers, layer_indices: list[int]):
        self.layer_indices = layer_indices
        self.states: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
        self._prefill_done: dict[int, bool] = {i: False for i in layer_indices}
        self._handles = []
        for idx in layer_indices:
            h = decoder_layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(h)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            hidden = out[0]
            if not self._prefill_done[layer_idx]:
                self._prefill_done[layer_idx] = True
                return
            h = hidden[:, -1, :].detach().float().cpu()  # [1, D]
            self.states[layer_idx].append(h)
        return hook_fn

    def reset(self):
        for idx in self.layer_indices:
            self.states[idx].clear()
            self._prefill_done[idx] = False

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_stacked(self, layer_idx: int) -> torch.Tensor:
        """Returns [num_steps, D] for a given layer."""
        if not self.states[layer_idx]:
            return torch.empty(0)
        return torch.cat(self.states[layer_idx], dim=0)


# ═══════════════════════════════════════════════════════════════
#  A-OSP Intervention Hook with Energy Trajectory Recording
# ═══════════════════════════════════════════════════════════════
class AOSPTrajectoryHook:
    """
    Combines A-OSP intervention with per-step energy recording.
    Avoids torch.compile for this diagnostic mode to prevent
    graph recompilation issues in the multi-hook setup.
    """

    def __init__(self, model, decoder_layers, V_bias: torch.Tensor,
                 layer_idx: int = INTERVENTION_LAYER,
                 alpha: float = 0.5, mu: float = 1.5, beta: float = 0.9,
                 epsilon_steady: float = 0.1, L_prior: float = 100.0):
        self.model = model
        self.V = V_bias.clone()
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.epsilon_steady = epsilon_steady
        self.L_prior_val = L_prior

        self.trajectory: list[dict] = []
        self.t = 0
        self.L_bar = L_prior
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0

        self._handle = decoder_layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        hidden = out[0]
        if not self._prefill_done:
            self._prefill_done = True
            return
        if hidden.shape[1] != 1:
            return

        self.t += 1
        H = hidden[:, -1, :]     # [1, D]
        V_dev = self.V.to(H.device, dtype=H.dtype)

        # ── Entropy burn-in ──
        burned_in = self.N_adaptive is not None
        if not burned_in:
            ent = self._entropy_from_hidden(H)
            if ent is not None and self.prev_entropy is not None:
                if abs(ent - self.prev_entropy) < self.epsilon_steady:
                    self.N_adaptive = self.t
                    burned_in = True
            if ent is not None:
                self.prev_entropy = ent
            if not burned_in:
                self.trajectory.append({
                    "step": self.t, "L_t": 0.0, "L_bar": self.L_bar,
                    "triggered": False, "burn_in": True,
                })
                return

        # ── Projection energy ──
        proj = H @ V_dev.T                                          # [1, K]
        L_t = torch.sqrt((proj ** 2).sum(dim=-1)).item()            # scalar

        triggered = L_t > self.mu * self.L_bar

        if triggered:
            H_proj = H - self.alpha * (proj @ V_dev)
            orig_norm = torch.clamp(H.norm(dim=-1, keepdim=True), min=1e-6)
            proj_norm = torch.clamp(H_proj.norm(dim=-1, keepdim=True), min=1e-6)
            hidden[:, -1, :] = H_proj / proj_norm * orig_norm
            self.interventions += 1
        else:
            if self.t > (self.N_adaptive or 0):
                self.L_bar = self.beta * self.L_bar + (1.0 - self.beta) * L_t

        self.trajectory.append({
            "step": self.t, "L_t": L_t, "L_bar": self.L_bar,
            "triggered": triggered, "burn_in": False,
        })

    def _entropy_from_hidden(self, H: torch.Tensor):
        """Compute Shannon entropy via the model's lm_head."""
        try:
            norm_layer = self.model.model.language_model.norm
            lm_head = self.model.lm_head
            with torch.no_grad():
                normed = norm_layer(H.unsqueeze(1))
                logits = lm_head(normed)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        except Exception:
            return None

    def reset(self):
        self.trajectory.clear()
        self.t = 0
        self.L_bar = self.L_prior_val
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0

    def remove(self):
        self._handle.remove()


# ═══════════════════════════════════════════════════════════════
#  Utility: Download COCO Image
# ═══════════════════════════════════════════════════════════════
def download_coco_image(coco_id: int) -> Image.Image:
    import requests
    url = COCO_BASE_URL.format(coco_id)
    print(f"  Downloading COCO val2014 image ID {coco_id} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def get_coco_ids(num: int) -> list[int]:
    """Read first N COCO IDs from the calibration manifest."""
    manifest = json.load(open(MANIFEST_PATH))
    return [item["coco_image_id"] for item in manifest["items"][:num]]


# ═══════════════════════════════════════════════════════════════
#  Generation Helper
# ═══════════════════════════════════════════════════════════════
def build_inputs(image: Image.Image, processor):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           return_tensors="pt", padding=True)
    except ImportError:
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt", padding=True)
    return inputs


@torch.no_grad()
def generate_and_decode(model, processor, inputs, max_new_tokens=MAX_NEW_TOKENS):
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = output_ids[0, input_len:]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_tokens = [processor.tokenizer.decode([tid], skip_special_tokens=True)
                  for tid in gen_ids]
    return gen_text, gen_tokens


# ═══════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("Loading model ...")
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
        print("[WARN] flash_attn not found, using SDPA")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation=attn,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model.eval()

    decoder_layers = model.model.language_model.layers
    print(f"Decoder layers: {len(decoder_layers)}, UMAP layers: {UMAP_LAYERS}")

    # ── Load V_matrix ──
    ckpt = torch.load(str(V_MATRIX_PATH), map_location="cpu", weights_only=True)
    V_bias = ckpt["V_bias"]  # [K, D]
    L_prior = ckpt.get("L_prior", 1.0)
    print(f"V_bias loaded: {V_bias.shape}, EVR={ckpt['evr']:.4f}, L_prior={L_prior:.2f}")

    # ── Get COCO IDs ──
    coco_ids = get_coco_ids(args.num_samples)
    print(f"Processing {len(coco_ids)} sample(s): {coco_ids}")

    all_umap_features = {}
    all_trajectories = []

    for sample_idx, coco_id in enumerate(coco_ids):
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx+1}/{len(coco_ids)}: COCO ID {coco_id}")
        print(f"{'='*60}")

        image = download_coco_image(coco_id)
        inputs = build_inputs(image, processor)

        # ─────────── Pass 1: BASE mode ───────────
        print("\n[BASE] Free generation (no intervention) ...")
        capture_base = MultiLayerCapture(decoder_layers, UMAP_LAYERS)
        base_text, base_tokens = generate_and_decode(model, processor, inputs)
        print(f"  Generated {len(base_tokens)} tokens: {base_text[:120]}...")

        base_features = {}
        for li in UMAP_LAYERS:
            base_features[li] = capture_base.get_stacked(li)
            print(f"  Layer {li}: {base_features[li].shape}")
        capture_base.remove()

        # ─────────── Pass 2: A-OSP mode ───────────
        print("\n[A-OSP] Intervened generation ...")
        capture_aosp = MultiLayerCapture(decoder_layers, UMAP_LAYERS)
        traj_hook = AOSPTrajectoryHook(model, decoder_layers, V_bias, L_prior=L_prior)

        aosp_text, aosp_tokens = generate_and_decode(model, processor, inputs)
        print(f"  Generated {len(aosp_tokens)} tokens: {aosp_text[:120]}...")
        print(f"  Interventions triggered: {traj_hook.interventions}")

        aosp_features = {}
        for li in UMAP_LAYERS:
            aosp_features[li] = capture_aosp.get_stacked(li)
            print(f"  Layer {li}: {aosp_features[li].shape}")

        trajectory = traj_hook.trajectory
        capture_aosp.remove()
        traj_hook.remove()

        # ─────────── Save Energy Trajectory CSV ───────────
        csv_path = OUTPUT_DIR / f"energy_trajectory_sample_{sample_idx+1}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "L_t", "L_bar", "triggered", "burn_in", "token"
            ])
            writer.writeheader()
            for row in trajectory:
                step_idx = row["step"] - 1
                tok = aosp_tokens[step_idx] if step_idx < len(aosp_tokens) else ""
                writer.writerow({**row, "token": tok.strip()})
        print(f"\n  Energy trajectory → {csv_path}")

        # ─────────── Accumulate UMAP features ───────────
        sample_key = f"coco_{coco_id}"
        all_umap_features[sample_key] = {
            "base": {li: base_features[li] for li in UMAP_LAYERS},
            "aosp": {li: aosp_features[li] for li in UMAP_LAYERS},
            "base_tokens": base_tokens,
            "aosp_tokens": aosp_tokens,
            "base_text": base_text,
            "aosp_text": aosp_text,
        }
        all_trajectories.append({
            "coco_id": coco_id,
            "trajectory": trajectory,
            "aosp_tokens": aosp_tokens,
        })

    # ─────────── Save UMAP features ───────────
    suffix = "mini" if args.num_samples <= 5 else f"n{args.num_samples}"
    pt_path = OUTPUT_DIR / f"umap_features_{suffix}.pt"
    torch.save(all_umap_features, pt_path)
    print(f"\nUMAP features → {pt_path}")

    # ─────────── Summary ───────────
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for sample_key, feats in all_umap_features.items():
        print(f"\n{sample_key}:")
        print(f"  BASE text:  {feats['base_text'][:100]}...")
        print(f"  A-OSP text: {feats['aosp_text'][:100]}...")
        for li in UMAP_LAYERS:
            b_shape = feats["base"][li].shape
            a_shape = feats["aosp"][li].shape
            print(f"  Layer {li:>2}: base={b_shape}, aosp={a_shape}")
        traj = [t for t in all_trajectories if t["coco_id"] == int(sample_key.split("_")[1])][0]
        n_triggered = sum(1 for r in traj["trajectory"] if r.get("triggered"))
        n_burnin = sum(1 for r in traj["trajectory"] if r.get("burn_in"))
        print(f"  Trajectory: {len(traj['trajectory'])} steps, "
              f"{n_burnin} burn-in, {n_triggered} interventions")


if __name__ == "__main__":
    main()
