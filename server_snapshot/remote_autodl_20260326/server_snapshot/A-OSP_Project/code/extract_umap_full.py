"""
Full-scale UMAP Feature Extraction — Figure 3 Data
====================================================
For N (default=200) real COCO images, capture the entity-prediction-moment
hidden state at layers {5, 15, 24} under Base and A-OSP modes.

Output shape per file: {layer_idx: [N, 3584]} — one entity-token hidden state
per sample, ready for UMAP dimensionality reduction.

Usage:
  python extract_umap_full.py                     # 200 samples (full)
  python extract_umap_full.py --num_samples 10    # quick validation
"""

import os, sys, json, io, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(line_buffering=True)  # force line-buffered output

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_PATH = PROJECT / "models" / "Qwen2-VL-7B-Instruct"
V_MATRIX_PATH = PROJECT / "models" / "V_matrix.pt"
MANIFEST_PATH = PROJECT / "data" / "blurred_calibration" / "calibration_manifest.json"
OUTPUT_DIR = PROJECT / "data" / "micro_features"
COCO_BASE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

UMAP_LAYERS = [5, 15, 24]
INTERVENTION_LAYER = 24
MAX_NEW_TOKENS = 64
PROMPT = "Describe the image concisely:"

# ── Entity detection (reused from subspace_extractor) ──
DETERMINERS = frozenset({"a", "an", "the", "some", "one", "two", "three",
                         "several", "many", "this", "that", "these", "those"})
TEMPLATE_WORDS = frozenset({
    "image", "picture", "photo", "photograph", "scene", "view", "shot",
    "blurred", "blurry", "unclear", "indistinct", "hazy",
    "it", "is", "shows", "depicts", "appears", "seems", "features",
    "there", "are", "was", "were", "has", "have", "been",
    "not", "very", "quite", "rather", "somewhat",
})


def locate_entity_position(generated_ids, tokenizer):
    decoded = [tokenizer.decode([tid], skip_special_tokens=True).strip().lower()
               for tid in generated_ids]
    for i, tok in enumerate(decoded):
        if tok in DETERMINERS and (i + 1) < len(decoded):
            cand = decoded[i + 1]
            if len(cand) > 2 and cand.isalpha() and cand not in TEMPLATE_WORDS:
                return i + 1
    for i, tok in enumerate(decoded):
        if len(tok) > 3 and tok.isalpha() and tok not in TEMPLATE_WORDS and tok not in DETERMINERS:
            return i
    return max(1, len(decoded) // 3)


# ── Multi-layer capture (all steps, pick entity step post-hoc) ──
class MultiLayerCapture:
    def __init__(self, decoder_layers, layer_indices):
        self.layer_indices = layer_indices
        self.states = {i: [] for i in layer_indices}
        self._prefill_done = {i: False for i in layer_indices}
        self._handles = []
        for idx in layer_indices:
            h = decoder_layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(h)

    def _make_hook(self, li):
        def fn(mod, inp, out):
            h = out[0]
            if not self._prefill_done[li]:
                self._prefill_done[li] = True
                return
            self.states[li].append(h[:, -1, :].detach().float().cpu())
        return fn

    def reset(self):
        for i in self.layer_indices:
            self.states[i].clear()
            self._prefill_done[i] = False

    def remove(self):
        for h in self._handles:
            h.remove()

    def get_at_step(self, li, step):
        """Get hidden state at generation step (0-indexed)."""
        if step < len(self.states[li]):
            return self.states[li][step]   # [1, D]
        elif self.states[li]:
            return self.states[li][-1]
        return None


# ── Lightweight A-OSP hook (intervention only, no trajectory recording) ──
class LightAOSPHook:
    def __init__(self, model, decoder_layers, V_bias, L_prior,
                 alpha=0.5, mu=1.5, beta=0.9, eps=0.1):
        self.model = model
        self.V = V_bias.clone()
        self.alpha, self.mu, self.beta, self.eps = alpha, mu, beta, eps
        self.L_bar = L_prior
        self.L_prior_val = L_prior
        self.t = 0
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0
        self._handle = decoder_layers[INTERVENTION_LAYER].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        h = out[0]
        if not self._prefill_done:
            self._prefill_done = True
            return
        if h.shape[1] != 1:
            return
        self.t += 1
        H = h[:, -1, :]
        V_dev = self.V.to(H.device, dtype=H.dtype)

        if self.N_adaptive is None:
            try:
                norm_l = self.model.model.language_model.norm
                logits = self.model.lm_head(norm_l(H.unsqueeze(1)))
                probs = F.softmax(logits[:, -1, :], dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
                if self.prev_entropy is not None and abs(ent - self.prev_entropy) < self.eps:
                    self.N_adaptive = self.t
                self.prev_entropy = ent
            except Exception:
                self.N_adaptive = self.t
            if self.N_adaptive is None:
                return

        proj = H @ V_dev.T
        L_t = torch.sqrt((proj ** 2).sum(-1)).item()
        if L_t > self.mu * self.L_bar:
            H_proj = H - self.alpha * (proj @ V_dev)
            on = torch.clamp(H.norm(-1, keepdim=True), min=1e-6)
            pn = torch.clamp(H_proj.norm(-1, keepdim=True), min=1e-6)
            h[:, -1, :] = H_proj / pn * on
            self.interventions += 1
        elif self.t > (self.N_adaptive or 0):
            self.L_bar = self.beta * self.L_bar + (1 - self.beta) * L_t

    def reset(self):
        self.t = 0
        self.L_bar = self.L_prior_val
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0

    def remove(self):
        self._handle.remove()


# ── Image fetcher ──
def fetch_images(coco_ids):
    import requests
    images = {}
    sess = requests.Session()
    for cid in coco_ids:
        url = COCO_BASE_URL.format(cid)
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        images[cid] = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return images


def build_inputs(image, processor):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(msgs)
        return processor(text=[text], images=imgs, videos=vids,
                         return_tensors="pt", padding=True)
    except ImportError:
        return processor(text=[text], images=[image],
                         return_tensors="pt", padding=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    N = args.num_samples
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("Loading model ...")
    try:
        import flash_attn; attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"; print("[WARN] SDPA fallback")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model.eval()
    decoder_layers = model.model.language_model.layers
    tokenizer = processor.tokenizer

    ckpt = torch.load(str(V_MATRIX_PATH), map_location="cpu", weights_only=True)
    V_bias = ckpt["V_bias"]
    L_prior = ckpt.get("L_prior", 100.0)
    print(f"V_bias: {V_bias.shape}, L_prior={L_prior:.2f}")

    manifest = json.load(open(MANIFEST_PATH))
    coco_ids = [item["coco_image_id"] for item in manifest["items"][:N]]

    # ── Phase A: download all images to local cache first ──
    IMG_CACHE = OUTPUT_DIR / "coco_img_cache"
    IMG_CACHE.mkdir(exist_ok=True)
    print(f"Phase A: Downloading {len(coco_ids)} images to {IMG_CACHE} ...")
    import requests
    sess = requests.Session()
    for ci, cid in enumerate(coco_ids):
        fpath = IMG_CACHE / f"{cid}.jpg"
        if fpath.exists():
            continue
        url = COCO_BASE_URL.format(cid)
        resp = sess.get(url, timeout=60)
        resp.raise_for_status()
        fpath.write_bytes(resp.content)
        if (ci + 1) % 20 == 0:
            print(f"  Downloaded {ci+1}/{len(coco_ids)}")
    print(f"Phase A complete: all images cached locally.")

    # ── Phase B: extract features with checkpoint resume ──
    CKPT_PATH = OUTPUT_DIR / "umap_checkpoint.pt"
    start_idx = 0
    base_entity_h = {li: [] for li in UMAP_LAYERS}
    aosp_entity_h = {li: [] for li in UMAP_LAYERS}
    meta = []

    if CKPT_PATH.exists():
        ckpt_data = torch.load(str(CKPT_PATH), weights_only=False)
        start_idx = ckpt_data["next_idx"]
        base_entity_h = ckpt_data["base_entity_h"]
        aosp_entity_h = ckpt_data["aosp_entity_h"]
        meta = ckpt_data["meta"]
        print(f"Resumed from checkpoint: starting at index {start_idx}")

    capture = MultiLayerCapture(decoder_layers, UMAP_LAYERS)
    aosp_hook = LightAOSPHook(model, decoder_layers, V_bias, L_prior)

    t0 = time.time()
    for global_idx in range(start_idx, len(coco_ids)):
        cid = coco_ids[global_idx]
        image = Image.open(IMG_CACHE / f"{cid}.jpg").convert("RGB")
        inputs = build_inputs(image, processor)
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # ── BASE pass ──
        capture.reset()
        out_base = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_ids_base = out_base[0, input_len:]
        ent_pos_base = locate_entity_position(gen_ids_base, tokenizer)
        cap_step_base = max(0, ent_pos_base - 1)

        for li in UMAP_LAYERS:
            h = capture.get_at_step(li, cap_step_base)
            base_entity_h[li].append(h if h is not None else torch.zeros(1, 3584))

        # ── A-OSP pass ──
        capture.reset()
        aosp_hook.reset()
        out_aosp = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_ids_aosp = out_aosp[0, input_len:]
        ent_pos_aosp = locate_entity_position(gen_ids_aosp, tokenizer)
        cap_step_aosp = max(0, ent_pos_aosp - 1)

        for li in UMAP_LAYERS:
            h = capture.get_at_step(li, cap_step_aosp)
            aosp_entity_h[li].append(h if h is not None else torch.zeros(1, 3584))

        base_txt = tokenizer.decode(gen_ids_base, skip_special_tokens=True)
        aosp_txt = tokenizer.decode(gen_ids_aosp, skip_special_tokens=True)
        meta.append({"coco_id": cid, "base_text": base_txt, "aosp_text": aosp_txt,
                     "ent_pos_base": ent_pos_base, "ent_pos_aosp": ent_pos_aosp,
                     "aosp_interventions": aosp_hook.interventions})

        if (global_idx + 1) % 10 == 0 or global_idx == 0:
            elapsed = time.time() - t0
            done_in_this_run = global_idx - start_idx + 1
            eta = elapsed / done_in_this_run * (len(coco_ids) - global_idx - 1)
            print(f"[{global_idx+1:>3}/{len(coco_ids)}] "
                  f"ent_base={ent_pos_base:>2} ent_aosp={ent_pos_aosp:>2} "
                  f"intv={aosp_hook.interventions:>2} "
                  f"| {base_txt[:60]}... "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # ── Checkpoint every 20 samples ──
        if (global_idx + 1) % 20 == 0:
            torch.save({
                "next_idx": global_idx + 1,
                "base_entity_h": base_entity_h,
                "aosp_entity_h": aosp_entity_h,
                "meta": meta,
            }, str(CKPT_PATH))
            print(f"  [Checkpoint saved at {global_idx+1}]")

    capture.remove()
    aosp_hook.remove()
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()

    # ── Stack and save ──
    base_out = {li: torch.cat(base_entity_h[li], dim=0) for li in UMAP_LAYERS}
    aosp_out = {li: torch.cat(aosp_entity_h[li], dim=0) for li in UMAP_LAYERS}

    base_path = OUTPUT_DIR / "umap_features_base_full.pt"
    aosp_path = OUTPUT_DIR / "umap_features_aosp_full.pt"
    meta_path = OUTPUT_DIR / "umap_meta_full.json"

    torch.save(base_out, base_path)
    torch.save(aosp_out, aosp_path)
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("FULL UMAP EXTRACTION COMPLETE")
    print(f"{'='*60}")
    for li in UMAP_LAYERS:
        print(f"  Layer {li}: base={base_out[li].shape}, aosp={aosp_out[li].shape}")
    print(f"\n  {base_path}")
    print(f"  {aosp_path}")
    print(f"  {meta_path}")
    total = time.time() - t0
    print(f"  Total time: {total:.0f}s ({total/len(coco_ids):.1f}s/sample)")


if __name__ == "__main__":
    main()
