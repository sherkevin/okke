"""
BRA Smoke Test Matrix
======================
Comprehensive smoke test: model x dataset combinations.

Tests:
  1. Model loading + basic inference
  2. BRA integration (does it hook correctly, does generate() work?)
  3. Dataset loading (can we read samples?)

Models:  Qwen3-VL-2B, LLaVA-1.5-7B, InstructBLIP-Vicuna-7B, (MiniGPT-4 noted as incomplete)
Datasets: COCO, MMMU, HallusionBench, MMBench, MME, FREAK, MVBench, VidHalluc

Run:  python smoke_test_matrix.py
"""

from __future__ import annotations
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from dataclasses import dataclass

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path("/root/autodl-tmp/BRA_Project")
MODELS_DIR = PROJECT / "models"
DATASETS_DIR = PROJECT / "datasets"

COCO_DIR = DATASETS_DIR / "coco2014" / "val2014"
COCO_ANN = DATASETS_DIR / "coco2014" / "annotations"

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    status: str  # PASS, FAIL, SKIP
    message: str = ""
    elapsed: float = 0.0

RESULTS: list[TestResult] = []

def record(name, status, message="", elapsed=0.0):
    RESULTS.append(TestResult(name, status, message, elapsed))
    icon = {"PASS": "[OK]", "FAIL": "[!!]", "SKIP": "[--]"}[status]
    print(f"  {icon} {name}: {message}" if message else f"  {icon} {name}")

HAS_GPU = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Qwen3-VL-2B": {
        "path": MODELS_DIR / "Qwen3-VL-2B-Instruct",
        "class": "Qwen3VLForConditionalGeneration",
        "processor": "AutoProcessor",
        "module": "transformers",
        "supports_video": True,
        "input_builder": "qwen3vl",
    },
    "LLaVA-1.5-7B": {
        "path": MODELS_DIR / "llava-1.5-7b-hf",
        "class": "LlavaForConditionalGeneration",
        "processor": "AutoProcessor",
        "module": "transformers",
        "supports_video": False,
        "input_builder": "llava",
    },
    "InstructBLIP-7B": {
        "path": MODELS_DIR / "instructblip-vicuna-7b",
        "class": "InstructBlipForConditionalGeneration",
        "processor": "InstructBlipProcessor",
        "module": "transformers",
        "supports_video": False,
        "input_builder": "instructblip",
    },
}

# ---------------------------------------------------------------------------
# Dataset Definitions
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "COCO-val2014": {
        "type": "image_dir",
        "path": COCO_DIR,
        "glob": "*.jpg",
        "n_samples": 3,
    },
    "MMMU": {
        "type": "parquet_dir",
        "path": DATASETS_DIR / "MMMU_hf" / "Art",
        "glob": "validation-*.parquet",
        "n_samples": 2,
    },
    "HallusionBench": {
        "type": "parquet",
        "path": DATASETS_DIR / "HallusionBench_hf" / "data" / "image-00000-of-00001.parquet",
        "n_samples": 2,
    },
    "MMBench": {
        "type": "parquet",
        "path": DATASETS_DIR / "MMBench_EN_hf" / "data" / "dev-00000-of-00001-75b6649fb044d38b.parquet",
        "n_samples": 2,
    },
    "MME": {
        "type": "parquet",
        "path": DATASETS_DIR / "MME_hf" / "data" / "test-00000-of-00004-a25dbe3b44c4fda6.parquet",
        "n_samples": 2,
    },
    "FREAK": {
        "type": "parquet",
        "path": DATASETS_DIR / "FREAK_hf" / "data" / "test-00001-of-00005.parquet",
        "n_samples": 2,
    },
    "MVBench": {
        "type": "video_json",
        "path": DATASETS_DIR / "video" / "OpenGVLab_MVBench" / "json",
        "video_dir": DATASETS_DIR / "video" / "OpenGVLab_MVBench" / "video",
        "n_samples": 2,
    },
    "VidHalluc": {
        "type": "video_json",
        "path": DATASETS_DIR / "video" / "chaoyuli_VidHalluc",
        "json_files": ["ach_mcq.json", "sth.json"],
        "n_samples": 2,
    },
}


# ===========================================================================
# PHASE 1: Dataset Completeness Check (CPU-only, no model needed)
# ===========================================================================
def test_datasets():
    print("\n" + "=" * 70)
    print("PHASE 1: Dataset Completeness Check")
    print("=" * 70)

    for ds_name, cfg in DATASET_CONFIGS.items():
        t0 = time.time()
        try:
            if cfg["type"] == "image_dir":
                _test_image_dir(ds_name, cfg)
            elif cfg["type"] in ("parquet", "parquet_dir"):
                _test_parquet(ds_name, cfg)
            elif cfg["type"] == "video_json":
                _test_video_json(ds_name, cfg)
        except Exception as e:
            record(f"Dataset:{ds_name}", "FAIL", str(e), time.time() - t0)


def _test_image_dir(name, cfg):
    p = Path(cfg["path"])
    if not p.exists():
        record(f"Dataset:{name}", "FAIL", f"Directory not found: {p}")
        return
    imgs = sorted(p.glob(cfg["glob"]))[:cfg["n_samples"]]
    if not imgs:
        record(f"Dataset:{name}", "FAIL", "No images found")
        return
    for img_path in imgs:
        img = Image.open(img_path)
        img.verify()
    record(f"Dataset:{name}", "PASS",
           f"{len(list(p.glob(cfg['glob'])))} images, verified {len(imgs)}")


def _test_parquet(name, cfg):
    import pandas as pd
    if cfg["type"] == "parquet_dir":
        p = Path(cfg["path"])
        files = sorted(p.glob(cfg.get("glob", "*.parquet")))
        if not files:
            record(f"Dataset:{name}", "FAIL", f"No parquet files in {p}")
            return
        df = pd.read_parquet(files[0])
    else:
        p = Path(cfg["path"])
        if not p.exists():
            record(f"Dataset:{name}", "FAIL", f"File not found: {p}")
            return
        df = pd.read_parquet(p)

    n = min(cfg["n_samples"], len(df))
    cols = list(df.columns)
    has_image_col = any("image" in c.lower() for c in cols)
    record(f"Dataset:{name}", "PASS",
           f"{len(df)} rows, cols={cols[:6]}{'...' if len(cols)>6 else ''}, "
           f"has_image={has_image_col}, sampled {n}")


def _test_video_json(name, cfg):
    p = Path(cfg["path"])
    if not p.exists():
        record(f"Dataset:{name}", "FAIL", f"Path not found: {p}")
        return

    if "json_files" in cfg:
        for jf in cfg["json_files"]:
            jp = p / jf
            if not jp.exists():
                record(f"Dataset:{name}", "FAIL", f"JSON not found: {jp}")
                return
            with open(jp) as f:
                data = json.load(f)
            record(f"Dataset:{name}:{jf}", "PASS",
                   f"{len(data)} entries in {jf}")
    else:
        json_dir = p
        json_files = sorted(json_dir.glob("*.json"))[:2]
        if not json_files:
            record(f"Dataset:{name}", "FAIL", "No JSON files found")
            return
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            record(f"Dataset:{name}:{jf.name}", "PASS", f"{len(data)} entries")

    video_dir = cfg.get("video_dir")
    if video_dir:
        vd = Path(video_dir)
        if vd.exists():
            vids = list(vd.iterdir())
            record(f"Dataset:{name}:videos", "PASS" if vids else "FAIL",
                   f"{len(vids)} video files")
        else:
            record(f"Dataset:{name}:videos", "FAIL", "Video dir missing")


# ===========================================================================
# PHASE 2: Model Loading Check
# ===========================================================================
def test_model_loading():
    print("\n" + "=" * 70)
    print("PHASE 2: Model Loading Check")
    print("=" * 70)

    if not HAS_GPU:
        print("  No GPU detected. Performing import/config checks only.")

    for model_name, cfg in MODEL_CONFIGS.items():
        t0 = time.time()
        try:
            _test_model(model_name, cfg)
        except Exception as e:
            record(f"Model:{model_name}", "FAIL", f"{e}", time.time() - t0)
            traceback.print_exc()


def _test_model(model_name, cfg):
    model_path = cfg["path"]
    if not model_path.exists():
        record(f"Model:{model_name}:exists", "FAIL", f"Path not found: {model_path}")
        return

    record(f"Model:{model_name}:exists", "PASS", str(model_path))

    import transformers
    model_cls = getattr(transformers, cfg["class"], None)
    proc_cls = getattr(transformers, cfg["processor"], None)

    if model_cls is None:
        record(f"Model:{model_name}:class", "FAIL",
               f"{cfg['class']} not in transformers {transformers.__version__}")
        return
    record(f"Model:{model_name}:class", "PASS", cfg["class"])

    # Check for model weight files
    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    weight_files = [f for f in weight_files if f.stat().st_size > 1_000_000]
    if not weight_files:
        record(f"Model:{model_name}:weights", "FAIL", "No model weight files (>1MB) found")
        return
    total_gb = sum(f.stat().st_size for f in weight_files) / 1e9
    record(f"Model:{model_name}:weights", "PASS",
           f"{len(weight_files)} files, {total_gb:.1f} GB")

    if not HAS_GPU:
        record(f"Model:{model_name}:load", "SKIP", "No GPU")
        return

    # Actually load the model
    t0 = time.time()
    model = model_cls.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = proc_cls.from_pretrained(str(model_path))
    model.eval()
    load_time = time.time() - t0
    record(f"Model:{model_name}:load", "PASS", f"{load_time:.1f}s")

    # Single image inference
    _test_single_inference(model_name, model, processor, cfg)

    # BRA integration
    _test_bra_integration(model_name, model, processor, cfg)

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def _get_sample_image():
    imgs = sorted(COCO_DIR.glob("*.jpg"))[:1]
    if imgs:
        return Image.open(imgs[0]).convert("RGB")
    return Image.new("RGB", (336, 336), (128, 128, 128))


def _test_single_inference(model_name, model, processor, cfg):
    t0 = time.time()
    try:
        image = _get_sample_image()
        inputs = _build_inputs(cfg["input_builder"], processor, image, "Describe this image briefly.")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        gen_len = out.shape[1] - inputs["input_ids"].shape[1]
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record(f"Model:{model_name}:inference", "PASS",
               f"gen_len={gen_len}, text={text[:80]}...")
    except Exception as e:
        record(f"Model:{model_name}:inference", "FAIL", str(e))
        traceback.print_exc()


def _test_bra_integration(model_name, model, processor, cfg):
    t0 = time.time()
    bra = None
    try:
        sys.path.insert(0, str(PROJECT))

        # Use original BRA operator for Qwen3-VL, multi for others
        if "qwen" in model_name.lower():
            from bra_operator import BRAOperator, BRAConfig
            bra_cfg = BRAConfig(debug=True, warmup_steps=1)
            tokenizer = getattr(processor, "tokenizer", processor)
            bra = BRAOperator(model, bra_cfg, tokenizer=tokenizer)
            adapter_name = "qwen3_vl (original)"
        else:
            from bra_operator_multi import create_bra_operator, BRAConfig
            bra_cfg = BRAConfig(debug=True, warmup_steps=1)
            tokenizer = getattr(processor, "tokenizer", processor)
            bra = create_bra_operator(model, config=bra_cfg, tokenizer=tokenizer)
            adapter_name = bra.adapter.name

        record(f"Model:{model_name}:bra_hook", "PASS", f"Adapter: {adapter_name}")

        image = _get_sample_image()
        inputs = _build_inputs(cfg["input_builder"], processor, image, "Describe this image briefly.")

        bra.reset()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        gen_len = out.shape[1] - inputs["input_ids"].shape[1]
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        has_vision = bra._vision_features is not None
        n_vision = bra._vision_features.shape[0] if has_vision else 0
        record(f"Model:{model_name}:bra_generate", "PASS",
               f"gen_len={gen_len}, vision_tokens={n_vision}, text={text[:60]}...")
    except Exception as e:
        record(f"Model:{model_name}:bra", "FAIL", str(e))
        traceback.print_exc()
    finally:
        if bra is not None:
            try:
                bra.remove()
            except:
                pass


# ---------------------------------------------------------------------------
# Input builders per model architecture
# ---------------------------------------------------------------------------
def _build_inputs(builder_type, processor, image, prompt):
    if builder_type == "qwen3vl":
        return _build_qwen3vl_inputs(processor, image, prompt)
    elif builder_type == "llava":
        return _build_llava_inputs(processor, image, prompt)
    elif builder_type == "instructblip":
        return _build_instructblip_inputs(processor, image, prompt)
    else:
        raise ValueError(f"Unknown builder: {builder_type}")


def _build_qwen3vl_inputs(processor, image, prompt):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    # Filter out keys that generate() doesn't accept
    skip_keys = {"mm_token_type_ids"}
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items() if k not in skip_keys}


def _build_llava_inputs(processor, image, prompt):
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def _build_instructblip_inputs(processor, image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


# ===========================================================================
# PHASE 3: Cross-Model x Dataset Smoke Test (if GPU available)
# ===========================================================================
def test_cross_matrix():
    print("\n" + "=" * 70)
    print("PHASE 3: Model x Dataset Cross-Validation (1 sample each)")
    print("=" * 70)

    if not HAS_GPU:
        print("  Skipping (no GPU).")
        record("CrossMatrix", "SKIP", "No GPU")
        return

    for model_name, mcfg in MODEL_CONFIGS.items():
        model_path = mcfg["path"]
        if not model_path.exists():
            record(f"Cross:{model_name}", "SKIP", "Model not available")
            continue

        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        weight_files = [f for f in weight_files if f.stat().st_size > 1_000_000]
        if not weight_files:
            record(f"Cross:{model_name}", "SKIP", "No weights")
            continue

        print(f"\n  Loading {model_name}...")
        import transformers
        model_cls = getattr(transformers, mcfg["class"])
        proc_cls = getattr(transformers, mcfg["processor"])

        try:
            model = model_cls.from_pretrained(
                str(model_path), torch_dtype=torch.bfloat16, device_map="auto")
            processor = proc_cls.from_pretrained(str(model_path))
            model.eval()
        except Exception as e:
            record(f"Cross:{model_name}:load", "FAIL", str(e))
            continue

        for ds_name, dcfg in DATASET_CONFIGS.items():
            if dcfg["type"] in ("video_json",) and not mcfg.get("supports_video"):
                record(f"Cross:{model_name}x{ds_name}", "SKIP", "Model does not support video")
                continue

            try:
                image = _get_dataset_sample_image(ds_name, dcfg)
                if image is None:
                    record(f"Cross:{model_name}x{ds_name}", "SKIP", "No image sample available")
                    continue

                prompt = _get_dataset_sample_prompt(ds_name, dcfg)
                inputs = _build_inputs(mcfg["input_builder"], processor, image, prompt)

                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=30)
                gen_len = out.shape[1] - inputs["input_ids"].shape[1]
                text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                record(f"Cross:{model_name}x{ds_name}", "PASS",
                       f"gen={gen_len}, {text[:50]}...")
            except Exception as e:
                record(f"Cross:{model_name}x{ds_name}", "FAIL", str(e)[:100])

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def _get_dataset_sample_image(ds_name, cfg):
    """Extract a single image from any dataset for testing."""
    import pandas as pd
    from io import BytesIO

    if cfg["type"] == "image_dir":
        imgs = sorted(Path(cfg["path"]).glob(cfg.get("glob", "*.jpg")))[:1]
        return Image.open(imgs[0]).convert("RGB") if imgs else None

    elif cfg["type"] in ("parquet", "parquet_dir"):
        if cfg["type"] == "parquet_dir":
            files = sorted(Path(cfg["path"]).glob(cfg.get("glob", "*.parquet")))
            if not files:
                return None
            df = pd.read_parquet(files[0])
        else:
            p = Path(cfg["path"])
            if not p.exists():
                return None
            df = pd.read_parquet(p)

        image_cols = [c for c in df.columns if "image" in c.lower()]
        if not image_cols:
            return Image.new("RGB", (336, 336), (200, 200, 200))

        sample = df.iloc[0]
        img_data = sample[image_cols[0]]
        if isinstance(img_data, dict) and "bytes" in img_data:
            return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            return Image.open(BytesIO(img_data)).convert("RGB")
        else:
            return Image.new("RGB", (336, 336), (200, 200, 200))

    return None


def _get_dataset_sample_prompt(ds_name, cfg):
    """Get a relevant prompt for the dataset."""
    prompts = {
        "COCO-val2014": "Describe this image briefly.",
        "MMMU": "Answer the question shown in this image.",
        "HallusionBench": "Is the object described present in this image? Answer yes or no.",
        "MMBench": "Answer the following question about this image.",
        "MME": "Answer the question about this image.",
        "FREAK": "Describe what you see in this image in detail.",
        "MVBench": "Describe the action in this video.",
        "VidHalluc": "What is happening in this video?",
    }
    return prompts.get(ds_name, "Describe this image.")


# ===========================================================================
# MiniGPT-4 Note
# ===========================================================================
def note_minigpt4():
    print("\n" + "=" * 70)
    print("NOTE: MiniGPT-4")
    print("=" * 70)
    mgpt_path = MODELS_DIR / "MiniGPT-4-LLaMA-7B"
    if mgpt_path.exists():
        config_path = mgpt_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architectures", ["unknown"])
            print(f"  MiniGPT-4-LLaMA-7B exists but is bare {arch[0]}")
            print(f"  This is NOT a complete multimodal model.")
            print(f"  MiniGPT-4 requires:")
            print(f"    - Its own codebase (github.com/Vision-CAIR/MiniGPT-4)")
            print(f"    - EVA-CLIP visual encoder")
            print(f"    - Pretrained Q-Former + projection layer weights")
            print(f"  Status: REQUIRES ADDITIONAL SETUP")
            record("Model:MiniGPT-4", "SKIP",
                   "Bare LLaMA backbone only, needs MiniGPT-4 codebase + visual encoder")
    else:
        record("Model:MiniGPT-4", "FAIL", "Directory not found")


# ===========================================================================
# Summary Report
# ===========================================================================
def print_summary():
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    passes = sum(1 for r in RESULTS if r.status == "PASS")
    fails = sum(1 for r in RESULTS if r.status == "FAIL")
    skips = sum(1 for r in RESULTS if r.status == "SKIP")

    print(f"\n  PASS: {passes}  |  FAIL: {fails}  |  SKIP: {skips}")
    print(f"  Total: {len(RESULTS)} checks\n")

    if fails:
        print("  FAILED ITEMS:")
        for r in RESULTS:
            if r.status == "FAIL":
                print(f"    [!!] {r.name}: {r.message}")

    if skips:
        print("\n  SKIPPED ITEMS:")
        for r in RESULTS:
            if r.status == "SKIP":
                print(f"    [--] {r.name}: {r.message}")

    print("\n" + "=" * 70)
    return fails == 0


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("BRA SMOKE TEST MATRIX")
    print(f"GPU: {'Available (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'NOT AVAILABLE'}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except:
        print("Transformers: NOT INSTALLED")
    print("=" * 70)

    # Phase 1: datasets (CPU-only)
    test_datasets()

    # MiniGPT-4 note
    note_minigpt4()

    # Phase 2: model loading + BRA
    test_model_loading()

    # Phase 3: cross-matrix (GPU only, 1 sample each)
    test_cross_matrix()

    # Summary
    all_pass = print_summary()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
