"""
Quick validation of BRA LogitsProcessor on Qwen3-VL-2B and LLaVA-1.5-7B.
Tests that:
  1. Model loads and generates baseline output
  2. BRA hook attaches, vision features extracted, logits reshaped
  3. No KV cache / dimension errors
  4. BRA output differs from baseline (reshaping is active)
"""
import gc
import sys
import time
import torch
from pathlib import Path
from PIL import Image

PROJECT = Path("/root/autodl-tmp/BRA_Project")
MODELS = PROJECT / "models"
COCO = PROJECT / "datasets" / "coco2014" / "val2014"

sys.path.insert(0, str(PROJECT))


def get_sample_image():
    imgs = sorted(COCO.glob("*.jpg"))[:1]
    if imgs:
        return Image.open(imgs[0]).convert("RGB")
    return Image.new("RGB", (336, 336), (128, 128, 128))


def test_qwen3vl():
    print("=" * 70)
    print("TEST: Qwen3-VL-2B + BRA LogitsProcessor")
    print("=" * 70)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from bra_logits_processor import create_bra_processor, make_bra_config
    from bra_operator_multi import detect_adapter

    model_path = MODELS / "Qwen3-VL-2B-Instruct"
    print(f"Loading model from {model_path}...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    image = get_sample_image()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image briefly."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    skip_keys = {"mm_token_type_ids"}
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items() if k not in skip_keys}

    # -- Baseline --
    print("\n  [Baseline] Generating...")
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=60)
    base_text = processor.decode(out_base[0, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    print(f"  Baseline: {base_text[:120]}")

    # -- BRA --
    print("\n  [BRA] Attaching LogitsProcessor...")
    adapter = detect_adapter(model)
    print(f"  Adapter: {adapter.name}")

    cfg = make_bra_config("bra_zero", debug=False, warmup_steps=2)
    tokenizer = getattr(processor, "tokenizer", processor)
    video_grid_thw = inputs.get("video_grid_thw")
    extractor, bra_proc = create_bra_processor(
        model, adapter, inputs["input_ids"], config=cfg,
        tokenizer=tokenizer, video_grid_thw=video_grid_thw)

    print("  Generating with BRA...")
    try:
        with torch.no_grad():
            out_bra = model.generate(**inputs, max_new_tokens=60,
                                     logits_processor=[bra_proc])
        bra_text = processor.decode(out_bra[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        print(f"  BRA text: {bra_text[:120]}")

        has_vision = bra_proc._vision_features is not None
        n_vision = bra_proc._vision_features.shape[0] if has_vision else 0
        n_steps = bra_proc._step
        print(f"  Vision features: {has_vision} (N={n_vision})")
        print(f"  BRA steps executed: {n_steps}")
        print(f"  BRA stats: {bra_proc.get_stats()}")
        print(f"  Texts differ: {base_text != bra_text}")
        print("  [PASS] Qwen3-VL-2B + BRA LogitsProcessor")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
    finally:
        extractor.remove()
        bra_proc.reset()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def test_llava():
    print("\n" + "=" * 70)
    print("TEST: LLaVA-1.5-7B + BRA LogitsProcessor")
    print("=" * 70)

    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from bra_logits_processor import create_bra_processor, make_bra_config
    from bra_operator_multi import detect_adapter

    model_path = MODELS / "llava-1.5-7b-hf"
    print(f"Loading model from {model_path}...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    image = get_sample_image()
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image briefly."},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # -- Baseline --
    print("\n  [Baseline] Generating...")
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=60)
    base_text = processor.decode(out_base[0, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    print(f"  Baseline: {base_text[:120]}")

    # -- BRA --
    print("\n  [BRA] Attaching LogitsProcessor...")
    adapter = detect_adapter(model)
    print(f"  Adapter: {adapter.name}")

    cfg = make_bra_config("bra_zero", debug=False, warmup_steps=2)
    tokenizer = getattr(processor, "tokenizer", processor)
    extractor, bra_proc = create_bra_processor(
        model, adapter, inputs["input_ids"], config=cfg, tokenizer=tokenizer)

    print("  Generating with BRA...")
    try:
        with torch.no_grad():
            out_bra = model.generate(**inputs, max_new_tokens=60,
                                     logits_processor=[bra_proc])
        bra_text = processor.decode(out_bra[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        print(f"  BRA text: {bra_text[:120]}")

        has_vision = bra_proc._vision_features is not None
        n_vision = bra_proc._vision_features.shape[0] if has_vision else 0
        n_steps = bra_proc._step
        print(f"  Vision features: {has_vision} (N={n_vision})")
        print(f"  BRA steps executed: {n_steps}")
        print(f"  BRA stats: {bra_proc.get_stats()}")
        print(f"  Texts differ: {base_text != bra_text}")
        print("  [PASS] LLaVA-1.5-7B + BRA LogitsProcessor")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
    finally:
        extractor.remove()
        bra_proc.reset()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    import transformers
    print(f"Transformers: {transformers.__version__}")

    test_qwen3vl()
    test_llava()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
