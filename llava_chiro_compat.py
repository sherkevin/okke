import contextlib
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from safetensors.torch import load_file
from torch.cuda.amp import autocast as autocast

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel
from minigpt4.models.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
NUM_IMAGE_TOKENS = 576

HF_VISION_TOWER_FALLBACKS = [
    "/root/chiro_assets/clip-vit-large-patch14-336",
    "/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14",
    "/root/autodl-tmp/.cache/huggingface/hub/models--openai--clip-vit-large-patch14",
    "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14",
]
IGNORABLE_MISSING_SUFFIXES = ("rotary_emb.inv_freq",)


def _build_auto_max_memory() -> dict:
    override_gib = os.environ.get("CHORD_MAX_MEMORY_PER_GPU_GIB")
    first_gpu_override = os.environ.get("CHORD_MAX_MEMORY_GPU0_GIB")
    other_gpu_override = os.environ.get("CHORD_MAX_MEMORY_OTHER_GPU_GIB")
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return {"cpu": "64GiB"}
    if gpu_count >= 2:
        headroom_gpu = _get_headroom_gpu_index(gpu_count)
        headroom_gpu_mem = first_gpu_override or "4GiB"
        other_gpu_mem = other_gpu_override or override_gib or "12GiB"
        max_memory = {}
        for idx in range(gpu_count):
            max_memory[idx] = headroom_gpu_mem if idx == headroom_gpu else other_gpu_mem
        max_memory["cpu"] = "64GiB"
        return max_memory
    per_gpu = override_gib or "1GiB"
    return {0: per_gpu, "cpu": "64GiB"}


def _get_primary_gpu_index(gpu_count: int) -> int:
    raw_primary = os.environ.get("CHORD_PRIMARY_GPU")
    if raw_primary is None:
        return min(1, gpu_count - 1)
    try:
        primary_gpu = int(raw_primary)
    except ValueError:
        return min(1, gpu_count - 1)
    if 0 <= primary_gpu < gpu_count:
        return primary_gpu
    return min(1, gpu_count - 1)


def _get_headroom_gpu_index(gpu_count: int) -> int:
    raw_headroom = os.environ.get("CHORD_HEADROOM_GPU")
    if raw_headroom is None:
        return 0
    try:
        headroom_gpu = int(raw_headroom)
    except ValueError:
        return 0
    if 0 <= headroom_gpu < gpu_count:
        return headroom_gpu
    return 0


def _build_explicit_dual_gpu_map(model: LlavaLlamaForCausalLM) -> dict[str, int]:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise ValueError("Explicit dual-GPU mapping requires at least two visible GPUs.")

    primary_gpu = _get_primary_gpu_index(gpu_count)
    secondary_gpu = next(idx for idx in range(gpu_count) if idx != primary_gpu)
    vision_gpu = secondary_gpu
    primary_layer_count = int(os.environ.get("CHORD_PRIMARY_GPU_LAYER_COUNT", "16"))

    llama_layers = getattr(model.model, "layers", None)
    if llama_layers is None:
        raise ValueError("Could not locate LLaMA decoder layers for explicit dual-GPU placement.")
    layer_count = len(llama_layers)
    primary_layer_count = max(1, min(primary_layer_count, layer_count - 1))

    device_map: dict[str, int] = {
        "model.embed_tokens": primary_gpu,
        "model.vision_tower": vision_gpu,
        "model.mm_projector": vision_gpu,
        "model.norm": secondary_gpu,
        "lm_head": secondary_gpu,
    }
    for idx in range(layer_count):
        device_map[f"model.layers.{idx}"] = primary_gpu if idx < primary_layer_count else secondary_gpu
    return device_map


def _rebalance_dual_gpu_auto_map(device_map: dict[str, int], gpu_count: int) -> dict[str, int]:
    if gpu_count < 2:
        return device_map

    primary_gpu = _get_primary_gpu_index(gpu_count)
    headroom_gpu = _get_headroom_gpu_index(gpu_count)
    if primary_gpu == headroom_gpu:
        return device_map

    early_layer_count = int(os.environ.get("CHORD_REBALANCE_EARLY_LAYERS", "0"))
    hot_modules = ["model.embed_tokens"] + [f"model.layers.{idx}" for idx in range(max(0, early_layer_count))]
    for module_name in hot_modules:
        if device_map.get(module_name) == headroom_gpu:
            device_map[module_name] = primary_gpu
    return device_map


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_hf_llava_checkpoint(ckpt_dir: str) -> bool:
    cfg_path = Path(ckpt_dir) / "config.json"
    if not cfg_path.exists():
        return False
    cfg = _load_json(cfg_path)
    archs = cfg.get("architectures", [])
    return "LlavaForConditionalGeneration" in archs


def _resolve_vision_tower_path(vision_tower: str) -> str:
    if vision_tower and os.path.exists(vision_tower):
        return vision_tower
    for candidate in HF_VISION_TOWER_FALLBACKS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve a local vision tower path for {vision_tower!r}")


def _build_hf_llava_config(ckpt_dir: str, vision_tower: str, mm_vision_select_layer: int) -> LlavaConfig:
    raw_cfg = _load_json(Path(ckpt_dir) / "config.json")
    text_cfg = raw_cfg["text_config"]
    cfg = LlavaConfig(**text_cfg)
    cfg.architectures = ["LlavaLlamaForCausalLM"]
    cfg.model_type = "llava"
    cfg.mm_vision_tower = vision_tower
    cfg.mm_hidden_size = raw_cfg["vision_config"]["hidden_size"]
    cfg.mm_projector_type = "mlp2x_gelu"
    cfg.mm_vision_select_layer = raw_cfg.get("vision_feature_layer", mm_vision_select_layer)
    cfg.mm_vision_select_feature = "patch"
    cfg.mm_use_im_start_end = True
    cfg.mm_use_im_patch_token = True
    cfg.use_mm_proj = True
    return cfg


def _map_hf_weight_key(key: str) -> str | None:
    if key.startswith("language_model.model."):
        return "model." + key[len("language_model.model.") :]
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head.") :]
    if key.startswith("multi_modal_projector.linear_1."):
        return "model.mm_projector.0." + key[len("multi_modal_projector.linear_1.") :]
    if key.startswith("multi_modal_projector.linear_2."):
        return "model.mm_projector.2." + key[len("multi_modal_projector.linear_2.") :]
    return None


def _hf_shard_paths(ckpt_dir: str) -> list[Path]:
    index_path = Path(ckpt_dir) / "model.safetensors.index.json"
    index = _load_json(index_path)
    shard_names = sorted(set(index["weight_map"].values()))
    return [Path(ckpt_dir) / name for name in shard_names]


def _load_hf_llava_weights(model: LlavaLlamaForCausalLM, ckpt_dir: str) -> None:
    expected_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    for shard_path in _hf_shard_paths(ckpt_dir):
        tensors = load_file(str(shard_path))
        mapped = {}
        for key, value in tensors.items():
            new_key = _map_hf_weight_key(key)
            if new_key is None or new_key not in expected_keys:
                continue
            mapped[new_key] = value
            loaded_keys.add(new_key)
        model.load_state_dict(mapped, strict=False)
        del tensors
        del mapped
    missing = sorted(
        key for key in (expected_keys - loaded_keys)
        if not key.endswith(IGNORABLE_MISSING_SUFFIXES)
    )
    if missing:
        raise ValueError(f"HF Llava checkpoint mapping left {len(missing)} missing tensors, e.g. {missing[:8]}")


@registry.register_model("llava-1.5")
class LLaVa(BaseModel):
    """
    LLaVa-1.5 model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/llava-1.5_vicuna7b.yaml",
    }

    def __init__(
        self,
        vision_tower=r"openai/clip-vit-large-patch14",
        mm_vision_select_layer=-2,
        merged_ckpt="",
        cache_dir=None,
        model_max_length=2048,
        shikra_version="v1",
        freeze_backbone=False,
        mm_use_im_start_end=True,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        apply_fsdp=None,
        max_txt_len=128,
        max_output_txt_len=256,
        low_resource=False,
        bf16=False,
        fp16=True,
        system_message="",
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
    ):
        super().__init__()

        kwargs = {"device_map": device_map}
        self.system_message = system_message
        self._skip_module_to = False

        if load_8bit:
            kwargs["load_in_8bit"] = True
        elif load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        self.llama_tokenizer = AutoTokenizer.from_pretrained(merged_ckpt, use_fast=False)

        if _is_hf_llava_checkpoint(merged_ckpt):
            if load_8bit or load_4bit:
                raise NotImplementedError("HF Llava compatibility path currently supports fp16 loading only.")
            resolved_vision_tower = _resolve_vision_tower_path(vision_tower)
            compat_cfg = _build_hf_llava_config(
                merged_ckpt,
                vision_tower=resolved_vision_tower,
                mm_vision_select_layer=mm_vision_select_layer,
            )
            self.llama_model = LlavaLlamaForCausalLM(compat_cfg)
            _load_hf_llava_weights(self.llama_model, merged_ckpt)
        else:
            self.llama_model = LlavaLlamaForCausalLM.from_pretrained(
                merged_ckpt,
                low_cpu_mem_usage=True,
                **kwargs,
            )

        mm_use_im_start_end = getattr(self.llama_model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.llama_model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.llama_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        vision_tower = self.llama_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        if _is_hf_llava_checkpoint(merged_ckpt) and device_map == "auto":
            # Keep the rollback-aware OPERA path on the physical GPU, but
            # aggressively offload HF LLaVA weights so attention-heavy decode
            # paths leave headroom for OPERA/CHORD rollout activations.
            # When multiple GPUs are visible, spread the model across them.
            force_infer_auto_device_map = os.environ.get("CHORD_FORCE_INFER_AUTO_DEVICE_MAP") == "1"
            if torch.cuda.device_count() >= 2 and not force_infer_auto_device_map:
                auto_device_map = _build_explicit_dual_gpu_map(self.llama_model)
            else:
                max_memory = _build_auto_max_memory()
                auto_device_map = infer_auto_device_map(
                    self.llama_model,
                    max_memory=max_memory,
                    no_split_module_classes=["LlamaDecoderLayer", "CLIPEncoderLayer"],
                )
                auto_device_map = _rebalance_dual_gpu_auto_map(auto_device_map, torch.cuda.device_count())
                # The multimodal path calls the vision tower directly during
                # prepare_inputs_labels_for_multimodal(), so keeping the vision
                # stack materialized on-device avoids meta-tensor failures during
                # image feature extraction.
                vision_device = _get_primary_gpu_index(torch.cuda.device_count()) if torch.cuda.device_count() > 0 else 0
                auto_device_map["model.vision_tower"] = vision_device
                auto_device_map["model.mm_projector"] = vision_device
            self.llama_model = dispatch_model(
                self.llama_model,
                device_map=auto_device_map,
                offload_buffers=True,
            )
            self._skip_module_to = True
        else:
            vision_tower.to(device=device, dtype=torch.float16)

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def forward(self, samples):
        image = samples["image"]
        instruction = samples["prompt"] if "prompt" in samples else None
        bs = image.size(0)
        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [p.replace("<ImageHere>", "<image>") for p in instruction]
        instruction = [self.system_message + p for p in instruction]
        input_ids = self.tokenizer_image_token(instruction, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        max_new_tokens=300,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        output_attentions=False,
        return_dict_in_generate=False,
        opera_decoding=False,
        key_position=None,
        scale_factor=1.0,
        threshold=1,
        num_attn_candidates=5,
        penalty_weights=1.0,
        chord_visual_token_weights=None,
        chord_lambda_cur=0.0,
        chord_lambda_fut=0.0,
        chord_lambda_txt=1.0,
        chord_future_horizon=0,
        chord_future_topk=0,
        chord_diagnostics=None,
    ):
        self.llama_tokenizer.padding_side = "left"

        image = samples["image"]
        instruction = samples["prompt"] if "prompt" in samples else None
        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [self.system_message + p for p in instruction]

        chunks_before, chunks_after = [], []
        for p in instruction:
            chunk_before, chunk_after = p.split("<ImageHere>")
            chunks_before.append(chunk_before)
            chunks_after.append(chunk_after)

        tokens_before = self.llama_tokenizer(
            chunks_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to(image.device).input_ids

        tokens_after = self.llama_tokenizer(
            chunks_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to(image.device).input_ids

        bos = torch.ones([bs, 1], dtype=torch.int64, device=image.device) * self.llama_tokenizer.bos_token_id
        image_token = torch.ones([bs, 1], dtype=torch.int64, device=image.device) * IMAGE_TOKEN_INDEX

        with self.maybe_autocast():
            input_ids = torch.cat([bos, tokens_before, image_token, tokens_after], dim=1)

            if chord_diagnostics is not None:
                chord_diagnostics.append({"llava_generate_called": True})

            if key_position is None:
                key_position = {
                    "image_start": tokens_before.shape[1] + 1,
                    "image_end": tokens_before.shape[1] + NUM_IMAGE_TOKENS,
                    "response_start": input_ids.shape[1] + NUM_IMAGE_TOKENS - 1,
                }

            output_ids = self.llama_model.generate(
                input_ids=input_ids,
                use_cache=True,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                images=image,
                output_attentions=output_attentions,
                return_dict_in_generate=return_dict_in_generate,
                opera_decoding=opera_decoding,
                key_position=key_position,
                scale_factor=scale_factor,
                threshold=threshold,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
                chord_visual_token_weights=chord_visual_token_weights,
                chord_lambda_cur=chord_lambda_cur,
                chord_lambda_fut=chord_lambda_fut,
                chord_lambda_txt=chord_lambda_txt,
                chord_future_horizon=chord_future_horizon,
                chord_future_topk=chord_future_topk,
                chord_diagnostics=chord_diagnostics,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        output_text = self.llama_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        output_text = [text.split("###")[0].strip() for text in output_text]
        return output_text

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, "model"):
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def to(self, *args, **kwargs):
        if self._skip_module_to:
            return self
        return super().to(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        vision_tower = cfg.get("vit_model", r"openai/clip-vit-large-patch14")
        mm_vision_select_layer = cfg.get("mm_vision_select_layer", -2)
        merged_ckpt = cfg.get("merged_ckpt", "")
        cache_dir = cfg.get("cache_dir", None)
        model_max_length = cfg.get("model_max_length", 2048)
        shikra_version = cfg.get("version", "v1")
        freeze_backbone = cfg.get("freeze_backbone", False)
        mm_use_im_start_end = cfg.get("mm_use_im_start_end", True)
        pretrain_mm_mlp_adapter = cfg.get("pretrain_mm_mlp_adapter", None)
        tune_mm_mlp_adapter = cfg.get("tune_mm_mlp_adapter", False)
        freeze_mm_mlp_adapter = cfg.get("freeze_mm_mlp_adapter", False)
        apply_fsdp = cfg.get("apply_fsdp", None)
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        low_resource = cfg.get("low_resource", False)
        bf16 = cfg.get("bf16", False)
        fp16 = cfg.get("fp16", False)
        system_message = cfg.get("system_message", "")

        model = cls(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            merged_ckpt=merged_ckpt,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            shikra_version=shikra_version,
            freeze_backbone=freeze_backbone,
            mm_use_im_start_end=mm_use_im_start_end,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            tune_mm_mlp_adapter=tune_mm_mlp_adapter,
            freeze_mm_mlp_adapter=freeze_mm_mlp_adapter,
            apply_fsdp=apply_fsdp,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            low_resource=low_resource,
            bf16=bf16,
            fp16=fp16,
            system_message=system_message,
        )
        return model
