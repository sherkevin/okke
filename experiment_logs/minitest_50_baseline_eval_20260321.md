# Mini-Test 50 Baseline Evaluation Log
> Date: 2026-03-21
> Server: AutoDL RTX 5090 (32GB VRAM)
> Model: Qwen3-VL-8B-Instruct (bfloat16, device_map="auto")

## Experiment Configuration

- **mini_test**: 50 samples per run
- **max_new_tokens**: 128
- **do_sample**: False (greedy decoding)
- **POPE split**: random (`coco_pope_random.json`)
- **CHAIR prompt**: "Please describe this image in detail."
- **Timing**: `torch.cuda.Event` for GPU-precise ITL
- **VRAM**: `torch.cuda.max_memory_allocated()`

## Methods

| Method | Implementation | Key Parameters |
|--------|---------------|----------------|
| Base   | No intervention | — |
| VCD    | `VCDLogitsProcessor` (diffusion noise, shadow KV cache) | cd_alpha=1.0, cd_beta=0.1, noise_step=500 |
| OPERA  | `OPERALogitsProcessor` (attn hook on last layer, eager attn) | penalty_weight=1.0, scale_factor=50.0 |

## Results

### POPE (random split, 50 samples)

| Method | Accuracy | Precision | Recall | F1     | AGL    | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|----------|-----------|--------|--------|--------|--------------|----------------|--------|
| Base   | 0.8800   | 1.0000    | 0.7600 | 0.8636 | 86.34  | 26.98        | 17.648         | 0/50   |
| VCD    | 0.8800   | 1.0000    | 0.7600 | 0.8636 | 77.04  | 55.30        | 17.704         | 0/50   |
| OPERA  | 0.8889   | 1.0000    | 0.7778 | 0.8750 | 83.22  | 302.89       | 17.449         | 14/50  |

### CHAIR (50 images, detailed description)

| Method | CHAIR-s | CHAIR-i | AGL    | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|---------|---------|--------|--------------|----------------|--------|
| Base   | 0.1333  | 0.2879  | 128.00 | 21.95        | 17.681         | 0/50   |
| VCD    | 0.1197  | 0.2869  | 128.00 | 43.14        | 17.765         | 0/50   |
| OPERA  | 0.1319  | 0.2984  | 128.00 | 26.29        | 18.104         | 0/50   |

## Delta Analysis

### Base vs VCD

| Metric | POPE | CHAIR |
|--------|------|-------|
| Accuracy / CHAIR-s | 0.00 (no change) | -0.0136 (VCD slightly better) |
| F1 / CHAIR-i | 0.00 (no change) | -0.0010 (negligible) |
| AGL delta | -10.77% (77.04 vs 86.34) | 0.0% (both saturated at 128) |
| ITL delta | +104.97% (55.30 vs 26.98) | +96.54% (43.14 vs 21.95) |
| VRAM delta | +0.056 GB | +0.084 GB |

### Base vs OPERA

| Metric | POPE | CHAIR |
|--------|------|-------|
| Accuracy / CHAIR-s | +0.0089 | -0.0014 |
| F1 / CHAIR-i | +0.0114 | +0.0105 (worse) |
| AGL delta | -3.61% (83.22 vs 86.34) | 0.0% |
| ITL delta | +1022.9% (302.89 vs 26.98) | +19.77% (26.29 vs 21.95) |
| VRAM delta | -0.199 GB | +0.423 GB |
| Error rate | 28% (14/50 OOM) | 0% |

## Key Findings

1. **VCD doubles per-token latency** on both benchmarks (~2x ITL increase) due to the mandatory shadow forward pass with diffusion-noised image KV cache.

2. **VCD suppresses POPE verbosity** — AGL drops 10.8% (86.34 → 77.04 tokens). This confirms VCD makes the model more conservative on binary (yes/no) questions, potentially silencing useful explanations.

3. **VCD provides marginal CHAIR improvement** — CHAIR-s drops from 0.1333 to 0.1197 (-1.4pp), indicating slightly fewer hallucinated sentences. CHAIR-i is essentially unchanged.

4. **OPERA is incompatible with Qwen3-VL on RTX 5090** — Requires `attn_implementation='eager'` (FlashAttention cannot return attention weights). This causes:
   - POPE ITL explosion: 302.89 ms/tok (11.2x base)
   - 14/50 POPE samples OOM (`torch.OutOfMemoryError: Tried to allocate 1.16 GiB`)
   - The `output_attentions` flag is stripped by `model.generate()`, limiting the hook's effectiveness

5. **CHAIR AGL saturated at max_new_tokens=128** — All methods hit the ceiling. Need to increase `max_new_tokens` (e.g., 512) to measure true AGL differentiation for CHAIR.

6. **OPERA CHAIR ran without OOM** on the rebooted server (different PCI bus slot), suggesting memory fragmentation or competing processes were the original cause of POPE OOM errors.

## OPERA Error Traceback (representative)

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.
GPU 0 has a total capacity of 31.36 GiB of which 471.69 MiB is free.
Including non-PyTorch memory, this process has 16.63 GiB memory in use.
Process 19239 has 14.26 GiB memory in use.
Of the allocated memory 15.03 GiB is allocated by PyTorch, and 1.00 GiB
is reserved by PyTorch but unallocated.
```

Location: `modeling_qwen3_vl.py:1541` → `self.lm_head(hidden_states[:, slice_indices, :])` during eager attention prefill with large image tokens.

## Remote JSON Log Paths

```
/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260321_202446.json
/root/autodl-tmp/BRA_Project/logs/minitest/vcd_pope_20260321_202828.json
/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260321_204435.json
/root/autodl-tmp/BRA_Project/logs/minitest/base_chair_20260321_213138.json
/root/autodl-tmp/BRA_Project/logs/minitest/vcd_chair_20260321_213639.json
/root/autodl-tmp/BRA_Project/logs/minitest/opera_chair_20260321_213946.json
```

## Implementation Notes

### VCD (baseline_processors.py → VCDLogitsProcessor)
- Noise: exact diffusion schedule from `VCD/vcd_utils/vcd_add_noise.py` (sigmoid beta schedule, 1000 steps)
- Formula: `cd_out = (1 + cd_alpha) * logits_std - cd_alpha * logits_noisy`
- Plausibility constraint: `mask where logits_std < log(cd_beta) + max(logits_std)`
- Per-sample: new noisy KV cache created for each image (different images)

### OPERA (baseline_processors.py → OPERALogitsProcessor)
- Simplified single-pass penalty (no beam-search rollback)
- Hook: `model.model.language_model.layers[-1].self_attn.register_forward_hook`
- Requires: `attn_implementation='eager'` at model load time
- Known issue: `model.generate()` strips `output_attentions` kwarg

### DoLa (baseline_processors.py → DoLaLogitsProcessor)
- Hooks on premature layers (mid-range of decoder stack)
- JSD-based dynamic layer selection
- Not evaluated in this run (scheduled for next phase)

## Files on Server

| File | Path |
|------|------|
| BRA operator | `/root/autodl-tmp/BRA_Project/bra_operator.py` |
| Baseline processors | `/root/autodl-tmp/BRA_Project/baseline_processors.py` |
| Eval pipeline | `/root/autodl-tmp/BRA_Project/run_eval_pipeline.py` |
| VCD source | `/root/autodl-tmp/BRA_Project/baselines/VCD/` |
| OPERA source | `/root/autodl-tmp/BRA_Project/baselines/OPERA/` |
