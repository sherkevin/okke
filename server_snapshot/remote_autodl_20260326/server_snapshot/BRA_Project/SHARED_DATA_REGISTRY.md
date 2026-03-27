# SHARED DATA REGISTRY
> BRA Project — Centralized index of all datasets, models, logs, and experiment records.
> Last updated: 2026-03-21 21:44 CST

---

## 1. Datasets

| Dataset | Path | Notes |
|---------|------|-------|
| COCO val2014 images | `/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/` | ~40k images |
| COCO annotations | `/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/` | instances_val2014.json, captions_val2014.json |
| POPE splits | `/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/` | random / popular / adversarial |

## 2. Models

| Model | Path | Notes |
|-------|------|-------|
| Qwen3-VL-8B-Instruct | `/root/autodl-tmp/BRA_Project/models/Qwen3-VL-8B-Instruct/` | Primary eval model |
| Qwen3-VL-2B-Instruct | `/root/autodl-tmp/BRA_Project/models/Qwen3-VL-2B-Instruct/` | Dev / smoke test |

## 3. Baseline Source Repos

| Baseline | Path | Cloned |
|----------|------|--------|
| VCD (DAMO-NLP-SG) | `/root/autodl-tmp/BRA_Project/baselines/VCD/` | 2026-03-21 via network_turbo |
| OPERA (shikiw) | `/root/autodl-tmp/BRA_Project/baselines/OPERA/` | 2026-03-21 via network_turbo |

## 4. Core Code

| File | Path | Description |
|------|------|-------------|
| BRA operator | `/root/autodl-tmp/BRA_Project/bra_operator.py` | BRA algorithm (monkey-patch + forward hook) |
| Baseline processors | `/root/autodl-tmp/BRA_Project/baseline_processors.py` | VCD / OPERA / DoLa as LogitsProcessor |
| Eval pipeline | `/root/autodl-tmp/BRA_Project/run_eval_pipeline.py` | Unified --method --dataset --mini_test runner |
| BRA smoke test | `/root/autodl-tmp/BRA_Project/bra_smoke_test.py` | 20-sample AGL deviation test |

---

## 5. Experiment Logs

### Experiment #1: Mini-Test 50 Baseline Evaluation (2026-03-21)

**Config**: Qwen3-VL-8B, RTX 5090 (32GB), greedy decoding, max_new_tokens=128, mini_test=50

**Local report**: `experiment_logs/minitest_50_baseline_eval_20260321.md`

#### JSON Log Paths

| Method | Dataset | Server Path |
|--------|---------|-------------|
| base | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260321_202446.json` |
| vcd | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_pope_20260321_202828.json` |
| opera | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260321_204435.json` |
| base | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/base_chair_20260321_213138.json` |
| vcd | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_chair_20260321_213639.json` |
| opera | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_chair_20260321_213946.json` |

#### POPE Results (random split, 50 samples)

| Method | Accuracy | F1     | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|----------|--------|-------|--------------|----------------|--------|
| Base   | 0.88     | 0.8636 | 86.34 | 26.98        | 17.648         | 0      |
| VCD    | 0.88     | 0.8636 | 77.04 | 55.30        | 17.704         | 0      |
| OPERA  | 0.8889   | 0.8750 | 83.22 | 302.89       | 17.449         | 14     |

#### CHAIR Results (50 images)

| Method | CHAIR-s | CHAIR-i | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|---------|---------|-------|--------------|----------------|--------|
| Base   | 0.1333  | 0.2879  | 128.0 | 21.95        | 17.681         | 0      |
| VCD    | 0.1197  | 0.2869  | 128.0 | 43.14        | 17.765         | 0      |
| OPERA  | 0.1319  | 0.2984  | 128.0 | 26.29        | 18.104         | 0      |

#### Base vs VCD Delta (JSON)

```json
{
  "comparison": "Base vs VCD (50 mini-test samples)",
  "pope": {
    "base_agl": 86.34, "vcd_agl": 77.04, "agl_delta_pct": -10.77,
    "base_itl_ms": 26.98, "vcd_itl_ms": 55.30, "itl_delta_pct": +104.97,
    "accuracy_delta": 0.0
  },
  "chair": {
    "base_agl": 128.0, "vcd_agl": 128.0, "agl_delta_pct": 0.0,
    "base_itl_ms": 21.95, "vcd_itl_ms": 43.14, "itl_delta_pct": +96.54,
    "chair_s_delta": -0.0136, "chair_i_delta": -0.0010
  }
}
```

#### Key Findings

1. VCD doubles ITL (~2x) due to shadow forward pass with diffusion-noised KV cache
2. VCD reduces POPE AGL by 10.8% — model becomes more conservative
3. OPERA impractical on Qwen3-VL: requires eager attn → 11.2x ITL, 28% OOM rate on POPE
4. CHAIR AGL saturated at max_new_tokens=128 for all methods
5. OPERA's `output_attentions` flag stripped by `model.generate()`, limiting penalty effectiveness

---

*Add new experiments below this line.*

---

## 6. Resource Audit

### Resource Audit: 标杆V1 download readiness (2026-03-21)

- Local log: `experiment_logs/asset_audit_标杆V1_20260321.md`
- Remote log: `/root/autodl-tmp/BRA_Project/logs/asset_audit_标杆V1_20260321.md`

#### Audit summary

- Ready now:
  - `Qwen3-VL-8B-Instruct`
  - `Qwen3-VL-2B-Instruct`
  - `COCO val2014 + annotations`
  - `POPE`
  - `CHAIR`
  - `MMBench_EN_hf` (version still worth confirming against V1.1)
  - `MME_hf`
  - `MMMU_hf` (full set only, not explicit Hard subset)
  - `FREAK_hf`

- Missing or not yet confirmed:
  - `DocVQA`
  - `Base + 5k LoRA`
  - `MMMU Hard` explicit split
  - explicit registration of `TLRA_calib` weights

#### Notes

- `Stage 0` is data-ready because `COCO val2014` and `instances_val2014.json` are present; only the held-out subset needs to be constructed at run time.
- `V_matrix.pt`, `V_matrix_q3.pt`, and `V_matrix_q3_mini.pt` exist under `models/`, but they are not clearly registered as the official `TLRA_calib` / `Phi_calib` assets yet.

### Download Update: DocVQA + Video-MME (2026-03-22)

- Local log: `experiment_logs/download_docvqa_videomme_20260321.md`
- Remote log: `/root/autodl-tmp/BRA_Project/logs/download_docvqa_videomme_20260321.md`

#### Status

- `DocVQA`: completed
  - path: `/root/autodl-tmp/BRA_Project/datasets/DocVQA_hf`
  - alias: `/root/autodl-tmp/BRA_Project/datasets/DocVQA`
  - file_count: `223`
  - size: `9,591,618,321 bytes`
  - summary: `/root/autodl-tmp/BRA_Project/logs/downloads/docvqa_summary.json`

- `Video-MME`: download started
  - path: `/root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf`
  - alias: `/root/autodl-tmp/BRA_Project/datasets/video/Video-MME`
  - repo size estimate: `~101.0 GB`
  - current mode: background resumable download with retries
  - log: `/root/autodl-tmp/BRA_Project/logs/downloads/videomme_resume.log`
  - summary: `/root/autodl-tmp/BRA_Project/logs/downloads/videomme_summary.json` (generated on completion)


---

# V3 Engineer B Closeout (2026-03-22)

## Remote Assets
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_smoke_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_meanpool.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_adaptivetopk.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_adaptivetopk.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/video_mme_tlra_full_smoke.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm.json`
- `/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json`

## Local Mirror
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/docvqa_tlra_full.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/docvqa_tlra_no_vasm.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/freak_tlra_meanpool.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/freak_tlra_adaptivetopk.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/mmmu_hard_tlra_full.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/mmmu_hard_tlra_no_vasm.json`

## DocVQA Negative Control
- `tlra_full`: accuracy `0.0000` -> `0.0000`, AGL `6.37` -> `6.38`, ITL `88.33` -> `87.64` ms/token, trigger rate `0.820`
- `tlra_no_vasm`: accuracy `0.0000` -> `0.0000`, AGL `6.45` -> `6.37`, ITL `87.12` -> `87.76` ms/token, trigger rate `0.815`
- Required field check: `normalized_exact_match` missing = `True`, `anls` missing = `True`.
- Trigger/audit fields are preserved: `intervention_rate=0.82`, sample audits present = `True`.
- Trace anomaly: output `bra_method` values are `bra_zero` and `bra_zero` rather than the requested `tlra_full` / `tlra_no_vasm` labels.

## FREAK Parity Review
- `meanpool accuracy`: baseline `0.2233` -> BRA `0.2100`
- `adaptivetopk accuracy`: baseline `0.2200` -> BRA `0.2200`
- `UNFROZEN_PROJECTOR / provisional`: projector identity is not frozen, so this pair is retained only as provisional evidence.
- On this run, `MeanPool` accuracy `0.2100` vs `AdaptiveTopK` `0.2200`; this does not support a strong local-evidence-superiority claim.

## MMMU Hard Formal Row
- `tlra_full accuracy`: baseline `0.1930` -> BRA `0.1930`
- `tlra_no_vasm accuracy`: baseline `0.1579` -> BRA `0.2193`
- Frozen manifest: `/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json`
- Loaded sample count: `114` across `26` subjects.
- Current evidence favors `tlra_no_vasm` over `tlra_full` on this frozen manifest (`0.2193` vs `0.1930`), while both runs show `intervention_rate=0.0`.

## Video Status
- `VidHalluc`: both current runs loaded `0 samples`; keep as appendix-only exploratory pilot and stop here.
- `Video-MME`: loader/index is auditable, but the current blocking point is the lower-level video decoding stack rather than data ingress; therefore it does not enter the main-paper benchmark and remains appendix-only diagnosis.

## Interpretation Guardrails
- `DocVQA` is useful as an OCR-concession probe, but the current JSONs are not fully compliant because `normalized_exact_match` and `anls` were not preserved in output.
- `FREAK` remains provisional because `Phi_calib` / projector identity is unfrozen.
- `MMMU Hard` is now materially upgraded from pilot to a frozen-manifest run, but its current result does not support the stronger `tlra_full > tlra_no_vasm` thesis.


---

# V3 Engineer B Contract-Ready Reruns (2026-03-22)

## Contract-Ready DocVQA
- Status: `contract-ready` after schema validation on both reruns.
- Accepted assets:
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log`
- `tlra_full`: bra_method `tlra_full`, baseline accuracy `0.6550`, BRA accuracy `0.6650`, baseline NEM `0.6550`, BRA NEM `0.6650`, baseline ANLS `0.7978`, BRA ANLS `0.8235`, intervention_rate `0.9000`, sample_audits `5`.
- `tlra_no_vasm`: bra_method `tlra_no_vasm`, baseline accuracy `0.6650`, BRA accuracy `0.6550`, baseline NEM `0.6650`, BRA NEM `0.6550`, baseline ANLS `0.8168`, BRA ANLS `0.8118`, intervention_rate `0.8950`, sample_audits `5`.
- Schema acceptance checks passed for both files: `bra_method` preserved, `normalized_exact_match` present, `anls` present, `intervention_rate` present, `sample_audits` non-empty.

## MMMU Hard Review
- Review assets:
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_review.log`
- `tlra_full`: accuracy `0.1754`, intervention_rate `0.3684`, avg_vasm_time_ms `0.0252`, avg_routing_time_ms `0.1831`, visual_state_provenance present `True`.
- `tlra_no_vasm`: accuracy `0.1667`, intervention_rate `0.3772`, avg_vasm_time_ms `0.0073`, avg_routing_time_ms `0.3074`, visual_state_provenance present `True`.
- Review conclusion: the old `intervention_rate = 0.0` reading is no longer valid under the corrected export protocol. Current reruns show substantial non-zero trigger rates.

## Guardrails
- `FREAK` remains `UNFROZEN_PROJECTOR / provisional` and was not rerun as a formal table.
- `VidHalluc` and `Video-MME` remain stopped and did not occupy GPU in this rerun phase.
