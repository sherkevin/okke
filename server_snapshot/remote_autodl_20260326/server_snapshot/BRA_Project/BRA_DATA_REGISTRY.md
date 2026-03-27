# BRA Project — Data Registry
**Project**: Bi-directional Resonance Anchoring (BRA)
**Last Updated**: 2026-03-21

---

## §0. Download Queue — 数据盘 (`/root/autodl-tmp/BRA_Project/datasets`)

由 `download_all.sh` **并行**拉取（与 COCO / MMMU / HallusionBench 等同阶段 `bg_run`）：

| 目录（绝对路径） | HuggingFace 仓库 | 说明 |
|------------------|------------------|------|
| `/root/autodl-tmp/BRA_Project/datasets/MMBench_EN_hf` | `lmms-lab/MMBench_EN` | MMBench 英文子集；全量多语言为 `lmms-lab/MMBench`（未默认加入队列） |
| `/root/autodl-tmp/BRA_Project/datasets/MME_hf` | `lmms-lab/MME` | MME 综合评测数据 |

---

## §1. Node 5 — Baseline Alignment Matrix

| File | Absolute Path | Supports | Status |
|------|--------------|---------|--------|
| `baselines/baseline_registry.csv` | `/root/autodl-tmp/BRA_Project/baselines/baseline_registry.csv` | Node 5 input — all competitive baselines for POPE/CHAIR/VidHalluc/HallusionBench/FREAK comparison | **VERIFIED** |
| `baselines/baseline_evidence_log.md` | `/root/autodl-tmp/BRA_Project/baselines/baseline_evidence_log.md` | Audit trail with verbatim table quotes and conflict resolutions for all values in baseline_registry.csv | **VERIFIED** |

### §1.1 Coverage Summary

| Benchmark | Classical Baselines | 2025-26 SOTAs | Primary Model | Status |
|-----------|---------------------|---------------|---------------|--------|
| **POPE (Adversarial)** | Regular=79.26, VCD=79.47 (F1) | ONLY=81.07 | LLaVA-1.5-7B | ✅ Verified |
| **POPE (Random)** | Regular=83.44, VCD=87.15, OPERA=88.85 | ONLY=89.10, FarSight=90.5\* | LLaVA-1.5-7B | ✅ Verified |
| **CHAIR_s (max128)** | Regular=55.0, VCD=54.4, OPERA=52.6 | ONLY=49.8, FarSight=41.6†, SCPO=7.0‡ | LLaVA-1.5-7B | ✅ Verified |
| **VidHalluc (Avg)** | Video-LLaVA=37.0%, VILA-1.5-13B=60.9% | GPT-4o=82.4% (ceiling) | Video-LLaVA-7B | ✅ Verified |
| **HallusionBench** | LLaVA-1.5=9.45%, Qwen-VL-2023=5.93% | Qwen2-VL-7B=43.95 qAcc, Qwen2.5-VL=47.25 | Mixed | ✅ Verified |
| **FREAK (6 cats)** | — | — | — | ⚠️ NOT_RELEASED (ICLR 2026 pending) |
| **Qwen3-VL-8B POPE-Adv** | — | — | Qwen3-VL-8B | ⚠️ NOT_REPORTED in tech report |

\* FarSight reports **Accuracy** not F1. † FarSight baseline=48.0 (different max_tok protocol). ‡ SCPO LLaVA-v1.6-7B; POPE not reported (uses AMBER-Disc F1=89.2).

### §1.2 Latency Overhead Reference (anchor for BRA's target +1.5%)

| Method | Latency Overhead | Source |
|--------|-----------------|--------|
| Regular (Base) | +0% | — |
| **ONLY (ICCV 2025)** | **+7%** | ONLY Table 5 (×1.07) |
| VCD (CVPR 2024) | **+101%** | ONLY Table 5 (×2.01) |
| M3ID | +107% | ONLY Table 5 (×2.07) |
| DoLa | +169% | ONLY Table 5 (×2.69) |
| OPERA (CVPR 2024) | **+612%** | ONLY Table 5 (×7.12) |
| **BRA Target** | **~+1.5%** (claimed) | Internal BRA design spec |

### §1.3 Critical Quality Flags

1. **DoLa is NOT a valid positive MLLM hallucination baseline**: MME=522 < Regular=562. DoLa hurts MLLM performance. (ONLY Table 3, verbatim).
2. **OPERA uses Beam Search (n=5)**: All other methods use greedy/sampling. OPERA comparisons need architectural asterisk.
3. **FarSight reports POPE Accuracy, not F1**: Direct F1 comparison requires conversion or separate acknowledgment.
4. **SCPO does not report POPE**: Uses AMBER-Discriminative F1 as substitute. Cross-metric comparison disallowed.
5. **FREAK leaderboard not yet public**: Cannot populate per-category columns until official release.
6. **VCD minimally helpful on POPE Adversarial**: F1=79.47 barely above Regular=79.26 (+0.21). VCD's strength is POPE Random/Popular.
