# BRA Project — Baseline Evidence Log
## Node 5 Input Verification — Chief Scientist Audit Trail
**Compiled**: 2026-03-20 | **Policy**: Zero-hallucination. Every number has a verbatim quote + stable URL.

---

## §A. ONLY (ICCV 2025) — Primary Cross-Verification Source

**PDF (directly downloaded and parsed)**:
`https://openaccess.thecvf.com/content/ICCV2025/papers/Wan_ONLY_One-Layer_Intervention_Sufficiently_Mitigates_Hallucinations_in_Large_Vision-Language_Models_ICCV_2025_paper.pdf`

This paper serves as the **primary source** for all classical baseline numbers (VCD, OPERA, DoLa, Regular) because:
1. It uses a **standardized protocol** (sampling decoding for all methods) across VCD, M3ID, and their own method.
2. Its Table 1 provides a unified POPE comparison with LLaVA-1.5, InstructBLIP, and Qwen-VL.
3. Table 5 provides latency overhead numbers for VCD, OPERA, M3ID and ONLY in a single efficiency table.

### A.1 POPE Table 1 Verbatim (p.3229 of PDF)

```
Setup   Method    LLaVA-1.5                         InstructBLIP                      Qwen-VL
                  Acc↑  Prec↑  Rec↑  F1↑           Acc↑  Prec↑  Rec↑  F1↑           Acc↑  Prec↑  Rec↑  F1↑

MS-COCO Random
        Regular   83.13 81.94  85.00 83.44          83.07 83.02  83.13 83.08          85.23 97.23  72.53 83.09
        VCD       87.00 86.13  88.20 87.15          86.23 88.14  83.73 85.88          87.03 97.36  76.13 85.45
        M3ID      87.50 87.38  87.67 87.52          86.67 88.09  84.80 86.41          86.40 98.23  74.13 84.50
        Ours      89.70 89.95  88.27 89.10          89.23 91.83  86.13 88.89          88.90 98.52  79.27 87.85

MS-COCO Popular
        Regular   81.17 78.28  86.27 82.08          77.00 73.82  83.67 78.44          84.53 94.50  73.33 82.58
        VCD       83.10 79.96  88.33 83.94          80.07 77.67  84.40 80.89          85.87 94.98  75.73 84.27
        M3ID      84.30 81.58  88.60 84.95          80.97 77.93  86.40 81.85          86.07 96.56  74.80 84.30
        Ours      86.00 84.44  88.27 86.31          83.27 81.46  86.13 83.73          87.47 95.63  79.48 86.81

MS-COCO Adversarial
        Regular   77.43 73.31  86.27 79.26          74.60 71.26  82.47 76.45          83.37 91.47  73.60 81.57
        VCD       77.17 72.18  88.40 79.47          77.20 74.29  83.20 78.49          83.73 89.84  76.07 82.38
        M3ID      78.23 73.51  88.27 80.22          77.47 73.68  85.47 79.14          83.37 91.19  73.87 81.62
        Ours      79.40 75.00  88.20 81.07          80.10 76.89  86.07 81.22          83.80 92.33  76.14 83.46
```
**Key extract for BRA registry**:
- `Regular MS-COCO Adversarial LLaVA-1.5: F1 = 79.26` ✓
- `VCD MS-COCO Adversarial LLaVA-1.5: F1 = 79.47` ✓ (barely above baseline — see conflict note)
- `ONLY MS-COCO Adversarial LLaVA-1.5: F1 = 81.07` ✓

### A.2 CHAIR Table 2 Verbatim (p.3230 of PDF)

```
Method       LLaVA-1.5                InstructBLIP             Qwen-VL
             max64          max128    max64          max128    max64          max128
             CHAIRS CHAIRI  CHAIRS CHAIRI  CHAIRS CHAIRI  CHAIRS CHAIRI  CHAIRS CHAIRI  CHAIRS CHAIRI
Regular      26.2   9.4     55.0   16.3    31.2   11.1    57.0   17.6    33.6   12.9    52.0   16.5
VCD          24.4   7.9     54.4   16.6    30.0   10.1    60.4   17.8    33.0   12.8    50.2   16.8
M3ID         21.4   6.3     56.6   15.7    30.8   10.4    62.2   18.1    32.2   11.5    49.5   17.2
Woodpecker   24.9   7.5     57.6   16.7    31.2   10.8    60.8   17.6    31.1   12.3    51.8   16.3
HALC         21.7   7.1     51.0   14.8    24.5   8.0     53.8   15.7    28.2   9.1     49.6   15.4
Ours         20.0   6.2     49.8   14.3    23.5   8.2     52.2   15.5    27.3   8.4     48.0   14.3
```
**Key extracts**:
- `Regular max128 LLaVA-1.5: CHAIRS = 55.0, CHAIRI = 16.3` ✓
- `VCD max128 LLaVA-1.5: CHAIRS = 54.4, CHAIRI = 16.6` ✓ (marginal improvement; CHAIRI slightly worse)

### A.3 MME-Hallucination Table 3 Verbatim (p.3230 of PDF, LLaVA-1.5 only)

```
Method      Existence↑  Count↑   Position↑  Color↑   MME Score↑
Regular     173.75       121.67   117.92     149.17   562.50
DoLa        176.67       113.33    90.55     141.67   522.22   ← WORSE than Regular
OPERA       183.33       137.22   122.78     155.00   598.33
VCD         186.67       125.56   128.89     139.45   580.56
M3ID        186.67       128.33   131.67     151.67   598.11
Woodpecker  187.50       125.00   126.66     149.17   588.33
HALC        183.33       133.33   107.92     155.00   579.58
Ours        191.67       145.55   136.66     161.66   635.55
```
**CRITICAL FINDING**: `DoLa MME = 522.22 < Regular 562.50`. DoLa **degrades** MLLM performance. Not a valid positive baseline for BRA.

### A.4 Efficiency Table 5 Verbatim (p.3231 of PDF)

```
Method    Avg Latency↓     GPU Memory↓      CHAIRS↓  MME↑    POPE↑   MMBench↑  MM-Vet↑
Regular   3.46s (×1.00)    14944MB (×1.00)   55.0    562.5   83.44    64.5      31.6
DoLa      9.32s (×2.69)    14951MB (×1.00)   55.4    522.2   82.62    62.5      30.9
OPERA     24.70s (×7.12)   22706MB (×1.52)   52.6    598.3   88.85    64.4      32.0
VCD       6.97s (×2.01)    15749MB (×1.05)   54.4    580.6   87.15    64.6      30.9
M3ID      7.18s (×2.07)    15726MB (×1.05)   50.0    577.2   88.03    64.2      31.4
Ours      3.70s (×1.07)    14951MB (×1.00)   49.8    635.6   89.10    65.0      32.8
```
**Key extracts**:
- `VCD latency: ×2.01 (+101%), Memory: ×1.05 (+5%)` ✓
- `OPERA latency: ×7.12 (+612%), Memory: ×1.52 (+52%)` ✓
- `OPERA POPE (Random) F1 = 88.85` ✓
- `DoLa latency: ×2.69 (+169%), POPE = 82.62 (below Regular 83.44!)` ✓ — DoLa even HURTS POPE

---

## §B. OPERA Paper (arXiv:2311.17911v3) — OPERA Own Numbers

**HTML**: `https://arxiv.org/html/2311.17911v3`

### B.1 CHAIR Table 1 Verbatim (max_tokens=512, n=500 COCO val 2014)

```
Method       InstructBLIP      MiniGPT-4        LLaVA-1.5        Shikra
             CS     CI         CS     CI         CS     CI         CS     CI
Greedy       [not parsed due to HTML rendering issue — see beam values below]
Beam Search  [baseline values visible at max512]
DoLa         [see Table 2 below]
OPERA(Ours)  46.4   14.2       26.2   9.5        44.6   12.8       36.2   12.1
```
**Extraction note**: OPERA Table 1 HTML renders poorly; only OPERA row was fully extracted. LLaVA-1.5 OPERA CHAIRS=44.6 at max_tokens=512.

**⚠️ CITATION CONFLICT (Protocol Variance)**:
- OPERA own paper (max_tokens=512, Beam Search): LLaVA-1.5 CHAIRS=44.6
- ONLY Table 5 cross-eval (max_tokens=128, Sampling): LLaVA-1.5 OPERA CHAIRS=52.6
- **Resolution**: These are not contradictory — different token limits produce different CHAIR scores. Use 52.6 for cross-method comparison at max128; use 44.6 when citing OPERA's own protocol.

---

## §C. FarSight (CVPR 2025) — Previously Verified

**PDF**: `https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Seeing_Far_and_Clearly_Mitigating_Hallucinations_in_MLLMs_with_Attention_CVPR_2025_paper.pdf`

### C.1 FarSight Table 2 Verbatim (p.26152 of PDF)

Column order: `[MMBench / LLaVAW / MM-Vet / VizWiz / SQA / CHAIRS / CHAIRI / POPE-R / POPE-P / POPE-A]`
```
LLaVA-1.5           64.3  72.5  30.5  48.5  64.5  48.0  13.9  87.0  82.8  76.6
+ FarSight (Ours)   66.0(+1.7) 74.7(+2.2) 32.5(+2.0) 50.8(+2.3) 67.4(+2.9) 41.6(+6.4) 13.2(+0.7) 90.5(+3.5) 86.1(+3.3) 80.4(+3.8)
```
⚠️ FarSight reports **POPE-R Accuracy** (not F1). Baseline=87.0 (Accuracy), FarSight=90.5 (+3.5).
⚠️ FarSight baseline CHAIRS=48.0 (vs ONLY baseline=55.0). Protocol variance: likely different max_tokens.

---

## §D. SCPO (ICLR 2026) — Previously Verified

**HTML**: `https://arxiv.org/html/2509.24491v1`

### D.1 SCPO Table 1 Verbatim (§4.1, vs LLaVA-v1.6-7B backbone)

```
| LLaVA-v1.6-7B § | 12.0 | 6.8 | 8.7 | 61.1 | 49.7 | 4.2 | 73.2 | 82.8 | 87.0 | 2.80 | 0.43 |
| +SCPO-7B (ours) | 7.0(↓6.6) | 4.4(↓2.9) | ... | 85.4(↑2.0) | 89.2(↑1.8) | ... |
```
CHAIRS = 7.0(↓5.0), AMBER-Disc F1 = 89.2(↑1.8). **POPE NOT REPORTED.**

### D.2 SCPO Table 2 Verbatim (§4.2, vs LLaVA-v1.5-7B backbone)

```
| LLaVA-v1.5-7B § | 54.7 | 26.5 | ...
| +Hard(ours)      | 20.3(↓34.4) | 11.6(↓14.9) | ...
```
Self-consistency check: 34.4/54.7 = **62.9%** = abstract's "reducing hallucination by up to 62.9%" ✓

---

## §E. VidHalluc (CVPR 2025) — Official Project Page

**Source**: `https://vid-halluc.github.io/` (official project website)
**arXiv**: `https://arxiv.org/abs/2412.03735`

### E.1 Official Leaderboard Table (from project website)

```
Model               Params  Binary QA  MCQ    TSH    STH    Avg
Human               —       95.14      93.29  90.17  87.43  91.51
GPT-4o              —       81.17      90.97  83.42  74.17  82.43
Gemini-1.5-Pro      —       75.02      79.04  82.67  64.11  75.21
VILA 1.5            13B     57.75      81.95  68.84  35.04  60.90
VideoLLaMA2         7B      48.23      83.79  22.50  65.22  54.94
LLaVA-NeXT-Video    34B     26.04      77.57  20.67  44.39  42.17
PLLaVA              13B     35.04      77.31  17.83  32.94  40.78
Video-LLaVA         7B      23.88      65.18  28.83  30.12  37.00
Chat-UniVi          13B     23.20      55.07  32.50  31.55  35.58
SharGPT4Video       8B      29.58      44.83  49.00  17.08  35.12
Video-ChatGPT       7B       9.36      23.25  29.83   8.13  17.64
```
**Video-LLaVA 7B Avg = 37.00** ✓ — Directly verified from official project page.

---

## §F. HallusionBench (CVPR 2024) — Original Leaderboard

**Source**: `https://github.com/tianyi-lab/HallusionBench` (official repository README)

### F.1 Original Leaderboard (from GitHub README)

```
Model                               Question Pair Acc  Figure Acc  Easy Q Acc  Hard Q Acc  Question Acc
GPT4V Sep2023 (Human Eval)          31.42              44.22       79.56       38.37       67.58
LLaVA-1.5 (Human Eval)              9.45               25.43       50.77       29.07       47.12
Qwen-VL (GPT Eval)                  5.93                6.65       31.43       24.88       39.15
```
**TERMINOLOGY NOTE for BRA project**:
The user's "Language-Bind" / "Vision-Bind" maps to HallusionBench's:
- **VD (Visual Dependent)** = "Vision-Bind" — questions requiring genuine image comprehension
- **VS (Visual Supplement)** = "Language-Bind" — questions answerable via language priors without image
- Hard Question Accuracy ≈ VD performance (higher = model truly uses visual input)

### F.2 VLMEvalKit Results (March 2025 data)

**Source**: VLMEvalKit GitHub issue #825 (https://github.com/open-compass/VLMEvalKit/issues/825)
```
Model                   aAcc    fAcc    qAcc    Avg (~)
Qwen2-VL-7B-Instruct   67.92   37.86   43.95   49.91
Qwen2.5-VL-7B-Instruct 71.81   47.97   47.25   55.67
```
**VLMEvalKit metric mapping**: aAcc = answer accuracy; fAcc = figure/consistency accuracy; qAcc = question-pair accuracy.

---

## §G. FREAK (ICLR 2026) — Status: Leaderboard Not Yet Public

**OpenReview**: `https://openreview.net/forum?id=YeagC09j2K`
**GitHub**: `https://github.com/Hans-M-Yin/FREAK`

As of 2026-03-20, the FREAK paper is accepted to ICLR 2026. The GitHub repository states the leaderboard and per-model scores will be released publicly via VLMEvalKit integration. **No numerical per-model scores are available in public literature.** The paper's own evaluation (which includes Qwen2-VL) was submitted during double-blind review and per-model details are in the PDF, which was not accessible via WebFetch (PDF directly behind OpenReview wall).

**Action item for BRA Node 5**: Monitor https://github.com/Hans-M-Yin/FREAK and https://huggingface.co/datasets for FREAK dataset + leaderboard release. Estimated: Q1-Q2 2026.

---

## §H. Citation Conflicts Summary

| Conflict | Methods | Source A | Source B | Resolution |
|---------|---------|---------|---------|------------|
| VCD CHAIR_s at max128 | VCD CVPR'24 vs ONLY ICCV'25 | VCD own paper: improvement reported | ONLY Table 2: CHAIRS=54.4 (marginal -0.6) | Use ONLY's standardized protocol value 54.4 |
| OPERA CHAIR at different max_tok | OPERA own paper vs ONLY Table 5 | OPERA own (max512): 44.6 | ONLY Table 5 (max128): 52.6 | Use max128 value (52.6) for cross-method comparison |
| FarSight baseline CHAIRS | FarSight vs ONLY | FarSight: baseline=48.0 | ONLY: baseline=55.0 | Protocol variance: different max_tokens. Not a numerical error. |
| DoLa on POPE | Expected baseline | ONLY Table 5: DoLa POPE=82.62 (below Regular=83.44) | — | DoLa HURTS POPE. Not a valid positive baseline. |
| OPERA uses Beam Search | OPERA vs all others | OPERA: Beam Search n=5 | VCD/ONLY/Regular: greedy/sampling | Add asterisk to OPERA numbers in Table 1; beam search fundamentally different from greedy |

---

## §I. NOT_FOUND Items — Action Items for BRA Team

| Item | Reason Not Found | Action |
|------|-----------------|--------|
| Qwen3-VL-8B POPE Adversarial (official) | Not in tech report arXiv:2511.21631 | Run locally on adversarial split; cite as "Internal Eval" |
| Qwen3-VL-8B FREAK per-category | FREAK leaderboard not yet public | Monitor GitHub/VLMEvalKit; check back Q2 2026 |
| OPERA POPE Adversarial | Not reported in OPERA paper or ONLY cross-eval | Use Random split F1=88.85 with caveat |
| DoLa POPE + CHAIR for MLLM | Paper focused on text LLMs; VLM adaptation not evaluated on POPE | Use MME proxy (522.22) with "NOT RECOMMENDED AS POSITIVE BASELINE" note |
| Qwen3-VL-8B HallusionBench (8B) | 8B not in Qwen3-VL tech report's HallusionBench; 32B present | Run VLMEvalKit locally or cite Qwen2-VL as proxy |
| VidHalluc Qwen3-VL | Not tested on VidHalluc (video benchmark) | Optional: run Vid-HEAL baseline; cite Video-LLaVA as cross-arch anchor |
