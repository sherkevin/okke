# Revision_Log_V46
- **Title & Branding Shrink:** Directly followed the reviewer's advice to strip the grand "Bounded Region Alignment (BRA)" buzzword, replacing it with the functionally accurate "Token-Local Visual Intervention (TLVI)".
- **Methodology Fallback Mechanisms:** 
  - Added a strict diagnostic fallback in 3.1: if `TLVI_zero` fails, it is mathematically invalidated and minimized in reporting.
  - Added an InfoNCE sampling fallback in 3.1 to skip anchors in highly dense/overlapping images where no negative meets the spatial exclusion rule.
  - Explicitly detailed VASM's greedy string matching for non-prefix tokenizers like Qwen's `tiktoken` in 3.3.
- **Latency Honesty:** Acknowledged the massive TTFT spike caused by the 85k dictionary prefill scan. Added a mandatory "ITL Multiplier" column to Table 2 (Chain C) to enforce the reviewer's 3-5% gap threshold over `MeanPool`.
- **Ablation & Appendices:** Added an ablation study for the Top-$k$ sparsity parameter $\rho$. Expanded Appendix D to explicitly demand qualitative failure cases for polysemy and dense overlaps.
- **Retained Highlights:** Maintained the strict 2D Image-LLM scope, orthogonal framing of baselines, `Base + 5k LoRA` control, and the VASM inaction conflation honesty.