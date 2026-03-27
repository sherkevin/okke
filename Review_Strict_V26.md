# Review_Strict_V26
## Overall Score
Score: 3/5

## Verdict
The paper proposes a highly structured, falsifiable protocol for decode-time token-local visual intervention in MLLMs. The mathematical formulation of the problem (input-output embedding asymmetry, post-hoc entanglement, structural collapse) is exceptionally clear. However, because the empirical execution is currently pending, the paper is evaluated strictly on the soundness of its hypotheses and the rigor of its experimental plan. While the proposed Vocabulary-Anchored Semantic Masking (VASM) and the 3-chain evidence protocol are robust, the methodology requires critical clarifications regarding logit normalization before Top-$k$ pooling, and the overarching claims must be slightly de-escalated. If the proposed protocol is executed rigorously without gaming the baselines, this will be a strong contribution to ACM Multimedia.

## Summary
The paper introduces Bounded Resonance Anchoring (BRA), a decode-time framework designed to inject token-local visual evidence into MLLM text generation while preserving linguistic structure and reasoning capabilities. It addresses structural challenges via three mechanisms: (1) native or lightweight calibrated (`BRA_calib`) projection of final-layer visual states to the terminal vocabulary space; (2) Adaptive Top-$k$ pooling to isolate localized patch support; and (3) VASM, a deterministic BPE-aware masking strategy that protects syntax and multi-token entities. The paper presents a pre-registered experimental protocol divided into three evidence chains: Hallucination Reduction, Structure Preservation, and Local Evidence Value. 

## WhatShouldBeKept
1. **The VASM Mechanism:** The combination of a WordNet-augmented dictionary with dynamic BPE continuation inheritance (e.g., tracking `Ġ` or subword fragments) is the most practically valuable and structurally sound innovation in the paper. It elegantly solves a major flaw in current hallucination mitigation techniques.
2. **The Honest `BRA_zero` vs. `BRA_calib` Split:** Acknowledging the "post-hoc entanglement" and spatial washout in 1D sequences (like LLaVA), and transparently introducing a frozen 5k-image linear projector as a "calibrated" track is intellectually rigorous. Keep this honest framing.
3. **The Scoping to Image-LLMs:** Explicitly restricting the scope to 2D spatial grids and dropping video is a mature scientific decision. Do not artificially force video back into the paper just to meet perceived ACM MM narrative expectations; your spatial density arguments are sufficient.
4. **AGL Tracking in Chain A:** Mandating the reporting of Average Generated Length (AGL) alongside POPE/CHAIR to prevent "truncation gaming" is an excellent evaluation standard.
5. **The Baseline Framing:** Treating DoLa, VCD, and OPERA as "competitive, orthogonal regularizers" rather than strawmen. Do not slide back into claiming these baselines rely on global pooling; your current framing of them as purely language-side/attention-side heuristics is accurate and safe.

## MajorWeaknesses
1. **Mathematical Ambiguity in $S_{raw}$ Calculation:** In Section 3.2, you compute $S_{raw}(c) = \frac{1}{k}\sum_{j} \frac{logit^{(v_j)}[c]}{\tau_{sim}}$. You are summing raw, unbounded logits across different patches $v_j$. If one specific visual patch has an inherently higher magnitude in its hidden state, it will dominate the Top-$k$ sum, skewing the resonance. You must specify whether $logit^{(v_j)}$ is temperature-normalized or softmaxed *per-patch* across the vocabulary before being aggregated for candidate $c$.
2. **The "Resonance" Overclaim:** The terminology "Bounded Resonance Anchoring" is overly grandiose for what is fundamentally "Token-Local Logits Intervention with Structural Masking." While the acronym BRA can stay, the narrative should be aggressively down-scoped to focus on the concrete mechanics (VASM + local evidence) rather than inventing a new grand theoretical paradigm.
3. **Lack of Detail on $\Phi_{calib}$ Dimensionality:** Section 3.1 states $\Phi_{calib}$ is a "single linear layer." You must explicitly state whether it projects the visual hidden state dimension $D_{vision}$ to the LLM's final hidden dimension $D_{llm}$ (so it can be dot-producted with $W_{vocab}$) or directly to the $|V|$-dimensional logit space. The former is a drastically smaller parameter space and is highly preferred.

## SectionBySectionComments
*   **Abstract & Introduction:** Strong problem definition. The explicit recognition of embedding asymmetry is well articulated.
*   **Section 3.1 (`BRA_zero` vs `BRA_calib`):** The InfoNCE loss formulation is solid. The strict negative sampling (IoU < 0.1) is exactly what is needed to avoid penalizing overlapping bounding boxes. 
*   **Section 3.2 (Adaptive Top-$k$):** The lower bound $k_{min}$ is practically necessary. However, as noted in Major Weaknesses, the aggregation of $logit$ requires explicit normalization steps to be mathematically sound.
*   **Section 3.3 (VASM):** The explanation of `_rhi`, `no`, `cer`, `os` is brilliant and demonstrates a deep understanding of autoregressive decoding failure modes. 
*   **Section 4 (Evaluation Protocol):** The structural division into Evidence Chains A, B, and C is the strongest part of the paper. It leaves no room for ambiguous claims.

## RequiredRevisions
1. **Fix Equation 2 ($S_{raw}$):** Explicitly define a normalization step for $logit^{(v_j)}$ prior to aggregation so that patches with massive L2-norms do not mathematically override the Top-$k$ selection.
2. **Clarify $\Phi_{calib}$:** Add one sentence explicitly defining the input and output tensor dimensions of the linear layer $\Phi_{calib}$. Confirm it projects to the latent dimension prior to $W_{vocab}$, not the full vocabulary dimension.
3. **Refine Terminology:** Strip out the rhetorical fluff. Rely on the strength of "token-local visual evidence" and "structure-preserving masking."

## SuggestedFiguresTablesExperiments
Since your experiments are pending execution, I am providing strict guidance to ensure your protocol passes peer review once completed:
1. **Chain C Crux (`BRA_MeanPool` setup):** When executing Table 3, `BRA_MeanPool` *must* use the exact same $\Phi_{calib}$ projector and the exact same VASM mask as `BRA_AdaptiveTopK`. The *only* difference must be that `MeanPool` averages all $N_v$ patches while `AdaptiveTopK` selects the subset. If you cripple the MeanPool baseline, Evidence Chain C will be rejected.
2. **Execution of Defense Line 1 (Locality Check):** For Appendix A, when showing the Cosine Similarity heatmap of the LLaVA 1D sequence reshaped to 2D, you must provide a side-by-side comparison of `lm_head` (native) vs $\Phi_{calib}$ mapping the word "dog" to a picture of a dog. This will visually prove why `BRA_calib` was necessary.
3. **Table 1 (AGL Reporting):** Do not hide the AGL metric in the appendix. It must be a primary column in Table 1 next to the POPE F1 score. 
4. **Table 2 (`BRA_no_VASM` Ablation):** This is your most important ablation. Ensure `BRA_no_VASM` applies the exact same visual penalty weight $\alpha$ to *all* tokens (including functional words). This will cause the reasoning collapse you hypothesize on MMMU, perfectly validating the necessity of VASM.
5. **Figure 1 (DocVQA Failure Case):** The proposed failure case must include three elements: (a) The input image with tiny text; (b) The attention/resonance heatmap of `MeanPool` showing diffuse activation over the whole page; (c) The heatmap of `AdaptiveTopK` showing a sharp spike exactly on the text bounding box.

## AcceptanceOutlook
The methodology and experimental protocol are rigorously designed and address a genuine, mathematically complex problem in MLLM decoding. If the authors correct the minor mathematical ambiguities in logit aggregation, explicitly clarify the projector dimensions, and execute the pre-registered 3-chain evidence protocol *exactly* as proposed without compromising baseline fairness, this paper has a very high probability of acceptance at ACM Multimedia. Focus solely on empirical execution and defending the token-local vs. global pooling hypothesis.