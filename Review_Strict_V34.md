# Review_Strict_V34

## Overall Score
Score: 4/5

## Verdict
This is a highly mature, conceptually rigorous experimental proposal. The authors have successfully defined a tight, falsifiable hypothesis centered purely on token-local visual evidence and structure-preserving decode-time intervention. By explicitly dropping unsupported spatiotemporal video claims, introducing the `TSLI_zero` vs. `TSLI_calib` boundary, and framing DoLa/VCD/OPERA as orthogonal baselines rather than a problem definition, the paper is structurally sound. The acceptance of this work hinges entirely on the flawless execution of the proposed experimental syllabus.

## Summary
The paper proposes Thresholded Spatial Logits Intervention (TSLI), a decode-time method to mitigate MLLM hallucination by directly injecting token-local visual evidence into candidate logits. To handle architectural spatial washout and language structure preservation, it introduces three mechanisms: a dual-track feature projection (`TSLI_zero` vs a lightweight `TSLI_calib`), Threshold-Gated Adaptive Top-$k$ pooling, and Vocabulary-Anchored Semantic Masking (VASM) with BPE continuation inheritance. The paper is currently in a pre-registered state, outlining a strict 3-chain evaluation protocol (Hallucination Reduction, Structure Preservation, Local Evidence Value) alongside explicit VRAM/latency constraints.

## WhatShouldBeKept
1. **The Affirmative Framing:** Maintain the current framing. Treating DoLa, VCD, and OPERA as orthogonal regularizers rather than a generalized "global pooling" strawman is intellectually honest and strengthens your core proposition.
2. **Domain Bounding:** The explicit restriction to Image-LLMs and 2D spatial reasoning must remain. Do not reintroduce video experiments; it would dilute the precise local-evidence claims made here.
3. **The `Base + 5k LoRA` Baseline:** This is a brilliant experimental design choice. Matching the LoRA parameter count to $\Phi_{calib}$ perfectly isolates the architectural benefit of decode-time logits intervention versus mere exposure to spatial training data.
4. **VASM BPE Inheritance:** The dynamic subword inheritance mechanism addressing raw byte fallbacks and multi-token entities is technically sound and vital for reproducible decode-time interventions.
5. **Acknowledge the "Arrogance of Language Priors":** Tracking and reporting the false negatives of the entropy gate (highly confident hallucinations) is excellent scientific transparency. Keep this constraint explicit.

## MajorWeaknesses
While the theoretical design is strong, the *execution plan* harbors severe operational risks that must be mitigated before final submission:
1. **The `TSLI_calib` InfoNCE Negative Sampling Risk:** Training $\Phi_{calib}$ on bounding box crops using InfoNCE is perilous. Bounding boxes inherently contain background pixels. If your negative samples ($c^-$) share background textures with the positive target, the contrastive loss will destroy the spatial gradient. You must refine the sampling logic.
2. **Brittleness of the $0.90$ Entropy Gate:** Hardcoding an absolute threshold of 0.90 for the base LLM's confidence is naive. MLLM calibration varies wildly across architectures (e.g., LLaVA vs Qwen-VL). A static threshold will either trigger TSLI unnecessarily (destroying latency) or miss critical hallucinations.
3. **The Top-$M$ Truncation vs. VASM Collision:** You propose bounding intervention to the top-$M$ candidates. However, if a valid visual noun candidate is ranked at $M+1$ by the base model, VASM cannot rescue it. The interaction between the Top-$M$ cutoff and VASM's dictionary must be empirically audited.
4. **Latency Profiling in Batched Environments:** Reporting A100 Tokens/Sec and VRAM at Batch Size = 1 is insufficient for ACM MM. Decode-time interventions that require cross-referencing vocabulary $V$ and visual patches $N_v$ often cause catastrophic Out-Of-Memory (OOM) errors when batched. 

## SectionBySectionComments
- **Abstract & Intro:** Excellent scoping. The transition from identifying the problem to defining the structural barriers (asymmetry, entanglement, collapse) is logical and sets up the method perfectly.
- **Section 3.1 (`TSLI_calib`):** You state $\Phi_{calib}$ is a "single linear layer". If the post-hoc entanglement in the LLM's final layers is highly non-linear, a single linear layer may fail to project the contextualized visual states back into a linearly separable spatial grid. You should prepare a 2-layer MLP fallback ablation just in case the linear layer underfits the 5k COCO subset.
- **Section 3.2 (Threshold Gating):** Using the moving median across spatial patches for $\theta_{noise}$ is clever, but requires justification. Why median and not mean + $1\sigma$? You need an ablation showing how this dynamic threshold behaves during dense vs. sparse image contexts.
- **Section 3.3 (VASM):** WordNet relies on disambiguated synsets. How do you handle polysemy at decode time when the context is still being generated? (e.g., distinguishing "mouse" the animal vs "mouse" the device before the sentence finishes). You must clarify the disambiguation logic or admit it as a brute-force superset.

## RequiredRevisions
1. **Refine InfoNCE Sampling (Chain A prep):** Instead of raw bounding boxes for the 5k COCO pairs, apply an off-the-shelf segmentation model (e.g., SAM) to the bounding boxes to extract strictly foreground visual tokens. If you must use bounding boxes, institute a strict Intersection-over-Union (IoU) penalty to ensure negative patches have zero overlap with the positive object's bounding box.
2. **Dynamic Entropy Calibration:** Replace the static $0.90$ threshold. Implement a calibration step: run the base model on a small validation set, plot Confidence vs. Hallucination Probability, and dynamically set the entropy gate where the hallucination rate statistically diverges.
3. **Extend Latency Constraints:** The latency and VRAM profiling (Figures 2 & 3) *must* include lines for Batch Size = 1, 4, and 8. If TSLI cannot scale beyond BS=1, state clearly in the limitations that it is an offline analytical tool.
4. **Address Top-$M$ Unrecoverability Explicitly:** In your evaluation, track exactly how many target visual nouns were natively ranked outside the top-$M$ window by the base model. This sets the absolute mathematical ceiling for your method.

## SuggestedFiguresTablesExperiments
To complete your execution syllabus, adhere strictly to these deliverables:
- **Chain A (Table 1):** Execute exactly as planned. Ensure the `Base + 5k LoRA` is run across 3 different random seeds to report variance, proving your decode-time intervention is statistically superior to fine-tuning noise.
- **Chain B (Scatterplot):** The proposed MMMU Performance vs. OOV Rate scatterplot is mandatory. Ensure the axes are clearly legible and categorized by question type (e.g., spatial layout vs. abstract logic).
- **Chain C (Failure Analysis / Heatmaps):** In your 4-way visualization, you *must* include an "Entanglement Washout" failure case. Show a specific image where `TSLI_zero` highlights the wrong area (e.g., top-left corner bias) but `TSLI_calib` correctly highlights the object. This empirically proves the necessity of your methodology's dual-track design.
- **Semantic Distractor (Appendix D):** Execute the proposed "dog vs. cat" patch collision analysis. This is critical to prove whether TSLI acts as a pure spatial pointer or if it leaks semantic conflation.

## AcceptanceOutlook
The methodology is exceptionally tight, the claims are properly bounded, and the experimental protocol is rigorous and hostile to its own hypotheses. If the authors execute this exact syllabus—particularly overcoming the $\Phi_{calib}$ training risks and reporting honest latency/VRAM figures at scale—this paper will be a highly compelling, top-tier contribution to ACM Multimedia.