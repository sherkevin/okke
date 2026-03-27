# Review_Strict_V73

## Overall Score
Score: 4/5

## Verdict
This is a highly disciplined, scientifically mature experimental blueprint. The authors have successfully pivoted from a generic critique of existing baselines to a legally binding, positive hypothesis: injecting token-local visual evidence into decode-time logits adjustment without destroying language structure. By explicitly defining operational boundaries—such as the "OCR Concession," the relegation of temporal video to an exploratory appendix, and the mathematical parity ablation against global pooling—the paper sets up a rigorously falsifiable framework. My score reflects the strength of this methodology. However, before the massive compute jobs are executed, several critical loopholes in the evaluation protocol (specifically regarding calibrator leakage, latency tracking, and BPE measurement) must be aggressively sealed to prevent the empirical results from becoming artifacts.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a decode-time intervention to reduce physical entity hallucination in Multimodal Large Language Models (MLLMs). It maps token-local visual patches to vocabulary logits via a frozen calibrator (`TLRA_calib`). To prevent structural collapse, it introduces Vocabulary-Anchored Semantic Masking (VASM) using a WordNet whitelist, intervening only on valid BPE prefixes and relying on the base model for suffix completion. The paper currently stands as an experimental contract, defining three strict evidence chains: Hallucination Reduction, Structure Preservation (including a DocVQA negative control), and Local Evidence Parity (proving Top-$k$ routing outperforms Mean Pooling). 

## WhatShouldBeKept
1. **The Positive Framing & Baseline Taxonomy**: Keep the current framing of `VCD`, `DoLa`, and `OPERA` as legitimate, highly competitive baselines for global trajectory modulation. Do not revert to claiming they "fail because they use global pooling." Your setup of TLRA as solving a completely different, orthogonal problem (token-local injection) is currently correct and must remain.
2. **The OCR Concession & DocVQA Negative Control**: Explicitly abandoning OCR intervention to guarantee 0% structural degradation is a fantastic scientific trade-off. Using `DocVQA` purely as a negative control (targeting a 0% VASM trigger rate) is mathematically elegant. Keep this exactly as planned.
3. **Video as an Exploratory Appendix**: Keep video out of the main 2D image-centric claims. The current framing acknowledges the lack of frame-specific temporal priors; do not bloat the main narrative with mechanically forced video benchmarks.
4. **The `TLRA_MeanPool` Parity Ablation**: This is the scientific crown jewel of the paper. Using the *exact same* external calibrator weights ($\Phi_{calib}$) for both global pooling and local Top-$k$ routing is the only mathematically valid way to prove your method isn't just a stealth object detector. 
5. **The AGL Asterisk Rule**: Penalizing methods that inflate CHAIR scores simply by collapsing the Average Generated Length (AGL) is excellent.

## MajorWeaknesses
1. **The "Zero Domain Overlap" Fallacy**: In Section 3.1, you claim to mathematically guarantee "zero spatial/domain image overlap" between your 100K calibrator subset (CC3M/Visual Genome) and POPE/CHAIR (which often rely on COCO). This is a mathematical fiction in zero-shot VLM evaluation; the semantic domains overlap massively. If you proceed with this definition, reviewers will tear it apart.
2. **BPE-CSR Metric is Too Loose**: In Section 4.2, BPE Completion Success Rate is defined using a character-level Levenshtein similarity $\ge 0.8$. This is fundamentally flawed. If your prefix intervention forces "table" and the model completes it as "tablet" (Levenshtein distance is close), the physical entity is entirely different. 
3. **Latency is Mentioned but Not Formalized**: You mention the Pareto frontier for candidate bounding ($M$), but you do not formalize the actual wall-clock metrics. Given that computing logits against $N_{active}$ patches at every step is highly expensive, failing to explicitly track standard system metrics will raise immediate red flags.
4. **The `TLRA_zero` Disconnect**: Section 3.1 mentions `TLRA_zero` (using raw visual embeddings without the calibrator) yields near-random overlap, but it is entirely missing from the Evidence Chain execution plan (Table 1).

## SectionBySectionComments
*   **Abstract/Intro**: The scope restriction ("designed exclusively for grounding physical entities in natural scenes") is excellent. You have successfully shrunk the claim to a defensible perimeter.
*   **Method (3.1 Calibration Protocol)**: The dual deduplication pipeline is solid, but the reliance on GroundingDINO introduces a massive external prior. This places absolute, non-negotiable weight on your Parity Ablation (Section 4.3). If `TLRA_AdaptiveTopK` does not statistically crush `TLRA_MeanPool`, your method is a failure.
*   **Method (3.2 Fallback)**: Prompt-Agnostic Normalized Entropy is a smart inclusion to bypass the prefill blindspot. However, what is the exact mechanism for setting $\theta_{fallback}$? If this requires per-dataset tuning, it breaks the zero-shot claim.
*   **Evaluation Protocol (4.1)**: The inclusion of AGL StdDev alongside AGL is a rigorous touch to catch models that collapse into robotic, uniform short answers.

## RequiredRevisions
1. **Redefine the Leakage Audit**: Drop the impossible claim of "zero domain overlap." Change this strictly to a **"Category Leakage Audit"**. Stratify your POPE/CHAIR results into `Entities Seen in Calibrator Pseudo-labels` vs. `Entities Unseen in Calibrator Pseudo-labels`. This is falsifiable and mathematically sound.
2. **Tighten BPE-CSR**: Discard Levenshtein $\ge 0.8$. BPE-CSR must be defined as an **Exact Semantic Match** of the target subword sequence. If the prefix $\gamma=1$ triggers, the resulting suffix must complete the exact intended physical entity, otherwise, it is a collapse. 
3. **Formalize Latency Metrics**: You must explicitly include **Time-To-First-Token (TTFT)** and **Time-Per-Output-Token (TPOT)** (in milliseconds/token) in your Pareto frontier evaluation. Bounding $M$ is meaningless if the baseline TPOT is fundamentally ruined.
4. **Clarify Fallback Threshold**: State explicitly in the protocol that $\theta_{fallback}$ will be fixed universally across all datasets, proving the prompt-agnostic normalization actually works.

## SuggestedFiguresTablesExperiments
To ensure your execution phase is airtight, modify your planned deliverables as follows:

*   **Table 1 Expansion (`TLRA_zero`)**: Add `TLRA_zero` (no $\Phi_{calib}$) to Table 1. This establishes the absolute floor of your embedding asymmetry claim and proves how much heavy lifting the calibrator is doing.
*   **Table 3 (System Overhead - New)**: Create a dedicated table reporting TTFT, TPOT, and peak GPU VRAM usage for `Base`, `VCD` (which requires batched forward passes), and `TLRA_AdaptiveTopK` (at $M=10, 50, 100$). This proves your method is a viable decode-time plugin.
*   **Figure 1 Parity Ablation Rigor**: For the Y-axis (Object Existence Accuracy vs Intervention Strength $\alpha$), you *must* plot error bands over at least 3 distinct prompt phrasing variants to prove that the gap between `AdaptiveTopK` and `MeanPool` is not prompt-fragile.
*   **Failure Analysis Visualization**: Include a direct text-generation trace showing a **Suffix Collapse**. Show Step $t$ (Prefix intervened, e.g., "re-") and Step $t+1, t+2$ (Model hidden states corrupt, generating gibberish or stuttering like "re-re-re"). Transparently showing your theoretical risk model occurring in practice builds massive reviewer trust.

## AcceptanceOutlook
The experimental blueprint is scientifically rigorous, defensively scoped, and refreshingly honest about its limitations. If the authors execute this exact experimental contract, strictly enforce the OCR negative controls, pass the `TLRA_MeanPool` parity ablation with a statistically significant gap, and tighten the BPE/Leakage definitions as requested, this paper will be a highly compelling, definitive contribution to ACM Multimedia. Execute the plan without inflating the narrative.