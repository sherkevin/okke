# Review_Strict_V81
## Overall Score
Score: 4/5

## Verdict
This draft presents a highly disciplined, methodologically sound framework for decode-time intervention in Multimodal LLMs. Unlike many contemporaneous submissions that rely on macro-level critiques of existing baselines or bloated, untestable claims, this paper correctly scopes its contribution: a verifiable proposition that token-local visual evidence can be injected into decode-time logits without destroying language structure. The introduction of Vocabulary-Anchored Semantic Masking (VASM) as an $O(1)$ latency bypass is an elegant engineering solution. The proposed evaluation protocol—specifically the Average Generation Length (AGL) audit and the strictly bounded use of DocVQA as a negative control—is exceptionally rigorous. If the authors execute the proposed experimental plan exactly as written, while addressing a few critical gaps in baseline parity, this paper will be a strong candidate for acceptance.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a method to inject localized visual patch support directly into the logit distributions of MLLMs at decoding time to reduce physical hallucination. To overcome the latency of dynamic part-of-speech tagging, it introduces VASM, a pre-computed WordNet static dictionary with an $O(1)$ BPE inheritance rule. The authors explicitly divide the method into a pure zero-shot probe (`TLRA_zero`) and a lightweight Test-Time Adaptation module (`TLRA_calib`). The paper outlines a strict experimental protocol based on three evidence chains: hallucination reduction bounded by an AGL audit, structure preservation validated by a negative control (DocVQA), and a spatial parity test strictly separating seen and unseen object categories to validate the Adaptive Top-$k$ mechanism against a MeanPool baseline.

## WhatShouldBeKept
1. **The Framing of Baselines:** Your treatment of DoLa, VCD, and OPERA as "highly competitive sequence-level heuristics" is mature and scientifically accurate. Do not change this. You have successfully avoided the trap of framing them as "failing due to global pooling." 
2. **The Test-Time Adaptation (TTA) Concession:** Explicitly labeling `TLRA_calib` as TTA and separating it from pure zero-shot methods is your strongest shield against reviewer attacks regarding parameter isolation. 
3. **The AGL Audit:** This is a brilliant methodological inclusion. Many intervention methods artificially boost POPE scores by acting as "silence engines" (forcing the model to output short, truncated answers). Measuring AGL eliminates this confounder.
4. **The DocVQA Negative Control:** Acknowledging the "OCR Concession" and using DocVQA purely as a flat-line negative control perfectly aligns your method's mechanics with its claims. 

## MajorWeaknesses
1. **The Disappearance of `TLRA_zero` in the Protocol:** You introduce `TLRA_zero` in Section 3.1 as the "Fairness Boundary" and the *only* variant directly comparable to VCD/DoLa regarding parameter isolation. However, `TLRA_zero` completely vanishes in your Experimental Design (Section 4). Tables 1, 2, and 3 only discuss `TLRA_calib` (MeanPool and AdaptiveTopK). This is a critical failure in experimental closure. `TLRA_zero` must be evaluated, even if its performance is weak, to establish the base viability probe.
2. **Ambiguity in VASM's "Physical Object" Definition:** You state that tokens "unambiguously mapped to physical objects/nouns are assigned $\gamma=1$." WordNet is vast and nuanced. Without a strict programmatic definition (e.g., tracing hypernym paths to `physical_entity.n.01` or `object.n.01`), this mask is irreproducible and vulnerable to cherry-picking.
3. **Calibrator ($\Phi_{calib}$) Training Bias:** In Evidence Chain C (Parity Test), you pit `TLRA_MeanPool` against `TLRA_AdaptiveTopK` using the *same frozen $\Phi_{calib}$ weights*. However, you do not specify how $\Phi_{calib}$ is trained. If $\Phi_{calib}$ is trained by pooling visual features, `MeanPool` gains an unfair advantage. If it is trained using local feature alignment, `TopK` gains an unfair advantage. The training objective of this projection layer must be strictly neutral.
4. **Video Modality Distraction:** You mention video in the limitations. Given your current benchmark scope, do not attempt to shoehorn video into the main narrative. If it does not form a true spatial-temporal token-local evidence argument, it will dilute your structural claims. 

## SectionBySectionComments

**1. Introduction:** Excellent restraint. You define the positive proposition clearly without macro-bashing baselines. Ensure you do not accidentally slip OCR motivations back into this section in future drafts. Keep the motivation strictly on "physical entities—such as small-object recognition, spatial positioning, and counting."

**3.3 VASM:** The $O(1)$ BPE inheritance is a highly defensible approach to the autoregressive latency paradox. The explicit concession regarding hyphenated words and extreme BPE fragmentation adds credibility. 

**4. Evaluation Protocol:** The three-chain structure is robust. However, as noted, it requires modification to include the zero-shot probe.

## RequiredRevisions
1. **Integrate `TLRA_zero` into Chain A:** Table 1 must include `TLRA_zero`. Your baseline block should be: Base, VCD, DoLa, OPERA, `TLRA_zero`. Your TTA block should be: `TLRA_MeanPool`, `TLRA_AdaptiveTopK`. This provides absolute fairness.
2. **Define the WordNet Masking Rule mathematically:** Add a footnote or a brief appendix reference explicitly stating the WordNet hypernym paths used to define $\gamma=1$. 
3. **Clarify $\Phi_{calib}$ Training:** Add a sentence in Section 3.1 explaining exactly how the 2-layer MLP is trained. It should ideally be trained via standard next-token prediction (cross-entropy) on the 50k captions *without* the TLRA logit intervention active, simply acting as a raw feature translator. This ensures the weights are neutral before the decode-time intervention is applied.
4. **Restrict Claims:** Maintain your core claim exactly as "token-local logits intervention + VASM + fair zero-shot/calibrated split." Do not invent new narratives as your experiments conclude. 
5. **Video:** If you run video experiments, relegate them strictly to a secondary pilot in the Appendix. Do not let them consume main text real estate.

## SuggestedFiguresTablesExperiments
To guarantee a watertight execution of your proposed protocol, structure your results exactly as follows:

*   **Table 1 (Hallucination & Truncation):** 
    *   *Columns:* POPE (Acc, F1), CHAIRi, CHAIRs, **AGL (Words/Tokens)**.
    *   *Rows:* (Zero-Shot Group): Base, VCD, DoLa, OPERA, `TLRA_zero`. (TTA Group): `TLRA_MeanPool`, `TLRA_AdaptiveTopK`.
    *   *Target:* `TLRA_AdaptiveTopK` dominates the TTA group; AGL remains within $\pm 5\%$ of the Base model.
*   **Table 2 (Structure & Negative Control):**
    *   *Columns:* MMBench, MME, MMMU(Hard), **DocVQA**.
    *   *Rows:* Base, `TLRA_no_VASM` (Ablation), `TLRA_full`.
    *   *Target:* `TLRA_no_VASM` crashes on MMMU. `TLRA_full` exactly matches Base on DocVQA ($\Delta \approx 0$).
*   **Table 3 (The Parity Test - FREAK):**
    *   *Columns:* Spatial Relations (Seen/Unseen), Existence (Seen/Unseen), Counting (Seen/Unseen).
    *   *Rows:* `TLRA_MeanPool`, `TLRA_AdaptiveTopK`.
    *   *Target:* `AdaptiveTopK` shows statistically significant $\Delta$ specifically on the *Unseen* splits, proving spatial routing (not just weight calibration) is doing the heavy lifting.
*   **Figure 2 (Latency Pareto):** Scatter plot. X-axis: TPOT (ms). Y-axis: POPE F1. Plot the $M \in \{10, 50, 100\}$ curve. This will perfectly illustrate the memory-bandwidth tax you hypothesized.
*   **Figure 3 (Failure Mode):** A step-by-step line graph over a single generated sentence. Y-axis is the rank of the target visual token. Show the horizontal threshold at $M=50$. Highlight the exact step where the token falls below $M$, permanently locking TLRA out of recovering it. This is a beautiful piece of scientific transparency.

## AcceptanceOutlook
The methodology is exceptionally well-scoped and the theoretical framing is mature. The reliance on structural commitments (VASM, bounded windows, negative controls) separates this from typical heuristic-chasing papers. If the authors execute the proposed experiments with the inclusion of `TLRA_zero` and maintain strict neutrality in their $\Phi_{calib}$ training, I will strongly champion this paper for acceptance.