# Review_Strict_V75
## Overall Score
Score: 3/5

## Verdict
This manuscript presents a highly disciplined, intellectually honest framework for decode-time visual grounding intervention. It correctly abandons grandiose claims of "defeating all baselines" in favor of a falsifiable, positive proposition: injecting token-local visual evidence at decode-time can reduce hallucination without destroying language structure, provided adequate safeguards (VASM) are in place. As a methodological blueprint, it is rigorous. However, because the empirical execution is currently in the planning/pilot stage, the methodology still harbors critical ambiguities—specifically regarding the nature of the `TLRA_calib` training pipeline and the mathematical stability of the logits penalty. If the authors execute the proposed experimental contract strictly and address the methodological holes, this can be a solid, narrow, but highly credible ACM Multimedia contribution. 

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a decode-time intervention for Multimodal Large Language Models (MLLMs). It reweights candidate tokens based on localized visual support using an Adaptive Top-$k$ mechanism. To prevent structural collapse of the generated text, it introduces Vocabulary-Anchored Semantic Masking (VASM) to protect functional syntax and multi-token entities via BPE inheritance. Recognizing the difficulty of zero-shot embedding alignment, the authors split the method into a strict zero-shot probe (`TLRA_zero`) and a lightweight calibrated plug-in (`TLRA_calib`). The experimental design is framed around three evidence chains: Hallucination Reduction, Structure/Reasoning Preservation, and Local Evidence Value, with an explicit "OCR concession" treating document-heavy tasks as negative controls.

## WhatShouldBeKept
1. **The `TLRA_zero` vs. `TLRA_calib` Separation:** This is the most intellectually honest part of the paper. Acknowledging that visual and lexical states often do not natively align, and explicitly isolating the calibrator, prevents the "fake zero-shot" trap that plagues this subfield.
2. **The Parity Ablation Rule:** Mandating that `TLRA_AdaptiveTopK` must be compared against `TLRA_MeanPool` using the *exact same* calibrator weights is excellent. This is the only way to prove that the *routing* mechanism is working, rather than the calibrator just injecting better global priors.
3. **VASM’s BPE Continuation Inheritance:** Intervening on root tokens and inheriting masks for subword continuations is a mechanically sound solution to the suffix-collapse problem. 
4. **Video as a Secondary Pilot:** Retaining video only as an exploratory appendix item is the correct scope bound for ACM MM. Do not elevate it back to the main text unless you have comprehensive spatio-temporal baselines, which you currently lack.

## MajorWeaknesses
1. **The Calibrator Blackbox (`Phi_calib`):** You state `TLRA_calib` trains a small projection to improve alignment. *How?* On what data? With what objective function? If `Phi_calib` is trained on high-quality dense grounding datasets, it is absorbing external domain priors. Even if the base model is frozen, you are injecting learned multimodal alignment. The paper currently waves this away with a "Category Leakage Audit." This is insufficient. You must explicitly define the training data and objective of `Phi_calib` in Section 3.1. 
2. **The Logits Penalty Formulation:** Your equation $L_{final}(c) = L_{orig}(c) - \alpha \cdot \gamma(c) \cdot (1 - \hat S(c))$ is a pure subtractive penalty. If $\alpha$ is large, you risk pushing the logits of visual tokens entirely out of the active dynamic range, heavily skewing the softmax temperature of the base model. Have you considered scaling the original logits instead of subtracting an absolute scalar? 
3. **The OCR / Document Contradiction:** You explicitly introduce an "OCR concession," stating that arbitrary OCR strings receive near-zero VASM coverage, and designate `DocVQA` as a negative control. Yet, you mention "small-object recognition, spatial positioning, and counting" as your motivation. If your method actively bypasses text-in-image tokens, you must scrub any lingering motivation related to dense document reasoning or rich-text environments from your Introduction. You cannot claim to solve fine-grained grounding if you mathematically ignore text pixels.
4. **Pre-registration vs. Paper:** Currently, this reads like an excellent pre-registration document. The scientific value depends entirely on whether `TLRA_zero` actually survives Stage 0, and whether `AdaptiveTopK` actually beats `MeanPool`. If both fail, your contribution shrinks to "VASM is a good mask," which is insufficient for an oral or strong accept at this tier.

## SectionBySectionComments
* **Abstract & Intro:** The pivot away from macro-critiques of `DoLa`, `VCD`, and `OPERA` is refreshing. Keep them strictly as competitive alternatives. However, ensure the term "fine-grained grounding" is scoped to physical entities, not text.
* **Section 3.2 (Adaptive Top-k):** The definition of $k = \max(k_{min}, \lceil \rho \cdot N_v \rceil)$ is sound, but $\tau_{sim}$ needs empirical justification. If $\tau_{sim}$ is too high, the distribution flattens, and Adaptive Top-$k$ degenerates into MeanPool.
* **Section 3.4 (VASM):** The OCR concession is logical given your deterministic mask, but make sure Table 2 actually proves that DocVQA performance *does not drop*. A negative control means performance should be strictly identical to the base model, proving VASM successfully bypassed the intervention.
* **Section 4.1 (Stage 0):** This is a harsh but necessary pilot. If `TLRA_zero` yields random noise, be prepared to completely rewrite Section 3 to position `TLRA_calib` as the core method.

## RequiredRevisions
1. **Define the Calibrator:** Provide the exact mathematical formulation, training dataset, and loss function for `Phi_calib`. 
2. **Enforce Claim Contraction if Necessary:** If your Stage 0 pilot shows `TLRA_zero` is non-viable, you must explicitly state: "Decode-time local intervention requires minimal calibration to bridge the embedding asymmetry." Do not try to hide a failed zero-shot probe in an appendix.
3. **Internal Control Expansion:** In Evidence Chain C (Local Evidence), `TLRA_RandomK` is currently listed as "optional." Make it **mandatory**. To prove *spatial locality* matters, you must show that taking the Top-$k$ visual tokens is statistically better than taking a random $k$ subset of visual tokens. Otherwise, your performance gain might just be a regularization effect of using a subset of visual tokens.

## SuggestedFiguresTablesExperiments
Since the experiments are actively being executed, follow this strict matrix for your results section:

* **Table 1 (Evidence Chain A - Hallucination):** 
  * Columns: POPE (Accuracy, F1), CHAIR (CHAIRi, CHAIRs), Average Generation Length (AGL), and % of tokens intervened upon.
  * Rows: Base, VCD, OPERA, DoLa, `TLRA_MeanPool`, `TLRA_AdaptiveTopK`. 
  * *Critical Check:* If `TLRA_AdaptiveTopK` reduces AGL by >15% compared to Base, your hallucination reduction is confounded by response truncation.
* **Table 2 (Evidence Chain B - Structure):**
  * Columns: MMBench, MME, MMMU, DocVQA (Neg Control), Exact-Match Suffix Completion %.
  * Rows: Base, `TLRA_full`, `TLRA_no_VASM`.
  * *Critical Check:* `TLRA_no_VASM` should show a catastrophic drop in MMMU and Exact-Match Suffix. DocVQA must remain flat across Base and `TLRA_full`.
* **Table 3 (Evidence Chain C - Local Value):**
  * Columns: FREAK (Spatial, Existence, Counting splits).
  * Rows: Base, `TLRA_MeanPool`, `TLRA_RandomK`, `TLRA_AdaptiveTopK`.
  * *Critical Check:* All three TLRA variants here must use the exact same `Phi_calib`. 
* **Figure 1 (Pareto Frontier):** 
  * X-axis: Throughput (Tokens/sec) or TPOT. Y-axis: POPE F1 or FREAK Spatial score. 
  * Plot the trajectory as the candidate window $M$ varies $\in \{10, 50, 100\}$.
* **Failure Case Analysis (Visual):** 
  * Show at least two examples where the correct visual token was *not* in the base model's Top-$M$ logits, proving the fundamental limitation of decode-time reweighting. 
  * Show one example of suffix-collapse even with VASM active (edge cases where BPE splitting defies logical linguistic boundaries).

## AcceptanceOutlook
The manuscript lays down an exemplary methodological framework. It avoids the toxic trend of bashing baselines and instead constructs a highly falsifiable evaluation matrix. Its acceptance will hinge entirely on the execution of the proposed experiments. If the authors provide Table 1, 2, and 3 exactly as specified, validating the positive claim of local-evidence routing while explicitly defining the calibrator's training, this paper will be a strong candidate for acceptance. If the empirical results fail to separate `AdaptiveTopK` from `MeanPool`, or if the calibrator data introduces massive leakage, the claims must be aggressively downgraded to a purely engineering paper about VASM.