# Review_Strict_V79
## Overall Score
Score: 4/5

## Verdict
This is a methodologically mature, exceptionally well-constrained experimental proposal. The authors demonstrate an unusual level of scientific restraint by pre-registering a "Fallback Clause," strictly compartmentalizing zero-shot versus calibrated capabilities, and designing explicit negative controls. However, the theoretical framing of existing baselines remains slightly blurred, and the introduction teases modalities (OCR) that the method explicitly bypasses. If the authors execute this experimental plan exactly as proposed, without shifting goalposts when the data arrives, this will be a highly valuable contribution to ACM Multimedia.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a decode-time intervention to reduce physical entity hallucination in MLLMs. It dynamically reweights the logits of the top-$M$ candidate tokens based on localized visual support (Adaptive Top-$k$). To avoid the catastrophic latency of autoregressive part-of-speech tagging and to prevent the degradation of language structure, TLRA introduces Vocabulary-Anchored Semantic Masking (VASM)—an offline, static dictionary using WordNet combined with dynamic BPE inheritance. Because visual and lexical states lack native alignment, the paper evaluates a pure zero-shot probe (`TLRA_zero`) and a test-time adaptation variant (`TLRA_calib`). The methodology relies on a strict set of evidence chains: hallucination reduction (audited by Average Generation Length), structure preservation (audited by a DocVQA negative control), and local evidence parity (stratified by Seen vs. Unseen objects).

## WhatShouldBeKept
1. **Vocabulary-Anchored Semantic Masking (VASM):** This is your strongest, most indisputable technical asset. Solving the autoregressive NLP latency paradox via an offline static mask and BPE inheritance is highly practical and elegant.
2. **The "Fallback Clause" and Exact-Calibrator Parity:** The commitment to downgrade the core claim to `MeanPool + VASM` if spatial routing fails on Unseen objects is a gold standard for rigorous evaluation. Keep this front and center.
3. **The Average Generation Length (AGL) Audit:** Using AGL to prove that hallucination reduction isn't simply a byproduct of premature truncation is excellent.
4. **The DocVQA Negative Control:** Utilizing OCR as a flat-line mathematical negative control to prove VASM operates correctly is a brilliant piece of structural validation.
5. **Video as an Appendix Pilot:** You correctly identified that your formulation is spatially bound ($H \times W$). Keep video strictly in the appendix; do not let it contaminate your core narrative.

## MajorWeaknesses
1. **Mischaracterization of Baselines (DoLa, VCD, OPERA):** You refer to DoLa, VCD, and OPERA as "global heuristics." This terminology is dangerous because it flirts with the incorrect notion that these methods rely on "global visual pooling." They do not. DoLa is a layer-contrastive method; VCD is a noise-contrastive distribution method; OPERA is an attention-penalty method. You must clearly state that they are *sequence-level or distribution-level language heuristics*, rather than framing them as "global" in a way that implies spatial globality. They are your competitors, not the setup for your spatial argument.
2. **The OCR Motivation Contradiction:** In the Introduction (Bottleneck 4), you cite "dense document text (OCR)" as a modality mismatch bottleneck that makes your intervention non-trivial. Yet, in Section 3.4, you explicitly (and correctly) concede that VASM is mathematically blind to text-in-image tokens and bypasses them entirely. **You cannot use a problem to motivate your paper if your solution is to explicitly ignore it.** Remove OCR from the Introduction's bottlenecks and introduce it solely in the evaluation as a necessary boundary condition / negative control.
3. **The Asymmetry of `TLRA_calib` vs. Baselines:** You acknowledge that `TLRA_calib` uses a 50k-caption projection layer, turning it into a lightweight TTA. However, you are still placing it in Table 1 against DoLa and VCD, which are strictly zero-shot. While your transparency is appreciated, the *only* mathematically fair baseline for the spatial routing claim in `TLRA_calib` is `TLRA_MeanPool` (using the exact same 50k calibrator). You must ensure the text heavily weights the head-to-head against `MeanPool` rather than over-celebrating a win against DoLa using external parameters.

## SectionBySectionComments
*   **Abstract:** Excellent framing. The distinction between the zero-shot probe and the calibrated TTA is clear.
*   **1. Introduction:** Fix Bottleneck 4. Restrict your motivation to the physical object grounding problem where token-local evidence actually matters. Do not tease document understanding.
*   **3.1 Fairness Boundary:** Strong. Emphasize that `TLRA_zero` is functioning primarily as a viability probe. 
*   **3.4 VASM:** The explanation of the offline dictionary and $O(1)$ BPE inheritance is compelling. Ensure you explicitly state how hyphenated words that break WordNet exact-matches are handled (or admit them as a failure mode).
*   **3.5 Fallback Clause:** If your results show that `AdaptiveTopK` is not significantly better than `MeanPool` on the Unseen split, execute this clause without hesitation. A paper claiming "Decode-time token-local logits intervention + VASM + fair calibrated split" is still highly publishable at ACM MM. Do not artificially inflate the spatial claim if the data doesn't support it.
*   **4.5 Efficiency Pareto:** The TPOT audit is crucial. Extracting spatial support for $M=50$ candidates per step will hit memory bandwidth hard. Be prepared for this chart to look unfavorable compared to base decoding, and frame it honestly as the "cost of intervention."

## RequiredRevisions
1. **Refine Baseline Terminology:** Remove the word "global" when describing DoLa, VCD, and OPERA. Describe them accurately based on their actual mechanisms (layer contrast, noise contrast, attention penalty).
2. **Scrub OCR from Motivation:** Remove dense document parsing from the introductory bottlenecks. Introduce DocVQA exclusively in Section 3/4 as a negative control for lexical rigidity.
3. **Commit to the `TLRA_MeanPool` Showdown:** In your upcoming results text, the core scientific conclusion must be drawn from the delta between `TLRA_MeanPool` and `TLRA_AdaptiveTopK` on the Unseen split. Wins against VCD/DoLa by `TLRA_calib` should be treated as secondary, as they are confounded by the 50k calibrator priors.

## SuggestedFiguresTablesExperiments
Since you are executing a pre-registered protocol, your final tables must look exactly like this to satisfy this review:

*   **Table 1 (Hallucination & Truncation):** 
    *   *Columns:* POPE (Acc, F1), CHAIR (i, s), **AGL (Average Generation Length)**.
    *   *Rows:* Base, VCD, OPERA, DoLa, `TLRA_zero` (if viable), `TLRA_MeanPool` (Calib), `TLRA_AdaptiveTopK` (Calib). 
    *   *Check:* If AGL drops massively for TLRA, you must report it as a truncation artifact.
*   **Table 2 (Structure & Negative Control):**
    *   *Columns:* MMBench, MME, MMMU (Hard), DocVQA.
    *   *Rows:* Base, `TLRA_full_pipeline`, `TLRA_no_VASM`.
    *   *Check:* `TLRA_full_pipeline` must equal Base on DocVQA. `TLRA_no_VASM` must crash on MMMU.
*   **Table 3 (The Ultimate Parity Test - FREAK/Spatial):**
    *   *Columns:* Seen Objects (50k overlap), Unseen Objects (Zero overlap).
    *   *Rows:* `TLRA_MeanPool`, `TLRA_RandomK`, `TLRA_AdaptiveTopK`. (All using identical frozen TTA weights).
    *   *Check:* If AdaptiveTopK does not beat MeanPool on Unseen, execute the Fallback Clause in the text.
*   **Figure 1 (Latency Pareto):** Scatter plot of TPOT (ms/token) vs. POPE F1. Show the trajectory of $M=10, M=50, M=100$. 
*   **Failure Analysis:** Provide one clear visual example of the "Out-of-Candidate Bound" where the correct token falls to rank $M=51$ at step $t-1$, proving post-hoc intervention was mathematically impossible.

## AcceptanceOutlook
If the authors populate the proposed tables honestly, adhere to the Fallback Clause if the data dictates it, fix the framing of the baselines, and remove the OCR motivational contradiction, this paper will be a strong Accept. The field desperately needs this level of methodological transparency, separating zero-shot illusions from calibrated realities, and auditing systemic latency costs. Execute the plan.