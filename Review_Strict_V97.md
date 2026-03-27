# Review_Strict_V97
## Overall Score
Score: 3/5

## Verdict
The paper presents an exceptionally self-aware, pre-registered experimental plan for mitigating MLLM hallucinations via Token-Local Resonance Anchoring (TLRA). The commitment to strict ablation, matched-budget baselines, and falsifiable claims is highly commendable. However, the author attempts to preemptively dictate the terms of the review by framing the paper as an "executable contract." As an Area Chair, I evaluate the scientific validity of the method and the feasibility of its execution, not the rhetorical framing. 

The current methodology has two massive systemic risks that the proposed experiments do not adequately stress-test: the algorithmic brittleness of the Vocabulary-Anchored Semantic Masking (VASM) and the likely prohibitive $O(M \times N_v)$ per-step latency overhead. If VASM relies on static vocabulary lookups, it ignores contextual polysemy; if the latency penalty is an order of magnitude higher than base decoding, the method is practically dead on arrival. The experimental design is robust, but it must be expanded to aggressively probe these two specific failure boundaries before acceptance.

## Summary
The paper proposes TLRA, a decode-time intervention to reduce MLLM hallucination by adjusting token logits based on localized visual evidence. It extracts token-local support from cached visual states, applying an Adaptive Top-k resonance penalty to the logits of a bounded candidate set (Top-$M$). To prevent the degradation of syntax and multi-token entities, it introduces VASM, an offline precomputed mask. The paper heavily emphasizes methodological hygiene, splitting the method into a training-free probe (`TLRA_zero`) and a calibrated variant (`TLRA_calib`), mandating matched-budget comparisons against a `Base + LoRA` model. Currently, the paper is in a pre-registered state with empty tables (TBF).

## WhatShouldBeKept
1. **The Strict Zero vs. Calib Split:** Acknowledging that `TLRA_calib` uses trained parameters (`Phi_calib`) and explicitly refusing to hide it among training-free baselines (like VCD or DoLa) is excellent scientific practice. 
2. **The Matched-Budget Baseline (`Base + LoRA`):** This is a brilliant and necessary control. If `TLRA_calib` cannot beat a simple LoRA trained on the same calibration data, the decode-time complexity is unjustified. Keep this front and center.
3. **Top-M Hijacking Audit:** Acknowledging that decode-time interventions cannot recover tokens that fall outside the base model's Top-$M$ logits is mathematically sound. The "Hijacking CDF" must be kept.
4. **OCR as a Negative Control:** Treating DocVQA as a guaranteed failure domain due to the VASM design is a fair, bounded claim.

## MajorWeaknesses
1. **The Semantic Fragility of VASM:** You describe VASM as an "offline precomputed vocab lookup." This is a massive red flag. Language is contextual. "Apple" is a physical, groundable entity in a fruit bowl, but a structural/named entity in "Apple Store" or "Apple CEO." If VASM is a static dictionary mask, it will routinely trigger false positives on polysemous words and idioms, corrupting the decoding stream. You explicitly ban online POS taggers to save latency, but a static lookup is a heuristic hack, not a structural safeguard.
2. **The "Decode-Time" Computational Wall:** You are proposing to compute the similarity between $M$ candidate tokens and $N_v$ visual states *at every single decoding step*. If $N_v$ is large (e.g., modern MLLMs with 1000+ visual tokens) and $M=50$, this is a heavy dense matrix multiplication injected into the autoregressive loop. If `TLRA` cuts `tokens_per_second` by 80%, the method is a non-starter for deployment, regardless of POPE improvements.
3. **The Identity Crisis of `TLRA_calib`:** If you are willing to train `Phi_calib` on a conceptual-caption subset, you are injecting external alignment knowledge. Why do this at decode time? If the knowledge is in the calibrator, why not just use those weights as an adapter during the prefill phase? You must prove that *decode-time intervention* specifically yields something that *prefill-time adaptation* mathematically cannot.
4. **Author's Meta-Framing:** The prose is too defensive. Stop telling the reviewer what you "explicitly do not claim" and what "the strongest version of TLRA is." Present the formulation, prove it, and let the data speak. 

## SectionBySectionComments
* **Abstract & Intro:** Remove the meta-commentary about "claiming premature state-of-the-art" and "scientifically honest." It reads as insecure. State the problem, the mechanism, and the boundaries cleanly. 
* **3.1 Fairness Boundary:** Excellent setup. However, specify *exactly* what `Phi_calib` is. Is it a linear layer? An MLP? If it's too heavy, the latency claim is completely invalidated.
* **3.2 Bounded Candidate Filtering:** The formula $k = \max(k_{min}, \lceil \rho \cdot N_v \rceil)$ makes sense, but you need to define how visual state provenance is tracked if the base model uses perceiver resamplers or Q-Formers where visual tokens no longer map 1:1 to spatial patches. TLRA implies a grid-like visual token structure. Does this fail on models like InstructBLIP?
* **3.3 Resonance Penalty:** The formulation $\Delta_L \cdot (1 - \hat S(c))$ is sensitive to $\Delta_L$. If the base model is extremely overconfident (spiky distribution), $\Delta_L$ might be huge, causing aggressive distortion. You need a clipping mechanism here.
* **3.4 VASM:** As noted, the offline lookup is mathematically dangerous. The BPE continuation inheritance is smart, but the root identification is flawed.

## RequiredRevisions
1. **Contextual vs. Static VASM Proof:** You must add a stress-test evaluating VASM's failure rate on polysemous words (e.g., words that are visual entities in some contexts but abstract in others). 
2. **Complexity Analysis Formulation:** Add a formal Big-O complexity analysis of the per-token latency overhead in Section 3.2. Define exactly how many FLOPs are added per decoding step.
3. **Dynamic Logit Clipping:** Introduce a mathematical bound on the penalty in Section 3.3 to prevent complete distribution destruction when $\Delta_L$ is anomalously large.
4. **Clarification of Visual Architectures:** Explicitly state the compatible MLLM architectures. Does this require a strict ViT-to-LLM patch mapping (like LLaVA), or does it work through cross-attention/Q-Formers? If the former, state it as a hard scope boundary.

## SuggestedFiguresTablesExperiments
* **Modify Table 1 (Latency Breakdown):** "ITL/TPOT" is not enough. You must split the latency penalty into two columns: *Projector Overhead* (time spent running `Phi_calib`) and *Routing Overhead* (time spent computing Top-K similarity). 
* **Add VASM Oracle Ablation to Table 2:** Add a row `TLRA_Oracle_VASM` (using a heavy, perfect online POS tagger). This is required to show the upper bound of your method if the offline dictionary lookup wasn't so primitive. The delta between `TLRA_full` and `TLRA_Oracle_VASM` will quantify the damage done by your static lookup heuristic.
* **Figure 1 Expansion:** The Pareto frontier of Local Evidence Cost is critical. You must plot `tokens_per_second` on the X-axis against `POPE F1` on the Y-axis. Include a vertical dashed line representing the base model's speed. If your curve lives entirely in the "unacceptably slow" domain, the paper fails.
* **Visual Parity Test:** In Table 3 (Local Evidence Parity), add a row for `Global_Visual_State` (using the pooled image embedding instead of token-local patches). If `Global_Visual_State` performs exactly as well as `TLRA_AdaptiveTopK`, then spatial locality does not matter, and your core claim is falsified.

## AcceptanceOutlook
The philosophical foundation of the experimental plan is superb, but it teeters on the edge of system-level impracticality. If the final TBF tables reveal that `TLRA_AdaptiveTopK` is statistically indistinguishable from `TLRA_MeanPool` or `Global_Visual_State`, or if the latency drops by >50%, the paper must pivot entirely to an analysis of *why* localized decode-time interventions fail in modern MLLMs. As a positive claim, acceptance is highly contingent on surviving the Latency vs. Hallucination-Reduction Pareto curve and proving VASM isn't destroying contextual syntax. Execute the plan strictly.