# Review_Strict_V82

## Overall Score
Score: 3/5

## Verdict
The paper defines a highly disciplined, falsifiable experimental protocol for decode-time multimodal intervention. By explicitly bounding the intervention space (VASM, Top-$M$ filtering) and setting up rigorous negative controls (AGL audits, DocVQA flat-lining, unseen category parity tests), the authors have constructed a review-proof methodology. The current draft is evaluated purely on its theoretical soundness and experimental planning. The final acceptance will hinge entirely on the faithful execution of Evidence Chain C (the Parity Test). If the spatial routing fails to outperform the calibrated baseline on unseen objects, the contribution must cleanly devolve, but the scientific rigor of the attempt remains high.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a decode-time intervention method designed to inject token-local visual evidence into MLLM logits to mitigate physical hallucination. To bypass the catastrophic latency of autoregressive POS tagging, it introduces Vocabulary-Anchored Semantic Masking (VASM), a static WordNet-based mask with $O(1)$ BPE inheritance to protect syntax and reasoning tokens. The methodology is split into a zero-shot probe (`TLRA_zero`) and a Test-Time Adaptation module (`TLRA_calib`). Crucially, the authors outline a strict, hypothesis-driven evaluation protocol comprising three evidence chains: Hallucination Reduction (audited by Average Generation Length), Structure Preservation (using DocVQA as a negative control), and a Local Evidence Parity Test (`AdaptiveTopK` vs `MeanPool` on unseen categories). 

## WhatShouldBeKept
1. **The Framing of Baselines**: Referring to DoLa, VCD, and OPERA as "sequence-level or distribution-level language heuristics" is accurate and respectful. You successfully avoided the trap of mischaracterizing them as relying on "global pooling."
2. **The Fairness Boundary**: Explicitly separating `TLRA_zero` (as a pure parameter-isolated probe) from `TLRA_calib` (as a TTA module) prevents "calibrator blackbox" ambiguity.
3. **VASM and BPE Inheritance**: The offline static dictionary mapping to `physical_entity.n.01` combined with the dynamic BPE continuation logic is an elegant, highly practical engineering solution to the latency paradox.
4. **The Strict Evaluation Protocol**:
   - The **AGL Audit** in Chain A is a brilliant mechanism to prevent "silence engine" truncation artifacts from masquerading as hallucination reduction.
   - Using **DocVQA** strictly as a mathematical flat-line negative control in Chain B resolves previous contradictions regarding OCR capabilities.
   - The **Parity Test** in Chain C (testing `AdaptiveTopK` vs `MeanPool` on strictly *unseen* categories) is the absolute correct way to prove the value of local spatial routing over mere parameter leakage.
5. **Video Demotion**: Relegating spatio-temporal extensions to an exploratory appendix pilot correctly tightens the paper's focus on static spatial grounding.

## MajorWeaknesses
1. **High Risk of Negative Results in Chain C**: The `AdaptiveTopK` routing relies entirely on the assumption that the native visual patch states can cleanly localize *unseen* physical entities. If the base vision encoder lacks fine-grained patch-text alignment for out-of-distribution objects, `AdaptiveTopK` will pool spatial noise, and `MeanPool` will trivially win. 
2. **Undefined Alignment Scoring**: In Section 3.2, the formulation uses $\text{score}(v_j, c)$. This is the most critical mathematical operation in the paper, yet it is entirely undefined. Is this a cosine similarity? A dot product? Does it use the LM's language head weights or the $\Phi_{calib}$ projection space? 
3. **Logit Scaling Instability**: In Section 3.4, the penalty scaling relies on $\Delta_L = \max(L_{orig}) - \min(L_{orig})$ within the candidate set. Relying on the absolute minimum logit in a Top-$M$ slice is highly unstable; a single mathematically improbable token in the top 50 candidates will massively inflate $\Delta_L$, causing wild fluctuations in the intervention magnitude.
4. **Polysemy Brittleness in VASM**: The strict programmatic definition relies on the *primary* synset in WordNet. Many tokens (e.g., "watch", "monitor", "bat") have primary synsets that might not be physical entities, or vice versa. The method's lexical rigidity here might cause it to miss obvious hallucinations or incorrectly penalize verbs.

## SectionBySectionComments
- **Abstract & Introduction**: Excellent setup. The transition from acknowledging the strength of sequence-level heuristics to questioning the viability of token-local injection is logically seamless.
- **Methodology (3.1 - 3.2)**: The structural bounds ($M$ candidates, $\rho$ spatial ratio) are well-motivated. However, the exact mechanics of how a text candidate $c$ is projected into the same space as visual state $v_j$ must be explicitly detailed.
- **Methodology (3.3)**: BPE fragmentation defaulting to $\gamma=0$ is a highly appreciated, conservative fail-safe. 
- **Methodology (3.4)**: The formulation $L_{final}(c) = L_{orig}(c) - \alpha \cdot \gamma(c) \cdot \Delta_L \cdot (1 - \hat S(c))$ is fundamentally sound in its masking logic, but numerically precarious due to $\Delta_L$.
- **Section 4 (Protocol)**: This is one of the strongest evaluation plans I have reviewed. Tying specific hypotheses to strict pass/fail validation criteria sets a gold standard for reproducibility.

## RequiredRevisions
1. **Define $\text{score}(v_j, c)$**: You must explicitly provide the equation for how spatial support is calculated. Clarify whether $c$ uses the base model's embedding matrix or if it passes through a secondary projection.
2. **Stabilize Logit Intervention**: Replace the raw range $\Delta_L$ with a more robust dispersion metric. Consider using the standard deviation of the Top-$M$ logits, the interquartile range, or simply normalizing the intervention via a softmax temperature relative to the target token's original probability.
3. **Address Polysemy in VASM**: Add a brief paragraph acknowledging how words with multiple distinct senses are handled. If the system is strictly bound to the primary synset, explicitly declare this as a limitation of the static mask approach.
4. **Contingency Planning for Claims**: If, upon running the experiments, `AdaptiveTopK` fails to statistically beat `MeanPool` on the Unseen split, you must heavily shrink the paper's claim. The narrative must pivot from "spatial routing is superior" to "Token-Local Logit Intervention + VASM provides a safe, fast, structure-preserving TTA." Do not force the spatial routing claim if the data refutes it.

## SuggestedFiguresTablesExperiments
1. **Figure 1 (Teaser)**: Create a visual contrasting sequence-level heuristics (which adjust the entire vocabulary distribution) against TLRA. Show a specific decoding step where TLRA intervenes on "cat" ($\gamma=1$) but explicitly bypasses the next token "is" ($\gamma=0$).
2. **Table 1 Design**: Ensure the AGL (Average Generation Length) column is placed immediately adjacent to the POPE F1/Accuracy metrics to force readers to contextualize hallucination drops against sequence length.
3. **Table 3 Design**: Explicitly divide this table into two macro-columns: "Seen Objects (Leakage Control)" and "Unseen Objects (True Parity)". Include a $\Delta$ column showing (`AdaptiveTopK` - `MeanPool`).
4. **Figure 2 (TPOT Pareto)**: When plotting the memory-bandwidth tax, ensure you are using a batch size of 1, as decode-time interventions are typically deployed in interactive, unbatched inference scenarios.
5. **New Ablation**: Add a small ablation table varying the candidate window $M \in \{10, 20, 50, 100\}$. This will empirically justify your TPOT vs. Out-of-Candidate Failure tradeoff.

## AcceptanceOutlook
The proposed framework is highly scientifically mature. If the authors execute this exact experimental plan, report the metrics transparently (especially the negative controls and AGL audits), and adjust their final claims to match whatever reality Table 3 (the parity test) dictates, this paper will be a strong, methodologically rigorous contribution to ACM Multimedia.