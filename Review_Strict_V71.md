# Review_Strict_V71
## Overall Score
Score: 3/5

## Verdict
The paper presents a highly self-aware and methodologically bounded blueprint for decode-time visual grounding. The transition away from attacking baseline architectures toward formulating a positive, falsifiable proposition—token-local resonance via VASM—is a significant maturation. However, because the empirical execution is pending, the scientific validity of this work currently rests on a knife-edge. The experimental protocol is rigorously designed, but its success will depend entirely on whether the authors can execute the `TLRA_MeanPool` parity ablations and the Vocabulary Leakage Audit without introducing experimental artifacts. It is a "Weak Accept" purely as an experimental design, contingent on the strict delivery of the promised evidence chains.

## Summary
The authors propose Token-Local Resonance Anchoring (TLRA), a decode-time logits intervention framework aimed at mitigating physical entity hallucinations in MLLMs. The method utilizes a frozen, lightweight calibrator ($\Phi_{calib}$) to project visual patch tokens into the LLM's vocabulary space, allowing for token-local evidence weighting. To prevent structural collapse of the language model, TLRA employs Prompt-Conditioned Pruning, an Entropy-Driven Fallback, and a deterministic Vocabulary-Anchored Semantic Masking (VASM) algorithm based on WordNet. The paper is currently structured as an experimental blueprint, defining three strict evidence chains (Hallucination Reduction, Structure Preservation, and Local Evidence Value) to validate the method.

## WhatShouldBeKept
1. **The OCR Concession:** Your explicit admission that VASM bypasses arbitrary text strings and your use of DocVQA as a *negative control* (aiming for a 0% trigger rate) is excellent scientific practice. Do not walk this back. Keep the motivation strictly scoped to "physical entities in natural scenes."
2. **The Calibrator Parity Constraint:** The inclusion of `TLRA_MeanPool` and `TLRA_RandomK` using the *exact same calibrator weights* as the Top-$k$ method is the single most important ablation in the paper. It isolates the value of local routing from the external knowledge injected by the calibrator. 
3. **The Vocabulary Leakage Audit:** Stratifying POPE F1 into "Seen in Calibrator" vs. "Unseen in Calibrator" is mandatory. Keep this explicit.
4. **Treatment of Baselines:** Your framing of `DoLa`, `VCD`, and `OPERA` as "highly competitive baselines for modulating global generation trajectories" is fair and accurate. Do not regress to claiming they "fail because they use global pooling." They are simply solving a different formulation of the decoding problem.

## MajorWeaknesses
1. **Calibrator Identity Crisis and Unfair External Priors:** You state the base model is 100% frozen, but $\Phi_{calib}$ is trained on generic noun chunks via GroundingDINO and CLIP. This injects a powerful external object-detection prior into the inference phase. If `TLRA_AdaptiveTopK` only beats baselines because $\Phi_{calib}$ acts as a stealth object detector, the claim of "decode-time intervention" is misleading. The `TLRA_MeanPool` comparison is your only defense here, and it must be executed flawlessly.
2. **BPE-CSR Feasibility and Formalization:** You define the BPE Completion Success Rate (BPE-CSR) as forcing a prefix and checking if the suffix generates a valid semantic continuation (exact match or WordNet synonym). Automating this across thousands of generations using WordNet is highly brittle and prone to false negatives. 
3. **The Candidate Bounding Risk ($M=10$):** In modern LLMs with vocabulary sizes of 32k-128k, bounding candidate evaluation to the Top-10 logits is extremely aggressive. A strong hallucination prior can easily push the visually grounded token to rank 20 or 30.
4. **Video Pilot Distraction:** You propose a "bounded secondary video pilot" on Video-MME. A method explicitly designed for spatial token-local resonance does not naturally translate to temporal resonance without a formal temporal alignment mechanism. This risks diluting your core claim.

## SectionBySectionComments
*   **Abstract & Intro:** The scoping is sharp. The claim is well-bounded. Ensure that you do not over-promise "zero-shot" capabilities without immediately clarifying the use of the generic calibration dataset.
*   **Method (3.1):** You must specify the training objective/loss function for $\Phi_{calib}$. Is it contrastive? Cross-entropy over the vocabulary? This fundamentally alters how the logits intervention behaves.
*   **Method (3.2):** The Shannon entropy fallback ($H_{attn} > \theta_{fallback}$) calibrated on VisDial is clever but hypersensitive. If attention is inherently peaky, entropy will remain low even when looking at the "wrong" patches.
*   **Evaluation Protocol (4):** The commitment to the "Asterisk Rule" (flagging AGL collapse) is exactly what ACs want to see. 

## RequiredRevisions
1. **Refine BPE-CSR:** You must provide a concrete, algorithmic definition of how BPE-CSR will be computed automatically. Consider using a lightweight auxiliary metric (e.g., character-level edit distance of the completed subword sequence against the target) rather than relying solely on WordNet synonym matching for fragmented suffixes.
2. **Expand the Leakage Audit:** Beyond vocabulary overlap, you must prove that the spatial domain of POPE/CHAIR does not overlap with the CC3M/VG data used for the calibrator.
3. **Downgrade Video-MME:** Treat the video evaluation strictly as an exploratory appendix. Do not let it consume main-paper real estate unless you can mathematically prove that TLRA intervenes *exclusively* on the specific frames where the entity appears. If it just averages across frames, it falsifies your "local evidence" claim.
4. **Relax the Candidate Bound:** Run sensitivity analyses for $M \in \{10, 50, 100\}$. Show the Pareto frontier of latency vs. POPE F1.

## SuggestedFiguresTablesExperiments
To help you execute this blueprint, format your results precisely as follows:

*   **Table 1 (The Core Claim):** POPE and CHAIR metrics. Columns: Model, Method, F1/Accuracy, AGL, AGL StdDev, VASM Trigger Rate. Baselines must include `Base`, `VCD`, `DoLa`, `TLRA_MeanPool`, `TLRA_RandomK`, and `TLRA_AdaptiveTopK`.
*   **Table 2 (The Safety Audit):** MMBench, MMMU (Hard), PPL, DocVQA, and BPE-CSR. This table must prove that DocVQA trigger rate $\approx 0\%$ and PPL $\le +5\%$.
*   **Figure 1 (The Parity Ablation - Crucial):** A line chart plotting *Object Existence Accuracy (FREAK/HalBench)* on the Y-axis against *Intervention Strength ($\alpha$)* on the X-axis. Plot three lines: `TLRA_AdaptiveTopK`, `TLRA_MeanPool`, and `TLRA_RandomK`. The Top-$k$ line must show a statistically significant gap above MeanPool.
*   **Figure 2 (BPE Survival Curve):** Plot BPE-CSR (Y-axis) vs. $\alpha$ (X-axis) to visually demonstrate the exact threshold where KV-cache momentum breaks down and suffix generation stutters.
*   **Qualitative Failure Analysis:** Provide a specific figure showing the logits distribution at step $t$ (prefix forced by TLRA) and step $t+1$ (suffix collapse), exactly as you proposed in 4.4. Show the LM head probabilities failing.

## AcceptanceOutlook
If the executed experiments follow this blueprint and confirm the hypotheses—specifically that `TLRA_AdaptiveTopK` beats `TLRA_MeanPool` on object existence without inflating PPL or triggering on DocVQA—this will be a strong, methodologically sound paper. If `TLRA_MeanPool` achieves parity with Top-$k$, or if PPL spikes uncontrollably, the core proposition fails, and the paper should be rejected. Execute the plan strictly.