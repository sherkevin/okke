# Review_Strict_V32

## Overall Score
Score: 3/5

## Verdict
The paper presents a highly rigorous, structurally defensive methodological framework for decode-time visual intervention. By correctly abandoning the false premise that competitive baselines (DoLa, VCD, OPERA) rely on "global pooling" and instead framing its own contribution as a positive proposition for token-local visual evidence, the paper establishes a scientifically sound foundation. Furthermore, restricting the scope strictly to 2D spatial reasoning rather than forcing an unsupported video narrative is a strong, mature decision. However, as the empirical execution is currently pending, the submission is effectively a pre-registered experimental plan. The final verdict will heavily depend on whether the proposed evidence chains (A, B, and C) mathematically validate the hypotheses without incurring catastrophic latency or structural collapse.

## Summary
The paper proposes Thresholded Spatial Logits Intervention (TSLI), a decode-time method designed to inject token-local visual evidence directly into MLLM logits adjustment. It tackles embedding space asymmetry via an explicit zero-shot vs. calibrated boundary (`TSLI_zero` vs. `TSLI_calib`). It mitigates background noise dilution using Threshold-Gated Adaptive Top-$k$ Pooling, and protects functional syntax and multi-token entities via Vocabulary-Anchored Semantic Masking (VASM). The evaluation protocol is pre-registered across three rigid evidence chains: Hallucination Reduction, Structure Preservation, and Local Evidence Value.

## WhatShouldBeKept
1. **The Corrected Baseline Framing:** Your explicit acknowledgment that VCD, DoLa, and OPERA are orthogonal regularizers of language/attention dynamics—rather than flawed "global pooling" methods—is excellent. Keep this exact framing. It prevents the paper from being rejected for attacking strawmen.
2. **The 2D Spatial Scope:** Deliberately avoiding spatiotemporal video domains tightens your claims and makes the paper much more defensible for ACM MM. Keep this boundary strict.
3. **The Zero vs. Calib Boundary (`TSLI_zero` vs `TSLI_calib`):** Explicitly acknowledging the "post-hoc entanglement gamble" (spatial washout in 1D sequences) is scientifically mature.
4. **The Parameter-Matched LoRA Baseline:** Testing `TSLI_calib` against a `Base + 5k LoRA` trained on the exact same data is the single most critical experimental design choice in this paper. It perfectly isolates the decode-time mechanism from mere data exposure.
5. **The Three Evidence Chains (A, B, C):** This structured evaluation framework is highly legible and sets a high bar for multimodal evaluation.

## MajorWeaknesses
1. **Pending Empirical Execution:** The most obvious weakness is that a methodological plan, no matter how rigorous, is inherently incomplete without data. 
2. **The Latency Trap ($O(L \times |V| \times N_v)$):** You acknowledge this in Section 5, but let's be blunt: computing similarities across $V$ tokens and $N_v$ patches at *every single decoding step* could absolutely tank throughput. If TSLI drops A100 tokens/sec by 80%, it ceases to be a decoding strategy and becomes an offline analytical tool. 
3. **VASM's Noun-Centric Blindspot:** By restricting $\gamma=1$ only to WordNet visual nouns, you protect syntax, but you completely surrender the ability to correct visual verbs (actions like "running", "throwing") and visual attributes ("red", "shiny"). This is a severe capability ceiling that needs to be explicitly measured.
4. **InfoNCE Negative Sampling Constraints:** Your negative sampling relies on IoU $< 0.1$. In densely packed images (e.g., overlapping objects, crowds), an IoU $< 0.1$ patch might still contain fragments of the target semantic class, poisoning the contrastive signal for $\Phi_{calib}$.

## SectionBySectionComments
- **Abstract & Intro:** Very strong. The positive proposition ("How can we inject strictly token-local visual evidence...") is clear and actionable. Do not inflate this further; your current claim ("token-local logits intervention + VASM + fair split") is exactly the right size for a top-tier conference.
- **Method 3.1 (`TSLI_calib`):** The linear constraint on $\Phi_{calib}$ is necessary to prevent it from becoming a memory bank. Ensure the learning rate and epochs for this projection are explicitly stated in the appendix so reviewers can verify it isn't overfitted.
- **Method 3.2 (Adaptive Top-$k$):** The threshold-gated Sigmoid is sound, but $\theta_{noise}$ introduces a brittle hyperparameter. You need to prove this threshold generalizes across datasets without manual per-dataset tuning.
- **Method 3.3 (VASM):** BPE Continuation Inheritance is a highly practical engineering solution to a fatal decode-time flaw. Commendable.
- **Evaluation Protocol:** Chains A, B, and C form a closed analytical loop. Stick to this script exactly.

## RequiredRevisions
1. **Execute the Experimental Plan:** Deliver the results for Tables 1, 2, and 3. Do not alter the hypotheses if the data fails to support them; report the negative results transparently (especially if `TSLI_zero` collapses).
2. **Quantify the Verb/Attribute Gap:** Since VASM ignores non-nouns, you must evaluate CHAIR or a similar metric broken down by object hallucinations vs. attribute/relation hallucinations to prove that leaving attributes unprotected doesn't cascade into wider failures.
3. **Strict Latency Reporting:** You must report absolute A100 Tokens/Sec for Base, VCD, DoLa, and TSLI. If TSLI is unacceptably slow, you must introduce an optimization (e.g., caching the $W_{vocab}$ projection, limiting $M$ further, or applying TSLI only at high-entropy steps) or accept its status as an offline tool.

## SuggestedFiguresTablesExperiments
*   **Table 1 (Chain A):** Execute exactly as planned. Must include POPE, CHAIR, AGL variance. The `Base + 5k LoRA` vs `TSLI_calib` row is your make-or-break result.
*   **Table 2 (Chain B):** Execute as planned. The OOV tracking column is crucial. If OOV > 10%, VASM is fundamentally broken.
*   **Table 3 (Chain C):** Must clearly contrast `TSLI_MeanPool` with `TSLI_AdaptiveTopK`. If mean-pooling performs identically, your core "local evidence" thesis dies.
*   **Figure 1 (3-Way Heatmaps):** Do exactly as you proposed. Native attention vs MeanPool vs Adaptive Top-$k$. Ensure you include the mandatory semantic distractor failure case where the infoNCE projection locks onto the wrong object of the same class.
*   **Figure 2 (Out-of-Candidate Unrecoverability):** Show the logit rank plot where the correct visual token sits at rank 150, proving that no top-$M$ post-hoc adjustment could save it. This transparency builds massive reviewer trust.
*   **New Figure (Latency vs. M):** A line chart showing A100 Tokens/Sec on the y-axis and Candidate Window Size ($M$) on the x-axis, proving at what threshold the latency overhead becomes catastrophic.

## AcceptanceOutlook
The framing, scoping, and methodological rigorousness of this draft are currently at an Accept level. The authors have successfully avoided the trap of macro-criticism and built a strictly bounded, falsifiable mechanism. If the pending experiments validate the claims—specifically that `TSLI_calib` beats `Base + 5k LoRA`, VASM preserves reasoning on MMMU, and Adaptive Top-$k$ geometrically beats Mean-Pooling without catastrophic latency—this will be a very strong paper for ACM MM. Execute the plan strictly.