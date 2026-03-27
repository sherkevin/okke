# Review_Strict_V35
## Overall Score
Score: 2/5

## Verdict
The manuscript presents an exceptionally rigorous and theoretically sound experimental blueprint for decode-time visual intervention, but it is currently an incomplete submission (a "pre-registered protocol" lacking executed results). As an Area Chair, I cannot recommend acceptance for a paper with zero empirical evidence. However, the methodology—specifically the positive proposition of injecting token-local visual evidence via `TSLI_zero`/`TSLI_calib`, threshold-gated pooling, and VASM—is highly structurally defensible. If the proposed experimental plan is fully executed with the rigor described, this work has the potential to be a top-tier contribution.

## Summary
The paper proposes Thresholded Spatial Logits Intervention (TSLI), a decode-time method to reduce object hallucination in MLLMs by explicitly injecting token-local visual evidence into the terminal logits space. To overcome input-output asymmetry and post-hoc entanglement, it establishes a dual-track approach (`TSLI_zero` and a contrastively trained `TSLI_calib`). It mitigates the risk of structural language collapse via Vocabulary-Anchored Semantic Masking (VASM) and dynamic BPE continuation inheritance. Furthermore, it addresses decode-time latency and VRAM bottlenecks through a dynamically calibrated entropy gate and candidate window bounding. The paper currently outlines a pre-registered evaluation protocol across three evidence chains (Hallucination, Structure, and Local Evidence) without presenting completed experimental results.

## WhatShouldBeKept
1. **The Positive Proposition Framing:** Your treatment of DoLa, VCD, and OPERA as "highly competitive, orthogonal regularizers" is exactly correct. Do not regress into falsely claiming these methods fail because they "rely on global pooling." Maintain the current framing: they discipline overarching language/attention dynamics, whereas TSLI provides explicit spatial grounding.
2. **The `TSLI_zero` vs. `TSLI_calib` Boundary:** Acknowledging the "post-hoc entanglement gamble" is excellent. The strict separation prevents you from claiming zero-shot alignment if the underlying 1D/2D architecture fundamentally washes out spatial fidelity.
3. **VASM and BPE Inheritance:** The dynamic BPE continuation inheritance (especially handling raw byte fallbacks) is technically superb and correctly identifies a major flaw in token-level logits interventions. 
4. **Deliberate Exclusion of Video:** Your decision to strictly bound the scope to Image-LLMs and 2D spatial reasoning is a mature scientific choice. Do not add a spatiotemporal/video main line unless you can mathematically prove token-local spatiotemporal evidence extraction, which is highly unlikely given current compute bounds.

## MajorWeaknesses
1. **Absence of Empirical Execution:** The paper is currently a plan, not a completed scientific claim. All hypotheses remain unvalidated.
2. **The Polysemy Superset Vulnerability in VASM:** Using a "brute-force superset" for WordNet synsets is a severe operational risk. If the word "mouse" triggers intervention globally, TSLI will penalize the computer device context if the visual evidence only supports an animal. This false-positive triggering could cause severe linguistic degradation that your current MMMU evaluation might not catch.
3. **The Batched Latency Trap:** Computing moving medians for $N_v$ patches and mapping logits dynamically at every uncalibrated step is highly unoptimized for GPU tensor cores. Even with entropy gating, batched inference (BS > 1) might easily OOM or severely degrade Tokens/Sec. 
4. **LoRA Baseline Confounding:** In Chain A, comparing `TSLI_calib` to `Base + 5k LoRA` is a good start, but if the LoRA is trained with a standard autoregressive loss while $\Phi_{calib}$ is trained with strict InfoNCE (IoU penalized), the performance delta might stem from the loss function, not the decode-time mechanism.

## SectionBySectionComments
- **1. Introduction:** The framing is sharp. If your final executed results only strongly support `TSLI_calib` + VASM + the fair zero-shot/calibrated split, be prepared to contract your grand claims. Do not over-claim if `TSLI_zero` entirely fails on 1D sequences like LLaVA.
- **3.1 Overcoming Post-Hoc Entanglement:** The SAM/IoU assisted negative sampling for InfoNCE is a massive strength. Ensure the exact IoU threshold (<0.1) and SAM parameters are fully documented in the final appendix.
- **3.2 Threshold-Gated Pooling:** Using a moving median is statistically robust against sparse/dense image contexts, but computationally heavy. You must profile the overhead of this specific operation in your latency section.
- **4. Evaluation Protocol:** The three chains are perfectly aligned with the claims. However, see the required revisions for specific ablations.

## RequiredRevisions
1. **Execute the Plan:** This is non-negotiable. Complete all proposed tables and figures.
2. **Refine Chain A (Hallucination Reduction):** Ensure the `Base + 5k LoRA` baseline is trained on the *exact* same 5k COCO pairs, and explicitly state its loss objective. Report standard POPE (Random, Popular, Adversarial) and CHAIR metrics.
3. **Refine Chain B (Structure Preservation):** You must include an exact breakdown of the "BPE Collision Rate" and report the absolute unrecoverability limits (targets naturally falling outside Top-$M$). 
4. **Fix the Baseline Vocabulary in Chain C:** As per standard practice in spatial evaluation, ensure you explicitly evaluate the `BRA_zero` vs `BRA_MeanPool` ablations (or explicitly map your `TSLI_zero` vs `TSLI_MeanPool` to this protocol) on FREAK and DocVQA to isolate the true value of local evidence against naive global averaging.
5. **Eliminate the "Pre-registered" Framing:** For the final ACM MM submission, remove the "planned execution" language. Present it as a completed, empirical study.

## SuggestedFiguresTablesExperiments
To form your execution roadmap, adhere strictly to these deliverables:
- **Table 1 (Chain A):** POPE (Acc/F1) and CHAIR (Obj/Attr) for Base, VCD, OPERA, DoLa, `TSLI_zero`, `Base+5k LoRA` (3 seeds), `TSLI_calib`. Add a column for "Entropy Gate Override %".
- **Table 2 & Scatterplot (Chain B):** MMMU(Hard) vs. OOV Rate. This scatterplot must clearly show if structural reasoning collapse correlates directly with VASM dictionary exhaustion.
- **Table 3 (Chain C):** FREAK / DocVQA comparing `TSLI_zero` vs `TSLI_MeanPool` vs `TSLI_AdaptiveTopK` (mirroring the `BRA_zero` vs `BRA_MeanPool` standard).
- **Figure 1 (Failure Case - Entanglement Washout):** 4-way visual showing Native Attention vs MeanPool vs AdaptiveTopK vs Entropy Gate over time. Crucially, show a case where 1D architecture (LLaVA) highlights the wrong region in `TSLI_zero` but `TSLI_calib` fixes it.
- **Figure 2 (Failure Case - Semantic Distractor):** Explicit visual proof of whether `TSLI_AdaptiveTopK` fails by locking onto an object in the same super-category (e.g., dog vs. cat).
- **Latency Profiling:** A rigorous dual-axis plot for Tokens/Sec and Peak VRAM vs. Candidate Window ($M$) at BS=1, 4, 8.

## AcceptanceOutlook
The core proposition is scientifically mature and structurally protected. If the authors return with this exact plan fully executed, demonstrating clear local evidence value (`TSLI_calib` > `Base+LoRA`, `AdaptiveTopK` > `MeanPool`), with transparent reporting of OOV/BPE constraints and latency overheads, this paper will easily cross the threshold for an Oral/Strong Accept. In its current state, it is an executable roadmap.