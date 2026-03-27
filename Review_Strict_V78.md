# Review_Strict_V78

## Overall Score
Score: 3/5

## Verdict
This paper presents a highly mature, defensively structured experimental proposal for decode-time hallucination mitigation. The framing is conceptually rigorous, properly designating global decoding heuristics (DoLa, VCD, OPERA) as legitimate competitors rather than strawmen, and treating text-heavy visual tasks (OCR) strictly as negative controls. However, because the empirical execution is pending, the current methodology reveals a potentially fatal mechanical flaw regarding the autoregressive latency and contextual feasibility of the VASM module. If the authors can execute the proposed "contract" while addressing the real-time POS-tagging bottleneck, this could be a highly impactful, falsifiable contribution.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), an inference-time intervention designed to reduce MLLM hallucination on physical entities. It extracts token-local visual evidence using an Adaptive Top-$k$ strategy applied to a bounded candidate window ($Top-M$) at decoding time. To prevent the destruction of linguistic structure, it introduces Vocabulary-Anchored Semantic Masking (VASM) via WordNet and BPE inheritance. Acknowledging native alignment gaps, the method relies on a frozen 2-layer MLP projector (`TLRA_calib`) trained on 50k captions. The paper explicitly commits to a stringent evaluation protocol, including a seen/unseen category audit, strict hyperparameter locking, and a "Fallback Clause" if spatial routing fails to outperform global mean pooling.

## WhatShouldBeKept
1. **The Framing of Baselines:** Treating VCD, OPERA, and DoLa as strong, legitimate global heuristics—rather than falsely claiming they "rely on global pooling"—is scientifically honest and must remain.
2. **The "Pre-registered" Contract Structure:** The explicit definition of Evidence Chains A, B, and C, particularly the Fallback Clause, is a breath of fresh air for ACM MM. 
3. **The Negative Control Concession:** Explicitly stating that TLRA is mathematically blind to OCR/text-in-image tokens and using `DocVQA` purely as a flat-line safety check is brilliant. Do not revert to claiming dense document understanding.
4. **The `TLRA_zero` vs. `TLRA_MeanPool` Parity Ablation:** Forcing the spatial routing claim (`AdaptiveTopK`) to beat `MeanPool` using the *exact same calibrator weights* on *unseen* objects is the absolute linchpin of this paper. Keep this exact design.

## MajorWeaknesses
1. **The VASM Autoregressive Paradox (Critical Flaw):** You propose automated POS tagging and WordNet lookups to isolate physical entity roots. Standard POS taggers (e.g., spaCy, NLTK) require complete sentences to resolve ambiguity (e.g., is "watch" a noun or a verb?). In decode-time intervention, you only have the *prefix*. How are you reliably executing POS tagging token-by-token on incomplete text? Furthermore, running external NLP pipelines inside the decode loop for $M$ candidates per step will incur catastrophic latency.
2. **The Identity of `TLRA_calib`:** By training $\Phi_{calib}$ on 50k captions via InfoNCE, TLRA ceases to be a purely "inference-time" method. It is a lightweight Test-Time Adaptation / PEFT method. Comparing it directly to pure zero-shot methods (DoLa/VCD) introduces an unfair advantage (new weights vs. no new weights). The paper masks this by calling it a "calibrator blackbox," but it is effectively a trained adapter.
3. **The "Single-Configuration Lock" Gamble:** While scientifically noble, locking $\tau_{sim}, \tau_{evidence}, \alpha, M, \rho$ on a single validation set and applying them to vastly different distributions (POPE vs. MMMU-Hard) may result in universally mediocre performance. The intervention scale ($\alpha$) needed to fix a hallucination in POPE will likely butcher the complex reasoning chains in MMMU.

## SectionBySectionComments
- **Abstract & Intro:** The tone is excellent. The boundary definitions are clear. 
- **Section 3.1 ($\Phi_{calib}$):** You must explicitly define the training cost, GPU hours, and exact nature of the 50k captions. If these captions contain the *exact distributions* of the POPE evaluations, the hallucination reduction is trivial. 
- **Section 3.4 (VASM):** As mentioned, the mechanical implementation of prefix-only POS tagging needs a detailed algorithmic explanation. If you pre-compute a static dictionary of "always-noun physical objects" from WordNet, state that clearly—it solves the latency issue but introduces lexical rigidity.
- **Section 4.4 (Seen vs. Unseen):** Ensure the 50k calibrator dataset is rigorously purged of the "Unseen" test split. This requires cross-referencing WordNet synsets, not just string matching.
- **Appendix (Video Pilot):** Retaining video as a bounded secondary pilot is the correct decision. Do not move it to the main text; it distracts from the spatial $H \times W$ argument.

## RequiredRevisions
1. **Clarify VASM Mechanics:** You must explicitly detail how VASM is computed at step $t$ given only tokens $[0...t-1]$. If it's a pre-computed static vocabulary mask, state it. If it's a dynamic prefix-based tagger, you must report the latency overhead of this specific module.
2. **Reframe the Baseline Comparison:** Acknowledge upfront that `TLRA_calib` uses external training data (50k captions), making it structurally different from pure zero-shot methods like VCD. The true zero-shot baseline is `TLRA_zero`.
3. **Execute the Fallback Clause if necessary:** If your ongoing experiments show `AdaptiveTopK` fails the unseen test against `MeanPool`, do not manipulate the data. Embrace the Fallback Clause. A paper titled "MeanPool + VASM is All You Need for Grounded Decoding" is still highly publishable if proven rigorously.

## SuggestedFiguresTablesExperiments
Since the experiments are currently a "contract," here is exactly how they must be executed to pass peer review:

*   **Table 1 (Hallucination):** Include a hard column for **AGL (Average Generation Length)**. If TLRA reduces POPE hallucination but drops AGL from 15 tokens to 3 tokens, you have built a truncation engine, not a grounding engine. 
*   **Table 2 (Structure):** `DocVQA` must be presented. I want to see `Base: 65.2` vs `TLRA_full: 65.1`. If `TLRA_full` drops to 40.0, VASM is destroying text tokens.
*   **Table 3 (The Parity Ablation):** 
    *   Split: FREAK (Seen Objects) | FREAK (Unseen Objects)
    *   Row 1: `TLRA_MeanPool` (Frozen $\Phi_{calib}$)
    *   Row 2: `TLRA_RandomK` (Frozen $\Phi_{calib}$)
    *   Row 3: `TLRA_AdaptiveTopK` (Frozen $\Phi_{calib}$)
    *   *Requirement:* Row 3 must beat Row 1 and 2 on the *Unseen* split.
*   **Figure 1 (Latency Pareto Front):** A scatter plot. X-axis: TPOT (ms/token) on an A100. Y-axis: POPE F1. Plot Base, VCD, DoLa, `TLRA_M=10`, `TLRA_M=50`. This will visually quantify the memory-bandwidth tax of $Top-M$ spatial routing.
*   **Failure Analysis:** Show a sequence probability graph where the correct physical entity logit starts at rank 12, drops to rank 55 at step $t-1$, and TLRA fails to recover it because $M=50$. This proves the "Out-of-Candidate" limitation is real.

## AcceptanceOutlook
The conceptual framework is highly defensible and perfectly aligned with the rigor expected at ACM MM. The acceptance of this paper rests entirely on the empirical execution of the proposed tables. If the authors can produce the data for Tables 1-3 without violating their own pre-registered rules, and successfully explain the VASM prefix-latency issue, I will strongly champion this paper for acceptance. If the results force the Fallback Clause, it remains a borderline-to-accept paper based on methodological honesty.