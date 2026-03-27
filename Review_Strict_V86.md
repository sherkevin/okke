# Review_Strict_V86
## Overall Score
Score: 3/5

## Verdict
Provisional. The paper presents an unusually rigorous, pre-registered experimental contract, which is highly commendable. However, the core identity of the method is conflicted, and the proposed evaluation protocol contains a fundamental category error regarding baselines. If the authors execute the plan *as amended below*, the resulting paper will be a solid, falsifiable contribution. If they ignore the structural baseline mismatch and latency realities, it will be rejected.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a decode-time intervention for MLLMs aimed at reducing physical grounding hallucinations. It selects top-M candidate tokens and reweights their logits based on localized visual support extracted via an "Adaptive Top-k" mechanism. To prevent structural degradation, it applies Vocabulary-Anchored Semantic Masking (VASM) using WordNet and BPE rules. The method is split into a pure zero-shot probe (`TLRA_zero`) and a version with a trained visual-to-lexical projector (`TLRA_calib`). The current draft outlines a strict experimental contract rather than reporting completed results.

## WhatShouldBeKept
1. **The Falsifiable Contract Structure:** The framing of "Evidence Chains" and explicit fallback claims is excellent scientific practice. Do not soften this.
2. **DocVQA as a Strict Negative Control:** Acknowledging the "OCR concession" and using document VQA to prove VASM works by *failing to intervene* is an intellectually honest and robust design choice. 
3. **The Internal Parity Ablations:** The requirement that `TLRA_AdaptiveTopK` must beat `TLRA_MeanPool` and `TLRA_RandomK` under the *exact same* frozen calibrator weights is the most important methodological safeguard in the paper. Keep this mandatory.
4. **The Seen vs. Unseen Leakage Split (Table 3):** Crucial for proving `Phi_calib` is learning routing, not just memorizing dataset priors.

## MajorWeaknesses
1. **The Identity Crisis and Baseline Category Error:** You explicitly state `TLRA_calib` is "not a training-free method in the same sense as VCD, DoLa, or OPERA." Yet, in Table 1, you set up a direct comparison against exactly these methods. This is an unfair fight. `TLRA_calib` uses an InfoNCE-trained projector on a conceptual-caption subset. If it beats VCD, we do not know if decode-time local anchoring is better, or if you simply fed the model more supervised alignment data. **You are missing a critical, mandatory baseline: a lightweight LoRA fine-tuning of the base MLLM using the exact same conceptual-caption subset and training budget as `Phi_calib`.** If a standard LoRA outperforms `TLRA_calib`, your complex decode-time intervention is rendered moot.
2. **The VASM Implementation Reality:** The methodology casually mentions "Automated root-token lookup: WordNet synsets and POS filtering." Doing this accurately on partial, autoregressively generated prefixes is notoriously brittle. Is VASM running a POS tagger at every decoding step? If so, the latency will be catastrophic. If it relies on pre-computed vocab lookups, how does it handle polysemy (words that are nouns in one context, verbs in another)? The paper entirely glazes over the computational and logical mechanics of online VASM.
3. **System Cost and Complexity (The $O(M \times H \times W)$ Problem):** At every decoding step, for $M$ candidates, you compute similarity against $H \times W$ visual states. For modern high-res MLLMs, visual tokens can exceed 4000. Doing $M \times 4000$ dense operations *inside the autoregressive loop* will likely destroy `tokens_per_second`. The current contract mentions `peak_vram` and `ITL`, but the theoretical FLOPs/token overhead must be explicitly modeled.
4. **The "Zero-shot" Definition:** You define `TLRA_zero` as using the "model-native readout." This assumes the base model's visual states naturally map to the LM's embedding space in a way that allows direct dot-product similarity. For models like LLaVA, the visual tokens are projected into the input space, but comparing final-layer visual states to output vocabulary logits directly is mathematically unsound without a specific shared embedding space (which most autoregressive LLMs do not enforce). Stage 0 is highly likely to fail for architectural reasons, not semantic ones.

## SectionBySectionComments
*   **Abstract & Introduction:** The tone is refreshingly honest, but you must stop calling `TLRA_calib` a purely "decode-time intervention" if it requires an offline training phase. Call it a "hybrid projection-routing method."
*   **Section 3.1 (`TLRA_zero` vs `TLRA_calib`):** Correct the factual assumption that final-layer visual states can be directly compared to lexical logits in standard MLLMs without a projector. If you are using the base model's pre-trained vision-language connector, specify that.
*   **Section 3.2 (Bounded Candidate Filtering):** The equation $k = \max(k_{min}, \lceil \rho \cdot N_v \rceil)$ is fine, but you must define exactly *which* visual states $h_L^{(v_j)}$ are used. Are these the visual tokens before the LLM cross-attention/self-attention, or after? If after, they are already heavily contextualized by the generated text prefix, which risks circular logic (the text prefix dictating the visual state, which then validates the next token).
*   **Section 4.1 (Stage 0):** This is highly likely to fail as noted above. You should prepare the fallback claim immediately.
*   **Section 4.2 & Baselines:** Fact check: DoLa is a layer-contrast method for *language models*, not natively multimodal. If you are using an adapted MM-DoLa, state so. VCD injects visual noise. Again, neither utilizes external training data.

## RequiredRevisions
1. **Mandatory Baseline Addition:** You *must* add a `Base + LoRA (Phi_calib data)` row to Tables 1, 2, and 3. Without this, any success of `TLRA_calib` can be attributed to the extra training data, invalidating the "decode-time intervention" claim.
2. **VASM Clarification:** You must add a subsection detailing the exact algorithmic implementation of VASM during generation. State explicitly whether it requires an external parser/tagger in the loop, and address polysemy resolution for partial sentences.
3. **Visual State Definition:** Clearly define $h_L^{(v_j)}$. Is it the output of the vision encoder, the output of the vision-language connector, or the hidden states of the LLM corresponding to the visual token positions? This completely changes the meaning of the method.
4. **Latency Measurement:** In Table 1 and Table 2, raw `ITL` (Inter-Token Latency) is not enough. You must report the latency multiplier (e.g., $1.5\times$ base model latency) and explicitly isolate the time spent running the VASM logic vs. the Adaptive Top-K routing.

## SuggestedFiguresTablesExperiments
*   **Modify Table 1 & Table 3:** Add the `Base + LoRA (calib data)` row. 
*   **New Mandatory Table/Figure (VASM Cost Breakdown):** Add an ablation showing the generation speed (tokens/s) of: Base vs. `TLRA_AdaptiveTopK` (No VASM) vs. `TLRA_AdaptiveTopK` (With VASM). This will expose the true cost of your structure-preserving mechanism.
*   **Modify Figure 1 (Pareto Frontier):** The Y-axis should be Hallucination Reduction (e.g., POPE F1) and the X-axis *must* be Relative Latency Overhead (%). Ensure the LoRA baseline is plotted here as a single point to see if your method actually lies on the optimal frontier.
*   **Appendix Figure A1 Expansion:** Ensure this figure explicitly demonstrates a failure case where VASM incorrectly masks a valid physical entity due to POS tagging errors on a partial sentence.

## AcceptanceOutlook
The philosophical framework of this paper is top-tier for a multimedia conference. However, the experimental design is currently rigged to allow `TLRA_calib` to win against baselines that haven't seen its training data. If the authors implement the required LoRA baseline to isolate the actual decode-time contribution, clarify the architectural extraction of visual states, and honestly report the latency overhead of the VASM module, this paper has a very high chance of acceptance—even if the fallback claims must be invoked. If they dodge the LoRA comparison or hide VASM's computational cost, it should be a strong Reject.