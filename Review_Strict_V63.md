# Review_Strict_V63
## Overall Score
Score: 4/5

## Verdict
This manuscript presents a highly mature, logically bounded, and structurally sound experimental blueprint. The authors have correctly abandoned the flawed strategy of broadly attacking baselines, instead proposing a mathematically specific, positive proposition: injecting token-local visual evidence at decode-time while using a deterministic vocabulary mask (VASM) to prevent structural language collapse. The rigorous pre-registration of evidence chains—particularly the inclusion of Average Generated Length (AGL) and Perplexity (PPL) as control metrics—demonstrates a strong grasp of the actual pitfalls in MLLM hallucination mitigation. If the executed experiments faithfully populate this protocol, this will be a strong contribution to ACM Multimedia. However, minor tensions regarding the definition of "training-free," the mechanics of the calibrator, and the total excision of OCR motivation must be addressed before final execution.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), an inference-time (base-model-frozen) logits intervention method to reduce multimodal hallucinations. It operates via a lightweight calibrator (`TLRA_calib`) to map visual tokens to vocabulary space, an Adaptive Top-$k$ patch selection bounded by prompt-conditioned pruning for high-resolution efficiency, and a Vocabulary-Anchored Semantic Masking (VASM) module driven by WordNet to bypass intervention on functional syntax and out-of-vocabulary tokens. The evaluation is structured as an experimental contract requiring three evidence chains: hallucination reduction (controlled by length), structure preservation (controlled by perplexity), and the isolated value of local over global evidence.

## WhatShouldBeKept
1. **The Positive Proposition Framing:** Do not alter the current tone regarding baselines (`DoLa`, `VCD`, `OPERA`). Treating them as highly competitive alternatives rather than strawmen strengthens your paper. Continue to avoid the false narrative that these baselines "rely on global pooling." 
2. **The Control Metrics (AGL and PPL):** The inclusion of Average Generated Length alongside POPE/CHAIR, and WikiText Perplexity alongside MMBench, is excellent. It prevents the common illusion where a model achieves "state-of-the-art hallucination reduction" simply by generating truncated, broken English.
3. **The Explicit OCR Concession:** Acknowledging the "OCR Paradox" and mathematically defining VASM to bypass OOV text is scientifically honest and saves the paper from unsupportable claims.
4. **Efficiency Audits:** Bounding the intervention to $O(M \times N_{active})$ via prompt-conditioned pruning is practically necessary for modern $>2000$ token vision encoders. Figure 1 must be kept as planned.

## Major Weaknesses
1. **The "Training-Free" Misnomer:** The abstract states you want to "ensure fair comparison against training-free baselines," but you train a projection matrix ($\Phi_{calib}$) on CC3M. While the base MLLM is frozen, your pipeline is strictly *not* training-free. It introduces external paired data (CC3M) that baselines like `DoLa` or `OPERA` do not see at inference. You must re-label this as a "Base-Model-Frozen plug-in" rather than implying it is zero-shot in the same vein as VCD.
2. **Residual OCR/Document Ambiguity:** You explicitly state in Section 4.3 that Document/OCR tasks are excluded per the VASM boundary. This is the correct methodological decision. However, you must ensure that *nowhere* in the Introduction or related work do you motivate this paper using dense text, document understanding, or complex charts. If the method explicitly bypasses OCR, motivating the paper with OCR-heavy scenarios is a logical contradiction.
3. **Calibrator Mechanism Opacity:** The mechanism for training $\Phi_{calib}$ via "token-to-patch InfoNCE" on generic image-caption data is dangerously underspecified. How do you extract bounding boxes or patch-level alignments from generic captions? If you rely on coarse CLIP similarity to generate pseudo-labels for InfoNCE, the "local" evidence might just be an artifact of CLIP's global biases.

## SectionBySectionComments
*   **Abstract/Intro:** Highly effective scoping. The claim limitation to "spatial object grounding and counting" directly aligns with the method's capabilities. 
*   **3.1 Calibration:** Clarify the InfoNCE setup. Is this a dense contrastive loss? How do you prevent the projector from collapsing into a bag-of-words representation?
*   **3.4 VASM:** The WordNet fallback (`physical_entity.n.01` and `color.n.01`) is clever but brittle against polysemy (e.g., "crane" the bird vs. "crane" the machine vs. "crane" one's neck). You should briefly address how context-free BPE mapping handles word sense disambiguation.
*   **4.3 Evidence Chain C:** You hypothesize that `TLRA_zero` will fail due to embedding asymmetry. You need to ensure that `TLRA_MeanPool` utilizes the *same* calibrator ($\Phi_{calib}$) as `TLRA_AdaptiveTopK`. If `MeanPool` uses zero-shot and `TopK` uses the calibrator, the ablation is entirely invalid.

## Required Revisions
1. **Scrub OCR/DocVQA Motivation:** Completely remove any references to reading text, document parsing, or complex charts in the motivation, as VASM renders the method inert on these tasks. Focus purely on fine-grained visual grounding, spatial relations, and counting.
2. **Clarify Baseline Data Fairness:** Add a disclaimer acknowledging that while `TLRA_calib` keeps the base model frozen, it leverages external CC3M distribution knowledge, making direct comparison to purely unsupervised methods (like DoLa) slightly asymmetrical.
3. **Video Confinement:** As noted in your discussion, keep the spatio-temporal video extensions strictly as a secondary pilot in the Appendix. Do not attempt to weave it into the main narrative, as it will break the rigorous spatial formulation you have built.
4. **Ensure Strict Ablation Parity:** Explicitly state in the methodology that `TLRA_MeanPool` uses the exact same $\Phi_{calib}$ weights as `TLRA_AdaptiveTopK`.

## SuggestedFiguresTablesExperiments
To support the planned experimental execution, adhere to the following strict guidelines:
*   **Table 1 (Chain A):** Include standard deviations for AGL to prove that TLRA isn't just increasing variance in output length.
*   **Table 3 (Chain C - Local Evidence):** Break down the FREAK evaluation into "Spatial Relations" vs. "Object Existence." Local evidence should show a massive delta on Spatial Relations compared to MeanPool.
*   **Figure 1 (Efficiency):** Ensure the $x$-axis scales up to $N_v = 4096$ (or whatever the maximum sequence length of your chosen base model is, e.g., LLaVA-1.5-HD or LLaVA-Next).
*   **Figure 3 (Failure Analysis):** Your proposed heatmaps are excellent. For the "VASM Masking Error," explicitly show the raw $S_{raw}(c)$ score that *would* have been applied if VASM hadn't erroneously zeroed out the $\gamma$ mask. This proves the mechanics work even when the heuristic fails.

## AcceptanceOutlook
Assuming the authors execute the experiments exactly as proposed in Section 4—without moving the goalposts, hiding perplexity spikes, or artificially deflating baseline scores—this paper presents a highly scientifically sound, positive contribution to decode-time intervention. Execute the blueprint, fix the "training-free" nomenclature, and it will be a strong Accept.