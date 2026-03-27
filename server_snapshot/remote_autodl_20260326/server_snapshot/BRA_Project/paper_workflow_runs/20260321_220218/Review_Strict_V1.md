# Review_Strict_V1
## Overall Score
Score: 2/5

## Verdict
Reject (with encouragement for fundamental methodology restructuring). The current draft identifies a highly relevant problem (the "Pooling Paradox" in high-resolution MLLM inference-time intervention) but suffers from severe theoretical disconnects, unverified mathematical assumptions, and unprofessionally bombastic rhetoric. The experimental plan is ambitious but requires significant methodological correction before execution.

## Summary
The paper aims to mitigate multimodal hallucinations in MLLMs during inference time without incurring the heavy latency of Test-Time Compute (TTC) or the spatial information loss ("Pooling Paradox") of traditional hidden-state intervention methods. The authors propose Bi-directional Resonance Anchoring (BRA), which directly computes cosine similarity between candidate token embeddings ($W_{vocab}$) and unpooled visual features ($Z_{vision}$), using the maximum similarity to reshape output logits. An extension using feature differences ($Z_\Delta$) is proposed for video action hallucination.

## WhatShouldBeKept
1. **The critique of the "Pooling Paradox":** The identification that modern high-resolution MLLMs (using 2D-RoPE and dynamic AnyRes) are fundamentally incompatible with traditional mean-pooling representation engineering (RepE/ITI) is astute and should remain the core motivation.
2. **The "Zero-Pooling" objective:** Attempting to anchor text to the full uncompressed visual token grid is mathematically sounder than compressing $N_v$ tokens into a single vector.
3. **The 4+2 Benchmark Matrix Plan:** The proposed evaluation protocol (POPE, CHAIR, MMBench, MME, FREAK, MMMU, VIDHALLUC) is rigorous and comprehensively covers spatial, logical, and temporal hallucination axes. Keep this evaluation framework.

## MajorWeaknesses
1. **Title-Content Disconnect & Theoretical Inconsistency:** The title prominently features "Adaptive Orthogonal Subspace Projection", yet the text explicitly claims to have "ruthlessly abandoned" SVD and projection in favor of Logits-space Resonance Anchoring. You cannot have a title describing a method that the paper explicitly rejects. 
2. **The $W_{vocab}$ vs. $Z_{vision}$ Space Mismatch (Critical Flaw):** The equation $S_{res}(c) = \max_j (\cos(w_c, \bar{z}_j))$ relies on an unverified mathematical leap of faith. $W_{vocab}$ lives in the final layer's unembedding space. Exactly *which* layer is $Z_{vision}$ extracted from? If it is from the vision encoder, it is not in the same manifold as $W_{vocab}$ and cosine similarity is meaningless. If $Z_{vision}$ is the visual tokens at the *final* transformer layer, they have already undergone massive bidirectional self-attention with the text prefix! By the final layer, $z_j$ is no longer a "pure visual feature" but a highly contextualized multimodal state. The claim of a "natural semantic filter" collapses if the extraction layer is not rigorously defined and theoretically defended.
3. **Naive Temporal Difference ($Z_\Delta$):** Defining action semantics as a simple spatial difference $z_{vision}^{(t)} - z_{vision}^{(t-1)}$ is extremely naive. In deep feature spaces, camera motion, object occlusion, and scaling do not linearly subtract to form an "action vector" that cleanly aligns with the $W_{vocab}$ of verbs like "running". 
4. **Flawed Triggering Mechanism:** Triggering the intervention based strictly on entropy delta ($\Delta E_t > \epsilon$) assumes hallucinations only occur when the model is uncertain. Abundant literature shows that language-prior-induced hallucinations are often generated with *high* confidence. This threshold may completely miss the most stubborn hallucinations.
5. **Bombastic Rhetoric:** The text is littered with dramatic, unscientific phrasing ("dead knots," "降维打击," "ruthlessly eliminated"). This tone is inappropriate for ACM Multimedia and masks the actual scientific mechanism.

## SectionBySectionComments
* **Abstract & Intro:** Rewrite the title immediately. Remove claims of "zero spatial loss" until you mathematically prove that the max-pooling operation over $\cos(w_c, z_j)$ doesn't implicitly ignore complex multi-token object representations.
* **Section 2.1 (TTC critique):** While comparing with Test-Time Compute (o1/Self-Correction) is trendy, it is fundamentally a strawman. TTC is designed for deep System-2 reasoning, not visual grounding. Be critical but fair; do not claim TTC is "mindless" (无脑).
* **Section 3.1 & 3.2:** You must explicitly define $\bar{z}_j$. Is it pre-LLM? Post-LLM layer $L/2$? Post-LLM layer $L$? If it's pre-LLM, how is it aligned with $W_{vocab}$? If post-LLM, how do you strip the text attention out of it?
* **Section 3.4:** The assumption that $\cos(w_{\text{verb}}, z_t - z_{t-1})$ peaks for correct verbs needs an isolated toy-experiment proof before being deployed in the main architecture.
* **Hardware Note:** You mention using "RTX 5090". Assuming this is a typo for 4090 or a placeholder, fix it. If you have early silicon access, state it clearly, otherwise it looks fabricated.

## RequiredRevisions
1. **Unify the terminology:** Remove all lingering mentions of "Orthogonal Subspace Projection" if the method is purely Logits Resonance.
2. **Formalize the Extraction Layer:** Add a dedicated subsection explicitly detailing the extraction of $\bar{z}_j$. Provide an ablation study showing the effect of extracting $Z_{vision}$ from different layers of the LLM.
3. **Revise the Entropy Trigger:** Either remove the $\Delta E_t$ trigger and apply BRA continuously, or provide a statistical distribution showing that multimodal hallucinations uniquely correlate with $\Delta E_t > \epsilon$.
4. **Tone down the language:** Replace all emotive, aggressive adjectives with objective scientific descriptions. Let the data speak for itself.

## SuggestedFiguresTablesExperiments
Since your experiments are currently placeholders, your execution plan must address the following to be accepted:
1. **For Table 1 (Iso-Compute):** Instead of just comparing against TTC, you must compare against state-of-the-art training-free methods: VCD, OPERA, and DoLa. Add a column for memory overhead (Peak VRAM), not just latency, to prove your $\mathcal{O}(K \cdot N_v \cdot d)$ claim.
2. **Missing Ablation (The "Max" vs "Mean" vs "Attention" Pool):** You claim Zero-Pooling (Max) is superior. You must include an ablation table on the FREAK benchmark where you replace your $\max$ operation with a mean pool and a top-K pool. If max doesn't strictly beat mean-pool on OCR/Counting, the entire paper's premise fails.
3. **Missing Ablation (Hyperparameters):** Provide a heatmap for $\alpha$ and $\beta$. If the method only works within a razor-thin margin of $\alpha \in [0.19, 0.21]$, it is too brittle.
4. **For Fig 6 (Appendix - Topological Orthogonality):** This placeholder is critical. You must plot the $S_{res}$ distribution for *actual* outputs. Furthermore, show the distribution of $S_{res}$ for heavily hallucinated words vs. visually grounded words. If the distributions overlap by more than 20%, your $\alpha/\beta$ logic will destroy fluent text.
5. **Video Experiment (Table 4):** To prove $Z_\Delta$ works, you must include a baseline where you use traditional Optical Flow features instead of $Z_\Delta$. If $Z_\Delta$ is just capturing noisy pixel shifts, optical flow will easily beat it. Prove your latent difference is semantically meaningful.

## AcceptanceOutlook
In its current state, the paper is conceptually fractured and rests on unproven assumptions about the alignment between the unembedding matrix and deep contextualized visual features. If the authors can rigorously define the feature extraction layer, mathematically justify the $Z_{vision} \cdot W_{vocab}$ dot product, and successfully execute the demanding experimental plan outlined above without cherry-picking, this could be a strong submission in the next cycle. For now, it is a Reject.