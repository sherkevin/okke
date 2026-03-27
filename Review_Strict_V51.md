# Review_Strict_V51

## Overall Score
Score: 3/5 (Provisional score based on experimental blueprint; requires full execution to reach a valid 4 or 5).

## Verdict
This draft presents a highly mature, theoretically sound, and refreshingly honest methodological framework. The conceptualization of the problem—injecting token-local visual evidence into decode-time logits adjustment without destroying structural language—is framed correctly as a positive proposition rather than relying on strawman attacks against existing baselines. The pre-registered experimental plan is one of the most rigorous diagnostic roadmaps I have seen in recent multimodal submissions. However, a paper with zero executed experiments remains a hypothesis. My review will therefore focus entirely on hardening your experimental blueprint to ensure that when the data comes in, the scientific loop is fully closed and your claims are bulletproof.

## Summary
The paper proposes a "Token-Local Visual Intervention" for 2D Image-LLMs. It acknowledges that deep self-attention washes out spatial fidelity (`BRA_zero`), necessitating a lightweight, 5k-sample trained projection (`BRA_calib`) optimized via compositional-aware contrastive learning. To apply this at decode-time, the authors introduce an Entropy-Scaled Adaptive Top-$k$ mechanism with a Rolling Decay threshold to prevent step-function generative collapse. Finally, to protect functional language and structural tokens, the intervention is gated by a Visual-Aware Semantic Masking (VASM) module using WordNet and deterministic regex. The evaluation is structured around three falsifiable chains (Hallucination, Structure, Local Evidence) and five defense lines.

## WhatShouldBeKept
1. **The 2D Strict Scoping:** Your explicit decision to discard "unsupported generalizations to spatiotemporal video domains" is excellent. Do not add video experiments back in just to appease standard ACM MM narratives. The spatial mapping in video requires temporal tracking mechanisms that this paper does not possess. Keep the scope strictly bounded to 2D.
2. **The Framing of Baselines:** Your acknowledgment that DoLa, VCD, and OPERA are "highly successful, state-of-the-art regularizers" for global distributions is scientifically mature. Keep this framing. It elevates your paper by defining your work as an orthogonal solution to explicit spatial routing.
3. **The Compositional Coherence Margin (Graph + Semantic Veto):** This is the strongest methodological insight in Section 3.1. Standard spatial hard-negative mining almost always destroys co-occurring objects (e.g., horse/rider). Your dual-veto system is theoretically very sound.
4. **Defense Line 2 (The Inaction Boundary):** Openly acknowledging that VASM might preserve reasoning simply by doing nothing (defaulting to $\gamma=0$) shows high analytical rigor. Keep this scatterplot plan.

## MajorWeaknesses
1. **The "Decode-Time" Definition is Strained:** You frame this as an "inference-time mitigation" and "decode-time intervention." However, `BRA_calib` requires training a 2-layer MLP on 5,000 Visual Genome samples. This makes your method a *lightweight hybrid fine-tuning* approach, not a pure training-free inference method like DoLa or VCD. 
2. **VASM is Algorithmically Brittle:** Relying on an offline 85k WordNet dictionary and deterministic regex is structurally archaic. While functional, it is highly sensitive to tokenization quirks. If your tokenizer splits a word in a way that breaks the greedy string match, the mask fails.
3. **Missing Mathematical Formalization:** The text describes the Compositional InfoNCE loss conceptually but lacks the precise mathematical formulation. 

## SectionBySectionComments
- **Abstract & Intro:** Very strong. The pivot from critiquing baselines to proposing a functional framework is clear. 
- **3.1 Embedding Asymmetry:** You need to provide the exact equation for the Compositional InfoNCE loss. Show how the indicator functions for the Graph Veto and Semantic Veto alter the denominator of the contrastive loss.
- **3.2 Entropy-Scaled Top-k:** The rolling decay formula is logical, but $\tau_{target}$ and the window size $W$ are introduced without defining their sensitivity. If $W$ is too small, the threshold will plummet too fast. 
- **3.3 VASM:** Explicitly state whether the $\gamma(c)$ bitmask is applied at the token level (post-BPE) or string level (pre-BPE). The mapping between subwords and WordNet synsets is notoriously messy.

## RequiredRevisions
1. **Clarify the Claim:** You must explicitly concede early in the introduction that your method is a *hybrid* test-time adaptation because of the `BRA_calib` training requirement. Stop calling it purely "decode-time" in a way that equates it with zero-shot methods.
2. **Mathematical Rigor:** Add the explicit mathematical formulation for the dual-veto InfoNCE loss in Section 3.1.
3. **Formalize the BPE-to-WordNet Mapping:** Explain the exact algorithmic step where a generated BPE token (e.g., `[" base", "ball"]`) triggers a WordNet match. If "ball" generates independently, does it trigger the visual check for the spherical object, or does it wait for the full word? 

## SuggestedFiguresTablesExperiments (Execution Blueprint)
Since you are about to execute the experiments, strictly follow these mandates to ensure acceptance:

**For Chain A (Hallucination Reduction):**
- **Crucial:** Execute *Defense Line 1* exactly as planned. The `Base + 5k LoRA` baseline is the make-or-break experiment of this paper. If the 5k LoRA achieves 95% of your `BRA_calib` performance, your decode-time logits adjustment is mathematically redundant, and the paper will be rejected. 
- POPE must be reported in Random, Popular, and Adversarial splits.

**For Chain B (Structure Preservation):**
- Execute the proposed scatterplot (OOV Rate vs. Acc Drop). 
- Add a table for MMMU(Hard) showing performance stratified by discipline (e.g., Math vs. Biology). VASM regex will likely preserve Math (numbers), but WordNet might fail on highly specialized Biology terms. Document this variance.

**For Chain C (Local Evidence):**
- The `BRA_zero` vs `BRA_MeanPool` ablation on DocVQA is excellent. 
- **Mandatory Visual:** Move the "Visual Proof of Locality" (currently planned for Appendix D) into the **Main Text**. You must show a side-by-side heatmap of an image where `BRA_MeanPool` highlights the whole image, but `BRA_calib` successfully highlights a localized bounding box for a specific token. This is the only way to prove your core claim to a reviewer visually.

**For Defense Lines 4 & 5 (Bottlenecks):**
- **ITL Graph:** In Appendix F (or main if space permits), you must plot Inter-Token Latency. Logits manipulation across the full vocabulary using spatial routing will cause CUDA bottlenecks. Quantify the exact ms/token overhead compared to standard greedy decoding.
- **Failure Case Analysis:** Execute your plan to map the logit trajectory of a broken OCR token. Show a graph of the top-5 logits across step $t$ where the correct localized token was suppressed because $\gamma=0$ bypassed the visual intervention.

## AcceptanceOutlook
This is a highly promising blueprint. If you execute the pre-registered protocol exactly as you have described it, and if Defense Line 1 (the LoRA boundary) proves that the decode-time routing is genuinely superior to simple parametric injection, this paper will be a very strong candidate for acceptance at ACM MM. Focus on empirical execution and do not change your methodological framing.