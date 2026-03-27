# Review_Strict_V7

## Overall Score
Score: 2/5

## Verdict
Strong Reject in its current form, but with a clear path to Major Revision. The theoretical identification of the "Pooling Paradox" is exceptionally sharp, and the architectural pivot to logits-space intervention is promising. However, the manuscript severely violates the implicit fairness contract of decoding-time baselines by introducing external grounding supervision (RefCOCO), completely ignores the "video" half of the expected ACM Multimedia narrative, and lacks critical stress-testing on reasoning-heavy benchmarks. The experimental plan is structurally sound but practically incomplete. 

## Summary
The paper proposes Bi-directional Resonance Anchoring (BRA), a decoding-time intervention to mitigate hallucinations in MLLMs. It critiques existing hidden-state methods (like DoLa) for falling into the "Pooling Paradox," where spatial information is destroyed by pooling high-resolution visual tokens. BRA mitigates this by intervening directly in the logits space via a learned Patch-to-Vocab projector ($\Phi$), utilizing Adaptive Top-$k$ Resonance to handle dynamic resolutions, and applying a Probabilistic Vocabulary-Anchored Semantic Masking (VASM) based on POS distributions to protect syntax and BPE tokens from semantic collapse.

## WhatShouldBeKept
1. **The "Pooling Paradox" Framing:** This is a highly compelling and mathematically sound critique of hidden-state spatial corruption. Do not dilute this narrative.
2. **Adaptive Top-$k$ Resonance (Eq 2 & 3):** Scaling $k$ dynamically based on visual resolution ($N_v$) is an elegant and necessary engineering choice for models like InternVL. 
3. **Probabilistic VASM (Eq 5):** Replacing predictive entropy with offline, POS-weighted continuous masks is a mathematically rigorous way to escape the "Entropy Trap" while maintaining $O(1)$ runtime complexity.
4. **The Logits-Space Pivot:** Bypassing the hidden space entirely is the correct architectural instinct for high-resolution dense tasks.

## MajorWeaknesses

**1. The "Training-Free" Illusion and Baseline Unfairness**
Your method claims to be an inference-time intervention (competing with DoLa, VCD, OPERA). Yet, Section 3.1 mandates training a projector $\Phi$ using RefCOCO (bounding box grounding data). This fundamentally breaks the comparison. DoLa and VCD operate purely zero-shot on the base model's intrinsic representations. By injecting grounding supervision, you are evaluating *added knowledge* vs. *zero-shot decoding*. Unless you explicitly benchmark against models fine-tuned on RefCOCO, your performance gains on dense spatial tasks will be dismissed as a supervision artifact rather than an algorithmic breakthrough.

**2. Complete Absence of the Video Narrative**
This paper targets ACM Multimedia, intending to maintain an "Image + Video" dual narrative. Yet, the entire methodology and experimental protocol (Sections 3 and 4) refer solely to 2D-RoPE, static image patches, and image datasets (DocVQA, POPE). Video tokens are simply a 3D extension (Spatial-Temporal) of dynamic resolutions. If BRA is truly modality-calibrated and adaptive, it must explicitly handle temporal token dimension expansion. Your experimental blueprint currently guarantees failure on this front.

**3. Mathematical Ambiguity in Modality Calibration**
In Section 3.1, you align $h_L^{(v_j)}$ (last-layer visual tokens) to $W_{vocab}$ (the text unembedding matrix). But in standard MLLMs (like LLaVA), visual tokens are processed *through* the LLM layers. Are you applying $\Phi$ to the vision encoder's output (pre-LLM), or the LLM's final hidden states corresponding to visual token positions (post-LLM)? If pre-LLM, the semantic gap to $W_{vocab}$ is massive. If post-LLM, the LLM has already entangled them with text context via self-attention, making localized alignment messy. This needs exact mathematical clarification.

## SectionBySectionComments

- **Abstract:** Strongly written, but oversells the "zero-pooling" supremacy without acknowledging the heavy lifting done by the supervised $\Phi$ projector.
- **Section 1:** Excellent setup. The conceptualization of the "Pooling Paradox" is publication-worthy on its own.
- **Section 3.1:** As noted above, the InfoNCE loss over RefCOCO introduces an unfair advantage. You must either prove $\Phi$ can be trained strictly on self-supervised signal (e.g., contrastive alignment of image captions without bounding boxes) or place BRA in a different track of "lightweight fine-tuning" rather than "training-free decoding."
- **Section 3.2:** What happens when $N_v$ grows to 50,000 for a 1-minute video? Does $\rho = 0.01$ (500 tokens) still isolate fine-grained features, or does it re-introduce the Pooling Paradox across temporal frames? You need a spatial-temporal decay factor.
- **Section 3.4:** Static POS weights are clever but brittle against contextual polysemy. For instance, the token "lead" could be a metal (noun, $\gamma=1.0$) or an action (verb, $\gamma=1.0$), which works here. But what about "like" (verb vs. preposition)? The static expected value $\mathbb{E}[\gamma(c)]$ might penalize functional syntax if the corpus distribution skews heavily toward semantic usage.

## RequiredRevisions

1. **Decouple $\Phi$ from Supervised Grounding:** You *must* implement Protocol 0 to show the performance of BRA utilizing a completely unsupervised or zero-initialized identity mapping for $\Phi$. If the method collapses without RefCOCO supervision, the core claim is invalidated.
2. **Extend Formulation to Video:** Update Eq 2 and Eq 3 to account for temporal dimensions. Define $N_v = T \times H \times W$. Discuss how Adaptive Top-$k$ operates across frames (does it find resonance in a single frame, or average across temporal patches?).
3. **Clarify the Token Pipeline:** Provide a precise diagram or equation showing exactly where $h_L^{(v_j)}$ is extracted (pre-LLM vs post-LLM). 

## SuggestedFiguresTablesExperiments

Your proposed 6-stage protocol is a good start, but it fails to provide the closed-loop defense required for a top-tier venue. Overhaul your experimental design to enforce the following **Five Defense Lines**:

*   **Defense Line 1: Basic Hallucination (POPE / CHAIR)**
    *   Keep Protocol 2, but strictly control for response length. Report Length-Normalized CHAIR metrics. DoLa often artificially inflates scores by truncating outputs; ensure BRA does not.
*   **Defense Line 2: General Instruction Following (MMBench / MME)**
    *   You must prove that penalizing logits based on visual matching does not lobotomize the model's general conversational ability. Add MMBench to Protocol 2.
*   **Defense Line 3: Spatial & Temporal Stress Test (DocVQA / VIDHALLUC)**
    *   *Image:* Execute Protocol 1 (DocVQA/ChartQA) to prove the Pooling Paradox. 
    *   *Video:* Introduce **VIDHALLUC** (or Video-MME). Prove that BRA's Top-$k$ can retrieve the correct frame/patch for a temporal query without temporal pooling destroying the signal.
*   **Defense Line 4: Deep Reasoning Check (MMMU - Hard Subset)**
    *   Add MMMU (Hard). Does localized visual resonance break down when the token generation requires deep external knowledge rather than just "pointing" at a visual patch? This tests the boundary where Language Inertia is actually *beneficial* (e.g., reciting a physics formula triggered by an image).
*   **Defense Line 5: Factuality vs. Hallucination (FREAK)**
    *   Use the FREAK benchmark to isolate whether BRA suppresses true hallucinations or merely suppresses factual statements that happen to not be visibly present in the image.

**Table Layout Requirement:** 
Create a Master Table grouped by Baseline Type (Zero-Shot Decoding vs. Lightweight Supervision). BRA with RefCOCO-trained $\Phi$ must be clearly separated from pure zero-shot methods like DoLa. Add a row for `BRA (Identity-Phi)` in the zero-shot section.

## AcceptanceOutlook
To achieve a competitive score (4+), the authors must successfully navigate the $\Phi$ supervision dilemma. If you can prove that `BRA (Identity-Phi)` or an unsupervised $\Phi$ still beats DoLa/VCD on dense tasks (DocVQA/VIDHALLUC) due to the Top-$k$ mechanism alone, the paper will be a slam dunk. The expansion into the video dimension is mathematically trivial for your Top-$k$ design but fundamentally necessary to survive the ACM MM review pool. Execute the Five Defense Lines strictly, and this will be an exceptionally strong contribution.