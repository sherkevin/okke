# Review_Strict_V94
## Overall Score
Score: 3/5

## Verdict
The paper proposes an admirably rigorous, self-falsifying experimental protocol for a hybrid visual routing head. The authors deserve high praise for explicitly decoupling the training objective from the architectural routing mechanism (via the `Base + LoRA` and `ContinuousAdd_Gated` baselines). However, the theoretical mechanism harbors a critical flaw: by dot-producting static prefill visual states with static LM-head vocabulary embeddings, TLRA fundamentally operates as a **context-agnostic "Bag-of-Words" noun booster**. The experimental plan is currently heavily biased toward object-existence benchmarks (POPE, CHAIR) which will mask this flaw. The paper is conditionally strong, provided the experimental contract is expanded to expose the method's boundaries on relational/spatial tasks and strictly ablate the highly brittle VASM heuristic.

## Summary
The authors present Token-Local Resonance Anchoring (TLRA), a hybrid auxiliary routing head designed to mitigate MLLM hallucination. A trained MLP projector maps intermediate-layer visual tokens into the pre-LM-head lexical space. During autoregressive decoding, these static visual representations are dot-producted against the Top-$M$ candidate tokens to dynamically adjust logits. To protect grammar, the method employs a statically hardcoded English dictionary mask (VASM). The paper outlines a strictly bounded evaluation protocol, including parameter/data-matched baselines, explicit latency audits, and mathematically quantified ceilings (the "Hijacking" problem). Currently, empirical results are pending (TBF).

## WhatShouldBeKept
1. **The Evaluation Contract (Section 4):** This is the strongest aspect of the paper. The mandatory parity ablation against an objective-matched `Base + LoRA` and the `ContinuousAdd_Gated` baseline sets a gold standard for MLLM architectural claims. 
2. **The "Hijacking Problem" CDF (Figure B):** Explicitly acknowledging and measuring the mathematical ceiling of your intervention window is excellent scientific practice.
3. **Hardware Latency Audit (Table 2):** Transparently breaking down the ms/token cost of the Top-$k$ kernel prevents the method from hiding behind unmeasured wall-clock degradation.
4. **The Tone of Self-Skepticism:** The explicit framing of VASM as a "crutch" and the static prefill extraction as a limitation builds immense trust. Keep this framing.

## MajorWeaknesses

**1. The Method Identity Challenge: TLRA is a Context-Agnostic Noun Booster, not Contextual Grounding.**
The paper describes "Token-Local Resonance Anchoring" via the equation $score(\Phi_{calib}(h_{prefill}^{(v_j)}), c)$. Assuming $c$ represents the static LM-head weights $W_{head}[c]$ for candidate $c$, this dot product is fundamentally **context-free**. It measures the static affinity between a visual patch (e.g., an apple) and a word ("apple"). It has zero awareness of the generated prefix. Therefore, TLRA will indiscriminately boost the token "apple" whether the prompt asks "What is on the table?" or "What is the man eating?". It cannot resolve relational queries (e.g., "the apple *left* of the cup") because the static visual state is decoupled from the dynamic text query at step $t$. You must redefine the method's identity or prove it does not suffer from this fatal context-blindness.

**2. The Illusion of POPE and CHAIR (Experiment Design Bias):**
Because TLRA is structurally biased toward context-free noun boosting, it will artificially inflate POPE (object polling) and CHAIR (object hallucination) scores simply by upweighting visible nouns. This is not true visual grounding; it is visual existence voting. Your evaluation protocol completely lacks compositional, spatial, or relational hallucination benchmarks, which is exactly where this method is theoretically predicted to fail.

**3. VASM is a Fatal Bottleneck, not just an "Engineering Crutch":**
If $\Phi_{calib}$ cannot map visual states to the lexical manifold without destroying grammatical syntax (requiring a static, C4-derived English dictionary mask to survive), the continuous mapping objective has fundamentally failed. A multi-modal LLM cannot rely on an English BPE lookup table. If VASM is required for the method to beat the baseline, the claim of a "learned routing head" is severely compromised. 

## SectionBySectionComments

*   **Abstract & Intro:** You claim TLRA injects "token-local visual evidence". This is a misnomer. The evidence is spatially local (from visual patches), but it is *not* temporally/contextually local to the dynamic token $t$, because it is extracted statically at prefill. Rephrase to reflect "static local visual evidence."
*   **Section 3.2 (Manifold Mismatch):** The mathematical definition of $c$ in the $score$ function must be explicitly defined. Is it the row of the final LM unembedding matrix? State this plainly. 
*   **Section 3.3 (Temperature Anchoring):** Bounding the penalty to $\min(\Delta_L / T, \beta)$ is clever, but if $T \to 0$ (greedy decoding, standard for POPE), the spread boundary explodes. You need to clarify how the routing behaves under greedy decoding.
*   **Section 4.1 (Continuous Baseline):** The `ContinuousAdd_Gated` baseline is a brilliant inclusion. Ensure the gating scalar $\gamma$ is initialized to 0, but verified to have non-zero gradients during training, to prove it wasn't just dead on arrival.

## RequiredRevisions

1.  **Introduce Relational/Compositional Benchmarks:** You must add benchmarks that penalize context-free noun boosting. e.g., MMHal, AMBER, or a specific spatial/relational subset of GQA/VisDial. If TLRA degrades performance on relational questions while boosting POPE, you must explicitly narrow your paper's claim to "Object-Existence Hallucination Mitigation" rather than general "MLLM Grounding."
2.  **Mandatory Ablation of VASM:** Table 1 must include a `TLRA w/o VASM` row. You must expose exactly how catastrophic the grammar collapse is without the hardcoded dictionary. If PPL explodes without VASM, the community needs to see the raw numbers.
3.  **Clarify the Score Function:** Explicitly define the vector representation of $c$. If it is the unembedding weight, explicitly acknowledge the context-agnostic limitation in the methodology section.

## SuggestedFiguresTablesExperiments

*   **Experiment - The Temperature Sweep:** Since your logit adjustment $\Delta_L / T$ scales inversely with Temperature, plot CHAIR-s vs. Temperature ($T \in [0.1, 1.0]$) for Base vs. TLRA to prove the routing head doesn't destabilize at the extremes.
*   **Experiment - Top-M Sensitivity:** In Table 2, alongside latency, show POPE F1 vs. $M \in \{10, 50, 100, \text{all}\}$. This will prove whether your $M=50$ bound is a harmless optimization or a severe performance constraint.
*   **Failure Case Analysis (Crucial):** Devote a qualitative figure specifically to **Relational/Spatial failures**. Show a case where TLRA incorrectly boosts a visually present object because it matches the candidate token, even though the grammatical/spatial context of the sentence should have suppressed it.

## AcceptanceOutlook
The experimental *plan* is one of the most intellectually honest and rigorous protocols submitted to this venue. However, the mechanism's inherent limitation as a "bag-of-words" noun-booster is a major structural weakness. If the authors execute the proposed tables, add the required relational benchmarks, and honestly report where the context-free routing breaks down, this paper will be a highly valuable, well-bounded contribution. If they attempt to hide the context-blindness by only evaluating on POPE/CHAIR, the paper should be rejected. Conditional Accept pending rigorous execution of the expanded contract.