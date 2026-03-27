# Review_Strict_V95

## Overall Score
Score: 3/5

## Verdict
The paper presents an intellectually honest but architecturally brittle approach to MLLM hallucination. The authors construct a highly rigorous evaluation contract to test a fundamentally limited mechanism: a context-agnostic, late-fusion static logit bias (framed here as "hybrid auxiliary routing"). While the scientific rigor of the proposed ablations (especially the objective-matched LoRA control) is commendable and rare, the reliance on a hardcoded English dictionary (VASM) to prevent grammar collapse severely limits the method's claim as a robust MLLM architecture. Acceptance hinges strictly on the unmodified execution of the proposed experimental contract, plus the inclusion of a "distractor-prompt" test to quantify the method's vulnerability to its own context-blindness.

## Summary
The authors propose Token-Local Resonance Anchoring (TLRA), an inference-time logit-adjustment technique for MLLMs. By projecting intermediate visual states directly into the LM's static vocabulary space, TLRA boosts the generation probability of visual nouns. To mitigate the inherent dangers of this context-blind "bag-of-words" approach, the method restricts interventions to a Top-$M$ candidate window and applies a hardcoded BPE-based semantic mask (VASM) to prevent interference with grammatical tokens. The current draft outlines a strict falsifiability protocol, including comparisons against objective-matched LoRA training, continuous-fusion baselines, and explicit audits of relational/spatial hallucination benchmarks where the method is theoretically expected to fail.

## WhatShouldBeKept
1. **The `Base + LoRA` Objective-Matched Baseline:** This is an excellent, mathematically sound control. Confounding routing architecture with fine-tuning data is a pervasive flaw in MM literature; your isolation of this variable must remain in the final paper.
2. **The "Hijacking Problem" Framing (Figure B):** Explicitly acknowledging the mathematical ceiling of Top-$M$ bounded routing is strong science. Keep the CDF plot.
3. **The Relational Blindspot Audit:** Openly testing on MMHal/GQA to expose the failures of context-agnostic noun boosting is a mandatory inclusion. Do not soften this section even if the results are poor.
4. **The VASM Ablation (`TLRA w/o VASM`):** You must keep the raw PPL explosion ablation. It provides critical insight into the manifold dynamics of LLMs.

## MajorWeaknesses
1. **The Identity of the Method is Oversold:** You define this as a "hybrid auxiliary routing head." stripped of the jargon, TLRA is a **static, late-fusion visual logit bias**. Because it bypasses the LLM's autoregressive contextualization, it is fundamentally an image-to-bag-of-words projector glued to the end of an LLM. 
2. **The VASM "Bottleneck" is a Fatal Heuristic, Not Just a Constraint:** You acknowledge VASM as a bottleneck, but you underestimate its theoretical damage. A multimodal language model that relies on an English C4-derived static $\{0,1\}$ BPE mask to prevent "catastrophic grammar collapse" is highly brittle. This completely breaks down for multilingual prompts, morphologically rich languages, and nuanced subword tokenizations. It proves that $\Phi_{calib}$ has not learned to route; it has only learned to spike, and you are manually muting the spikes.
3. **The `ContinuousAdd_Gated` Baseline is a Strawman:** You attempt to prove discrete logit routing is superior to continuous fusion by adding projected visual states directly to the *final* hidden state ($h_{t, final}$) before the LM head. This is destined to fail. The final hidden state of an LLM is a highly contextualized manifold optimized for syntax and semantics; injecting raw, uncontextualized visual states here will obviously destroy perplexity. A fairer continuous baseline would be injecting these local visual features via cross-attention at an earlier layer.
4. **Latency Claims vs. PyTorch Reality:** You claim $M=50$ optimizes CUDA thread synchronization. However, computing $O(M \cdot N_v)$ dot products dynamically at *every autoregressive step* in native PyTorch will cause massive GPU idle time due to kernel launch overhead, not just thread sync. Unless you have written a custom fused Triton/CUDA kernel for this, your wall-clock latency (Table 2) will likely be unacceptable for production.

## SectionBySectionComments
*   **Abstract & Introduction:** The framing is aggressive and refreshing. However, tone down "hybrid trained auxiliary routing head" and be upfront that this is a late-fusion logit bias. 
*   **Section 3.1:** The objective-matching is brilliant. However, as noted above, the continuous addition baseline is mechanically flawed.
*   **Section 3.2:** The equation for $S_{raw}(c)$ relies on $\tau_{sim}$. You must define how sensitive the model is to this hyperparameter. If $\tau_{sim}$ requires per-image tuning, the method is dead on arrival.
*   **Section 3.3:** Bounding by local logit spread ($\Delta_L$) is clever, but relying on generation temperature $T$ to stabilize it feels hacky. What happens during beam search, where temperature semantics change or are absent?
*   **Section 4 (Evaluation Protocol):** The plan to evaluate POPE/CHAIR vs. MMHal is exactly what is needed. However, you are missing a critical test for context-blindness: Distractor Prompts.

## RequiredRevisions
1. **Redesign the Continuous Baseline:** If you want to claim discrete logit routing is superior to continuous fusion, you must compare against a standard continuous integration method (e.g., prefix tuning or a single cross-attention layer initialized to zero), not merely adding raw visual vectors to the final syntactic hidden state.
2. **Acknowledge/Test the Multilingual Breakdown:** You must add a brief discussion or a small-scale experiment showing what happens to TLRA when the prompt is in a non-English language. Given VASM is English-derived, I expect TLRA to either fail to boost nouns (if the target language tokens aren't in VASM) or destroy grammar. Document this boundary.
3. **Clarify the Implementation of the Top-M Kernel:** Explicitly state in the methodology whether the $O(M \cdot N_v)$ routing is implemented as a custom fused kernel or native PyTorch tensor operations, as this directly impacts the legitimacy of Table 2.

## SuggestedFiguresTablesExperiments
*   **Mandatory New Experiment: The Distractor Prompt Test.** 
    *   *Setup:* Pass an image containing two prominent objects (e.g., a Dog and a Cat).
    *   *Prompt:* "Describe the Dog in detail. Do not mention the Cat."
    *   *Hypothesis:* Because TLRA is context-blind and computes static affinity, it will indiscriminately boost the logits for "Cat" simply because the cat is visually present, forcing the LLM to hallucinate the cat into the text despite negative prompt instructions. 
    *   *Action:* Quantify this failure rate compared to the Base model. This is the true test of the "Context-Agnostic Bag-of-Words" limitation.
*   **Enhance Table 1:** Add standard deviations for POPE and CHAIR-s across at least 3 random seeds for the `Base + LoRA` and `TLRA` training phases. Logit-adjustment methods are notoriously high-variance.
*   **Enhance Table 2:** Add a column for "VRAM Overhead" to Table 2. Caching $\Phi_{calib}(h_{prefill}^{(v_j)})$ for 4000 tokens per image has a memory footprint that must be reported alongside latency.

## AcceptanceOutlook
If the authors execute the proposed experimental contract exactly as written in V95, and add the "Distractor Prompt Test" to honestly quantify the negative side-effects of context-blind routing, this paper will be a strong, highly cited boundary-defining work, earning an Accept. If they back down from the `TLRA w/o VASM` ablation, or if the method fails to beat the `Base + LoRA` objective-matched control, the paper must be rejected as the core routing hypothesis will have been falsified. Execute the plan without fear of negative results on relational tasks.