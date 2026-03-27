# Review_Strict_V48
## Overall Score
Score: 4/5

## Verdict
This is a methodologically mature and rigorously scoped proposal. The explicit abandonment of the "purely training-free" illusion in favor of a formally calibrated `Local_Calib` pipeline, combined with the stringent `Base + 5k LoRA` control baseline, sets a high standard for evaluation integrity. However, the pre-registered experimental plan requires tightening in its ablation of the semantic mask and the measurement of threshold attrition to ensure the execution matches the strength of the hypotheses.

## Summary
The paper proposes a decode-time logits adjustment framework for Image-LLMs that injects strictly token-local visual evidence. Acknowledging spatial washout in deep self-attention, the authors introduce a hybrid adaptation (`Local_Calib`) optimized on a 5k-sample prior. To mitigate computational overhead and protect linguistic structures, the method employs an Entropy-Scaled Adaptive Top-$k$ pooling mechanism (derived during prefill) and a dual-tier Dictionary-Gated Semantic Masking module (WordNet + deterministic OCR regex). The paper outlines a pre-registered evaluation protocol spanning three evidence chains: hallucination reduction, structure preservation, and local evidence value.

## WhatShouldBeKept
1. **The Scoping:** Restricting the claims strictly to 2D Image-LLMs and explicitly discarding unsupported spatiotemporal video generalizations is a strong, defensible scientific decision. Do not add video back into this manuscript.
2. **The `Base + 5k LoRA` Control:** This is the most critical element of Chain A. Isolating the inference-time routing mechanism from the simple parameter-exposure benefits of the 5k Visual Genome prior is brilliant and mandatory.
3. **The OOV Bypass Tracking:** Openly profiling the "Intervention Trigger Rate (%)" on MMBench/MMMU to avoid conflating structure preservation with mere algorithmic inaction is exactly the type of honesty expected at ACM MM.
4. **The Framing of Baselines:** Correctly classifying DoLa, VCD, and OPERA as successful but orthogonal global regularizers, rather than attacking them for not being spatial, is accurate and prevents unnecessary adversarial reviews.

## MajorWeaknesses
1. **Unmeasured Prefill Threshold Attrition:** The authors correctly identify that calculating the activation threshold strictly during the prefill stage risks "attrition" (drift) during long autoregressive decoding. However, the experimental protocol lacks a specific test to quantify this drift. If the visual hidden states shift significantly by token 200, the prefill-derived $\theta_{active}$ becomes garbage.
2. **Conflation of Tier 1 and Tier 2 Masking Validations:** DocVQA is proposed in Chain C, but performance on DocVQA is almost entirely dependent on the Tier 2 Regex triggers (since WordNet lacks numbers/dates). If the regex is too aggressive, it behaves like global pooling; if too weak, the method bypasses. The current plan does not cleanly isolate the efficacy of Tier 1 vs. Tier 2.
3. **Negative Sampling Naivety:** The InfoNCE negative sampling relies on an absolute spatial exclusion zone of $\text{IoU} < 0.1$. In dense scenarios (e.g., a rider on a horse, a logo on a shirt), objects overlap heavily. Filtering these out as invalid negatives prevents the MLP from learning to differentiate highly entangled local features—which is precisely the stated goal of the paper.

## SectionBySectionComments
- **Abstract & Intro:** The narrative is tight. The formalization of the three structural barriers is clear. The claim is appropriately constrained to "token-local visual intervention + dynamic gating." Do not inflate the claims further; keep the focus strictly on the execution of this specific pipeline.
- **Section 3.1 (Spatial Washout & `Local_Calib`):** The transition from `Local_Zero` to `Local_Calib` is well-justified. However, the semantic negative sampling strategy requires a fallback for highly overlapping objects of different semantic classes. 
- **Section 3.2 (Entropy-Scaled Pooling):** Using visual hidden-state entropy to scale the threshold is mathematically sound for prefill. But how do you handle dynamic image resolutions (e.g., in Qwen-VL or LLaVA-AnyRes) where the spatial grid size varies drastically? Ensure $\mathcal{H}_{vis}$ is normalized by sequence length.
- **Section 3.3 (Semantic Masking):** The BPE continuation inheritance via pre-compiled bitmasks is a strong engineering detail. However, deterministic regex for OCR is brittle. Acknowledge that regex cannot handle multi-line fragmented text seamlessly.

## RequiredRevisions
1. **Threshold Drift Measurement:** You must add an experiment or analytical plot tracking the valid activation rate (or cosine similarity of visual states to the prefill anchor) across decoding steps (e.g., $t=1$ to $t=250$).
2. **Masking Ablation on DocVQA:** Update the Chain C protocol to explicitly ablate the semantic mask: Base vs. `Local_Calib` (WordNet Only) vs. `Local_Calib` (WordNet + Regex). This is required to prove the Regex isn't just gaming the metric.
3. **Refine InfoNCE Negative Criteria:** Modify the spatial exclusion zone logic. Include a subset of "Hard Negatives" where $\text{IoU} > 0.5$ but the semantic super-category is distinctly different, to force the projector to disentangle overlapping features.
4. **Success Criteria Definition:** In Section 4.1, explicitly state the required $\Delta$ between `Base + 5k LoRA` and `Local_Calib`. If the LoRA closes 90% of the gap, the decode-time routing is rendered practically irrelevant. Establish a falsifiable boundary here.

## SuggestedFiguresTablesExperiments
To form your final execution roadmap, adhere to the following updates in your pre-registered plan:
- **Table 1 (Hallucination):** Keep as planned. Ensure the FPR is reported for POPE. 
- **Table 2 (Dense Spatial - Chain C):** Add the ablation columns for Masking Tiers: `Base` | `Local_MeanPool` | `Local_Calib (Tier 1)` | `Local_Calib (Tier 1+2)`.
- **Figure 1 (OOV Scatterplot):** Excellent design. Ensure the marker color represents the task type (e.g., MMBench vs MMMU) to show domain-specific bypass trends.
- **NEW Figure 4 (Threshold Attrition):** A line graph plotting Sequence Length (X-axis) against the % of Dictionary Terms that still surpass $\theta_{active}$ (Y-axis). This will empirically prove whether the prefill threshold holds up during autoregression.
- **Appendix D (Failure Cases):** Ensure you include a specific failure case where a word polysemy (e.g., "bank") triggered an incorrect local visual mapping, exposing the limitation of context-blind deterministic gating.

## AcceptanceOutlook
The framing, baseline construction, and scope are well within the expectations of top-tier ACM MM papers. If the empirical execution adheres strictly to this pre-registered protocol, survives the `Base + 5k LoRA` control, and successfully isolates the proposed latency/OOV trade-offs without collapsing, this paper will be highly competitive for acceptance. Execute exactly as written, with the required revisions added.