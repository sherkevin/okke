# Pilot Comparison

## Workflow Settings
- Model: `[L]gemini-3.1-pro-preview`
- Pilot scope: `V1 -> Review_V1 -> V2 -> Review_V2`
- Goal: validate that the adversarial workflow can improve paper quality without discarding the core idea

## Score Tracking
- `V1`: 3.0/5
- `V2`: 3.0/5
- Delta: +0.0

## Review V1 Snapshot
# Review_Strict_V97
## Overall Score
Score: 3/5

## Verdict
The paper presents an exceptionally self-aware, pre-registered experimental plan for mitigating MLLM hallucinations via Token-Local Resonance Anchoring (TLRA). The commitment to strict ablation, matched-budget baselines, and falsifiable claims is highly commendable. However, the author attempts to preemptively dictate the terms of the review by framing the paper as an "executable contract." As an Area Chair, I evaluate the scientific validity of the method and the feasibility of its execution, not the rhetorical framing. 

The current methodology has two massive systemic risks that the proposed experiments do not adequately stress-test: the algorithmic brittleness of the Vocabulary-Anchored Semantic Masking (VASM) and the likely prohibitive $O(M \times N_v)$ per-step latency overhead. If VASM relies on static vocabulary lookups, it ignores contextual polysemy; if the latency penalty is an order of magnitude higher than base decoding, the method is practically dead on arrival. The experimental design is robust, but it must be expanded to aggressively probe these two specific failure boundaries before acceptance.

## Summary
The paper proposes TLRA, a decode-time intervention to reduce MLLM hallucination by adjusting token logits based on localized visual evidence. It extracts token-local support from cached visual states, applying an Adaptive Top-k resonance penalty to the logits of a bounded candidate set (Top-$M$). To prevent the degradation of syntax and multi-token entities, it introduces VASM, an offline precomputed mask. The paper heavily emphasizes methodological hygiene, splitting the method into a training-free probe (`TLRA_zero`) and a calibrated variant (`TLRA_calib`), mandating matched-budget comparisons against a `Base + LoRA` model. Currently, the paper is in a pre-registered state with empty tables (TBF).

## WhatShouldBeKept
1. **The Strict Zero vs. Calib Split:** Acknowledging that `TLRA_calib` uses trained parameters (`Phi_calib`) and explicitly refusing to hide it among training-free baselines (like VCD or DoLa) is excellent scientific practice. 
2. **The Matched-Budget Baseline (`Base + LoRA`):** This is a brilliant and necessary control. If `TLRA_calib` cannot beat a simple LoRA trained on the same calibration data, the decode-time complexity is unjustified. Keep this front and center.
3. **Top-M Hijacking Audit:** Acknowledging that decode-time interventions cannot recover tok

## Revision Log V2 Snapshot
# Revision_Log_V117
- **Addressed the "Geometric Manifold" Fallacy ($W_{in}$ vs $W_{out}$):** Explicitly acknowledged the depth gap between layer 0 and layer $N$. Reframed the $W_{out}$ projection as a "Shallow Semantic Bypass" and added the `TLRA_W_in` ablation to the pre-registered protocol (Chain C).
- **Reframed Negation Kill-Switch:** Downgraded the heuristic to a "Lexical Fallback Constraint", acknowledging it as a brittle limit of the architecture. Added "False Positive Negation Rate" to Chain A.
- **Added `TLRA_Blind` Baseline:** Added a mandatory blind-adapter ablation to Table 1 to prove hallucination reduction stems from token-local spatial resonance, not just memorizing the Visual Genome long-tailed prior.
- **Clarified Hardware/Trie Masking Reality:** Specified byte-level tokenizer dependencies (e.g., Tiktoken leading spaces) and added "TPOT Jitter (Std Dev)" to Table 3 to explicitly audit the branch-divergence cost of batched Trie traversal on GPUs.
- **Mandated Collateral Damage Visualizations:** Updated Chain C visual requirements to explicitly mandate failure cases where occlusion causes false penalties.
- **Retained Core Strengths:** Maintained VASM, dynamic activation pooling, TTFT vs TPOT disaggregation, and the brutally honest tone regarding system boundaries.

## Review V2 Snapshot
# Review_Strict_V117
## Overall Score
Score: 3/5

## Verdict
The paper proposes an exceptionally self-aware, pre-registered experimental protocol for a late-fusion MLLM hallucination mitigation module (TLRA). However, the core methodology—projecting layer-0 visual features directly into layer-$N$ unembedding weights—is theoretically suspect and fundamentally reduces the proposed module to a context-blind, parallel multi-label image classifier. While the brutal honesty regarding hardware latency, BPE fragmentation, and negation brittleness is refreshing and intellectually rigorous, the structural limitations of the method may prove fatal once the experiments are actually executed. Acceptance hinges entirely on whether the pre-registered protocol, plus several mandatory controls added in this review, can prove the system doesn't collapse under its own architectural compromises.

## Summary
The authors propose Token-Local Resonance Anchoring (TLRA), a supervised late-fusion adapter that extracts visual features post-connector (Layer 0) and maps them directly to the LLM's final vocabulary unembedding matrix ($W_{out}$) to penalize hallucinated physical nouns during decoding. To address the inherent flaws of decode-time logit manipulation, the authors introduce Dynamic Activation Pooling, Deterministic Trie Masking (for BPE collateral damage), Visual-Aware Syntax Maintenance (VASM), and a lexical Negation Fallback. The paper currently reads as a pre-registered protocol, explicitly outlining five major architectural vulnerabilities and proposing three experimental chains to test them.

## WhatShouldBeKept
1. **The Falsifiable Protocol Structure:** The explicit definition of the "Five Hard Boundaries" and the pre-registered baseline hypotheses (especially `TLRA_Blind` and `TLRA_no_VASM`) are exemplary. Do not water this down in the final version.
2. **Deterministic Vocabulary Trie Masking:** Exposing the fallacy of byte-level BPE collateral damage (e.g., Tiktoken leading spaces) is a highly valuable contribution to decode-time intervention literature, which often sweeps this under the rug.
3. **The Batched Inference Latency Audit (Chain C):** Disaggregating TTFT, TPOT, and especially *TPOT Jitter* at Batch Size 32 is a critical system-level reality check that must remain in the paper.
4. **VASM ($\tau$-Abort):** The mathematical acknowledgment that hallucination suppression must sometimes concede to the language prior to prevent grammatical destruction is a matur

## Preliminary Judgment
- If `V2` improves the score or shifts criticism from fatal flaws to actionable weaknesses, the workflow is functioning.
- If the score does not improve, inspect whether the scientist over-rewrote the draft or failed to address the harshest reviewer concerns.
- Use the next rounds to focus on claim calibration, experimental closure, fairer related-work positioning, and clearer contribution boundaries.
