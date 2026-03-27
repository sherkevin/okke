# Review_Strict_V29
## Overall Score
Score: 3/5

## Verdict
This is a highly mature, self-aware, and structurally rigorous experimental protocol. By strictly bounding the problem to 2D token-local visual evidence and explicitly confronting embedding asymmetry, the paper presents a falsifiable and scientifically defensible proposition. The score of 3 reflects the current "pending execution" state; if the empirical results fulfill the pre-registered success criteria without moving the goalposts, this is on a direct trajectory to a strong Accept.

## Summary
The paper proposes Bounded Resonance Anchoring (BRA), a decode-time intervention to reduce MLLM hallucination by directly injecting token-local visual evidence into the terminal logits space. To navigate the structural hazards of this intervention, it introduces three mechanisms: (1) a calibrated (`BRA_calib`) or zero-shot (`BRA_zero`) vocabulary projection to bridge embedding asymmetry; (2) Threshold-Gated Adaptive Top-$k$ Pooling to isolate local evidence and prevent background dilution; and (3) Vocabulary-Anchored Semantic Masking (VASM) with BPE continuation inheritance to protect language syntax and multi-token entities. The paper outlines a strict 3-chain evaluation protocol (Hallucination, Structure, Local Evidence) awaiting execution.

## WhatShouldBeKept
1. **The Framing of Baselines**: Your decision to treat DoLa, VCD, and OPERA as orthogonal, highly competitive regularizers rather than setting them up as strawmen ("they rely on global pooling") is exactly what I expect at the AC level. Do not change this mature framing.
2. **The `Base + 5k LoRA` Fairness Baseline**: This is a brilliant and necessary inclusion in Chain A. Decode-time interventions that require *any* trained projector (`BRA_calib`) often cheat by claiming zero-shot superiority over base models. This strict equivalent-data baseline keeps the methodology honest.
3. **BPE Continuation Inheritance**: This is a genuinely insightful engineering solution to structural collapse. Preserving the exact $\gamma$ weight across token fragments (e.g., `_rhi`, `no`, `cer`, `os`) prevents pathological subword fragmentation.
4. **Scoping Out Video**: The explicit disclaimer avoiding unsupported generalizations to spatiotemporal video domains is a strength, not a weakness. It tightens your narrative around 2D dense spatial reasoning.

## MajorWeaknesses
1. **The Subword Averaging Assumption in InfoNCE**: In Section 3.1, you define the positive target $W_{vocab}[c^+]$ as the L2-normalized mean of the subword embeddings comprising the ground-truth label. LLM subword embeddings in the terminal vocabulary matrix encode heavy syntactic and structural priors, not pure semantics. Simply averaging them does not guarantee a semantically coherent target vector for $\Phi_{calib}$ to learn from. 
2. **VASM's Noun-Only Limitation**: You state VASM uses a dictionary of "visual object nouns". This structurally blinds BRA to attribute (color, size) and relational/action hallucinations (e.g., "running", "red"). While safe for CHAIR (which focuses on objects), this limitation may severely bottleneck performance on MMBench or MMMU where multi-modal reasoning relies on attributes.
3. **Latency is Marginalized**: Token-level intervention that requires running Top-$k$ pooling, thresholding, and dynamic BPE mask lookups at *every single generation step* will significantly impact autoregressive decoding speed. Relegating latency to Appendix E is unacceptable for a system-level paper.
4. **Post-Hoc Entanglement Assumption**: A single linear layer ($\Phi_{calib}$) may mathematically lack the capacity to reverse the deep spatial washout caused by 32+ layers of dense attention mixing, especially in LLaVA-1.5's 1D sequence architecture. 

## SectionBySectionComments
- **Abstract & Intro**: Excellent tight narrative. The transition from the grand hallucination problem to the specific structural barriers (asymmetry, entanglement, collapse) is textbook.
- **Section 3.1**: The semantic negative constraint (IoU < 0.1 + non-matching class) is well-designed. However, you must specify what happens when an image has *no* negative patches meeting the criteria.
- **Section 3.2**: Mathematically sound. The dynamic top-$k$ rule $k = \max(k_{min}, \lceil \rho \cdot N_v \rceil)$ correctly prevents small objects from being outvoted by a sea of mathematically irrelevant background tokens.
- **Section 4.3 (OOV Tracking)**: Tracking the exact percentage of ground-truth targets where VASM defaults to $\gamma=0$ is a highly commendable self-accountability metric.

## RequiredRevisions
1. **Elevate Latency to Main Text**: Move the A100 Tokens/Sec overhead benchmark into the main evaluation (perhaps Table 1 or a dedicated subsection). A decode-time method's viability is intrinsically tied to its throughput penalty.
2. **Clarify VASM Scope**: Explicitly address the "noun-only" limitation. If VASM ignores verbs and attributes, add a paragraph in the Limitations section acknowledging that BRA targets *object-level* hallucination but explicitly cedes attribute-level hallucination to the base model's priors.
3. **Specify the `Base + 5k LoRA` Setup**: You must define exactly what the LoRA tunes (e.g., Q/V projections in the LLM, or the vision-language MLP projector?). It must have a comparable or greater parameter budget than $\Phi_{calib}$ to validate the claim that the decode-time mechanism is the true source of performance gains.
4. **Subword Fallback Plan**: If the L2-normalized mean of subwords fails to converge during $\Phi_{calib}$ training, prepare a fallback (e.g., using the hidden state of the final subword after passing the full word through the frozen LLM).

## SuggestedFiguresTablesExperiments
1. **Chain C Heatmap Extension**: For the DocVQA failure analysis (Figure 1), do not just show `BRA_AdaptiveTopK` vs `BRA_MeanPool`. You must also show the base model's *native cross-attention / self-attention* map for that token. You need to geometrically prove that `BRA` is supplying *new* localized evidence that the base model's deep attention failed to focus on.
2. **The "Out-of-Candidate" Failure Case**: Devote one qualitative example to demonstrating the out-of-candidate limit. Show an image where the correct visual object exists, but the base model's strong language prior pushes the correct token to rank 150 (outside $M=100$). Show how BRA mathematically fails here. Admitting this limitation with a visual example will vastly increase your credibility as scientists.
3. **Ablate $\Phi_{calib}$ Complexity**: As a minor experiment, test if a 2-layer MLP for $\Phi_{calib}$ is required to un-mix the attention, or if the proposed linear layer is genuinely sufficient.

## AcceptanceOutlook
The proposition is clear: "token-local logits intervention + VASM + fair zero-shot/calibrated split". Do not inflate this claim further. If your executed experiments confirm that `BRA_calib` beats the LoRA baseline on Chain A, strictly preserves MMMU on Chain B, and `AdaptiveTopK` beats `MeanPool` on Chain C, this paper will easily clear the bar for ACM Multimedia. Execute the protocol exactly as written.