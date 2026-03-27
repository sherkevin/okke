# Review_Strict_V21
## Overall Score
Score: 3/5

## Verdict
The paper presents a highly structured, mathematically rigorous protocol for a very specific problem: injecting token-local visual evidence into decode-time logits adjustment while preserving language structure. The framing is refreshing—it avoids setting up false strawmen out of existing baselines (VCD, OPERA, DoLa) and instead focuses on a defensible, positive methodological proposition. However, the current manuscript suffers from a severe disconnect between its theoretical formulation (which includes video) and its proposed experimental plan (which completely ignores video). Furthermore, the fallback to a calibrated projection (`BRA_calib`) risks shifting the paper's identity from a decode-time strategy to a fine-tuning strategy. The experimental execution plan is generally excellent, but requires strict tightening of claims and benchmark alignment before it can be considered for acceptance at ACM Multimedia.

## Summary
The paper proposes Bounded Resonance Anchoring (BRA), a decode-time intervention framework for MLLMs. It aims to reduce hallucination by adjusting output logits based on token-local visual evidence extracted from the final-layer visual hidden states. To handle embedding asymmetry, it proposes a dual-track evaluation (`BRA_zero` and `BRA_calib`). To prevent structural collapse during inference, it introduces Vocabulary-Anchored Semantic Masking (VASM), utilizing static dictionaries and BPE continuation inheritance. The paper is currently structured as an extensive hypothesis-driven evaluation protocol, laying out planned experiments across hallucination, reasoning, and local evidence benchmarks.

## WhatShouldBeKept
1. **The Core Framing:** The explicit focus on a positive hypothesis (how to inject token-local evidence without hurting structure) rather than a grand critique of external baselines is scientifically mature. Keep treating VCD/OPERA/DoLa as competitive baselines, not as flawed global-pooling methods.
2. **VASM and BPE Inheritance:** This is perhaps the most practical and mathematically sound contribution of the paper. Explicitly protecting multi-token entities and functional syntax via inference-time BPE inheritance is a critical insight for logits intervention.
3. **The `AGL` Mandate:** Tracking Average Generated Length (AGL) alongside CHAIR/POPE is excellent. It preemptively closes the loophole of methods achieving high hallucination-reduction scores simply by generating truncated, uninformative text.
4. **The Three Evidence Chains:** The logical flow from Hallucination Reduction (Chain A) to Structure Preservation (Chain B) to Local Evidence Value (Chain C) is a textbook example of how to validate an MLLM inference strategy. Keep this exact structure.

## MajorWeaknesses
1. **The Video Disconnect:** Section 3.2 introduces a Spatio-Temporal Extension with a temporal decay penalty ($\lambda$) for video. However, Section 4 (Evaluation Protocol) does not list a single video benchmark. You cannot claim a mathematical formulation for video and then validate it solely on static benchmarks like POPE, FREAK, and DocVQA. 
2. **The `BRA_calib` Identity Crisis:** The author correctly identifies that `BRA_zero` (using the native `lm_head` on visual prefixes) carries a "severe theoretical risk of yielding pure noise" in decoder-only MLLMs. If Defense Line 1 fails (which it almost certainly will for models like LLaVA), the paper pivots to `BRA_calib`. While forcing baselines to be LoRA-tuned on the same 5k COCO dataset is a mathematically fair fallback, it structurally changes the paper from "training-free decode-time intervention" to "lightweight visual-alignment fine-tuning." This fundamentally alters the baseline comparison dynamic.
3. **Model Architecture Ambiguity:** The methodology speaks generically of MLLMs. The behavior of $h_L^{(v_j)}$ projected through $W_{vocab}$ depends entirely on the architecture. Does the MLLM use an MLP projector (LLaVA), cross-attention (Flamingo), or 2D-RoPE (Qwen-VL)? The protocol needs to specify exactly which base models are being tested, as the post-hoc entanglement behaves differently across them.

## SectionBySectionComments
- **Abstract & Intro:** The claims are well-bounded, but the mention of video temporal decay feels bolted on. If your best claim is "token-local logits intervention + VASM + fair zero-shot/calibrated split," own that. Do not artificially inflate the scope to video unless you are prepared to prove it.
- **Section 3.1 (`BRA_zero` vs `calib`):** The transparency regarding post-hoc entanglement is commendable. However, if `BRA_zero` fails, you must ensure that the InfoNCE loss used to train $\Phi_{calib}$ does not inadvertently leak task-specific formatting into the model.
- **Section 3.2 (Temporal Decay):** The equation $S_{raw}(c)$ is mathematically sound but empirically orphaned. Drop it, or add the required benchmarks.
- **Section 3.3 (VASM):** The distinction between SentencePiece (`_`) and byte-level BPE (`Ġ`) is an excellent, grounded engineering detail. Expand on how the static root-token expectation dictionary is constructed (e.g., NLTK POS tagging?).
- **Section 4 (Protocol):** The protocol is incredibly strong. Defense Line 4 (Chain C) comparing `BRA_AdaptiveTopK` vs `BRA_MeanPool` on FREAK/DocVQA is the linchpin of the paper. If this ablation fails, the entire premise of the paper collapses.

## RequiredRevisions
1. **Scoping the Video Claim:** Either entirely remove the video formulation (Section 3.2) and focus strictly on high-resolution static images (which is sufficient for an ACM MM paper), or mandate the inclusion of temporal-specific benchmarks (e.g., MVBench, Video-MME) in Defense Line 4 to prove that $\lambda$ temporal decay actually isolates local evidence. I strongly suggest downgrading video to Future Work and shrinking the claim to static spatial locality.
2. **Base Model Specification:** Explicitly select and list 2-3 specific MLLM families for this protocol (e.g., LLaVA-1.5, Qwen-VL-Chat, InstructBLIP) to prove the framework is not an artifact of a single specific architecture's residual stream.
3. **Clarify VASM Dictionary:** Provide a concrete, reproducible definition in the methodology of exactly how the $O(1)$ static dictionary is generated (what constitutes a "visual noun" vs "functional syntax").

## SuggestedFiguresTablesExperiments
To successfully execute this planned protocol, I expect the final manuscript to deliver the following exact structures:
- **Table 1 (Chain A):** Baselines vs `BRA` on POPE and CHAIR. **Must include an explicit column for AGL.** If AGL drops $>10\%$, the hallucination win is invalid.
- **Table 2 (Chain B):** MMBench, MME, MMMU (Hard). Must include the `BRA_no_VASM` ablation to prove that without BPE inheritance, complex reasoning collapses.
- **Table 3 (Chain C):** FREAK and DocVQA (ANLS). Must show `Base` vs `BRA_MeanPool` vs `BRA_AdaptiveTopK` to prove that *local* evidence is strictly necessary over global aggregation.
- **Figure 1 (Qualitative Heatmap):** Show 2-3 examples of dense OCR or small objects. Display the original image, the generated text, and a heatmap highlighting the exact visual patches selected by the Adaptive Top-$k$ mechanism when decoding the specific grounded noun.
- **Failure Case Analysis:** Execute the proposed "Out-of-Candidate Unrecoverability" limit. Show a clear failure case where the correct visual word was pushed to Rank $> M$ by the base LLM, proving the absolute bounds of this decode-time intervention.

## AcceptanceOutlook
The core proposition is scientifically sound and the evaluation protocol is highly rigorous. If the authors execute the proposed experiments—specifically validating the `BRA_AdaptiveTopK` vs `BRA_MeanPool` ablation and proving VASM's necessity—while reining in the unsupported video claims, this will be a strong, defensible paper. Proceed with the experimental execution strictly adhering to the established evidence chains.