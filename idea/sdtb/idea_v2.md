# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework, the core engine behind **ViHallu**. To fundamentally address the cascading errors and logical paradoxes prevalent in naive pipeline concatenations, CVLA shifts the paradigm from stochastic hallucination-mining to deterministic, provenance-driven counterfactual generation. The framework consists of two highly coupled phases: 1) Morphologically-Aware Visual Variation Generation, which ensures physical and semantic compatibility to prevent structural artifacts; and 2) Deterministic Counterfactual Instruction Construction, which synthesizes zero-hallucination ground-truth responses directly from edit provenance rather than relying on error-prone multimodal inference.

## 3.1 Morphologically-Aware Visual Variation Generation

The visual variation phase aims to construct high-fidelity counterfactual image pairs while strictly mitigating structural distortions caused by semantic-mask contradictions. This is achieved through morphology-constrained concept substitution and adaptive mask relaxation.

### Morphological Constraint Validation and Caption Editing
To initialize the editing process, we extract foundational semantics using Tag2Text [10] and precise object masks via MobileSAM [42]. Unlike conventional pipelines that perform unconstrained concept substitution, we introduce a **Morphological Compatibility Constraint (MCC)** module. 

When replacing objects or backgrounds, an LLM (DeepSeek-chat V2) acts as a structural validator. Instead of forcibly mapping an incompatible semantic concept to a rigid mask (e.g., mapping a "car" onto a "zebra" mask), the MCC restricts substitutions to objects belonging to the same broad morphological family (e.g., quadrupeds to quadrupeds, dense foliage to rock formations). We design specialized prompt templates equipped with structural rule checklists (detailed in the Appendix) to guide the LLM. The LLM outputs both the edited caption and a structured **Modification Log**, recording the exact source-to-target semantic mapping.

### Adaptive Mask-Relaxation for Controllable Generation
Forcing a text-to-image model to strictly adhere to an original pixel-perfect mask during concept substitution inherently leads to visual artifacts. To resolve this physical-visual paradox, we introduce a **Semantic-Edge Mask Relaxation** strategy before feeding the masks into the controllable T2I model (ControlNet++ [27]).

Rather than passing the hard binary mask from MobileSAM, we apply morphological dilation to the object boundaries based on the target concept's complexity. This relaxation provides ControlNet++ with localized spatial degrees of freedom, allowing the model to naturally synthesize the target concept's distinct contours (e.g., fur textures or variable geometric edges) while maintaining the original macro-spatial layout.

### Cross-modal Filtering against Error Cascades
To further prevent the propagation of generation artifacts into the instruction tuning phase, we abandon arbitrary static thresholding. Instead, we compute a **Relative Cross-Modal Decay Score**. We measure the CLIP-based image-text alignment for both the original image-caption pair ($S_{orig}$) and the generated counterfactual pair ($S_{cf}$). Variations are only retained if $S_{cf} \geq \alpha \cdot S_{orig}$ (where $\alpha = 0.85$ based on empirical tuning), ensuring the structural integrity of the generated variation is comparable to the real-world baseline.

## 3.2 Deterministic Counterfactual Instruction Construction

Existing methods often depend on the unpredictable hallucinations of large vision-language models (LVLMs) to construct negative samples, which is inefficient and uncontrollable. Furthermore, evaluating counterfactual images with standard LVLMs introduces systematic biases. We bypass these limitations by utilizing a deterministic, provenance-driven logic engine.

### Graph-based Deterministic Counterfactual Query Generation
Rather than hoping for a generic LVLM to spontaneously hallucinate, we construct targeted trap questions systematically. We first extract a formal Scene Graph based on the original image’s ground-truth tags. By cross-referencing this Scene Graph with the **Modification Log** generated in Section 3.1, the LLM systematically formulates deterministic counterfactual queries. 

These queries deliberately probe three hallucination vulnerability dimensions: *Existence* (asking about the original object that was removed), *Attribute* (querying the altered state of the new concept), and *Relation* (testing physical interactions in the counterfactual scene). This graph-based strategy ensures that the trap questions are comprehensively and specifically targeted at the injected visual modifications.

### Provenance-driven Ground-Truth Synthesis
A critical paradox in previous datasets is relying on strong LVLMs to generate ground-truth answers for complex counterfactual images, which often induces secondary hallucinations. We fundamentally eliminate this "blind leading the blind" problem via **Provenance-driven Answer Synthesis**.

The ground-truth answers in ViHallu are strictly synthesized on the text side. The LLM engine takes the *Original Scene Graph*, the *Modification Log*, and the *Target Trap Question* as inputs. By applying deductive logic rules, the LLM generates a 100% accurate, zero-hallucination ground-truth response without ever "looking" at the potentially challenging generated image. This ensures the text-side ground-truth is mathematically aligned with the injected visual edits.

### Fine-grained Feature Alignment Verification
To replace the flawed majority-voting mechanism of LVLMs for final quality assessment, we implement an objective, feature-level verification. We utilize Grounded-SAM [13,23,30] guided by the target concepts in the Modification Log to extract the local bounding box of the altered region in the generated image. We then compute the fine-grained visual-semantic alignment score between this localized visual crop and the text descriptor of the target concept. QA pairs are finalized only if this localized deterministic alignment confirms the successful visual instantiation of the text-side logical edits, completely severing the cycle of LVLM self-validation bias.