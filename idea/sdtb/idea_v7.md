# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To fundamentally eliminate the cascading errors of multi-model pipelines and the mathematical ill-posedness of single-view 3D projections, CVLA operates entirely within the unified continuous latent space of a pre-trained diffusion model. By replacing hard-coded geometric assumptions and rigid skeleton templates with implicit structural manifold preservation and open-vocabulary generative reasoning, CVLA synthesizes highly harmonized, topologically consistent counterfactual datasets without relying on brittle deterministic heuristics.

## 3.1 Unified Latent Interaction Synthesis

Generating extreme counterfactual substitutions (e.g., swapping a "horse" for a "car" under an active interaction) fundamentally requires topological flexibility and lighting consistency, which strict spatial constraints and multi-step pose extraction models fail to provide. We achieve this via implicit semantic-driven latent modulation, inherently solving the error cascade and perspective distortion problems.

### Implicit Structural Manifold Preservation
Explicitly estimating absolute scale and local ground planes from monocular images is an ill-posed inverse problem that inevitably leads to severe perspective distortions. Instead of relying on flawed metric depth estimators and pseudo-camera intrinsics, we introduce **Implicit Structural Manifold Preservation**. 

We leverage the inherent physical and perspective priors embedded within the self-attention layers of a frozen foundational diffusion model. Given a source image inversion, we extract the dense self-attention maps $\mathcal{S}_{src}$, which implicitly encode the continuous spatial layout and relative object scales without explicit 3D instantiation. When substituting the target object, we inject the topological footprint of $\mathcal{S}_{src}$ into the target generation process. The target concept's spatial extent is constrained softly by aligning the unnormalized attention density of the target tokens with the original object's attention manifold. This allows the target object to naturally inherit the correct perspective scaling and depth ordinal relations directly from the latent structure of the scene, entirely bypassing the fragile geometry projection pipeline.

### Open-Vocabulary Topological Blending
Hard-coding pose transitions via external pose estimation models (e.g., 2D skeletons) destroys the open-vocabulary generation capability and forces the system into an unscalable paradigm of rigid heuristics. To enable unconstrained physical co-adaptation, we propose **Open-Vocabulary Topological Blending**.

Rather than explicitly freezing or modifying a skeleton, we rely on the compositional reasoning capabilities of the diffusion model. We construct a unified compositional prompt describing the desired state (e.g., "A person sitting on the roof of a car"). During the denoising process, we softly anchor the subject's identity latents while allowing their structural latents to be entirely driven by the cross-attention gradients of the target interaction tokens. By avoiding strict bounding box or skeleton constraints, the diffusion model is granted the generative freedom to autonomously hallucinate the kinematically correct human-object interaction (HOI) boundaries—such as naturally wrapping legs or shifting weight—guided solely by the pre-trained text-to-image semantic priors. 

### Frequency-Aware Latent Harmonization
Confining target generation to strict spatial masks causes severe boundary artifacts, lighting inconsistencies, and a "pasted" appearance. To guarantee photorealistic harmonization, we replace region-restricted attention control with **Frequency-Aware Latent Harmonization**.

At each denoising step $t$, we decouple the intermediate latent representation $z_t$ into low-frequency $\mathcal{F}_{low}(z_t)$ and high-frequency $\mathcal{F}_{high}(z_t)$ components using a spatial Fourier transform. The low-frequency components, which dictate broad structural layout and object identity, are driven by the target text prompt. Concurrently, the high-frequency components—which encode localized lighting, shadows, and edge transitions—are smoothly interpolated with the corresponding high-frequency features of the source image background. This dual-frequency modulation mathematically guarantees that the newly hallucinated object naturally absorbs the ambient lighting and perspective shadows of the original scene, ensuring seamless boundary harmonization without the need for discrete spatial masking.

## 3.2 Open-World Counterfactual Alignment

To construct QA pairs that accurately reflect the generated counterfactuals without falling into the "Verification Paradox" caused by closed-set detection models, CVLA adopts an open-world semantic entailment strategy.

### Generative Relational Graph Synthesis
Instead of relying on rigid, template-based graph substitutions that fail to capture the nuanced realities of open-vocabulary generation, we utilize a Large Language Model (LLM) as an unconstrained physical reasoning engine. 

Given the source interaction (e.g., $e_{ride}(v_{person}, v_{horse})$) and the target substitution ($v_{car}$), the LLM is prompted to predict the most physically plausible resulting state based on real-world affordances (e.g., $e_{sit\_on}(v_{person}, v_{car\_roof})$ or $e_{stand\_next\_to}(v_{person}, v_{car})$). This generates a dynamic Target Scene Graph $\mathcal{G}_{tgt}$ that naturally accommodates structural shifts rather than forcing an artificial 1:1 mapping. The LLM then derives *Existence*, *Attribute*, and *Relation* instructions directly from this dynamically reasoned graph, ensuring the logical deductions are grounded in natural physical affordances.

### Zero-Shot Visual Entailment Verification
Traditional HOI detectors are trained on highly biased, standard interaction datasets (e.g., HICO-DET) and systematically fail to recognize out-of-distribution counterfactual interactions, creating a verification paradox. We discard task-specific detectors and introduce **Zero-Shot Visual Entailment Verification**.

We leverage a generalized, open-world Vision-Language Model (VLM) (e.g., LLaVA-1.5) acting as a visual entailment scorer. For each generated image $\mathcal{I}_{cf}$ and its corresponding predicted interaction state from $\mathcal{G}_{tgt}$, we formulate the verification as a Visual Question Answering (VQA) entailment task. The VLM is queried to verify the presence of the specific topological interaction (e.g., "Is the person physically supporting their weight on the roof of the vehicle?"). 
1. **Semantic Entailment:** We compute the localized SigLIP similarity between the interaction region and the target conceptual prompt to ensure identity preservation.
2. **Interaction Entailment:** The image is retained only if the VLM outputs a high-confidence affirmative response to the specific relational query.

By treating verification as a generalized semantic reasoning task rather than a closed-set bounding box detection problem, we reliably filter out generative failures (e.g., disconnected objects) while successfully preserving high-quality, extreme counterfactual scenes, maintaining high dataset yield and uncompromised multimodal difficulty.