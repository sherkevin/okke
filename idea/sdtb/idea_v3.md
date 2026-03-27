# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the inherent vulnerability of cascading errors in multimodal data synthesis, CVLA abandons the linear, open-loop pipeline in favor of a **Closed-Loop Generation and Verification** paradigm. We introduce Depth-Aware Layout Adaptation to resolve structural contradictions during complex semantic substitutions, and substitute naive visual-linguistic filtering with dense compositional validation to guarantee the physical and semantic fidelity of the counterfactual instructions.

## 3.1 Depth-Aware Visual Variation Generation

To construct challenging counterfactuals without introducing generative artifacts, we must reconcile the morphological differences between cross-category objects (e.g., replacing a "zebra" with a "car") while strictly preserving physical occlusion and scene lighting.

### Depth-Guided Layout Adaptation
Instead of artificially restricting semantic substitutions to morphologically similar categories, we implement a **Depth-Guided Layout Adaptation (DGLA)** module. Given a source image $\mathcal{I}$, we extract the foundational semantics, instance masks $\mathcal{M}_{src}$, and a monocular depth map $\mathcal{D}$. 

When the LLM proposes a challenging cross-category substitution, we recalculate the valid spatial footprint for the target concept. We define the target bounding volume by aligning the expected real-world aspect ratio of the target concept with the depth plane of the original object. The valid placement area is constrained by the relative depth gradients $\nabla \mathcal{D}$, ensuring that foreground objects ($d_{fg} < d_{src}$) naturally occlude the newly generated concept.

### Occlusion-Preserving Mask Re-projection
To prevent structural artifacts and blending boundary errors during generation, we replace heuristic mask dilation with **Occlusion-Preserving Mask Re-projection**. The generative mask $\mathcal{M}_{gen}$ provided to the controllable text-to-image model (ControlNet++) is mathematically defined as:
$$ \mathcal{M}_{gen} = (\mathcal{B}_{tgt} \cap \mathcal{M}_{plane}) \setminus \mathcal{M}_{occluders} $$
where $\mathcal{B}_{tgt}$ is the projected bounding box of the target concept based on its spatial footprint, $\mathcal{M}_{plane}$ represents the background depth plane, and $\mathcal{M}_{occluders}$ denotes the union of masks for all objects with a depth value strictly smaller than the target placement depth. This formulation grants the generative model the exact spatial degrees of freedom required to synthesize novel structures (e.g., wheels of a car) without bleeding into or overwriting overlapping foreground entities.

### Dense Compositional Filtering
To effectively intercept generative artifacts and compositional failures, we replace the global, spatially-agnostic CLIP score with a **Dense Compositional Alignment** mechanism. Specifically, we enforce a dual-check validation:
1.  **Background Preservation:** We compute the DINOv2 feature similarity $S_{DINO}$ between the unedited regions of $\mathcal{I}_{orig}$ and $\mathcal{I}_{cf}$ to ensure global structural integrity and lighting consistency are untouched.
2.  **Fine-Grained Semantic Binding:** We employ a localized Vision-Language scoring metric (e.g., VQAScore applied strictly to the cropped region of $\mathcal{M}_{gen}$) to verify the specific attributes of the new concept. 
A counterfactual image is only retained if $S_{DINO} > \tau_{bg}$ and the localized alignment score surpasses the source object's baseline score, ensuring artifact-free generation without relying on flawed static thresholding.

## 3.2 Closed-Loop Counterfactual Instruction Construction

Constructing ground-truth responses solely based on text-side deduction introduces hidden hallucinations if the generative model fails to perfectly instantiate the logic in pixel space. We resolve this by anchoring instruction generation in edit provenance, followed by rigorous pixel-level relational verification.

### Relational Scene Graph and Query Generation
We first extract a localized Relational Scene Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $\mathcal{V}$ are objects and edges $\mathcal{E}$ explicitly encode spatial and physical interactions (e.g., *supporting*, *holding*, *occluding*). Utilizing the edit provenance log from Section 3.1, the LLM systematically formulates deterministic counterfactual queries targeting three dimensions: *Existence* (the absence of the source object), *Attribute* (properties of the target object), and crucially, *Relation*. 

### Provenance-Anchored GT Synthesis and Visual Validation
To avoid the "blind leading the blind" paradox of using LVLMs for ground-truth synthesis, we propose a joint **Logical-Visual Verification** step. 

First, the LLM engine deductively generates a candidate Ground-Truth (GT) response strictly based on the altered Scene Graph $\mathcal{G}'$. Second, instead of blindly accepting this text-side GT, we introduce a **Relational Consistency Check** in the visual domain. We utilize a dense grounding model (e.g., Grounding DINO) combined with spatial heuristic checks to verify if the physical relationships proposed in the text genuinely exist in $\mathcal{I}_{cf}$. For instance, if the logic dictates a "person riding a car" (substituted from a horse), the visual verifier checks for bounding box intersection and correct relative depth positioning between the "person" and the "car" in $\mathcal{I}_{cf}$.

Only when the visual evidence strongly corroborates the logical deduction is the QA pair finalized. If structural conflicts or relational violations (e.g., floating objects or clipping) are detected by the visual verifier, the sample is immediately discarded. This closed-loop mechanism completely severs the cycle of LVLM self-validation bias and guarantees that the text-side logic is faithfully reflected in the pixel space.