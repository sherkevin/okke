# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To fundamentally resolve the cascading errors, prohibitive rejection rates, and topological paradoxes inherent in deterministic pipeline filtering, CVLA introduces an **Energy-Based Iterative Feedback Generation** paradigm. We replace mathematically flawed hard-coded 3D assumptions with an implicit perspective field adaptation, and alleviate structural contradictions via topology-aware generative relaxation. This enables the scalable synthesis of highly challenging counterfactual datasets with robust self-correction mechanisms.

## 3.1 Topology-Adaptive Visual Variation Generation

To construct extreme counterfactuals (e.g., substituting a "horse" with a "car") without inducing severe structural artifacts or perspective distortions, we propose a physically grounded yet topologically flexible generation process guided by iterative refinement.

### Perspective-Guided 2.5D Layout Adaptation
Directly estimating absolute 3D bounding volumes from monocular depth introduces severe perspective scaling errors without camera intrinsics. Instead, we propose **Perspective-Guided 2.5D Layout Adaptation**. Given a source image $\mathcal{I}$, we extract the relative depth ordinal relations $\mathcal{R}_D$ and estimate an implicit perspective field $\mathcal{P}$ utilizing vanishing point heuristics derived from mid-level image features. 

When a cross-category substitution is proposed, the target region $\mathcal{R}_{tgt}$ is not defined by strict geometric bounding boxes, but is dynamically warped according to the local gradient of the perspective field $\nabla \mathcal{P}$ and depth ordinals $\mathcal{R}_D$. This anchors the generated object into the existing 2.5D coordinate space of the scene, naturally mitigating the "floating object" artifact and resolving perspective distortions without requiring rigid 3D assumptions.

### Topology-Relaxed Generative Inpainting
Applying hard truncation masks to handle occlusion for objects with vastly different topological structures (e.g., saddle vs. car roof) inevitably forces generative models into severe failure modes, yielding distorted textures. To preserve structural integrity and interaction naturalness, we introduce **Topology-Relaxed Inpainting**. 

Instead of an absolute subtraction of occluders, we define the generative mask $\mathcal{M}_{gen}$ via a topological relaxation:
$$ \mathcal{M}_{gen} = \mathcal{M}_{tgt\_base} \cup \mathcal{M}_{transition} $$
where $\mathcal{M}_{tgt\_base}$ is the warped region guided by the perspective field, and $\mathcal{M}_{transition}$ is an adaptively dilated transition zone surrounding the interacting occluders (e.g., the rider's legs). By feeding $\mathcal{M}_{gen}$ into the diffusion model, we grant it the essential degrees of freedom to jointly optimize the target concept and perform *local reconstruction* of the occluder's boundaries. This allows the model to hallucinate physically plausible interactive structures (e.g., naturally shifting the rider's posture to sit on a car roof) rather than forcing pixels into structurally contradictory shapes.

### Energy-Based Feedback Loop for High-Yield Verification
To resolve the unsustainable rejection rates of linear dual-check filtering, we formulate the validation step as an **Energy Minimization Problem** integrated into an active feedback loop. We define the generation energy $\mathcal{E}_{gen}$ as a weighted sum of global semantic preservation and local physical consistency:
$$ \mathcal{E}_{gen} = \lambda_1 \mathcal{L}_{bg\_drift} + \lambda_2 \mathcal{L}_{topo\_conflict} $$
where $\mathcal{L}_{bg\_drift}$ measures the undesired shift in background latent representations, and $\mathcal{L}_{topo\_conflict}$ evaluates the localized semantic-structure mismatch using cross-attention map alignment within the diffusion process itself, completely bypassing the compounding errors of cascaded external API models (e.g., DINOv2 or VQAScore).

Crucially, if $\mathcal{E}_{gen} > \tau$, the sample is *not* blindly discarded. Instead, the specific energy gradients are mapped back into natural language error descriptions (e.g., "The target is blending with the background"). These descriptions serve as **Feedback Prompts** to dynamically adjust the layout constraints and trigger an iterative resampling phase. This closed-loop self-correction drastically improves the generative yield rate and ensures visual fidelity.

## 3.2 Dynamic Counterfactual Instruction Construction

Static text-side ground-truth deduction creates an unbridgeable semantic gap when the generative model inevitably introduces slight morphological adaptations during the topology-relaxed inpainting. We bridge this gap through dynamic, visually-conditioned provenance updating.

### Visually-Conditioned Relational Scene Graph
Instead of deriving ground-truth responses solely from the text-side modification log, we construct a **Post-Generation Relational Scene Graph** $\mathcal{G}_{post} = (\mathcal{V}_{post}, \mathcal{E}_{post})$. We extract the latent visual representations of $\mathcal{I}_{cf}$ and map them against the original text-side edits. 

If the topology-relaxed generation process slightly altered the physical relationship to ensure visual realism (e.g., changing the interaction from "riding a horse" to "standing next to a car" because the model failed to plausibly place the rider on the roof but generated a high-quality adjacent car), this structural shift is immediately captured and updated in $\mathcal{G}_{post}$. This ensures the logical foundation for the queries perfectly mirrors the actual pixel-space reality, fully eliminating the "blind leading the blind" text-side hallucination.

### Iterative Graph-to-Instruction Synthesis
Leveraging the dynamically updated $\mathcal{G}_{post}$, the LLM constructs highly targeted counterfactual queries spanning *Existence*, *Attribute*, and *Relation*. The ground-truth (GT) answers are mathematically derived directly from the visually-validated nodes and edges of $\mathcal{G}_{post}$.

By coupling the Energy-Based visual self-correction with the Visually-Conditioned graph updates, CVLA operates as a genuine closed-loop system. The physical generative constraints inform the text-side logic, and the logical requirements guide the visual resampling. This holistic algorithmic alignment isolates and eliminates cascading multi-model errors, resulting in a challenging, zero-hallucination dataset without relying on arbitrary hard-coded rules or yielding catastrophic sample discard rates.