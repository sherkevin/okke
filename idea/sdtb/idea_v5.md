# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the severe logical inconsistencies and cascading errors in prior generation pipelines, CVLA enforces strict structural preservation and deterministic text-to-image semantic alignment. Rather than relying on fragile 3D heuristics or compromising the difficulty of counterfactual interactions, we introduce Depth-Anchored Spatial Scaling and Pose-Conditioned Latent Optimization. This ensures the scalable synthesis of high-fidelity, extreme counterfactual datasets where the visual space strictly obeys the pre-defined deductive logic.

## 3.1 Structure-Preserving Counterfactual Generation

Generating extreme counterfactuals (e.g., substituting a "horse" with a "car" while preserving the "riding" interaction) requires resolving severe topological conflicts without destroying the original scene structure. We achieve this through physically anchored scaling and multi-conditional structural preservation.

### Depth-Anchored Spatial Scaling
Previous methods relying on vanishing points fail catastrophically in natural scenes lacking explicit geometric structures (e.g., forests, macro shots). Instead, we introduce **Depth-Anchored Spatial Scaling**, utilizing dense, scale-invariant relative depth maps extracted via robust monocular estimators (e.g., Depth Anything). 

Given a source object mask $\mathcal{M}_{src}$ and its corresponding median depth value $d_{src}$, we determine the target object's spatial footprint without requiring rigid 3D bounding volumes or camera intrinsics. The generative bounding box $\mathcal{B}_{tgt}$ is scaled based on the relative physical dimensions of the target concept anchored to the source depth plane. To prevent perspective distortion, the aspect ratio of $\mathcal{B}_{tgt}$ is constrained by the target's canonical real-world proportions, while its area $A_{tgt}$ is dynamically adjusted to ensure the relative depth gradient $\nabla \mathcal{D}$ across the boundary between $\mathcal{B}_{tgt}$ and the background remains continuous.

### Pose-Conditioned Boundary Adaptation
Unconstrained mask dilation for handling occlusions in cross-category substitutions leads to unpredictable generative artifacts and often destroys complex physical interactions (e.g., a rider's posture). To enforce structural fidelity, we propose **Pose-Conditioned Boundary Adaptation**.

Instead of allowing the diffusion model unconstrained freedom to hallucinate interactions, we strictly lock the interacting occluders (e.g., the human rider). We extract the dense pose and boundary edges of the interactant, encoding them as a deterministic structural condition $C_{struct}$ using a multi-conditional ControlNet architecture. The generative mask $\mathcal{M}_{gen}$ is strictly defined as the union of $\mathcal{B}_{tgt}$ and the local contact boundary, but the structural integrity of the interactant is frozen by $C_{struct}$. This forces the diffusion model to synthesize the target counterfactual concept (e.g., the car roof) *underneath* the preserved human posture, strictly adhering to the original topological interaction rather than arbitrarily erasing it.

### Latent Cross-Attention Optimization
To minimize generative failures without relying on unreliable external APIs mapping loss gradients to text, we introduce a **Latent Cross-Attention Optimization** mechanism directly within the diffusion denoising process. 

We define a localized alignment loss $\mathcal{L}_{align}$ based on the cross-attention maps of the target text tokens. Let $\mathcal{A}_t^k$ denote the cross-attention map for the target semantic token at denoising step $t$. We optimize the latent code $z_t$ to maximize the attention density strictly within $\mathcal{M}_{gen}$ and minimize it outside:
$$ \mathcal{L}_{align} = 1 - \frac{\sum_{(i,j) \in \mathcal{M}_{gen}} \mathcal{A}_t^k(i,j)}{\sum \mathcal{A}_t^k} + \lambda \cdot MSE(z_t^{bg}, z_{src}^{bg}) $$
The second term enforces background preservation by computing the Mean Squared Error (MSE) between the background latent representations of the generated and source images. By applying gradient updates to $z_t$ during the early denoising steps ($t > 0.8T$), we mathematically enforce semantic binding and structural preservation, significantly improving the yield of challenging counterfactuals without relying on subsequent VLM evaluation.

## 3.2 Deterministic Counterfactual Instruction Construction

To construct a genuinely challenging counterfactual dataset, the ground-truth text must not compromise to accommodate generative failures. If a generated image fails to depict the extreme interaction, altering the text to match the failure destroys the counterfactual intent. CVLA strictly enforces the pre-defined deductive logic.

### Deductive Relational Scene Graph Extraction
We begin by extracting a localized Relational Scene Graph $\mathcal{G}_{src} = (\mathcal{V}_{src}, \mathcal{E}_{src})$ from the source image, capturing objects and their explicit spatial/physical interactions. When an extreme counterfactual substitution is proposed, we deterministically update this graph on the text side to form the **Target Scene Graph** $\mathcal{G}_{tgt}$. For instance, if node $v_{horse}$ is replaced with $v_{car}$, the interaction edge $e_{ride}(v_{person}, v_{horse})$ is strictly translated to $e_{ride}(v_{person}, v_{car})$.

The LLM constructs the deterministic counterfactual queries spanning *Existence*, *Attribute*, and *Relation* based exclusively on $\mathcal{G}_{tgt}$. The ground-truth (GT) answers are derived through strict mathematical deduction from this graph, ensuring absolutely zero hallucination in the instruction generation phase.

### Strict Visual Compliance Verification
To guarantee that the generated image faithfully instantiates the extreme counterfactual defined in $\mathcal{G}_{tgt}$, we employ a hard-filtering **Visual Compliance Verification** protocol. We abandon the notion of "post-generation graph updates" which degrade dataset difficulty.

Instead, we extract the bounding boxes of the target concept and the interactant from the generated image $\mathcal{I}_{cf}$ using a foundational grounding model. We compute the geometric Intersection over Union (IoU) and relative spatial positioning (e.g., $y_{person} < y_{car}$ for "riding"). 
1. **Semantic Compliance:** The target region must yield a higher contrastive alignment score (e.g., CLIP/BLIP feature cosine similarity) with the target concept than with the source concept.
2. **Relational Compliance:** The localized bounding boxes must strictly satisfy the spatial heuristics dictated by the interaction edge $e \in \mathcal{E}_{tgt}$.

If the image fails these explicit mathematical checks—indicating the diffusion model failed to generate the required extreme interaction (e.g., placing the person "next to" the car instead of "riding" it)—the sample is outright rejected. While this strict compliance verification filters out failed generations, the yield rate is fundamentally sustained by the Latent Cross-Attention Optimization in Section 3.1. This uncompromising protocol ensures that the final dataset contains only rigorously aligned, highly challenging counterfactual scenarios that genuinely test multimodal model robustness.