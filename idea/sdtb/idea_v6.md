# 3 Methodology

This section presents the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the fundamental geometric paradoxes and interaction failures inherent in cross-category object substitution, CVLA departs from fragile 2D heuristics and rigid pose freezing. Instead, we formulate a pipeline grounded in metric depth estimation and kinematically plausible co-adaptation. This enables the synthesis of highly challenging counterfactual datasets where both the spatial geometry and the physical interactions remain robustly aligned with the intended multimodal logic.

## 3.1 Kinematic-Aware Counterfactual Generation

Generating extreme counterfactuals (e.g., substituting a "horse" with a "car" while adapting the "riding" interaction) requires simultaneously resolving scene-level perspective scaling and object-level topological conflicts. We achieve this through metric-anchored scaling and kinematic pose co-adaptation, ensuring physical plausibility without compounding inference overhead.

### Metric-Depth Guided Spatial Alignment
Relying on relative depth estimators for bounding box scaling introduces unresolvable scale ambiguities and perspective distortion, as relative depth lacks a global metric anchor. To establish a physically grounded spatial constraint, we introduce **Metric-Depth Guided Spatial Alignment**. 

We utilize a pre-trained metric depth estimator (e.g., ZoeDepth) to extract an absolute depth map $\mathcal{D}_{metric}$ from the source image $\mathcal{I}_{src}$. Utilizing the estimated camera intrinsics derived from the image field-of-view, we project the local region into a 3D coordinate space and explicitly estimate the local ground plane. The generative bounding volume $\mathcal{B}_{tgt}$ for the target concept is determined by projecting its canonical real-world metric dimensions onto this ground plane at the source object's metric depth $d_{src}$. This mathematically resolves the perspective ambiguity, ensuring the target object (e.g., a car) is correctly scaled relative to the scene context without relying on flawed 2D bounding box scaling.

### Kinematic Pose Co-Adaptation
Directly freezing the structural pose of an interactant (e.g., preserving a horse-rider's straddling posture) when substituting objects with vastly different topologies (e.g., a car roof) inevitably forces the generative model into severe topological failures, such as floating bodies or severe clipping. To maintain physical realism, we introduce **Kinematic Pose Co-Adaptation**.

Rather than strictly locking the interactant's posture, we extract the source 2D skeleton pose $\mathcal{P}_{src}$ using a pose estimation model (e.g., DWPose). We define a set of kinematically plausible pose transitions for cross-category interactions (e.g., transferring from "straddling" a horse to "sitting with legs forward" on a car roof). By conditioning a multi-modal generation framework (e.g., a dual-conditioned ControlNet) on both the metric-aligned target mask $\mathcal{M}_{tgt}$ and the adapted target skeleton $\mathcal{P}_{tgt}$, we explicitly guide the diffusion model to co-generate the target object and the modified human pose. This grants the model the necessary degrees of freedom to render physically valid interaction boundaries, fundamentally resolving the rigid mask-conflict problem.

### Efficient Region-Restricted Attention Control
To ensure the target semantics strictly bind to the designated spatial layout without the prohibitive latency of per-timestep latent gradient optimization, we adopt an efficient **Region-Restricted Attention Control** mechanism. 

During the cross-attention computation of the diffusion denoising process, we explicitly restrict the attention maps of the target concept tokens to the union of the generated object mask and the adapted interactant region. Specifically, the cross-attention scores for the target tokens are masked by the spatially expanded footprint $\mathcal{M}_{gen}$, forcing the model to synthesize the new object features strictly within the kinematically adapted region while the background latents remain guided by the unedited source image conditioning. This ensures high structural fidelity and semantic binding with minimal inference overhead.

## 3.2 Robust Counterfactual Instruction Construction

Constructing a genuinely challenging counterfactual dataset requires that the text-side logic precisely reflects the synthesized interaction without falling into the extremes of LLM hallucination or fragile heuristic filtering. CVLA enforces a structured relational translation followed by explicit interaction verification.

### Constrained Relational Graph Substitution
To construct the counterfactual QA pairs, we first extract a Relational Scene Graph $\mathcal{G}_{src} = (\mathcal{V}_{src}, \mathcal{E}_{src})$ from the source image. When a counterfactual substitution is proposed (e.g., replacing $v_{horse}$ with $v_{car}$), we employ a template-constrained substitution mechanism. 

Instead of relying on free-form LLM generation, which introduces uncontrollable variance and potential hallucinations in complex scenarios, we strictly map the interaction edge $e_{source}$ to a valid target edge $e_{tgt}$ based on the kinematic pose adaptation defined in Section 3.1. The updated Target Scene Graph $\mathcal{G}_{tgt}$ serves as a rigid schema. The LLM is then prompted with a constrained decoding strategy (e.g., JSON schema adherence) to generate QA pairs spanning *Existence*, *Attribute*, and *Relation*, ensuring the ground-truth text strictly adheres to the logically translated structural nodes without extraneous narrative drift.

### Explicit Human-Object Interaction (HOI) Verification
To verify that the generated image faithfully represents the extreme counterfactual defined in $\mathcal{G}_{tgt}$, relying on 2D spatial heuristics (e.g., relative bounding box coordinates) is fundamentally insufficient for capturing complex 3D relationships. We therefore introduce an **Explicit HOI Verification** protocol.

Instead of naive geometric checks, we process the generated image $\mathcal{I}_{cf}$ through a specialized Human-Object Interaction (HOI) detection model or a vision-language relation scorer configured to detect the explicit triplet $<Interactant, Relation, Target\_Object>$. 
1. **Semantic Verification:** We compute the localized contrastive alignment score to ensure the target region semantically aligns with the new object.
2. **Relational Verification:** The interaction model must explicitly detect the updated physical relationship $e_{tgt}$ defined in $\mathcal{G}_{tgt}$ (e.g., detecting the state of "sitting on" rather than merely "standing next to").

Images that fail to yield high-confidence HOI detections for the specified counterfactual relationship are rejected. By replacing fragile bounding box heuristics with dedicated interaction-level visual verification, we ensure the pipeline exclusively retains samples where the generative model successfully executed the complex, kinematically adapted counterfactual logic, thereby establishing a high-quality, rigorously aligned multimodal benchmark.