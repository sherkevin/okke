# 3 Methodology

This section details the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the critical flaws of arbitrary latent gradient interventions, geometric distortions, and VLM confirmation biases identified in previous pipelines, CVLA introduces a training-free, forward-pass diffusion intervention mechanism grounded strictly in attention feature routing and explicit kinematic verification. This ensures semantic flexibility and physical realism without violating the natural continuous manifold of the diffusion process or introducing prohibitive computational overhead.

## 3.1 Intrinsic Latent Interaction Synthesis

Substituting objects in active human-object interactions inherently requires structural adaptation without corrupting the diffusion latent trajectory. We achieve this by manipulating the attention mechanisms directly during the forward pass, entirely avoiding the destructive effects of latent gradient updates or noisy structural heuristics.

### Semantic-Prior Layout Adaptation
Arbitrarily expanding low-resolution attention maps ignores the vast aspect ratio differences between source and target objects (e.g., replacing a basketball with a ladder). Instead of heuristic dilation, we introduce **Semantic-Prior Layout Adaptation**, which leverages the natural dimensional priors embedded in the CLIP text encoder without requiring explicit 3D instantiation.

For a target concept, we extract its canonical aspect ratio $\gamma_{tgt}$ by querying a lightweight pre-trained shape-prior dictionary (or deriving it from average CLIP spatial feature dimensions). Given the source object's cross-attention map centroid $(c_x, c_y)$ and its relative scale $S_{src}$ in the source image, we construct a 2D Gaussian spatial prior $\mathcal{G}_{prior}$ centered at $(c_x, c_y)$ with covariance matching $\gamma_{tgt}$ and scaled proportionally to maintain the relative scene depth volume. 

During the early denoising steps ($t > 0.7T$), rather than updating $z_t$ with disruptive gradients, we modulate the unnormalized cross-attention scores $Q \cdot K^T$ of the target token by adding a logarithmic bias:
$$ \tilde{A}_{tgt}^{l,t} = \text{Softmax}\left( \frac{Q_l K_l^T}{\sqrt{d}} + \alpha \cdot \log(\mathcal{G}_{prior}) \right) $$
where $\alpha$ is a decay factor. This elegantly guides the target generation to adopt its correct canonical shape within a plausible spatial constraint purely via the forward-pass attention distribution, preventing both shape distortion and target bleeding without violating the latent manifold.

### Manifold-Preserving Attention Routing
Enforcing kinematic contact by calculating gradients on low-resolution attention maps introduces severe boundary artifacts and blending failures. Instead, we propose **Manifold-Preserving Attention Routing**, utilizing the diffusion model's inherent ability to synthesize structural continuity via its self-attention layers.

Physical interactions are manifested in the self-attention maps where the interactant and object features heavily attend to one another at boundary regions. In the intermediate denoising steps ($0.4T < t < 0.7T$), we intervene directly in the UNet's self-attention matrix $A_{self} \in \mathbb{R}^{N \times N}$. We identify the spatial tokens corresponding to the human interactant $\Omega_{human}$ and the target object $\Omega_{tgt}$ via their cross-attention activations. We introduce a directed routing bias $\beta$ to the self-attention scores between these two sets of tokens:
$$ A_{self}(i, j) \leftarrow A_{self}(i, j) + \beta, \quad \forall i \in \Omega_{human}, j \in \Omega_{tgt} $$
By synthetically increasing the query-key affinity between the interactant and the target object, we force the diffusion model to share structural features (values) across their boundaries. This allows the model's pre-trained natural image prior to organically render plausible physical contact (e.g., proper occlusion and shadow casting) without explicitly enforcing rigid gradients or merging pixel clusters.

### Zero-Overhead Feature Injection
Applying spatial frequency filtering on noisy latents destroys the diffusion noise schedule and produces severe ghosting artifacts, while recalculating latents doubles inference time. We solve background harmonization with **Zero-Overhead Feature Injection**, executed exclusively in the late, noise-free denoising stages ($t \le 0.3T$).

Ambient lighting and high-frequency textures are natively encoded in the self-attention Keys ($K$) and Values ($V$) of the diffusion model. During the DDIM inversion of the source image, we cache the self-attention $K_{src}^t$ and $V_{src}^t$ for $t \le 0.3T$. During target generation, we seamlessly replace the self-attention features of the unedited background regions with the cached source features:
$$ K_{gen}^t = \mathcal{M}_{bg} \odot K_{src}^t + (1 - \mathcal{M}_{bg}) \odot K_{gen}^t $$
$$ V_{gen}^t = \mathcal{M}_{bg} \odot V_{src}^t + (1 - \mathcal{M}_{bg}) \odot V_{gen}^t $$
where $\mathcal{M}_{bg}$ is the binarized background mask. Because this operation simply swaps internal representations during the standard UNet forward pass, it incurs strictly zero extra computational delay. Furthermore, since it operates directly on attention features rather than modifying the latent noise $z_t$, it perfectly preserves the structural integrity and seamlessly transfers background lighting and texture without generating high-frequency ghosting.

## 3.2 Physics-Grounded Counterfactual Alignment

To overcome the "Verification Paradox" caused by homogeneous MLLM generation and evaluation, CVLA strictly separates semantic reasoning from physical evaluation, utilizing deterministic kinematic models to replace biased visual language validators.

### Geometry-Conditioned Reasoning
Relying solely on LLMs for interaction deduction ignores the spatial reality of the scene. To prevent physical hallucinations, we constrain the Large Language Model using explicit geometric representations.

Before querying the LLM, we utilize a pre-trained monocular depth estimator to extract a coarse surface normal map $\mathcal{N}_{src}$ from the source image. We extract basic environmental affordances from $\mathcal{N}_{src}$ (e.g., "flat ground", "slanted surface", "narrow space") and feed these as hard textual constraints into the LLM prompt alongside the source graph $\mathcal{G}_{src}$. By conditioning the deduction on explicit physical affordances (e.g., an LLM is prevented from predicting "standing next to" if the object is situated in a narrow mid-air spatial constraint), the dynamic Target Scene Graph $\mathcal{G}_{tgt}$ becomes physically plausible and contextually anchored, heavily mitigating text-driven spatial hallucination.

### Kinematic Mesh-Based Verification
Using VLM visual question answering to verify structural modifications introduces severe confirmation bias. We discard VQA entirely for interaction verification and introduce an objective **Kinematic Mesh-Based Verification** protocol.

To explicitly verify the counterfactual physical connection without homogeneous bias, we deploy an off-the-shelf monocular 3D human mesh recovery model (e.g., SMPL-X) and a zero-shot metric depth estimator on the generated image $\mathcal{I}_{cf}$. 
1. **Mesh Penetration Penalty:** We compute the 3D collision volume between the generated human mesh vertices and the explicitly segmented target object's estimated depth point cloud. Severe mesh penetration (indicating fusion artifacts) or excessive distance (indicating floating) yields a high physical penalty.
2. **Support Surface Validation:** For relational edges implying weight-bearing (e.g., "sitting on"), we calculate the alignment between the human's center of mass projection and the target object's upward-facing surface normals.

Samples whose kinematic penalty exceeds a strict physical threshold are filtered out. By relying on deterministic 3D geometry and mesh heuristics rather than VLM subjective scoring, CVLA establishes a rigorous, scientifically sound verification pipeline that objectively guarantees the topological realism of the synthesized counterfactual dataset.