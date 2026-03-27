# 3 Methodology

This section details the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the critical flaws of arbitrary latent constraints, global feature pollution, and computationally prohibitive multi-step 3D verifications, CVLA introduces a perspective-aware, localized diffusion modulation scheme coupled with lightweight 2.5D ordinal verification. This ensures semantic flexibility, global illumination consistency, and topological realism without violating perspective geometry or relying on failure-prone monocular 3D meshes.

## 3.1 Intrinsic Latent Interaction Synthesis

Substituting objects in active human-object interactions requires adaptive structural expansion and illumination-consistent blending. We achieve this by locally modulating attention mechanisms along perspective vanishing lines, preserving global illumination while avoiding the destructive effects of rigid geometric priors.

### Perspective-Aware Latent Anchoring
Relying on canonical 2D aspect ratios or scaling based on source objects fundamentally ignores 3D camera pose and relative object volumes. To allow the target object to naturally adopt its correct morphological shape across varied viewpoints (e.g., side-view vs. top-view), we introduce **Perspective-Aware Latent Anchoring**.

Instead of enforcing hard 2D Gaussian bounds, we extract a relative depth map $\mathcal{D}_{src}$ and perspective vanishing lines $\mathcal{V}_{src}$ from the source image. We define a continuous spatial likelihood field $\mathcal{P}_{spatial}$ where the expansion probability follows the depth gradient: objects are encouraged to expand along the horizontal plane defined by $\mathcal{V}_{src}$ and scaled down proportionally as $\mathcal{D}_{src}$ recedes. 

During the early denoising steps ($t > 0.7T$), we apply a soft energy penalty to the target object's cross-attention map $\mathcal{A}_{tgt}^t$, scaled dynamically by the diffusion timestep to prevent "locking" the latent space:
$$ \mathcal{L}_{anchor} = \omega_t \sum_{i,j} \mathcal{A}_{tgt}^t(i,j) \cdot (1 - \mathcal{P}_{spatial}(i,j))^2 $$
Updating $z_t$ via $\nabla_{z_t} \mathcal{L}_{anchor}$ softly anchors the generation within a perspective-consistent volume. Because $\mathcal{P}_{spatial}$ respects depth gradients rather than absolute metrics or source object scales, substituting a "tennis ball" with a "car" allows the car to naturally expand into the 3D scene background along valid vanishing lines without being compressed into a toy-sized 2D bounding box.

### Frontier-Localized Attention Routing
Applying a global routing bias across all human and object tokens inevitably forces biologically implausible cross-attention (e.g., faces attending to car tires), resulting in severe texture bleeding and structural collapse. To render plausible physical contact, we propose **Frontier-Localized Attention Routing**.

Physical contact mathematically manifests as localized boundary intersections. At intermediate denoising steps ($0.4T < t < 0.7T$), we compute the cross-attention spatial activations for the human ($\mathcal{A}_{human}$) and the target object ($\mathcal{A}_{tgt}$). We define the *Interaction Frontier* $\mathcal{F}$ as the localized intersection of their morphologically dilated boundaries:
$$ \mathcal{F} = \text{Dilate}(\mathcal{A}_{human}, k) \cap \text{Dilate}(\mathcal{A}_{tgt}, k) $$
We then inject a directed self-attention bias $\beta$ strictly limited to the tokens residing within the frontier $\mathcal{F}$:
$$ A_{self}(i, j) \leftarrow A_{self}(i, j) + \beta, \quad \forall i, j \in \mathcal{F} $$
By confining the query-key affinity enhancement exclusively to the spatial contact boundaries (e.g., hands and steering wheels), we allow the diffusion model to seamlessly share structural features precisely where occlusion and contact occur, entirely avoiding global texture pollution and feature homogenization.

### Illumination-Aware Latent Blending
Overwriting background attention features using binary masks destroys global illumination, as it blindly erases the ambient occlusion and shadows cast by the newly generated object, resulting in a "floating sticker" artifact. We introduce **Illumination-Aware Latent Blending** to preserve both original textures and newly synthesized lighting.

Instead of directly swapping Keys ($K$) and Values ($V$), we decouple structure from illumination. The self-attention Keys primarily encode structural layout, while Values encapsulate lighting and texture details. At late denoising stages ($t \le 0.3T$), we compute a soft shadow mask $\mathcal{M}_{shadow}$ by thresholding the luminance difference between the intermediate decoded latents of the edited and source images. 
We then construct a hybrid blending mask $\mathcal{M}_{blend} = \mathcal{M}_{bg} \cdot (1 - \mathcal{M}_{shadow})$. We replace the Key features strictly in the background, but interpolate the Value features smoothly using the shadow-aware mask:
$$ K_{gen}^t = \mathcal{M}_{bg} \odot K_{src}^t + (1 - \mathcal{M}_{bg}) \odot K_{gen}^t $$
$$ V_{gen}^t = \mathcal{M}_{blend} \odot V_{src}^t + (1 - \mathcal{M}_{blend}) \odot V_{gen}^t $$
This specific decoupling ensures that the high-frequency structural background is accurately restored, while the newly generated shadows and environment reflections cast by the target object are strictly protected, maintaining global illumination consistency.

## 3.2 Physics-Grounded Counterfactual Alignment

To ensure physical validity without suffering from the lossy compression of text-only geometry constraints or the prohibitive noise of monocular 3D mesh reconstruction, CVLA employs multimodal spatial affordance reasoning and robust 2.5D topological verification.

### Multimodal Affordance Grounding
Summarizing complex 3D geometry into discrete text tokens (e.g., "flat ground") causes catastrophic information loss, leading to spatial hallucinations in LLMs. We discard text-bottlenecked reasoning and introduce **Multimodal Affordance Grounding**.

We utilize a Multi-modal Large Language Model (MLLM) capable of directly processing spatial tensors. Alongside the source image $\mathcal{I}_{src}$ and the substitution prompt, we feed the MLLM a visually encoded relative depth map $\mathcal{D}_{src}$. By directly perceiving the continuous RGB-D representation, the MLLM natively processes occlusions, surface gradients, and spatial volumes without relying on lossy textual proxies. The dynamically inferred Target Scene Graph $\mathcal{G}_{tgt}$ is thus rigorously conditioned on the uncompressed physical affordances of the scene, ensuring that the deduced interactions are geometrically viable.

### 2.5D Ordinal Kinematic Verification
Monocular 3D mesh recovery (e.g., SMPL-X) on generated 2D images is highly susceptible to occlusion-induced noise, making mesh-collision thresholds arbitrary and unreliable. We replace absolute 3D metric verification with robust **2.5D Ordinal Kinematic Verification**.

We leverage a robust 2D pose estimator to extract interaction keypoints $\mathcal{K}_{human}$ (e.g., hands, feet) and a generic segmentation model to extract the target object mask $\mathcal{M}_{tgt}$ from the generated counterfactual image $\mathcal{I}_{cf}$. We then estimate the relative depth map $\mathcal{D}_{cf}$. Rather than checking absolute 3D penetration, we verify two computationally lightweight, noise-resistant ordinal conditions:
1. **2D Topological Intersection:** For contact-based interactions (e.g., "holding", "stepping on"), the relevant keypoints in $\mathcal{K}_{human}$ must fall within the boundary gradient of $\mathcal{M}_{tgt}$. 
2. **Ordinal Depth Consistency:** We compare the relative depth values at the intersection points: $\Delta d = \mathcal{D}_{cf}(\mathcal{K}_{human}) - \mathcal{D}_{cf}(\mathcal{M}_{tgt}^{local})$. Depending on the semantic relation (e.g., "riding" implies the human is locally in front of or physically resting on the object's upper boundary), $\Delta d$ must satisfy the correct depth ordering threshold, avoiding strict metric collision checks.

By shifting from brittle absolute 3D meshes to 2.5D ordinal relations and 2D topological overlaps, CVLA guarantees that the verified interactions are physically coherent and accurately grounded, bypassing the "garbage in, garbage out" cycle of generative 3D reconstruction artifacts.