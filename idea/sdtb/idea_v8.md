# 3 Methodology

This section details the **Counterfactual Visual-Linguistic Alignment (CVLA)** framework. To address the fundamental flaws of rigid geometric projections, interaction hallucination, and naive latent modifications, CVLA introduces a mathematically constrained, energy-guided diffusion modulation scheme coupled with vision-grounded contrastive verification. This formulation ensures strict spatial consistency, physical interaction fidelity, and reliable evaluation without violating the underlying physics of diffusion models or succumbing to VLM evaluation biases.

## 3.1 Physically-Constrained Latent Interaction Synthesis

Substituting objects within active human-object interactions (HOIs) requires preserving the scene layout while mathematically accommodating the topological differences between the source and target objects. We formalize this within the cross-attention and latent spaces of the diffusion denoising process.

### Morphological Cross-Attention Adaptation
Directly aligning the target object's attention map with the source object's manifold inevitably forces the target to inherit an incorrect shape (e.g., generating a "horse-shaped car"). To preserve spatial locality while granting the model necessary topological freedom, we introduce **Morphological Cross-Attention Adaptation**.

Given the source image $\mathcal{I}_{src}$, we invert it using DDIM inversion to obtain the latent trajectory $\{z_t\}_{t=1}^T$ and the source cross-attention maps $\mathcal{A}_{src}^{l,t}$ at specific UNet resolution layers $l \in \{16 \times 16, 32 \times 32\}$. For the target object token, we compute an expanded spatial prior via a morphological dilation operation: $\mathcal{M}_{prior}^{l} = \text{Dilate}(\mathcal{A}_{src}^{l, t_{ref}}, \kappa)$, where $\kappa$ is a dynamic kernel size proportional to the target concept's canonical volume scale. 

During the target generation at early denoising steps ($t > 0.6T$), we apply a localized loss to guide the target attention map $\mathcal{A}_{tgt}^{l,t}$:
$$ \mathcal{L}_{shape} = \sum_{i,j} \left( \mathcal{A}_{tgt}^{l,t}(i,j) \cdot (1 - \mathcal{M}_{prior}^{l}(i,j)) \right)^2 $$
Updating $z_t$ via $\nabla_{z_t} \mathcal{L}_{shape}$ strictly confines the newly generated object within a structurally plausible expanded bounding region, preventing it from bleeding into the background while allowing it to naturally adopt its correct morphological shape inside the designated bounds.

### Energy-Guided Contact Optimization
Relying solely on textual prompts to "autonomously hallucinate" complex physical interactions leads to floating objects or severed limbs. To mathematically enforce physical contact without relying on rigid skeleton estimation, we propose **Energy-Guided Contact Optimization**.

Physical interaction fundamentally requires spatial adjacency between the interactant and the target object. We define a continuous contact energy function $\mathcal{E}_{contact}$ based on the cross-attention maps of the interactant token ($\mathcal{A}^{human}_t$) and the target object token ($\mathcal{A}^{tgt}_t$). To explicitly encourage topological binding, we compute the gradient magnitude of the target's attention map, $\nabla \mathcal{A}^{tgt}_t$, which represents the object's boundary. The contact energy is formulated as:
$$ \mathcal{E}_{contact}(z_t) = - \log \left( \sum_{i,j} \mathcal{A}^{human}_t(i,j) \odot \|\nabla \mathcal{A}^{tgt}_t(i,j)\|^2 + \epsilon \right) $$
At intermediate denoising steps ($0.3T < t < 0.8T$), we perform a gradient descent update on the latent state: $\hat{z}_t = z_t - \eta_t \nabla_{z_t} \mathcal{E}_{contact}(z_t)$. This explicitly pulls the generation of the interactant's extremities toward the boundary of the target object, ensuring a kinematically plausible connection (e.g., hands naturally resting on the car surface) directly through the diffusion prior.

### Timestep-Scheduled Asymptotic Latent Harmonization
Applying spatial Fourier transforms directly to noisy latents $z_t$ destroys the noise schedule and introduces severe artifacts. To achieve photorealistic boundary harmonization while respecting diffusion physics, we propose **Timestep-Scheduled Asymptotic Latent Harmonization**, executing exclusively at late denoising steps ($t \le 0.2T$) where the signal-to-noise ratio is sufficiently high.

Instead of modifying $z_t$, we operate on the predicted clean data $\hat{z}_0(t)$ computed at step $t$:
$$ \hat{z}_0(t) = \frac{z_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}} $$
We apply a Gaussian high-pass filter $\mathcal{H}(\cdot)$ to extract structural high-frequencies. The harmonized prediction $\hat{z}_0^{blend}(t)$ is constructed by interpolating the low-frequency semantics of the generated prediction with the high-frequency ambient lighting and textures of the source image:
$$ \hat{z}_0^{blend}(t) = \mathcal{L}(\hat{z}_0(t)) + \lambda_t \left[ \mathcal{M}_{bg} \odot \mathcal{H}(\hat{z}_{0, src}(t)) + (1 - \mathcal{M}_{bg}) \odot \mathcal{H}(\hat{z}_0(t)) \right] $$
where $\mathcal{M}_{bg}$ is the background mask and $\mathcal{L}(\cdot)$ extracts low frequencies. The corresponding noise $\epsilon_\theta$ is re-evaluated based on $\hat{z}_0^{blend}(t)$, and the standard DDIM step proceeds to $t-1$. This guarantees that lighting and shadows are seamlessly transferred without disrupting the foundational noise distribution.

## 3.2 Vision-Grounded Counterfactual Alignment

To eliminate both LLM contextual blindness and VLM "Yes-bias", CVLA constructs and verifies the relational scene graph through explicit visual grounding and contrastive entailment.

### Context-Aware Relational Graph Synthesis
Deriving physical interactions using text alone ignores the specific geometric constraints of the original image. We utilize a Multimodal Large Language Model (MLLM, e.g., LLaVA-1.5) as a **Context-Aware Reasoning Engine**.

The MLLM is provided with the source image $\mathcal{I}_{src}$, the source interaction edge $e_{source}$, and the target object $v_{tgt}$. The prompt explicitly instructs the MLLM to analyze the physical affordances of the specific scene layout (e.g., indoor constraints, ground terrain) to deduce the most physically valid target interaction $e_{tgt}$. The dynamically inferred Target Scene Graph $\mathcal{G}_{tgt}$ is thus rigorously conditioned on both the counterfactual intent and the actual visual background, preventing the generation of contextually impossible QA pairs.

### Grounded Contrastive Verification
Using VQA models for verification typically fails due to inherent confirmation bias (Yes-bias) and lack of precise localization. We replace heuristic spatial checks with a two-stage **Grounded Contrastive Verification** protocol.

**Step 1: Explicit Semantic Grounding.** We employ an open-world grounding model (e.g., Grounding DINO) to extract precise bounding boxes for both the interactant ($\mathcal{B}_{human}$) and the target object ($\mathcal{B}_{tgt}$) from the generated image $\mathcal{I}_{cf}$. If the model fails to detect the target object above a confidence threshold $\tau_{det}$, it indicates severe generation failure (e.g., object disappearance), and the sample is immediately discarded.

**Step 2: Contrastive Entailment.** Instead of querying a VLM with a binary yes/no question, we formulate a Multiple-Choice Contrastive Query. Given $\mathcal{I}_{cf}$ and the localized bounding regions, the VLM is provided with an explicitly defined target interaction $e_{tgt}$ (e.g., "sitting on") alongside semantically confounding distractors (e.g., "standing next to", "floating above"). We compute the log-likelihood $P_{\theta}(Y_c | \mathcal{I}_{cf}, Q)$ for each candidate choice $c$. The image is retained only if the derived target state $e_{tgt}$ obtains the maximum marginal probability by a predefined margin $\Delta_{margin}$. This mechanism strictly penalizes visual hallucination and ensures the generated dataset genuinely exhibits the complex topological interactions defined in the counterfactual logic.