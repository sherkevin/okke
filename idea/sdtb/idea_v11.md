# 3 Methodology

This section details the **Causally-Conditioned Visual-Linguistic Alignment (CVLA)** framework. To address the fundamental limitations of heuristic spatial masks, noisy attention interventions, and computationally fragile post-hoc verifications, CVLA reformulates human-object substitution as a rigorous causal intervention within the diffusion latent space. This approach guarantees structural coherence, mathematically sound global illumination, and deterministic geometric validation without relying on lossy textual proxies or failure-prone monocular 3D reconstructions.

## 3.1 Causal Intervention in Diffusion Latent Space

We define the image generation process via a Structural Causal Model (SCM), $\mathcal{G} = \langle U, V, F \rangle$, where the endogenous variables $V = \{B, H, O, I\}$ represent the Background, Human interactant, Object, and their Interaction, respectively. Modifying an active interaction entails a counterfactual query under the intervention $do(O = o_{tgt})$. Following Pearl’s do-calculus, computing this counterfactual requires three steps: (1) **Abduction:** recovering the exogenous noise $\epsilon_{src}$ via DDIM inversion; (2) **Action:** applying $do(o_{tgt})$ while holding non-descendant variables (background, human pose) constant; and (3) **Prediction:** simulating the forward denoising process. To enforce this causal graph mathematically without violating diffusion manifolds, we introduce the following continuous latent modulations.

### Depth-Projected Spatial Anchoring
Relying on vanishing lines or heuristic 2D bounding boxes fails in unconstrained environments. Instead, we formulate a spatial prior explicitly derived from the inverse projection of the source camera model. 

Given the source relative depth map $D_{src}$ extracted via a metric depth estimator, we define a continuous 3D coordinate field. For any pixel $p = (u,v)$, the projected 3D coordinate is parameterized as $\mathbf{X}_p = D_{src}(p) \cdot \mathbf{K}^{-1} [u, v, 1]^T$, assuming a canonical camera intrinsic matrix $\mathbf{K}$. Under the intervention $do(o_{tgt})$, the counterfactual object must obey the depth-scale inverse proportionality: its expected scale expansion $S(p)$ is strictly bounded by $S(p) \propto 1 / D_{src}(p)$.
We construct a soft energy field $\mathcal{E}_{anchor}(z_t)$ that penalizes target cross-attention $\mathcal{A}_{tgt}^t$ outside the depth-projected valid volume $V_{tgt}$:
$$ \mathcal{E}_{anchor}(z_t) = \sum_p \mathcal{A}_{tgt}^t(p) \cdot \left( 1 - \exp\left(-\frac{\|\mathbf{X}_p - \mu_{tgt}\|^2}{2\sigma^2 \cdot S(p)^2}\right) \right) $$
where $\mu_{tgt}$ is the 3D centroid of the source object. During the generation phase ($t > 0.6T$), we apply a localized latent update $\hat{z}_t = z_t - \eta \nabla_{z_t} \mathcal{E}_{anchor}(z_t)$. This strictly confines the latent expansion to valid 3D perspective geometry without locking it into arbitrary 2D constraints.

### Semantic-Affinity Attention Modulation
Directly adding a constant scalar bias to noisy intermediate cross-attention maps forces unnatural biological feature mappings and introduces grid artifacts. To achieve seamless structural integration across interaction boundaries, we utilize the noise-resistant, high-frequency structural correspondences from a frozen foundation model (e.g., DINOv2).

At intermediate denoising steps ($0.4T \le t \le 0.7T$), we extract dense feature maps $F \in \mathbb{R}^{HW \times d}$ from the intermediate UNet layers. We compute a structural affinity matrix $W = \text{Softmax}\left(\frac{F F^T}{\sqrt{d}}\right)$. Rather than manually defining an interaction frontier via morphological operations, we use $W$ to smoothly guide the self-attention matrix $A_{self}$ of the diffusion model. We apply a temperature-scaled Hadamard modulation:
$$ \hat{A}_{self} = A_{self} \odot \left( \mathbf{1} + \gamma \cdot W_{H \leftrightarrow O} \right) $$
where $W_{H \leftrightarrow O}$ represents the naturally emerging affinity block between the human and object spatial regions. By modulating attention exclusively via data-driven semantic affinities rather than arbitrary spatial intersections, the model organically shares features along physically valid occlusion boundaries, completely preventing global texture pollution.

### Latent Poisson Harmonization
Decoupling structural Keys and illumination Values via binary masks destroys the internal semantic consistency of the UNet and inherently creates seams. To perfectly preserve the global illumination and background structural integrity without expensive VAE decoding, we introduce **Latent Poisson Harmonization**.

We formulate background blending as a discrete Poisson equation solved directly within the latent space. Let $z_{src}^t$ be the inverted source latent and $z_{gen}^t$ be the currently generated latent. At late denoising steps ($t < 0.3T$), we seek a harmonized latent $z_{blend}^t$ whose gradients match the source background while accommodating the newly generated object $\Omega_{tgt}$:
$$ \min_{z} \iint_{\Omega_{bg}} \|\nabla z - \nabla z_{src}^t\|^2 \ dx dy \quad \text{s.t.} \quad z \big|_{\partial \Omega_{tgt}} = z_{gen}^t \big|_{\partial \Omega_{tgt}} $$
Rather than solving this iterative optimization explicitly at each timestep, we approximate it via a single-step gradient correction mask $\mathcal{M}_{\nabla}$:
$$ z_{blend}^t = z_{gen}^t + \mathcal{M}_{bg} \odot \left( z_{src}^t - z_{gen}^t \right) + \lambda \nabla^2(z_{gen}^t - z_{src}^t) $$
The Laplacian term $\nabla^2(\cdot)$ smooths the transition at the object boundaries directly in the latent representation. This inherently preserves ambient occlusion and global lighting distributions from the source image without invoking any VAE-level pixel computations or disrupting the model's key-value manifolds.

## 3.2 Closed-Loop Causal Verification & Guidance

Post-hoc verification pipelines utilizing lossy textual scene graphs or noise-sensitive monocular 3D reconstructions inherently succumb to the "garbage-in, garbage-out" paradigm. CVLA entirely replaces generate-and-discard filtering with programmatic spatial grounding and continuous, differentiable reward guidance.

### Programmatic Geometric Grounding
Feeding continuous RGB-D data directly to standard Large Language Models causes catastrophic hallucinations due to domain mismatch. Instead of forcing MLLMs to interpret depth colormaps, we decouple spatial reasoning into a deterministic programmatic constraint.

Using the source depth map $D_{src}$, we apply the RANSAC algorithm to extract the parameters of dominant 3D planes within the scene (e.g., ground plane normal $\mathbf{n}_g$, supporting surfaces). We construct a symbolic geometric dictionary $S_{geom} = \{P_i = (\mathbf{n}_i, d_i, \mathbf{B}_i)\}_{i=1}^N$, representing mathematically precise affordances. 
When querying the LLM for counterfactual interaction edges, the LLM is strictly constrained by this programmatic dictionary. For example, the action "standing on" is deterministically pruned if the bounding box of the target object does not mathematically intersect with any stable horizontal plane $P$ ($\mathbf{n}_p \approx [0, 1, 0]^T$). This ensures the deduced causal graph is strictly bounded by uncompressed metric reality, eradicating MLLM spatial hallucination at the source.

### Differentiable Interaction Reward Guidance
Post-hoc verification utilizing 2D pose estimators fails drastically on generated artifacts. We transition from discrete accept/reject thresholds to an active, closed-loop **Differentiable Interaction Reward**, optimizing the latent trajectory continuously.

We employ a pre-trained Vision-Language Reward Model $\mathcal{R}_{\theta}(x, c)$ (e.g., ImageReward) that computes a scalar alignment score between an image $x$ and the specific physical interaction text condition $c$ (e.g., "hands grasping the steering wheel"). To apply this during diffusion without breaking the noise schedule, we evaluate the expected clean prediction $\hat{z}_0(t)$ at intermediate steps ($0.2T < t < 0.6T$).
We compute the reward gradient $\nabla_{z_t} \mathcal{R}_{\theta}(\mathcal{D}(\hat{z}_0(t)), c)$, where $\mathcal{D}$ is an approximation of the VAE decoder (e.g., using Tiny AutoEncoder to minimize computational overhead). The diffusion sampling step is then modulated:
$$ \tilde{\epsilon}_{\theta}(z_t, t) = \epsilon_{\theta}(z_t, t) - \alpha_t \sqrt{1 - \bar{\alpha}_t} \nabla_{z_t} \mathcal{R}_{\theta}(\mathcal{D}(\hat{z}_0(t)), c) $$
By integrating the verification dynamically as a classifier-guidance gradient, CVLA steers the sampling process away from physical impossibilities in real-time. This eliminates the need for arbitrary thresholds, bypasses the failure modes of monocular mesh recovery, and guarantees that the final generated image mathematically converges toward the causally valid interaction state.