# 3 Methodology

This paper introduces **CoFi-Dec**, a training-free decoding framework designed to improve the reliability of Large Vision-Language Models (LVLMs) and mitigate visual hallucinations. We achieve this through parallel multi-granular visual conditioning and a computationally efficient, divergence-triggered contrastive barycenter fusion.

### Problem Setting
We consider an LVLM parameterized by $\theta$, which takes a sequence of visual tokens $v$ and a tokenized textual query $x$, and generates a textual response sequence $y$ autoregressively. The raw input image $I$ is processed by a vision encoder and mapped into the language embedding space. Formally, the token distribution at decoding step $t$ is:
$$
y_t \sim p_\theta(y_t \mid v, x, y_{<t}) = \text{Softmax}\big(f_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
$$
where $y_{<t}$ denotes the preceding tokens, and $f_\theta$ outputs the unnormalized logits over the vocabulary $\mathcal{V}$.

---

## 3.1 Pre-Computed Hierarchical Visual Decomposition

Existing decoding methods often rely on a single-scale visual input, leading to hallucination cascades when the global context overrides fine-grained details. To break this confirmation bias without incurring real-time generation overhead or temporal paradoxes during decoding, we perform a deterministic hierarchical visual decomposition *prior* to the autoregressive generation phase.

### Multi-Granular Visual Embeddings
Given an input image $I_0 \in \mathbb{R}^{H\times W\times 3}$ and a user query $x$, we extract three distinct views:
- **Global View ($I_0$)**: The original resized image capturing the holistic scene.
- **Coarse-grained View ($I_c$)**: A heavily downsampled and low-pass filtered version of $I_0$. This view retains only the macro spatial layout and intentionally obliterates fine-grained visual evidence, serving as an uncertainty baseline prone to generating hallucinated priors.
- **Fine-grained View ($I_f$)**: High-resolution localized crops. To avoid temporal misalignment and mid-decoding latency, $I_f$ is extracted based on the cross-attention map between the global image $I_0$ and the *initial user query* $x$ during a preliminary grounding step, effectively isolating query-salient semantic regions.

These images are independently passed through the vision encoder to obtain three visual token sequences: $v_0, v_c$, and $v_f$. 

### Parallel KV-Cache Initialization
To strictly maintain stream-decoding efficiency and eliminate the disastrous latency of context-switching, we process $(v_0, x)$, $(v_c, x)$, and $(v_f, x)$ simultaneously during the standard prefill phase. This initializes three parallel Key-Value (KV) caches. During the subsequent autoregressive decoding steps, the model simply updates these parallel caches with $O(1)$ complexity per step, completely avoiding the need to recompute historical embeddings $y_{<t}$.

---

## 3.2 Divergence-Triggered Contrastive Barycenter Decoding

Executing complex distribution fusion at every generation step is computationally redundant. Furthermore, standard uncertainty metrics like Shannon entropy are fundamentally flawed for LVLMs, as hallucinated tokens are frequently emitted with extreme overconfidence (low entropy), while high entropy often merely indicates synonym selection. 

### Cross-Granular Divergence Trigger
Instead of relying on token-level entropy, CoFi-Dec employs a **distributional divergence trigger**. At decoding step $t$, we initially compute the probability distributions from the global and coarse KV-caches:
$$
P_t^{(0)} = p_\theta(y_t \mid v_0, x, y_{<t}), \quad P_t^{(c)} = p_\theta(y_t \mid v_c, x, y_{<t}). \tag{2}
$$
We quantify hallucination risk using the symmetric Kullback-Leibler (KL) divergence between the global and coarse predictions:
$$
\mathcal{D}_t = \frac{1}{2} D_{KL}(P_t^{(0)} \parallel P_t^{(c)}) + \frac{1}{2} D_{KL}(P_t^{(c)} \parallel P_t^{(0)}). \tag{3}
$$
A low $\mathcal{D}_t$ implies that fine details are unnecessary for the current prediction, and we sample $y_t \sim P_t^{(0)}$ directly. Conversely, if $\mathcal{D}_t \geq \tau$ (a predefined semantic threshold), it indicates that the model is making a high-stakes decision where global layout and missing details strongly conflict—a prime condition for hallucination. We then trigger the fine-grained forward pass:
$$
P_t^{(f)} = p_\theta(y_t \mid v_f, x, y_{<t}). \tag{4}
$$

### Log-Linear KL-Barycenter (Contrastive Fusion)
Applying geometric Optimal Transport (e.g., Wasserstein distance) directly on BPE or SentencePiece token embeddings is linguistically unsound, as subwords often lack independent semantics. To robustly fuse multi-granular evidence in the discrete token space, we compute the **KL-Barycenter** of the distributions. This yields a closed-form weighted geometric mean, effectively executing a principled contrastive decoding objective.

To prevent missing correct tokens that might be suppressed by overconfident hallucinations in $P_t^{(0)}$, we define a dynamic candidate set $\mathcal{V}_K$ as the union of the Top-$K$ tokens from all three distributions: $\mathcal{V}_K = \text{TopK}(P_t^{(0)}) \cup \text{TopK}(P_t^{(c)}) \cup \text{TopK}(P_t^{(f)})$.

The log-linear barycenter distribution $\tilde{P}_t^{(\text{fused})}$ over $\mathcal{V}_K$ is formulated as:
$$
\tilde{P}_t^{(\text{fused})}(w) = \frac{1}{Z} \left( P_t^{(0)}(w) \right)^{\lambda_0} \left( P_t^{(f)}(w) \right)^{\lambda_f} \left( P_t^{(c)}(w) \right)^{-\lambda_c}, \quad \forall w \in \mathcal{V}_K, \tag{5}
$$
where $Z$ is the normalization constant, and $\lambda_0, \lambda_f, \lambda_c > 0$ are balancing coefficients. 

This formulation naturally functions as a rigorous error-correction mechanism: it amplifies tokens supported by fine-grained visual evidence ($P_t^{(f)}$) while explicitly penalizing tokens that rely purely on the blurred, hallucination-prone macro layout ($P_t^{(c)}$). Finally, the next token $y_t$ is sampled from $\tilde{P}_t^{(\text{fused})}$. By leveraging parallel KV-caching and closed-form log-linear fusion, CoFi-Dec decisively eliminates hallucinations while maintaining real-time decoding latency.