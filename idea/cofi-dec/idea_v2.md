# 3 Methodology

This paper introduces **CoFi-Dec**, a training-free framework designed to improve the reliability of Large Vision-Language Models (LVLMs) and mitigate hallucinations. We achieve this through uncertainty-triggered multi-granular visual conditioning and computationally efficient semantic optimal transport, as illustrated in Figure 2.

### Problem Setting
We consider an LVLM parameterized by $\theta$, which takes a sequence of visual tokens $v$ and a tokenized textual query $x$, and generates a coherent textual response sequence $y$ autoregressively. The raw input image $I$ is encoded by a vision encoder and mapped into the language embedding space via a vision-language alignment module. 

Formally, the token distribution at each decoding step $t$ is:
$$
y_t \sim p_\theta(y_t \mid v, x, y_{<t}) = \text{Softmax}\big(f_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
$$
where $y_{<t}$ denotes the preceding tokens, and $f_\theta$ outputs the unnormalized logits over vocabulary $\mathcal{V}$.

---

## 3.1 Hierarchical Visual Decomposition and Generative Calibration

Existing decoding methods often rely on single-scale visual inputs, which easily leads to hallucination when global context conflicts with fine-grained details. To address this, we decompose the visual input and introduce a generative semantic anchor to break the confirmation bias inherent in standard self-correction loops.

### Multi-Granular Visual Embeddings
Given an input image $I_0 \in \mathbb{R}^{H\times W\times 3}$, we extract multi-granular views to preserve both structural integrity and local discriminability:
- **Global View ($I_0$)**: The original resized image capturing the overall scene.
- **Coarse-grained View ($I_c$)**: Low-pass filtered or heavily downsampled patches that retain only the macro spatial layout, intentionally suppressing fine details.
- **Fine-grained View ($I_f$)**: High-resolution localized crops derived from the model's cross-attention maps, isolating salient semantic regions.

Instead of processing these independently, which destroys global-local spatial relationships, they are passed through the vision encoder and alignment module to obtain distinct visual token sequences: $v_0, v_c$, and $v_f$.

### Generative Semantic Anchor
To prevent the "hallucination cascade"—where feeding hallucinated LVLM responses into a Text-to-Image (T2I) model merely visualizes and reinforces errors—we restructure the generative feedback loop. We utilize a pre-trained T2I model $G$ (e.g., Stable Diffusion) conditioned *strictly on the user query* $x$ to synthesize a semantic visual prior:
$$
I_{t2i} = G(x). \tag{2}
$$
By encoding $I_{t2i}$ into visual tokens $v_{t2i}$, we obtain an independent, text-aligned visual anchor. This circumvents the domain shift and confirmation bias issues, as $v_{t2i}$ provides pure semantic layout expectations directly from the prompt, serving as an external regularizer rather than a self-generated echo chamber.

---

## 3.2 Uncertainty-Triggered Semantic Barycenter Decoding

Executing multiple forward passes and high-dimensional distribution fusions at every generation step is computationally infeasible. To ensure practical deployment, CoFi-Dec employs an **uncertainty-triggered** dynamic decoding strategy coupled with a dimension-reduced **Semantic Wasserstein Barycenter**.

### Uncertainty-Triggered Forwarding
At step $t$, we first compute the base probability distribution using only the original global view $v_0$:
$$
P_t^{(0)} = p_\theta(y_t \mid v_0, x, y_{<t}). \tag{3}
$$
We quantify the model's predictive uncertainty using Shannon entropy: $\mathcal{H}_t = -\sum_{w \in \mathcal{V}} P_t^{(0)}(w) \log P_t^{(0)}(w)$. 
If $\mathcal{H}_t < \tau$ (where $\tau$ is a predefined confidence threshold), the model is highly certain, and we directly sample $y_t \sim P_t^{(0)}$, skipping any further computation. 

If $\mathcal{H}_t \geq \tau$, indicating ambiguity or potential hallucination, we trigger the auxiliary forward passes using the fine-grained tokens $v_f$ and the generative anchor $v_{t2i}$:
$$
P_t^{(f)} = p_\theta(y_t \mid v_f, x, y_{<t}), \quad P_t^{(t2i)} = p_\theta(y_t \mid v_{t2i}, x, y_{<t}). \tag{4}
$$

### Semantic Wasserstein Barycenter Fusion on Top-K Simplex
Standard Wasserstein distance is undefined without a specific metric space, and computing it over the entire vocabulary $|\mathcal{V}|$ (e.g., 32k-128k) is mathematically hollow and computationally disastrous. 

To solve this, we restrict the fusion to a truncated vocabulary $\mathcal{V}_K$ containing only the Top-$K$ tokens from $P_t^{(0)}$ (e.g., $K=50$). The distributions $P_t^{(0)}, P_t^{(f)}$, and $P_t^{(t2i)}$ are normalized over $\mathcal{V}_K$ to form $\tilde{P}_t^{(0)}, \tilde{P}_t^{(f)}, \tilde{P}_t^{(t2i)} \in \Delta^{K}$.

Crucially, we define the underlying cost matrix $C \in \mathbb{R}^{K \times K}$ based on the semantic distance between the language model's word embeddings. Let $E_i, E_j$ be the continuous embedding vectors for tokens $i, j \in \mathcal{V}_K$. The transport cost is defined via cosine distance:
$$
C_{i,j} = 1 - \frac{E_i \cdot E_j}{\|E_i\| \|E_j\|}. \tag{5}
$$
With the cost matrix $C$ geometrically grounding the tokens, the fused distribution is obtained by computing the Wasserstein Barycenter:
$$
\tilde{P}_t^{(\text{fused})} = \arg\min_{P\in\Delta^{K}} \Big( \lambda_0 \mathcal{W}_C(P, \tilde{P}_t^{(0)}) + \lambda_f \mathcal{W}_C(P, \tilde{P}_t^{(f)}) + \lambda_{t2i} \mathcal{W}_C(P, \tilde{P}_t^{(t2i)}) \Big), \tag{6}
$$
where $\mathcal{W}_C$ is the Optimal Transport distance parameterized by $C$, and $\lambda_0, \lambda_f, \lambda_{t2i}$ are hyperparameters balancing the original context, fine-grained details, and generative text-alignment prior.

Because $K \ll |\mathcal{V}|$, Equation 6 is solved efficiently in milliseconds using the Sinkhorn-Knopp algorithm. Finally, $y_t$ is sampled from $\tilde{P}_t^{(\text{fused})}$. This lightweight, uncertainty-aware semantic fusion successfully mitigates hallucinations while maintaining near-standard decoding latency.