# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the fine-grained alignment of spatial priors without context bloat, the active mitigation of detector false positives via a cross-modal consistency bottleneck, and realistic system-level deployment.

## 4.1 Problem Formulation

LVLMs generate text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the preceding token sequence, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While external object-level priors (bounding boxes and class labels from a detector) can provide explicit grounding, naively concatenating these priors as raw text strings rapidly overwhelms the LLM's context window and leads to cascading errors when the detector inevitably produces false positives.

To address this, we reformulate the generation process to incorporate a highly compressed, hybrid multi-modal prior $\mathcal{P}_{hybrid}$:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \mathcal{P}_{hybrid}, Q, y_{<t}).
\tag{2}
$$
Here, $\mathcal{P}_{hybrid}$ is derived from a tightly coupled fusion of textual semantic claims and fine-grained visual evidence. Through joint optimization with an explicit cross-modal verification objective and Low-Rank Adaptation (LoRA), the LVLM learns to critically evaluate the consistency between the external prior and the visual reality, intrinsically filtering out hallucinated proposals.

## 4.2 The PATCH Framework

To achieve robust instance-aware visual understanding without exhausting the context window, PATCH completely decouples spatial coordinates from textual serialization, shifting the burden of spatial representation entirely to visually grounded tokens.

### 4.2.1 Token-Efficient Prior Abstraction

Serializing continuous bounding box coordinates into text strings introduces severe token bloat and exacerbates quadratic attention complexity. We argue that explicit coordinate strings are redundant if the spatial information can be fully absorbed by continuous visual tokens. 

Given an external detector, we first extract a set of objects. To prevent redundancy in dense scenes, we apply Distance-Penalized Non-Maximum Suppression (DP-NMS) to retain up to a maximum budget of $K$ objects, formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i$ is the bounding box, and $s_i \in [0, 1]$ is the confidence score. 

Instead of serializing $b_i$, we map the semantic class $c_i$ to its latent text embedding $E_{text}(c_i) \in \mathbb{R}^d$ using the LLM's embedding matrix. The confidence score $s_i$ is discretized into learnable semantic bins $E_{conf}(s_i) \in \mathbb{R}^d$. The purely semantic claim of the prior is defined as:
$$
\mathcal{C}_{semantic}^{(i)} = E_{text}(c_i) + E_{conf}(s_i).
\tag{3}
$$
This removes hundreds of coordinate-related tokens from the context window, leaving a highly compact semantic representation.

### 4.2.2 Continuous Spatial-Visual Anchors

To capture the spatial aspect of the prior, we generate visually grounded tokens conditioned on the exact continuous coordinates $B = \{b_i\}_{i=1}^K$. To retain high-frequency spatial details often lost by simple MLPs, we apply Fourier Feature Mapping to the coordinates to generate spatial queries $\mathcal{Q}_{cond} \in \mathbb{R}^{K \times d}$:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}(\gamma(B)),
\tag{4}
$$
where $\omega$ represents a set of learnable frequency bands. 

We extract multi-scale visual features $\mathcal{E}_{ms}$ by fusing intermediate blocks of the frozen vision encoder. The dynamic visual anchors $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are derived via a Multi-Scale Deformable Cross-Attention (MS-DeformAttn) mechanism:
$$
\mathcal{T}_{dynamic} = \text{MS-DeformAttn}(\text{Query}=\mathcal{Q}_{cond}, \text{Reference}=B, \text{Features}=\mathcal{E}_{ms}).
\tag{5}
$$
Crucially, $\mathcal{T}_{dynamic}$ now encapsulates the *actual visual evidence* present at the locations $B$. The final hybrid prior $\mathcal{P}_{hybrid}$ is formed by adding the visual anchors to the semantic claims:
$$
\mathcal{P}_{hybrid} = \{ \mathcal{C}_{semantic}^{(i)} + \mathcal{T}_{dynamic}^{(i)} \}_{i=1}^K,
\tag{6}
$$
which strictly requires only $K$ tokens in the LLM's context window.

### 4.2.3 Cross-Modal Hard-Negative Injection

To train the model to reject detector false positives, we introduce an Active Hard-Negative Injection strategy during training with a probability $p_{noise} = 0.3$. We simulate realistic detector failures:
1. **Semantic Hard Negatives**: The ground-truth class embedding $E_{text}(c_i)$ in Eq. (3) is replaced with an embedding of a visually similar, yet incorrect, category (e.g., confusing "dog" with "wolf"), based on cosine similarity in the text latent space.
2. **Spatial Hard Negatives**: The coordinate $b_i$ in Eq. (4) and (5) is shifted to an adjacent, visually ambiguous region ($0.1 < \text{IoU} < 0.3$).

This explicitly breaks the default assumption that the textual claim $\mathcal{C}_{semantic}$ always matches the visual evidence $\mathcal{T}_{dynamic}$.

## 4.3 Cross-Modal Consistency Verification Objective

A mathematically rigorous mechanism is required to teach the model to distinguish between authentic priors and injected hard negatives. Relying solely on autoregressive loss is insufficient. We introduce an explicit **Cross-Modal Consistency Bottleneck**.

The standard text generation objective is defined as:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \tilde{\mathcal{P}}_{hybrid}, Q, y_{<t} \big),
\tag{7}
$$
where $\tilde{\mathcal{P}}_{hybrid}$ incorporates the injected noise.

Simultaneously, we construct a Verification Head. Unlike previous flawed formulations that attempted to predict semantic authenticity using *only* visual features, our Verification Head explicitly computes the mutual agreement between the visual anchor $\mathcal{T}_{dynamic}$ and the textual claim $\tilde{\mathcal{C}}_{semantic}$ (which may be falsified). We concatenate these representations and pass them through a Multi-Layer Perceptron (MLP):
$$
\hat{y}^{(i)} = \sigma \Big( \text{MLP} \big( \mathcal{T}_{dynamic}^{(i)} \oplus \tilde{\mathcal{C}}_{semantic}^{(i)} \big) \Big).
\tag{8}
$$
This head predicts $y_{verif} \in \{0, 1\}$ (0 for injected noise, 1 for authentic prior) using a Binary Cross-Entropy (BCE) loss:
$$
\mathcal{L}_{verif} = - \frac{1}{K} \sum_{i=1}^{K} \big[ y_{verif}^{(i)} \log(\hat{y}^{(i)}) + (1 - y_{verif}^{(i)}) \log(1 - \hat{y}^{(i)}) \big].
\tag{9}
$$
The total optimization objective is:
$$
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{verif}.
\tag{10}
$$
By providing both the visual evidence and the semantic claim to the Verification Head, the objective becomes mathematically sound, forcing the network to learn a genuine cross-modal alignment metric rather than collapsing the visual latent space.

## 4.4 System-Level Deployment Architecture

We design PATCH to be practical for real-world LVLM deployment. While adding an external detector introduces computational overhead, we mitigate this through an asynchronous, parallelized pipeline. 

At inference, a lightweight detector (e.g., YOLOv10-N/S) processes the image, incurring negligible GPU latency ($\approx 10$ ms), which is executed asynchronously via CUDA streams during the LLM's initial image-encoding and prompt-prefill phase. Because we abstract priors into $K$ continuous tokens $\mathcal{P}_{hybrid}$ (where $K \le 20$), the KV-cache memory footprint and attention complexity are strictly bounded, resolving the context bloat typical of text-serialization methods. The Verification Head MLP is discarded post-training. The only active parameters introduced during the LVLM forward pass are the MS-DeformAttn weights and the merged LoRA matrices, allowing PATCH to effectively suppress hallucinations with minimal impact on autoregressive decoding throughput.