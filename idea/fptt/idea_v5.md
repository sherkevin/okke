# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the fine-grained multi-scale alignment of spatial priors, the active mitigation of detector noise via hard-negative verification, and system-level deployment efficiency.

## 4.1 Problem Formulation

LVLMs generate text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the token sequence preceding step $t$, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While external object-level priors from an off-the-shelf detector can provide explicit grounding, naively concatenating these priors leads to severe cascading errors. Detectors inevitably produce noisy predictions (missed detections or false positives). Without an explicit noise-rectification mechanism, the LVLM blindly trusts these priors, exacerbating text-bias.

To address this, we reformulate the generation process to incorporate external priors $D$ through a prior-guided spatial bridging module and a multi-task verification bottleneck:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \mathcal{T}_{dynamic}, D, Q, y_{<t}).
\tag{2}
$$
Here, $\mathcal{T}_{dynamic}$ serves as fine-grained visual prompts actively conditioned on the spatial layout of $D$. Through joint optimization with an explicit verification objective and Low-Rank Adaptation (LoRA) within the LLM, the model learns to critically evaluate the consistency between the external prior $D$ and the actual visual features, fundamentally breaking the textual shortcut.

## 4.2 The PATCH Framework

To achieve robust instance-aware visual understanding, PATCH introduces a high-resolution prior-guided token generation mechanism combined with a hard-negative noise injection training strategy.

### 4.2.1 Diversity-Aware Prior Serialization and Discretization

To translate detection outputs into a format interpretable by the LLM without overwhelming the context window or losing spatial diversity, we introduce a **Diversity-Aware Spatial Pruning** strategy. Relying solely on confidence scores often results in highly redundant bounding boxes clustered around a single salient object. Instead, we apply Distance-Penalized Non-Maximum Suppression (DP-NMS), which balances confidence scores with spatial distribution, ensuring that background and small objects are explicitly retained up to a maximum budget of $K$ objects.

For the pruned set of $K$ objects $\{(c_i, b_i, s_i)\}_{i=1}^K$, $c_i$ is the class name, and $b_i = (x_{1}, y_{1}, x_{2}, y_{2})$ represents the normalized bounding box coordinates. To accommodate the LLM's insensitivity to raw floating-point numbers, we discretize the continuous confidence score $s_i \in [0, 1]$ into $N$ distinct semantic bins (e.g., `<conf_high>`, `<conf_med>`, `<conf_low>`). $D$ is then serialized as:
$$
D = \text{Concat}_{i=1}^K \big( \text{"[OBJ]"} \oplus c_i \oplus \text{"[LOC]"} \oplus \text{str}(b_i) \oplus \text{"[CONF]"} \oplus \text{Bin}(s_i) \big).
\tag{3}
$$
This discretized representation provides the LLM with a highly intuitive initial heuristic to estimate the reliability of the prior.

### 4.2.2 High-Resolution Spatial Interaction via Fourier Mapping

To resolve the severe resolution mismatch between low-frequency coordinates and highly downsampled vision encoder outputs (e.g., $14 \times 14$ ViT feature maps), we redesign the spatial interaction mechanism. 

First, to prevent high-frequency spatial detail loss, we apply **Fourier Feature Mapping** to the bounding box coordinates $B = \{b_i\}_{i=1}^K$ before passing them to the MLP. The spatially-conditioned queries $\mathcal{Q}_{cond} \in \mathbb{R}^{K \times d}$ are generated as:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}(\gamma(B)),
\tag{4}
$$
where $\omega$ represents a set of learnable frequency bands. 

Second, we extract **multi-scale visual features** $\mathcal{E}_{ms}$ by fusing intermediate blocks of the frozen vision encoder, providing a high-resolution feature pyramid enriched with 2D Positional Encoding. The dynamic virtual tokens $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are derived via a Multi-Scale Deformable Cross-Attention layer:
$$
\mathcal{T}_{dynamic} = \text{MS-DeformAttn}(\text{Query}=\mathcal{Q}_{cond}, \text{Reference}=B, \text{Features}=\mathcal{E}_{ms}).
\tag{5}
$$
This allows the virtual tokens to directly sample fine-grained visual evidence from high-resolution feature maps based on the exact continuous coordinates, maintaining a strict one-to-one spatial correspondence with the objects in $D$. These tokens are then prepended to the embedded sequence of $D$.

### 4.2.3 Hard-Negative Noise Injection Strategy

To intrinsically cultivate robustness against detector hallucinations, we introduce an **Active Hard-Negative Injection** strategy. Instead of simplistic augmentations, we synthesize challenging noise with a probability $p_{noise}$ (set to 0.3) to simulate real-world detector failures:
1. **Semantic Hard Negatives**: Replacing the ground-truth class $c_i$ with a visually similar category (e.g., confusing "dog" with "wolf") sampled based on the cosine similarity of text embeddings.
2. **Spatial Hard Negatives**: Shifting the bounding boxes $b_i$ to visually ambiguous adjacent regions, constrained by an Intersection over Union (IoU) threshold of $0.1 < \text{IoU} < 0.3$.
This strategy forces the model to move beyond simple image-text matching and explicitly cross-check the claims in $D$ against the fine-grained visual evidence in $\mathcal{T}_{dynamic}$.

## 4.3 Multi-Task Verification Objective

Applying a contrastive loss directly to the visual tokens inherently disrupts their representation in the latent space. Instead, we explicitly train the model to recognize hallucinations by introducing a lightweight **Verification Head**. 

The first component of our optimization is the standard negative log-likelihood for text generation:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \mathcal{T}_{dynamic}, \tilde{D}, Q, y_{<t} \big),
\tag{6}
$$
where $\tilde{D}$ represents the external prior subjected to the Hard-Negative Injection.

Simultaneously, we pass the prior-guided visual tokens $\mathcal{T}_{dynamic}$ through a linear classification head to predict whether the corresponding explicit prior in $\tilde{D}$ is authentic ($y_{verif}=1$) or injected noise ($y_{verif}=0$). This is optimized using Binary Cross-Entropy (BCE) loss:
$$
\mathcal{L}_{verif} = - \frac{1}{K} \sum_{i=1}^{K} \big[ y_{verif}^{(i)} \log(\hat{y}^{(i)}) + (1 - y_{verif}^{(i)}) \log(1 - \hat{y}^{(i)}) \big],
\tag{7}
$$
where $\hat{y} = \sigma(\text{Linear}(\mathcal{T}_{dynamic}))$. The total optimization objective is defined as:
$$
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{verif},
\tag{8}
$$
with $\lambda$ controlling the verification strength. This multi-task approach explicitly penalizes the acceptance of false textual priors without distorting the continuous visual representations.

## 4.4 System-Level Deployment Architecture

While PATCH leverages an external detector, the system architecture is designed for decoupled, asynchronous deployment to minimize overhead. The detector operates as an independent pre-processing pipeline, which can be offloaded to CPUs or edge devices, thereby preventing VRAM bottlenecks on the primary GPU hosting the LVLM. 

Furthermore, the DP-NMS pruning strictly bounds the length of the serialized prior $D$, preventing quadratic computational surges in the LLM's self-attention mechanism. At deployment, the trained LoRA matrices are merged back into the frozen LLM weights, and the Verification Head is discarded. The only active overhead during the LVLM forward pass stems from the lightweight MLP and Deformable Cross-Attention layers, rendering PATCH highly practical for real-time, hallucination-resistant inference.