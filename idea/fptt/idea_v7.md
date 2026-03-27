# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the feature extraction from standard vision encoders, the active mitigation of detector false positives via a deployable cross-modal consistency bottleneck, and the realistic system-level execution pipeline.

## 4.1 Problem Formulation

LVLMs generate text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the preceding token sequence, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While external object-level priors (bounding boxes and class labels from a detector) can provide explicit grounding, naively concatenating these priors as raw text strings rapidly overwhelms the LLM's context window. Furthermore, without an active filtering mechanism, the LVLM is highly susceptible to cascading errors caused by the detector's inevitable false positives.

To address this, we reformulate the generation process to incorporate a highly compressed, hybrid multi-modal prior $\mathcal{P}_{hybrid}$:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \Phi(\mathcal{P}_{hybrid}), Q, y_{<t}).
\tag{2}
$$
Here, $\mathcal{P}_{hybrid}$ is derived from a cross-modal projection of textual semantic claims and fine-grained visual evidence. $\Phi(\cdot)$ denotes an active inference-time filtering function guided by our cross-modal consistency objective. Through joint optimization and Low-Rank Adaptation (LoRA), the LVLM intrinsically relies on verified visual realities rather than hallucinated textual proposals.

## 4.2 The PATCH Framework

To achieve robust instance-aware visual understanding without exhausting the context window, PATCH maps external spatial proposals directly into the continuous latent space, bridging the semantic gap through explicit projection networks.

### 4.2.1 Token-Efficient Prior Abstraction

Serializing continuous bounding box coordinates into text strings introduces severe token bloat. Given an external detector, we extract a set of objects. Instead of an arbitrary hard truncation that discards small objects, we apply Dynamic Thresholding Non-Maximum Suppression (DT-NMS), which retains all objects above a baseline confidence threshold, capped at a generous upper safety bound $K_{max}$ to prevent out-of-memory errors in extreme cases. The retained set is formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i$ is the bounding box, and $s_i \in [0, 1]$ is the confidence score.

To avoid modality collapse, we map the semantic class $c_i$ to its latent text embedding $E_{text}(c_i) \in \mathbb{R}^d$ using the LLM's embedding matrix. The confidence score $s_i$ is discretized into learnable semantic bins $E_{conf}(s_i) \in \mathbb{R}^{d/4}$. Instead of naive element-wise addition which disrupts the manifold, the semantic claim of the prior is constructed via concatenation and a Multi-Layer Perceptron (MLP):
$$
\mathcal{C}_{semantic}^{(i)} = \text{MLP}_{sem} \big( E_{text}(c_i) \oplus E_{conf}(s_i) \big).
\tag{3}
$$
This condenses the coordinate-free semantic prior into a single robust token.

### 4.2.2 Continuous Spatial-Visual Anchors

To capture the spatial aspect of the prior, we generate visually grounded tokens conditioned on the explicit continuous coordinates $B = \{b_i\}_{i=1}^K$. We apply Fourier Feature Mapping to the coordinates to prevent high-frequency detail loss, generating spatial queries $\mathcal{Q}_{cond} \in \mathbb{R}^{K \times d}$:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}_{sp}(\gamma(B)).
\tag{4}
$$

Since standard LVLM vision encoders (e.g., CLIP-ViT) output single-scale patch tokens rather than feature pyramids, we construct pseudo multi-scale features $\mathcal{E}_{ms}$ by extracting intermediate representations from specific transformer layers (e.g., $L_{1/4}, L_{1/2}, L_{last}$) and processing them through lightweight Convolutional blocks (SimpleFPN). 

The dynamic visual anchors $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are then derived via a Multi-Scale Deformable Cross-Attention (MS-DeformAttn) mechanism:
$$
\mathcal{T}_{dynamic} = \text{MS-DeformAttn}(\text{Query}=\mathcal{Q}_{cond}, \text{Reference}=B, \text{Features}=\mathcal{E}_{ms}).
\tag{5}
$$
To construct the final hybrid prior $\mathcal{P}_{hybrid}$ without destroying the distinct feature spaces, we fuse the semantic claims and visual anchors using concatenation followed by a projection network:
$$
\mathcal{P}_{hybrid}^{(i)} = \text{MLP}_{fuse} \big( \mathcal{C}_{semantic}^{(i)} \oplus \mathcal{T}_{dynamic}^{(i)} \big), \quad \text{for } i = 1, \dots, K.
\tag{6}
$$

### 4.2.3 Cross-Modal Hard-Negative Injection

To train the model to reject detector false positives, we introduce a Background-Verified Active Hard-Negative Injection strategy during training with a probability $p_{noise} = 0.3$:
1. **Semantic Hard Negatives**: The ground-truth class embedding $E_{text}(c_i)$ in Eq. (3) is replaced with a visually similar but incorrect category based on cosine similarity in the text latent space.
2. **Spatial Hard Negatives**: The coordinate $b_i$ in Eq. (4) is shifted to an adjacent region. To prevent the shifted box from accidentally capturing another valid object (which would create noisy false-negative gradients), the shift is strictly constrained to have an $\text{IoU} < 0.05$ with *all* ground-truth bounding boxes in the image, guaranteeing it lands on background or invalid regions.

## 4.3 Cross-Modal Consistency Verification Objective

Relying solely on autoregressive loss cannot impart a robust discriminative capability. We introduce an explicit **Cross-Modal Consistency Verification Head**.

The standard text generation objective is defined as:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \tilde{\mathcal{P}}_{hybrid}, Q, y_{<t} \big),
\tag{7}
$$
where $\tilde{\mathcal{P}}_{hybrid}$ incorporates the injected noise.

Simultaneously, the Verification Head explicitly computes the mutual agreement between the visual anchor $\mathcal{T}_{dynamic}$ and the corresponding textual claim $\tilde{\mathcal{C}}_{semantic}$. We concatenate these representations and pass them through a classification MLP:
$$
\hat{y}^{(i)} = \sigma \Big( \text{MLP}_{verif} \big( \mathcal{T}_{dynamic}^{(i)} \oplus \tilde{\mathcal{C}}_{semantic}^{(i)} \big) \Big).
\tag{8}
$$
This head predicts $y_{verif} \in \{0, 1\}$ (0 for injected noise/background, 1 for authentic prior) using a Binary Cross-Entropy (BCE) loss:
$$
\mathcal{L}_{verif} = - \frac{1}{K} \sum_{i=1}^{K} \big[ y_{verif}^{(i)} \log(\hat{y}^{(i)}) + (1 - y_{verif}^{(i)}) \log(1 - \hat{y}^{(i)}) \big].
\tag{9}
$$
The total optimization objective is:
$$
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{verif}.
\tag{10}
$$

## 4.4 System-Level Deployment Architecture

To guarantee the practical suppression of hallucinations during real-world deployment, the Verification Head is *retained* at inference. 

Functioning as the active filtering module $\Phi(\cdot)$ introduced in Eq. (2), the Verification Head evaluates every proposed token in $\mathcal{P}_{hybrid}$. If the predicted validity score $\hat{y}^{(i)}$ falls below a threshold $\tau_{verif}$, the corresponding prior token $\mathcal{P}_{hybrid}^{(i)}$ is explicitly pruned before entering the LLM's context window. This guarantees that detector false positives are deterministically intercepted.

Regarding latency, we explicitly define PATCH as a serial but highly efficient pipeline. At inference, a lightweight detector (e.g., YOLOv10-N) processes the image sequentially prior to the LLM prompt prefill. While this introduces a hard dependency, the execution latency of YOLOv10-N ($\approx 10$ ms) and the MS-DeformAttn module ($\approx 8$ ms) is negligible compared to the hundreds of milliseconds required for a single autoregressive decoding step of a 7B-parameter LVLM. By replacing hundreds of discrete text bounding-box strings with a dynamically filtered set of $K \le K_{max}$ continuous tokens, PATCH drastically reduces the LLM's KV-cache memory footprint and attention complexity, effectively offsetting the detector's pre-processing time.