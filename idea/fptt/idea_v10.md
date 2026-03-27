# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and introduce the refined **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. We specifically address the mathematically rigorous integration of external priors, the extraction of semantic-spatial balanced visual features, the mitigation of detector noise via genuine cross-instance grounding tests, and a realistic pipelined deployment mechanism.

## 4.1 Problem Formulation

LVLMs model the probability of a text sequence $Y$ given an input image $I$ and a prompt $Q$ autoregressively. Incorporating explicit object-level priors from an external detector can alleviate hallucination, provided they do not overwhelm the context window or introduce cascading omission errors from poorly calibrated hard-filtering.

To achieve this, we reformulate the generation process using a continuous hybrid prior sequence. We explicitly define the input sequence layout to prevent modality confusion while maintaining mathematical integrity before layer normalization:
$$
\text{Input} = \big[ \text{System Prompt}, \mathcal{E}_{img}, \tilde{\mathcal{P}}_{hybrid}, Q \big].
\tag{1}
$$
Here, $\mathcal{E}_{img}$ denotes the native visual tokens, and $\tilde{\mathcal{P}}_{hybrid}$ acts as an explicit set of grounded object tokens prepended immediately before the user query $Q$. $\tilde{\mathcal{P}}_{hybrid}$ is derived from raw detector proposals dynamically modulated via a Probabilistic Null-Interpolation mechanism, which smoothly degrades hallucinated proposals into benign placeholders without structurally altering the latent manifold. The autoregressive generation is modeled as:
$$
p(Y) = \prod_{t=1}^{T} p_\theta \big( y_t \mid \text{Input}, y_{<t} \big).
\tag{2}
$$

## 4.2 The PATCH Framework

PATCH replaces textual coordinate serialization with spatially-grounded visual anchors, querying a semantically rich yet spatially preserved feature space of the vision encoder.

### 4.2.1 Continuous Prior Abstraction

Given an external detector, we apply standard Non-Maximum Suppression (NMS) and retain top candidates formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i$ is the continuous bounding box, and $s_i \in (0, 1]$ is the scalar confidence score.

To map the semantic class into the continuous latent space, we extract its textual embedding $E_{text}(c_i) \in \mathbb{R}^d$. Rather than treating the non-linear semantic space as linearly additive, we project the confidence score $s_i$ using a lightweight coordinate-free sinusoidal positional encoding $\phi(s_i) \in \mathbb{R}^{d_{conf}}$ to preserve its continuous resolution. The semantic claim of the prior is then constructed via concatenation and a Multi-Layer Perceptron (MLP) to ensure robust non-linear fusion:
$$
\mathcal{C}_{semantic}^{(i)} = \text{MLP}_{sem} \big( E_{text}(c_i) \oplus \phi(s_i) \big).
\tag{3}
$$

### 4.2.2 Spatial-Aware Visual Anchoring

To capture the spatial geometry of the prior, we generate continuous queries based on the coordinates $B = \{b_i\}_{i=1}^K$ using Fourier Feature Mapping:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}_{sp}(\gamma(B)).
\tag{4}
$$

While deep ViT layers lose precise spatial localization, early layers lack object-level semantics. To extract highly precise semantic-spatial anchors, we extract the intermediate feature map $\mathcal{E}_{vit\_inter} \in \mathbb{R}^{N \times C}$ from a late-intermediate transformer block (e.g., at $75\%$ depth, such as layer $L_{18}$ in a 24-layer ViT). This layer optimally balances high-level semantic richness with retained spatial geometry. The dynamic visual anchors $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are extracted using $\mathcal{Q}_{cond}$ to cross-attend this map:
$$
\mathcal{T}_{dynamic} = \text{CrossAttn} \big( \text{Query}=\mathcal{Q}_{cond}, \text{Key}=\mathcal{E}_{vit\_inter}, \text{Value}=\mathcal{E}_{vit\_inter} \big).
\tag{5}
$$
The raw hybrid prior $\mathcal{P}_{hybrid}$ is fused via a projection network:
$$
\mathcal{P}_{hybrid}^{(i)} = \text{MLP}_{fuse} \big( \mathcal{C}_{semantic}^{(i)} \oplus \mathcal{T}_{dynamic}^{(i)} \big).
\tag{6}
$$

### 4.2.3 Contextual Hard-Negative Synthesis

To explicitly train the network to identify detector false positives, we introduce a Contextual Hard-Negative Injection strategy ($p_{noise} = 0.3$) that mirrors genuine failure modes:
1. **Object-Level Semantic Negatives**: We replace $c_i$ with a confusing category sampled based on the cosine similarity of pre-computed CLIP visual prototypes, simulating genuine detector classification errors.
2. **Cross-Instance Grounding Negatives**: To explicitly test visual-semantic alignment rather than trivial background regions, we swap the bounding box coordinates $b_i$ of object $A$ with the coordinates $b_j$ of a *different* ground-truth object $B$ present in the same image. This semantic mismatch forces the verification module to rigorously ground the text claim to the localized visual patch.

## 4.3 Cross-Modal Consistency Objective

The model must explicitly learn to evaluate the alignment between the textual claim and the visual evidence. We construct a Cross-Modal Consistency Verification Head alongside the standard autoregressive framework.

The generation objective using the noise-injected prior is:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid \text{Input}, y_{<t} \big).
\tag{7}
$$

Simultaneously, the Verification Head assesses the mutual agreement between the visual anchor $\mathcal{T}_{dynamic}$ and the potentially falsified textual claim $\tilde{\mathcal{C}}_{semantic}$:
$$
\hat{y}^{(i)} = \sigma \Big( \text{MLP}_{verif} \big( \mathcal{T}_{dynamic}^{(i)} \oplus \tilde{\mathcal{C}}_{semantic}^{(i)} \big) \Big).
\tag{8}
$$
This outputs a validity probability $\hat{y}^{(i)} \in (0, 1)$, optimized via Binary Cross-Entropy (BCE) against the ground-truth validity label $y_{verif}^{(i)}$:
$$
\mathcal{L}_{verif} = - \frac{1}{K} \sum_{i=1}^{K} \big[ y_{verif}^{(i)} \log(\hat{y}^{(i)}) + (1 - y_{verif}^{(i)}) \log(1 - \hat{y}^{(i)}) \big].
\tag{9}
$$
The total objective is $\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{verif}$.

## 4.4 Probabilistic Null-Interpolation and Pipelined Deployment

Directly scaling features or manipulating attention logits for soft-gating suffers from normalization negation or softmax calibration disruption. Instead, we implement a mathematically sound Probabilistic Null-Interpolation mechanism. 

During inference, the Verification Head predicts the validity score $\hat{y}^{(i)}$. We introduce a globally learnable null-prior token $\mathcal{P}_{null} \in \mathbb{R}^d$, which acts as a benign placeholder signifying "no reliable prior available". The final prior token injected into the LLM is computed as a continuous interpolation:
$$
\tilde{\mathcal{P}}_{hybrid}^{(i)} = \text{LayerNorm} \Big( \hat{y}^{(i)} \cdot \mathcal{P}_{hybrid}^{(i)} + (1 - \hat{y}^{(i)}) \cdot \mathcal{P}_{null} \Big).
\tag{10}
$$
This guarantees that when confidence is low, the representation gracefully transitions to a safe state without destroying the variance expected by the LLM's initial embedding projection, effectively preventing cascading omission errors.

Regarding latency, PATCH utilizes a heavily pipelined deployment architecture. A lightweight detector processes the image concurrently with the LVLM's initial native visual encoding. While the Cross-Attention and MLP Verification inherently introduce a minimal synchronization overhead prior to LLM prefill, executing these operations on the intermediate ViT feature map allows them to partially overlap with the computation of the final ViT layers. This heavily mitigates sequential stalls, establishing a favorable trade-off where the negligible latency added by PATCH is comprehensively offset by the elimination of excessive text-coordinate tokens from the dense attention matrix.