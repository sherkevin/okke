# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and introduce the refined **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. We specifically address the disruption of 2D spatial inductive biases caused by raw text serialization, the adaptation to standard single-scale vision encoders, the active mitigation of detector noise via genuine hard-negative synthesis, and a fault-tolerant soft-gating deployment mechanism.

## 4.1 Problem Formulation

LVLMs model the probability of a text sequence $Y$ given an input image $I$ and a prompt $Q$ autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}).
\tag{1}
$$
Incorporating explicit object-level priors (bounding boxes and class labels from an external detector) can alleviate hallucination. However, directly serializing continuous coordinates into discrete text tokens strips away the essential 2D spatial inductive bias, forcing the LLM to learn complex geometry from raw strings. Furthermore, applying hard-filtering heuristics to external priors often leads to irreversible cascading errors—if a true object is mistakenly discarded (a false negative), the LLM permanently loses that visual evidence, leading to severe omission errors.

To overcome this, we reformulate the generation process using a continuous, continuous hybrid prior $\mathcal{P}_{hybrid}$ equipped with a differentiable soft-gating mechanism:
$$
p(Y) = \prod_{t=1}^{T} p_\theta \big( y_t \mid I, \Psi(\mathcal{P}_{hybrid}), Q, y_{<t} \big).
\tag{2}
$$
Here, $\mathcal{P}_{hybrid}$ explicitly bridges textual semantics and continuous visual representations in the latent space. $\Psi(\cdot)$ denotes a parameterized soft-gating function optimized via a cross-modal verification objective. This enables the model to dynamically down-weight hallucinated proposals without executing risky hard truncations.

## 4.2 The PATCH Framework

PATCH replaces textual coordinate serialization with spatially-grounded visual anchors, directly querying the native feature space of the vision encoder.

### 4.2.1 Continuous Prior Abstraction

Given an external detector, we first apply standard Non-Maximum Suppression (NMS) with an Intersection-over-Union (IoU) threshold $\tau_{nms}$ to filter highly redundant overlapping proposals. We retain candidates with confidence scores above a baseline threshold, formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i$ is the continuous bounding box, and $s_i \in (0, 1]$ is the scalar confidence score. 

To map the semantic class into the continuous latent space, we extract its textual embedding $E_{text}(c_i) \in \mathbb{R}^d$. Recognizing that the confidence score possesses explicit physical meaning, we avoid destructive discretization. Instead, we project the continuous score $s_i$ into the latent space via a linear mapping layer $W_{conf} \in \mathbb{R}^{1 \times d/4}$. The initial semantic claim of the prior is formed by concatenation followed by a Multi-Layer Perceptron (MLP) to prevent manifold disruption:
$$
\mathcal{C}_{semantic}^{(i)} = \text{MLP}_{sem} \big( E_{text}(c_i) \oplus (s_i \cdot W_{conf}) \big).
\tag{3}
$$

### 4.2.2 Spatial-Aware Visual Anchoring

To capture the spatial geometry of the prior, we generate continuous queries based on the coordinates $B = \{b_i\}_{i=1}^K$. We apply Fourier Feature Mapping to $B$ to preserve high-frequency positional details:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}_{sp}(\gamma(B)).
\tag{4}
$$

Standard LVLM vision encoders (e.g., CLIP-ViT) inherently output single-scale patch tokens. Rather than forcibly applying multi-scale deformable attention mechanisms which are incompatible with standard ViT topologies, we design a Spatial-Aware Cross-Attention mechanism directly on the final native ViT feature map $\mathcal{E}_{vit} \in \mathbb{R}^{N \times C}$. The dynamic visual anchors $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are extracted by using the coordinate-conditioned $\mathcal{Q}_{cond}$ to query the global visual patches:
$$
\mathcal{T}_{dynamic} = \text{CrossAttn} \big( \text{Query}=\mathcal{Q}_{cond}, \text{Key}=\mathcal{E}_{vit}, \text{Value}=\mathcal{E}_{vit} \big).
\tag{5}
$$
To construct the hybrid prior $\mathcal{P}_{hybrid}$ without conflating distinct feature distributions, we fuse the textual semantic claims and the spatial-visual anchors via projection:
$$
\mathcal{P}_{hybrid}^{(i)} = \text{MLP}_{fuse} \big( \mathcal{C}_{semantic}^{(i)} \oplus \mathcal{T}_{dynamic}^{(i)} \big), \quad \text{for } i = 1, \dots, K.
\tag{6}
$$

### 4.2.3 Visually-Grounded Hard-Negative Synthesis

To explicitly train the network to identify detector false positives, we introduce a genuine Hard-Negative Injection strategy ($p_{noise} = 0.3$) mimicking real-world detector confusion:
1. **Visual Semantic Hard Negatives**: Replacing $E_{text}(c_i)$ based purely on textual similarity is flawed. Instead, we replace $c_i$ with a confusing category sampled based on the cosine similarity of their *visual prototypes* (extracted via the pre-trained CLIP visual encoder).
2. **Partial-Object Spatial Hard Negatives**: Shifting bounding boxes to pure background regions (IoU $\approx 0$) yields trivial easy negatives. To construct true spatial hard negatives, we perturb the coordinate $b_i$ to ensure a partial overlap ($0.3 \le \text{IoU} \le 0.5$) with the ground truth. This forces the model to discriminate between localized parts and complete object instances.

## 4.3 Cross-Modal Consistency Objective

The model must explicitly learn to evaluate the alignment between the textual claim and the visual evidence. We construct a Cross-Modal Consistency Verification Head alongside the standard autoregressive framework.

The generation objective using the noise-injected prior $\tilde{\mathcal{P}}_{hybrid}$ is:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \tilde{\mathcal{P}}_{hybrid}, Q, y_{<t} \big).
\tag{7}
$$

Simultaneously, the Verification Head assesses the explicit mutual agreement between the visual anchor $\mathcal{T}_{dynamic}$ and the potentially falsified textual claim $\tilde{\mathcal{C}}_{semantic}$:
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

## 4.4 System-Level Soft-Gating Deployment

To ensure robust execution during inference without introducing catastrophic omission errors, the Verification Head is retained to function as the continuous soft-gating mechanism $\Psi(\cdot)$ defined in Eq. (2). 

Instead of employing a hard pruning threshold that permanently drops tokens (which heavily penalizes the system if a false negative occurs), the predicted validity score $\hat{y}^{(i)}$ acts as a continuous modulation weight:
$$
\Psi(\mathcal{P}_{hybrid}^{(i)}) = \hat{y}^{(i)} \cdot \mathcal{P}_{hybrid}^{(i)}.
\tag{10}
$$
This smoothly attenuates the feature magnitude of hallucinated priors while preserving highly confident proposals. 

In terms of execution flow, PATCH explicitly operates as a serial, decoupled pipeline. A lightweight detector (e.g., YOLOv10-N) generates $B$ sequentially before the LLM's prompt prefill phase. By completely replacing discrete bounding-box text serialization with explicitly grounded continuous tokens, PATCH establishes a mathematically rigorous continuous space for prior injection. This enables the LVLM to inherently align visual and textual modalities, systematically resisting hallucination without brittle manual heuristics.