# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and introduce the refined **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. We specifically address the mathematically rigorous extraction of localized visual features, the dynamic integration of external priors, the decoupled mitigation of detector noise via strict cross-instance grounding tests, and a realistic computational-latency trade-off deployment mechanism.

## 4.1 Problem Formulation

LVLMs model the probability of a text sequence $Y$ given an input image $I$ and a prompt $Q$ autoregressively. Incorporating explicit object-level priors from an external detector can alleviate hallucination, provided they do not overwhelm the context window or introduce cascading omission errors from poorly calibrated hard-filtering.

To achieve this, we reformulate the generation process using a continuous hybrid prior sequence. We explicitly define the input sequence layout to prevent modality confusion while maintaining mathematical integrity:
$$
\text{Input} = \big[ \text{System Prompt}, \mathcal{E}_{img}, \mathcal{P}_{mod}^{*}, Q \big].
\tag{1}
$$
Here, $\mathcal{E}_{img}$ denotes the native visual tokens, and $\mathcal{P}_{mod}^{*}$ acts as an explicit set of grounded object tokens prepended immediately before the user query $Q$. $\mathcal{P}_{mod}^{*}$ is derived from raw detector proposals dynamically modulated via a Confident Semantic-Fallback mechanism. This mechanism smoothly degrades unverified proposals into safe, geometry-free textual embeddings without triggering normalization instability. The autoregressive generation is modeled as:
$$
p(Y) = \prod_{t=1}^{T} p_\theta \big( y_t \mid \text{Input}, y_{<t} \big).
\tag{2}
$$

## 4.2 The PATCH Framework

PATCH systematically establishes visual-semantic priors by extracting precise region-based features and fusing them with confidence-calibrated textual claims.

### 4.2.1 Confidence-Gated Semantic Abstraction

Given an external detector, we apply standard Non-Maximum Suppression (NMS) and retain top candidates formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i = [x_1, y_1, x_2, y_2]$ is the continuous 4D bounding box, and $s_i \in (0, 1]$ is the scalar confidence score.

To map the semantic class into the continuous latent space, we extract its textual embedding $E_{text}(c_i) \in \mathbb{R}^d$. Recognizing that the confidence score $s_i$ is a strictly monotonic scalar probability, over-engineering it into high-frequency positional encodings is mathematically unfounded. Instead, we project $s_i$ via a simple linear projection followed by a sigmoid activation to act as a continuous feature-scaling gate, ensuring the underlying semantic manifold remains uncorrupted:
$$
\mathcal{C}_{semantic}^{(i)} = E_{text}(c_i) \odot \sigma(W_s s_i + \beta_s),
\tag{3}
$$
where $W_s \in \mathbb{R}^{d \times 1}$ and $\beta_s \in \mathbb{R}^d$ are learnable parameters, and $\odot$ denotes element-wise multiplication.

### 4.2.2 Dynamic Layer Routing and RoI-Align Anchoring

Extracting spatial features from a rigidly specified, single intermediate ViT layer lacks generalizability, and using global Cross-Attention with box coordinates introduces severe feature confusion in densely overlapped regions. 

To resolve this, we first dynamically aggregate multi-level representations from the vision encoder's later stages (e.g., the last four layers $\mathcal{L}_{late}$), optimized via learnable softmax weights $\alpha_l$:
$$
\mathcal{E}_{aggr} = \sum_{l \in \mathcal{L}_{late}} \frac{\exp(\alpha_l)}{\sum_{k} \exp(\alpha_k)} \mathcal{E}_l.
\tag{4}
$$
By reshaping $\mathcal{E}_{aggr}$ back into its 2D spatial grid format, we entirely bypass error-prone global attention. Instead, we directly apply the precise and localized **RoI-Align** operation using the 4D coordinates $B = \{b_i\}_{i=1}^K$. This guarantees that the extracted visual anchors are strictly bounded by their corresponding physical geometries:
$$
\mathcal{T}_{local} = \text{RoI-Align}(\mathcal{E}_{aggr}, B).
\tag{5}
$$
The raw hybrid prior $\mathcal{P}_{hybrid}$ is then fused via a robust projection network:
$$
\mathcal{P}_{hybrid}^{(i)} = \text{MLP}_{fuse} \big( \mathcal{C}_{semantic}^{(i)} \oplus \text{Flatten}(\mathcal{T}_{local}^{(i)}) \big).
\tag{6}
$$

### 4.2.3 Strict Contextual Hard-Negative Synthesis

To explicitly train the Verification Head to identify detector false positives, we introduce a Hard-Negative Injection strategy ($p_{noise} = 0.3$) formulated to avoid false-positive penalizations:
1. **Object-Level Semantic Negatives**: We replace $c_i$ with a confusing category sampled based on the cosine similarity of pre-computed CLIP visual prototypes, simulating genuine detector classification errors.
2. **Strict Cross-Instance Negatives**: We swap the bounding box coordinates $b_i$ of object $A$ with $b_j$ of ground-truth object $B$. Crucially, to prevent generating pseudo-positive labels from overlapping instances or objects of the same class, this swap is *only* executed if object $B$ belongs to a different semantic category and their spatial intersection satisfies $\text{IoU}(b_i, b_j) < 0.1$. 

## 4.3 Decoupled Cross-Modal Consistency Objective

A critical flaw in prior multi-task designs is forcing the generation loss to backpropagate through noise-injected tokens, which directly trains the LLM to hallucinate. To prevent this, PATCH fundamentally decouples the training data streams.

The standard autoregressive generation objective is exclusively trained using the **clean**, uncorrupted hybrid priors $\mathcal{P}_{hybrid}$:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \mathcal{P}_{hybrid}, Q, y_{<t} \big).
\tag{7}
$$

Concurrently, the independent Verification Head is trained using the **noise-injected** representations $\tilde{\mathcal{P}}_{hybrid}$ (containing both genuine and forged $\tilde{\mathcal{C}}_{semantic}$) to rigorously assess alignment:
$$
\hat{y}^{(i)} = \sigma \Big( \text{MLP}_{verif} \big( \tilde{\mathcal{P}}_{hybrid}^{(i)} \big) \Big).
\tag{8}
$$
This outputs a validity probability $\hat{y}^{(i)} \in (0, 1)$, optimized via Binary Cross-Entropy (BCE) against the strict ground-truth label $y_{verif}^{(i)}$:
$$
\mathcal{L}_{verif} = - \frac{1}{K} \sum_{i=1}^{K} \big[ y_{verif}^{(i)} \log(\hat{y}^{(i)}) + (1 - y_{verif}^{(i)}) \log(1 - \hat{y}^{(i)}) \big].
\tag{9}
$$
The final end-to-end objective is $\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{verif}$.

## 4.4 Semantic-Fallback Deployment and Computation Trade-offs

Scaling features with LayerNorm introduces variance-explosion risks when the input magnitude approaches zero. Instead, we implement a **Confident Semantic-Fallback** mechanism. 

During inference, the Verification Head predicts the validity score $\hat{y}^{(i)}$ for each proposal. Rather than interpolating toward a learnable global null-token—which acts as unpredictable noise—we interpolate the prior back to its pure, geometry-free textual embedding $E_{text}(c_i)$:
$$
\mathcal{P}_{mod}^{*(i)} = \hat{y}^{(i)} \cdot \mathcal{P}_{hybrid}^{(i)} + (1 - \hat{y}^{(i)}) \cdot E_{text}(c_i).
\tag{10}
$$
This mathematically sound residual gating guarantees that when the detector's spatial grounding is deemed unreliable ($\hat{y} \rightarrow 0$), the representation safely degrades to a standard textual word token, entirely preserving the expected feature variance for the LLM while completely stripping away falsified spatial emphasis.

Regarding system latency, we acknowledge the inherent serial overhead introduced by the external detector and RoI-Align feature extraction. However, PATCH achieves computational feasibility by fundamentally shifting the workload bottleneck. Directly serializing 100 object bounding boxes into raw text introduces massive sequence bloat, exacerbating the $O(T^2)$ computational complexity of the LLM's Attention mechanism during the Prefill phase. By compressing these priors into exactly $K$ continuous, region-aligned hybrid tokens, PATCH significantly reduces the prompt length. The heavy reduction in the quadratic Attention computation comprehensively offsets the constant-time serial overhead introduced by the lightweight detector and linear projection components, yielding a highly accurate, hallucination-resistant system suitable for practical deployment.