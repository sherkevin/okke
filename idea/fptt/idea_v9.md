# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and introduce the refined **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. We specifically address the mathematical integration of priors bypassing normalization destruction, the extraction of spatially-retained intermediate visual features, the mitigation of detector noise via contextual hard-negative synthesis, and a low-latency concurrent deployment mechanism.

## 4.1 Problem Formulation

LVLMs model the probability of a text sequence $Y$ given an input image $I$ and a prompt $Q$ autoregressively. Incorporating explicit object-level priors from an external detector can alleviate hallucination, provided they do not overwhelm the context window or introduce cascading omission errors from brittle hard-filtering. 

To achieve this, we reformulate the generation process using a continuous hybrid prior sequence $\mathcal{P}_{hybrid}$. Crucially, we explicitly define the input sequence structure to avoid modality confusion. The layout is formatted as:
$$
\text{Input} = \big[ \text{System Prompt}, \mathcal{E}_{img}, \Psi(\mathcal{P}_{hybrid}), Q \big].
\tag{1}
$$
Here, $\mathcal{E}_{img}$ denotes the native visual tokens, and $\mathcal{P}_{hybrid}$ acts as an explicit set of grounded object tokens prepended immediately before the user query $Q$. $\Psi(\cdot)$ denotes a parameterized soft-attention biasing function optimized via a cross-modal verification objective, which gracefully reduces the influence of hallucinated proposals without structurally altering the latent manifold or triggering omission errors. The autoregressive generation is thus modeled as:
$$
p(Y) = \prod_{t=1}^{T} p_\theta \big( y_t \mid \text{Input}, y_{<t} \big).
\tag{2}
$$

## 4.2 The PATCH Framework

PATCH replaces textual coordinate serialization with spatially-grounded visual anchors, directly querying the spatially-retained feature space of the vision encoder.

### 4.2.1 Continuous Prior Abstraction

Given an external detector, we apply standard Non-Maximum Suppression (NMS) and retain top candidates formulated as $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i$ is the continuous bounding box, and $s_i \in (0, 1]$ is the scalar confidence score. 

To map the semantic class into the continuous latent space, we extract its textual embedding $E_{text}(c_i) \in \mathbb{R}^d$. Rather than unjustifiably expanding the 1D confidence score $s_i$ into a high-dimensional space, we utilize it to linearly scale a single learnable confidence direction vector $v_{conf} \in \mathbb{R}^d$. The coordinate-free semantic claim of the prior is thus efficiently formed:
$$
\mathcal{C}_{semantic}^{(i)} = E_{text}(c_i) + s_i \cdot v_{conf}.
\tag{3}
$$

### 4.2.2 Spatial-Aware Visual Anchoring

To capture the spatial geometry of the prior, we generate continuous queries based on the coordinates $B = \{b_i\}_{i=1}^K$ using Fourier Feature Mapping:
$$
\gamma(b_i) = \big[ \sin(2\pi \omega b_i), \cos(2\pi \omega b_i) \big], \quad \mathcal{Q}_{cond} = \text{MLP}_{sp}(\gamma(B)).
\tag{4}
$$

It is well-established that the final layers of standard ViTs (e.g., CLIP-ViT) lose precise spatial localization due to deep self-attention mixing. To extract highly precise spatial anchors, we bypass the final layer and instead extract the intermediate feature map $\mathcal{E}_{vit\_inter} \in \mathbb{R}^{N \times C}$ from an intermediate transformer block (e.g., layer $L_{2/3}$), which optimally balances semantic richness with preserved spatial geometry. The dynamic visual anchors $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are extracted using $\mathcal{Q}_{cond}$ to cross-attend this intermediate map:
$$
\mathcal{T}_{dynamic} = \text{CrossAttn} \big( \text{Query}=\mathcal{Q}_{cond}, \text{Key}=\mathcal{E}_{vit\_inter}, \text{Value}=\mathcal{E}_{vit\_inter} \big).
\tag{5}
$$
The hybrid prior $\mathcal{P}_{hybrid}$ is fused via a projection network:
$$
\mathcal{P}_{hybrid}^{(i)} = \text{MLP}_{fuse} \big( \mathcal{C}_{semantic}^{(i)} \oplus \mathcal{T}_{dynamic}^{(i)} \big).
\tag{6}
$$

### 4.2.3 Contextual Hard-Negative Synthesis

To explicitly train the network to identify detector false positives without penalizing natural phenomena like occlusion, we introduce a Contextual Hard-Negative Injection strategy ($p_{noise} = 0.3$):
1. **Visual Semantic Hard Negatives**: We replace $c_i$ with a confusing category sampled based on the cosine similarity of their pre-computed CLIP visual prototypes, simulating genuine detector classification errors.
2. **Instance-Swap Spatial Hard Negatives**: Instead of shifting boxes to random IoU ranges (which falsely penalizes partial occlusions), we swap the bounding box coordinates $b_i$ of object $A$ with the coordinates $b_j$ of a *different* ground-truth object $B$ present in the same image. This forces the model to detect severe visual-semantic mismatches rather than trivial background regions.

## 4.3 Cross-Modal Consistency Objective

The model must explicitly learn to evaluate the alignment between the textual claim and the visual evidence. We construct a Cross-Modal Consistency Verification Head alongside the standard autoregressive framework.

The generation objective using the noise-injected prior $\tilde{\mathcal{P}}_{hybrid}$ is:
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

## 4.4 Attention-Mask Biasing and Concurrent Deployment

Attempting to implement soft-gating by scaling the feature magnitude of $\mathcal{P}_{hybrid}^{(i)}$ is mathematically flawed, as subsequent LayerNorm/RMSNorm operations instantly undo the scaling. Instead, we implement $\Psi(\cdot)$ as an Attention-Mask Biasing mechanism. 

The Verification Head is retained during inference. The predicted validity score $\hat{y}^{(i)}$ is translated into an additive bias applied directly to the LLM's causal attention mask matrix $M$ whenever a subsequent token attends to the prior token $i$:
$$
M_{t, i} = \begin{cases} 
0, & \text{if token } t \text{ attends to standard tokens} \\
\alpha \log(\hat{y}^{(i)} + \epsilon), & \text{if token } t \text{ attends to prior token } i
\end{cases}
\tag{10}
$$
where $\alpha$ is a scaling hyperparameter. Because this bias is added *after* the dot-product query-key multiplication and immediately before the softmax operation, it completely bypasses the norm layers. It explicitly diminishes the attention weights assigned to hallucinated priors mathematically rigorously, preventing cascading omission errors.

To eliminate latency bottlenecks, PATCH operates asynchronously. A lightweight detector processes a downsampled image frame entirely in parallel with the initial visual encoding phase of the LVLM. Since $\mathcal{E}_{vit\_inter}$ is extracted midway through the ViT, the $K$-token Cross-Attention and MLP Verification execute concurrently with the deeper ViT layers. By the time the LLM initiates the prompt prefill, $\mathcal{P}_{hybrid}$ is already fully resolved, introducing zero sequential stall time to the overarching pipeline while replacing costly token bloat with an ultra-compact continuous prior.