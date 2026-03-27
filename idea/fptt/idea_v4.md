# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the fine-grained alignment of spatial priors, the active mitigation of detector noise, and the inference efficiency.

## 4.1 Problem Formulation

LVLMs generate text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the token sequence preceding step $t$, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While external object-level priors from an off-the-shelf detector can provide explicit grounding, naively concatenating these priors leads to severe cascading errors. External detectors inevitably produce noisy predictions (missed detections or false positives). Without an explicit noise-rectification mechanism, the LVLM blindly trusts these forced priors, exacerbating text-bias and hallucinations.

To address this, we reformulate the generation process to incorporate external priors $D$ through a prior-guided spatial bridging module and a dual-objective verification process:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \mathcal{T}_{dynamic}, D, Q, y_{<t}).
\tag{2}
$$
Here, $\mathcal{T}_{dynamic}$ serves as fine-grained visual prompts actively conditioned on the spatial layout of $D$. Through joint optimization with an explicit alignment loss and Low-Rank Adaptation (LoRA) within the LLM, the model learns to critically evaluate the consistency between the external prior $D$ and the actual visual features, thereby actively filtering detection noise.

## 4.2 The PATCH Framework

To achieve robust instance-aware visual understanding, PATCH introduces a prior-guided token generation mechanism combined with an active noise injection training strategy.

### 4.2.1 Confidence-Aware Prior Serialization and Pruning

To translate detection outputs into a format interpretable by the LLM without overwhelming the context window, we define a confidence-aware deterministic serialization function for $D$. Given a set of detected objects, we first apply a Confidence-based Top-$K$ Pruning strategy. This restricts the maximum number of explicitly incorporated objects to $K$ (e.g., $K=20$), effectively preventing catastrophic attention dilution and memory overflow in scenes with dense objects. 

For the pruned set of $K$ objects $\{(c_i, b_i, s_i)\}_{i=1}^K$, where $c_i$ is the class name, $b_i = (x_{1}, y_{1}, x_{2}, y_{2})$ represents the normalized bounding box coordinates, and $s_i \in [0, 1]$ is the detector's confidence score, $D$ is serialized as:
$$
D = \text{Concat}_{i=1}^K \big( \text{"[OBJ]"} \oplus c_i \oplus \text{"[LOC]"} \oplus \text{str}(b_i) \oplus \text{"[CONF]"} \oplus \text{str}(s_i) \big).
\tag{3}
$$
Crucially, preserving the confidence score $s_i$ provides the LLM with an essential initial heuristic to estimate the reliability of the prior, avoiding the logical regression of forcing the LLM to deduce noise levels from scratch.

### 4.2.2 Prior-Guided Fine-Grained Spatial Interaction

Standard 1D virtual tokens fail to align with explicit floating-point coordinates. To achieve genuine spatial awareness, we design $\mathcal{T}_{dynamic}$ to be directly modulated by the spatial layout of the priors. 

Let $\mathcal{E}_{img}^{pos} \in \mathbb{R}^{L \times d}$ denote the patch-level visual features enriched with explicit 2D Positional Encoding ($PE_{2D}$). Instead of using arbitrary static latent queries, we generate spatially-conditioned queries $\mathcal{Q}_{cond} \in \mathbb{R}^{K \times d}$ based on the bounding box coordinates $B = \{b_i\}_{i=1}^K$ using a Multi-Layer Perceptron (MLP):
$$
\mathcal{Q}_{cond} = \text{MLP}(B).
\tag{4}
$$
The dynamic virtual tokens $\mathcal{T}_{dynamic} \in \mathbb{R}^{K \times d}$ are then derived via a lightweight Cross-Attention layer, explicitly querying the visual features based on the detector's spatial proposals:
$$
\mathcal{T}_{dynamic} = \text{CrossAttention}(\text{Query}=\mathcal{Q}_{cond}, \text{Key}=\mathcal{E}_{img}^{pos}, \text{Value}=\mathcal{E}_{img}^{pos}).
\tag{5}
$$
By using prior-guided queries, $\mathcal{T}_{dynamic}$ maintains a strict one-to-one spatial correspondence with the objects in $D$. These tokens are then prepended to the embedded sequence of $D$. To ensure sufficient capacity for noise filtration, LoRA is applied to the self-attention projection matrices of the frozen LLM.

### 4.2.3 Prior Noise Injection and Perturbation Strategy

Simply dropping priors (Modality Dropout) only teaches the model to function without them, but fails to teach it how to rectify false positives. To intrinsically cultivate robustness against detector hallucinations, we introduce an **Active Noise Injection** strategy.

During training, for a subset of instances in $D$, we synthetically inject two types of noise with a probability $p_{noise}$ (set to 0.3):
1. **Semantic Perturbation**: Replacing the ground-truth class $c_i$ with a randomly sampled incorrect category from the vocabulary (simulating false classifications).
2. **Spatial Perturbation**: Adding Gaussian noise $\mathcal{N}(0, \sigma^2)$ to the coordinates $b_i$ or shifting the bounding boxes to background regions (simulating false localization).
This strategy forces the LLM to actively utilize $\mathcal{T}_{dynamic}$ to cross-check the claims made in $D$, fundamentally preventing the model from developing a textual shortcut dependency on external priors.

## 4.3 Dual Optimization Objectives

Relying solely on standard autoregressive loss is insufficient to guarantee that $\mathcal{T}_{dynamic}$ learns to discriminate between true and false positives. Therefore, we optimize the PATCH framework using a dual-objective loss. 

The first component is the standard negative log-likelihood for text generation:
$$
\mathcal{L}_{gen} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \mathcal{T}_{dynamic}, \tilde{D}, Q, y_{<t} \big),
\tag{6}
$$
where $\tilde{D}$ represents the external prior subjected to the Noise Injection strategy. 

To enforce explicit cross-modal verification, we introduce an Auxiliary Alignment Loss $\mathcal{L}_{align}$ based on contrastive learning. It pulls the prior-guided visual tokens $\mathcal{T}_{dynamic}$ closer to the textual embeddings of unperturbed (true) priors and pushes them away from the injected noisy (false) priors in the latent space:
$$
\mathcal{L}_{align} = - \log \frac{\exp(\text{sim}(\mathcal{T}_{dynamic}^+, E_{true}) / \tau)}{\sum \exp(\text{sim}(\mathcal{T}_{dynamic}^-, E_{false}) / \tau)},
\tag{7}
$$
where $\tau$ is a temperature hyperparameter. The total optimization objective is defined as:
$$
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{align},
\tag{8}
$$
with $\lambda$ controlling the verification strength. All base parameters of the vision encoder and LLM remain strictly frozen, optimizing only the MLP, Cross-Attention weights, and LLM LoRA matrices (approx. 25M trainable parameters).

## 4.4 Inference Efficiency and Practicality

While PATCH utilizes an external detector, the inference overhead remains highly manageable. The serialized prior $D$ is strictly limited to a maximum length via the Confidence-based Top-$K$ pruning, ensuring the LLM's context window is not saturated and preventing quadratic complexity surges in the self-attention mechanism. Furthermore, lightweight detectors (e.g., YOLOv8) incur a negligible latency (typically under 30ms per image), which is marginally fractional compared to the standard autoregressive decoding time of a 7B-parameter LVLM. The addition of LoRA parameters is absorbed into the base LLM weights during deployment, guaranteeing that PATCH introduces virtually zero computational penalty to the LLM forward pass itself.