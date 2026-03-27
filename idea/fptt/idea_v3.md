# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the integration of spatial priors, and parameter-efficient optimization strategies for robust hallucination mitigation.

## 4.1 Problem Formulation

LVLMs generate text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the token sequence preceding step $t$, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While external object-level priors (e.g., categories $C$ and bounding boxes $B$ from an off-the-shelf detector) can provide explicit grounding, naively concatenating these priors often leads to severe cascading errors. External detectors inevitably produce noisy predictions (missed detections or false positives), which the LVLM may blindly trust if directly forced into the prompt context.

To address this, we reformulate the generation process to incorporate external priors $D$ through a spatial-aware dynamic bridging module:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \mathcal{T}_{dynamic}, D, Q, y_{<t}).
\tag{2}
$$
Here, $D$ is the serialized representation of the external priors, and $\mathcal{T}_{dynamic}$ serves as soft visual prompts that encode spatially-grounded visual evidence. Through the joint optimization of $\mathcal{T}_{dynamic}$ and the LLM's internal representations (via LoRA), the model is trained to evaluate the consistency between the external prior $D$ and the actual visual features, thereby mitigating the negative impact of detection noise.

## 4.2 The PATCH Framework

Inspired by architecture paradigms like Perceiver Resampler and Q-Former, PATCH introduces a spatially-aware token generation mechanism. To ensure sufficient model capacity and prevent underfitting during complex multi-modal alignment, PATCH integrates Low-Rank Adaptation (LoRA) within the LLM. Furthermore, a modality dropout strategy is employed to enhance robustness against detector noise.

### 4.2.1 Serialization of External Priors

To translate detection outputs into a format interpretable by the LLM, we define a deterministic serialization function for $D$. Given a set of $K$ detected objects $\{(c_i, b_i)\}_{i=1}^K$, where $c_i$ is the class name and $b_i = (x_{1}, y_{1}, x_{2}, y_{2})$ represents the normalized bounding box coordinates, $D$ is serialized as:
$$
D = \text{Concat}_{i=1}^K \big( \text{"[OBJ]"} \oplus c_i \oplus \text{"[LOC]"} \oplus \text{str}(b_i) \big),
\tag{3}
$$
where `[OBJ]` and `[LOC]` are special indicator tokens added to the LLM's vocabulary. This structured representation allows the LLM to explicitly parse the semantic and spatial constraints provided by the external detector.

### 4.2.2 Spatially-Aware Dynamic Bridging Mechanism

To capture instance-level visual evidence and align it with the spatially-grounded prior $D$, we generate image-dependent virtual tokens $\mathcal{T}_{dynamic}$. Let $\mathcal{E}_{img} \in \mathbb{R}^{L \times d}$ denote the patch-level visual features extracted from the frozen vision encoder. Since object hallucination is highly correlated with spatial localization, we explicitly inject 2D Positional Encoding ($PE_{2D}$) into the visual features:
$$
\mathcal{E}_{img}^{pos} = \mathcal{E}_{img} + PE_{2D}.
\tag{4}
$$

We introduce a set of $n$ trainable static latent queries $\mathcal{Q} \in \mathbb{R}^{n \times d}$. The dynamic virtual tokens $\mathcal{T}_{dynamic} \in \mathbb{R}^{n \times d}$ are derived via a lightweight Cross-Attention layer:
$$
\mathcal{T}_{dynamic} = \text{CrossAttention}(\text{Query}=\mathcal{Q}, \text{Key}=\mathcal{E}_{img}^{pos}, \text{Value}=\mathcal{E}_{img}^{pos}).
\tag{5}
$$
These $n$ spatial-aware dynamic tokens are prepended to the embedded sequence of $D$. To ensure the model possesses the capacity to genuinely filter noise in $D$ based on $\mathcal{T}_{dynamic}$, we apply LoRA to the attention projection matrices ($W_q, W_k, W_v, W_o$) of the frozen LLM. The synergy between the spatial-aware $\mathcal{T}_{dynamic}$ and the tunable LoRA parameters empowers the LLM's self-attention layers to dynamically weigh the reliability of the external prior $D$ against the actual visual evidence.

### 4.2.3 Modality Dropout for Noise Robustness

To mitigate error propagation from off-the-shelf detectors and prevent the LVLM from developing a shortcut dependency on $D$, we apply a Prior Modality Dropout strategy during training. 

During each training iteration, there is a probability $p_{drop}$ (set to 0.5 to balance distributions) that the external prior $D$ is replaced with a standard empty prompt, effectively dropping the explicit detection guidance. This forces the LVLM to continuously rely on $\mathcal{T}_{dynamic}$ and the inherent visual features $\mathcal{E}_{img}$ for grounding, rather than blindly copying $D$. Consequently, this acts as a regularizer, training the model to be skeptical of false positives within $D$. At inference time, this strategy allows the model to perform prior-adaptive inference: it can leverage $D$ when available, or fall back smoothly to the standard vision-language generation pipeline without suffering severe distribution shifts.

## 4.3 Optimization Objective

The entire PATCH framework is trained end-to-end (with the vision encoder and LLM base parameters strictly frozen) using the standard autoregressive language modeling objective. The loss function is defined as the negative log-likelihood of the ground-truth target sequence $Y$:
$$
\mathcal{L} = - \sum_{t=1}^{T} \log p_\theta \big( y_t \mid I, \mathcal{T}_{dynamic}, \tilde{D}, Q, y_{<t} \big),
\tag{6}
$$
where $\tilde{D}$ represents the external prior after the Modality Dropout operation, and $\theta$ encapsulates the trainable parameters: the latent queries $\mathcal{Q}$, the Cross-Attention weights, and the LLM LoRA matrices. This combined parameter space accounts for approximately 25M trainable parameters (depending on the backbone size), providing sufficient capacity to learn complex cross-modal alignments while maintaining training efficiency.