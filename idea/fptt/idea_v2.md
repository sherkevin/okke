# 4 Methodology

In this section, we formulate the object hallucination problem in Large Vision-Language Models (LVLMs) and comprehensively introduce our proposed **PATCH** (Prior-Aware Token Conditioning Heuristic) framework. Specifically, we address the cross-modal semantic gap, the error propagation from external detectors, and the optimization strategies for robust, parameter-efficient fine-tuning.

## 4.1 Problem Formulation

LVLMs aim to generate coherent text responses by integrating continuous visual features with discrete textual semantics. The standard generation process for an input image $I$ and a text prompt $Q$ is modeled autoregressively:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
$$
where $y_{<t}$ denotes the token sequence preceding step $t$, and $\theta$ represents the LVLM parameters. 

A primary cause of object hallucinations is the misalignment between fine-grained visual evidence and the global semantic representation. While introducing external object-level priors (e.g., categories $C$ and bounding boxes $B$) can provide explicit grounding, naively concatenating these priors often leads to severe cascading errors. External detectors inevitably produce noisy predictions (missed detections or false positives). If directly forced into the prompt context, the LVLM blindly trusts these flawed priors, exacerbating hallucinations.

To address this, we reformulate the generation process to incorporate external priors $D = \text{Serialize}(\{(c_i, b_i)\}_{i=1}^K)$ through a noise-robust, dynamic verification bottleneck:
$$
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, \mathcal{T}_{dynamic}, D, Q, y_{<t}),
\tag{2}
$$
where $\mathcal{T}_{dynamic}$ acts as a dynamic bridging module that aligns the visual features of $I$ with the noisy textual priors $D$, conditionally verifying the existence of objects before text generation.

## 4.2 The PATCH Framework

To achieve robust instance-aware visual understanding without catastrophic forgetting or excessive computational overhead, we propose **PATCH**. Instead of using naive, static prompt tuning—which lacks image-awareness and suffers from shortcut learning—PATCH introduces an image-conditioned dynamic token generation mechanism coupled with a condition dropout training strategy. 

### 4.2.1 Image-Conditioned Dynamic Token Bridging

A fundamental paradox in standard prompt tuning for LVLMs is that static trainable parameters cannot adaptively align with highly variable image contents. To genuinely narrow the cross-modal feature representation gap, PATCH generates image-dependent virtual tokens. 

Let $\mathcal{E}_{img} \in \mathbb{R}^{L \times d}$ denote the patch-level visual features extracted from the frozen vision encoder, where $d$ is the hidden dimension. We introduce a small set of $n$ trainable static latent queries $\mathcal{Q} \in \mathbb{R}^{n \times d}$. The dynamic virtual tokens $\mathcal{T}_{dynamic} \in \mathbb{R}^{n \times d}$ are derived via a lightweight Cross-Attention mechanism:
$$
\mathcal{T}_{dynamic} = \text{CrossAttention}(\text{Query}=\mathcal{Q}, \text{Key}=\mathcal{E}_{img}, \text{Value}=\mathcal{E}_{img}).
\tag{3}
$$
By doing so, $\mathcal{T}_{dynamic}$ adaptively aggregates instance-specific visual evidence conditioned on the input image. These $n$ dynamic tokens are then prepended to the embedded sequence of the external prior $D$. 

Crucially, this design resolves the length-alignment dilemma. Regardless of whether the image contains $1$ or $K$ objects, the dynamic length of the detection prior $D$ is serialized into a single text sequence. The fixed $n$ dynamic tokens serve as a **global semantic bridge** and a noise-filtering bottleneck that evaluates the reliability of the serialized sequence $D$ against the visual evidence $\mathcal{E}_{img}$, thereby preventing the model from blindly mimicking the potentially flawed external detector.

### 4.2.2 Noise-Robust Training and Plug-and-Play Inference

To mitigate error propagation from off-the-shelf detectors (e.g., Grounding DINO) and prevent the LVLM from developing a rigid dependency on external priors (shortcut learning), we introduce a **Prior Condition Dropout** mechanism during the parameter-efficient fine-tuning phase.

During training, for each iteration, there is a probability $p_{drop}$ (empirically set to 0.15) that the external prior $D$ and the dynamic bridging tokens $\mathcal{T}_{dynamic}$ are entirely dropped from the input sequence. Under this condition, the model is optimized using only the standard input $(I, Q)$. 
This strategy serves two pivotal theoretical purposes:
1. **Regularization against Error Propagation:** By exposing the model to instances without explicit priors, it is forced to preserve and utilize its inherent visual reasoning capabilities rather than over-relying on $D$. This inherently regularizes the model to be skeptical of false positives in $D$.
2. **True Plug-and-Play Capability:** The condition dropout provides a rigorous theoretical foundation for flexible inference. At inference time, if no external detection information is provided or required by the downstream task, the framework can seamlessly revert to the standard LVLM pipeline without suffering the steep performance degradation typically associated with shortcut learning.

### 4.2.3 Parameter Efficiency

Throughout the training process, all original parameters $\theta$ of the LVLM (including the vision encoder, projection layer, and LLM backbone) remain strictly **frozen**. The only trainable parameters are the latent queries $\mathcal{Q}$ and the lightweight cross-attention weights. For a typical configuration ($n=20$, $d=4096$), the trainable parameter count is approximately $2.1\mathrm{M}$, accounting for less than **0.03%** of a standard 7B LVLM's total parameters. This guarantees high computational efficiency while significantly enhancing the model's resilience to object hallucinations.