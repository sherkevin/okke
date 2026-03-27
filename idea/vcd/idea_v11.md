# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$, taking a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. At time step $t$, the token $y_t$ is sampled from the conditional probability distribution:
$$
p_{\text{full}}(y_t) \equiv p_{\theta}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{full}}(y_t)\big). \tag{1}
$$
Object hallucinations during this process are widely attributed to an over-reliance on the language decoder's intrinsic statistical inertia, which can occasionally override fine-grained visual signals. Contrastive Decoding aims to mitigate this by isolating and penalizing the language-biased prior during generation. 

## 3.2 Efficient Prior Extraction via Low-Resolution Visual Conditioning
Previous contrastive decoding paradigms, such as Visual Contrastive Decoding (VCD), extract the prior by adding diffusion noise to the image. However, introducing noise or extreme patch masking can inject low-frequency artifacts or break spatial continuity, potentially shifting the features out-of-distribution (OOD) relative to the pre-training data. Furthermore, maintaining an auxiliary KV-cache sequence of identical length directly doubles the memory bandwidth and computational FLOPs, imposing severe system overhead.

To extract a reliable language-biased prior while mitigating both OOD shifts and computational costs, we employ **Low-Resolution Visual Conditioning**. Rather than masking or corrupting the image, we condition the auxiliary prior stream on a down-sampled version of the input image, denoted as $v_{\text{low}}$:
$$
p_{\text{prior}}(y_t) \equiv p_{\theta}(y_t \mid v_{\text{low}}, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{prior}}(y_t)\big). \tag{2}
$$
By feeding a low-resolution input (e.g., scaling the image to a fraction of its original dimensions), the model retains the global semantic anchor—preventing the text stream from collapsing into an OOD state when processing complex multimodal prefixes—but loses the high-frequency visual details required for fine-grained object identification. This naturally forces the model to rely more heavily on its language prior for specific entity predictions. 

Crucially, down-sampling the visual input quadratically reduces the number of visual tokens in the auxiliary stream (e.g., reducing the number of tokens from 576 to 144). This significantly decreases the KV-cache memory footprint and computational overhead of the prior extraction branch, making the parallel decoding process more feasible for standard inference systems.

## 3.3 Intermediate Attention-Guided Contrastive Decoding

### 3.3.1 Logit-Level Contrastive Calibration
To recalibrate the output distribution, we subtract the unnormalized prior logits from the full multimodal logits:
$$
\tilde{\operatorname{logit}}(y_t) = \operatorname{logit}_{\text{full}}(y_t) - \alpha_t \cdot \operatorname{logit}_{\text{prior}}(y_t), \tag{3}
$$
where $\alpha_t \ge 0$ is a dynamic penalty weight. This logit-level operation is performed before any non-linear transformations (e.g., Softmax) or structural truncation (e.g., Top-$p$/Top-$k$). 

### 3.3.2 Intermediate Attention Modulator
Applying a static penalty weight $\alpha_t$ across all tokens uniformly penalizes syntactically deterministic tokens (e.g., punctuation or functional words) where visual reliance is naturally low. To dynamically scale $\alpha_t$, we directly measure the model's internal structural reliance on the visual modality. 

Previous probing studies demonstrate that cross-modal alignment and deep semantic grounding primarily occur in the intermediate layers of the Transformer, whereas the final layers often specialize in vocabulary projection. Furthermore, because the sum of post-softmax attention weights over all visual tokens is constrained, computing a global average across hundreds of visual tokens dilutes the signal, leading to near-zero scores even when the model precisely localizes a specific patch.

To accurately capture visual grounding, we aggregate the **maximum** attention scores directed at visual tokens across a subset of intermediate layers $\mathcal{L}_{\text{mid}}$ (e.g., layers from $L/4$ to $3L/4$). Let $\mathbf{A}_{t, j}^{(l, h)}$ denote the attention weight from the target token at step $t$ to token $j$ in attention head $h$ of layer $l$. The **Visual Reliance Score** $\mathcal{S}_{\text{vis}}(t)$ is formulated as:
$$
\mathcal{S}_{\text{vis}}(t) = \frac{1}{|\mathcal{L}_{\text{mid}}| H} \sum_{l \in \mathcal{L}_{\text{mid}}} \sum_{h=1}^H \max_{j \in \mathcal{I}_{\text{vis}}} \mathbf{A}_{t, j}^{(l, h)}, \tag{4}
$$
where $H$ is the number of attention heads, and $\mathcal{I}_{\text{vis}}$ is the index set of the visual tokens. Using the maximum value over $j$ ensures that precise, localized visual attention on specific bounding boxes or regions yields a high score, without being averaged down by the large number of irrelevant background tokens.

We use this score to modulate the penalty weight dynamically:
$$
\alpha_t = \alpha_{\text{base}} \cdot \exp\left( - \frac{\mathcal{S}_{\text{vis}}(t)}{\tau} \right), \tag{5}
$$
where $\tau$ is a temperature hyperparameter. 
1. **High Visual Reliance Phase**: When the model strongly grounds its generation on specific visual patches, $\mathcal{S}_{\text{vis}}(t)$ is high, and $\alpha_t$ decays toward zero. This avoids penalizing correct visually-grounded entities.
2. **Language Inertia Phase**: When the intermediate layers fail to attend to any visual tokens, indicating that the prediction is primarily driven by the text prefix, $\mathcal{S}_{\text{vis}}(t)$ approaches zero. The penalty $\alpha_t$ then increases toward $\alpha_{\text{base}}$, suppressing the language prior.

The final token $y_t$ is sampled from the recalibrated probability distribution:
$$
p_{\text{final}}(y_t) = \operatorname{softmax}\big(\tilde{\operatorname{logit}}(y_t)\big), \tag{6}
$$
followed by standard sampling strategies. This formulation adapts the contrastive penalty based on internal multi-modal alignment signals rather than heuristic textual thresholds.