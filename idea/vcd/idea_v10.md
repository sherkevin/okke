# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$, which takes a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. At time step $t$, the token $y_t$ is sampled from the full multimodal conditional distribution:
$$
p_{\text{full}}(y_t) \equiv p_{\theta}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{full}}(y_t)\big). \tag{1}
$$
Object hallucinations fundamentally stem from a dynamic imbalance during this auto-regressive decoding: the language decoder's intrinsic statistical inertia occasionally eclipses the fine-grained visual grounding signals. When this happens, the model defaults to high-frequency linguistic co-occurrences that may plausibly fit the text prefix but severely contradict the actual visual context. Correcting this requires explicitly isolating the language-biased prior and systematically down-weighting it.

## 3.2 Manifold-Preserving Prior Extraction via Visual Degradation
Previous contrastive decoding paradigms typically extract the language prior by feeding the generated multimodal prefix $y_{<t}$ into an auxiliary text-only stream ($v=\emptyset$). However, this "forced teaching" fundamentally violates the continuous semantic manifold of the KV-cache. If $y_{<t}$ contains strongly visually-grounded entities, forcing a purely text-conditioned stream to process this prefix confronts it with an incomprehensible context. This triggers a catastrophic out-of-distribution (OOD) collapse, resulting in chaotic and uninterpretable prior distributions.

To rigorously extract the language-biased prior while strictly preserving the continuous multimodal representation manifold, we introduce **Visual Degradation Prior Extraction**. Instead of entirely masking the visual input—which aggressively severs the cross-modal attention pathways—we condition the prior stream on a heavily degraded visual anchor, denoted as $v_{\text{deg}}$ (e.g., achieved via extreme Gaussian blurring or high-ratio patch masking). The prior distribution is thus formulated as:
$$
p_{\text{prior}}(y_t) \equiv p_{\theta}(y_t \mid v_{\text{deg}}, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{prior}}(y_t)\big). \tag{2}
$$
By substituting the high-frequency fine-grained visual features with a degraded global semantic anchor, the model is compelled to rely heavily on its language inertia and accumulated text prefix $y_{<t}$ to predict the next token, effectively exposing the language bias. Crucially, because $v_{\text{deg}}$ retains the identical sequence length and structural modality characteristics as $v$, the self-attention mechanisms and KV-cache updating function exactly as they do in the full model. This guarantees that $p_{\text{prior}}$ remains strictly in-distribution, completely resolving the OOD collapse and temporal discontinuity issues that plague pure-text prior extraction. Both streams can be continuously processed via parallel batched decoding, fully leveraging optimized hardware kernels.

## 3.3 Attention-Routed Continuous Modality Decoding

### 3.3.1 Logit-Level Contrastive Calibration
To recalibrate the output distribution, we adopt a mathematically rigorous logit-space penalty formulation. We subtract the unnormalized prior logits from the full multimodal logits:
$$
\tilde{\operatorname{logit}}(y_t) = \operatorname{logit}_{\text{full}}(y_t) - \alpha_t \cdot \operatorname{logit}_{\text{prior}}(y_t), \tag{3}
$$
where $\alpha_t \ge 0$ is a dynamic penalty weight. Performing this operation in the continuous logit space strictly prior to any non-linear transformations (e.g., Softmax) or heuristic truncation (e.g., Top-$p$/Top-$k$) ensures that visually grounded tokens, whose initial probabilities might be temporarily suppressed by the language inertia, are systematically rescued and elevated.

### 3.3.2 Attention-Driven Continuous Modulator
A static penalty weight $\alpha_t$ inevitably disrupts the generation of syntactically deterministic tokens (e.g., punctuation or functional words) where visual reliance is naturally minimal. Previous attempts to dynamically scale $\alpha_t$ using the Shannon entropy of the prior or inter-distribution divergence metrics conflate natural linguistic diversity (e.g., in open-ended reasoning) with OOD anomalies, leading to severe mathematical discontinuities and erratic generation behavior.

To establish a coherent and theoretically sound modulation mechanism, we bypass superficial output probability statistics entirely. Instead, we directly probe the model's internal structural reliance on the visual modality. At time step $t$, we compute the aggregated attention score mapped from the current query token to all visual tokens in the final Transformer layer $L$. Let $\mathbf{A}_{t, j}^{(h)}$ denote the post-softmax attention weight from the target token to token $j$ in attention head $h$. The **Visual Reliance Score** $\mathcal{S}_{\text{vis}}(t)$ is defined as:
$$
\mathcal{S}_{\text{vis}}(t) = \frac{1}{H} \sum_{h=1}^H \sum_{j \in \mathcal{I}_{\text{vis}}} \mathbf{A}_{t, j}^{(h)}, \tag{4}
$$
where $H$ is the total number of attention heads, and $\mathcal{I}_{\text{vis}}$ represents the index set of the visual tokens in the KV-cache. 

$\mathcal{S}_{\text{vis}}(t)$ serves as an exact, internal metric of cross-modal engagement. When the model actively grounds its prediction in the image, $\mathcal{S}_{\text{vis}}(t)$ is high. Conversely, when the model succumbs to language inertia (the primary catalyst for hallucination), the attention pathways detach from the visual tokens, causing $\mathcal{S}_{\text{vis}}(t)$ to plummet. 

We formulate the continuous, attention-driven dynamic penalty as:
$$
\alpha_t = \alpha_{\text{base}} \cdot \exp\left( - \frac{\mathcal{S}_{\text{vis}}(t)}{\tau} \right), \tag{5}
$$
where $\tau$ is a scaling hyperparameter. The mathematical properties of Eq. (5) are fundamentally elegant and robust:
1. **High Visual Reliance Phase**: When generating strongly visually-grounded entities, $\mathcal{S}_{\text{vis}}(t)$ is maximized, causing $\alpha_t$ to smoothly decay toward zero. This safely deactivates the penalty, preserving the integrity of the correct visual reasoning.
2. **Language Inertia Phase**: When generating hallucinated entities driven by text, the model ignores the image ($\mathcal{S}_{\text{vis}}(t) \to 0$). The penalty $\alpha_t$ exponentially peaks at $\alpha_{\text{base}}$, applying targeted and severe suppression to the extracted prior $\operatorname{logit}_{\text{prior}}(y_t)$.
3. **Continuous Smoothness**: Unlike heuristic binary gates or absolute top-1 matching indicators that introduce catastrophic discontinuities into the decoding space, Eq. (5) offers a strictly smooth, differentiable transition governed directly by the latent attention mechanism.

The final token $y_t$ is seamlessly sampled from the recalibrated probability distribution:
$$
p_{\text{final}}(y_t) = \operatorname{softmax}\big(\tilde{\operatorname{logit}}(y_t)\big), \tag{6}
$$
followed by standard structural pruning (Nucleus Sampling). By unifying manifold-preserving visual degradation with continuous attention-routed calibration, this framework provides a highly stable, interpretable, and theoretically cohesive solution to hallucination mitigation.