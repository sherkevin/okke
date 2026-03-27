# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$. The model takes a textual query $x$ and a visual input $v$ as input, where $v$ provides contextual visual information to ground the generation of a relevant response $y$. The response $y$ is sampled auto-regressively from the probability distribution conditioned on the query $x$ and the visual context $v$. Mathematically, this is formulated as:
$$
\begin{align*}
y_t &\sim p_\theta(y_t \mid v, x, y_{<t}), \\
&\propto \exp\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
\end{align*}
$$
where $y_t$ denotes the token at time step $t$, and $y_{<t}$ represents the sequence of generated tokens up to time step $t-1$.

In the decoding phase of LVLMs, object hallucinations often emerge as a symptom of *cross-modal misalignment*, where the language decoder over-relies on its intrinsic statistical biases rather than faithfully anchoring to the visual evidence. To address this, our approach seeks to decouple the visual-driven predictions from the prior-driven biases during inference, directly contrasting the outputs to isolate the true visual contribution without requiring model retraining.

## 3.2 In-Distribution Modality Probing via Feature Masking
To accurately penalize language priors without harming visually grounded predictions, we must isolate the effect of these priors. Previous attempts utilizing Gaussian noise often push visual embeddings out-of-distribution (OOD), triggering unpredictable model behaviors rather than extracting pure language priors. 

To ensure the perturbed visual input remains within the natural manifold of the vision encoder, we introduce a simple yet effective in-distribution modality probe via **Spatial Feature Masking**. Instead of mathematical noise injection, we selectively drop a large proportion of fine-grained visual patch features:
$$
v' = M \odot v, \tag{2}
$$
where $M$ is a binary mask tensor that randomly zeroes out $90\%$ of the spatial visual tokens, and $\odot$ denotes element-wise multiplication. 

This severe degradation effectively strips away fine-grained visual evidence while keeping the remaining features well within the representation space the LVLM was trained on. Conditioned on this impoverished input $v'$, the LVLM is deprived of reliable detailed visual cues. Consequently, the predictions generated under $p_\theta(y \mid v', x)$ are forced to heavily rely on language priors and dataset statistical biases. Tokens that maintain anomalously high probabilities under $v'$ (e.g., predicting "yellow" for a masked banana) explicitly expose themselves as hallucination-prone concepts driven by linguistic inertia rather than visual facts.

## 3.3 Efficient Visual Contrastive Decoding (EVCD)
### 3.3.1 Cross-Modal Contrastive Calibration
Building upon the isolated prior distribution, we introduce **Visual Contrastive Decoding (VCD)** to calibrate the output distribution. VCD dynamically adjusts token probabilities by penalizing logits that stem primarily from language priors rather than visual evidence:
$$
p_{\text{VCD}}(y \mid v, v', x) = \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y \mid v, x) - \alpha\operatorname{logit}_\theta(y \mid v', x)\big], \tag{3}
$$
where $\alpha \geq 0$ serves as the contrastive strength parameter. 

For visually faithful tokens, the probability under the pristine image $v$ significantly outweighs that under $v'$, and Equation (3) amplifies their prominence. Crucially, for high-confidence hallucinated tokens—which typically exhibit high logits under *both* $v$ and $v'$ due to their heavy reliance on strong language priors—the subtraction effectively neutralizes their dominance, allowing genuinely visually-grounded tokens with lower prior probabilities to surface.

### 3.3.2 Selective Computation for Acceptable Overhead
A direct application of dual forward passes at every decoding step would double the computational latency, rendering the method impractical for large-scale LVLMs. However, hallucinations predominantly occur when generating concrete visual entities (e.g., nouns and adjectives) rather than functional or structural text. 

To eliminate the unacceptable computational overhead, we introduce a **Selective Computation Strategy**. We dynamically bypass the forward pass for $v'$ based on the entropy of the pristine distribution $p_\theta(y \mid v, x)$. The contrastive calculation is triggered only when the prediction uncertainty is high or when the top predicted tokens belong to visual semantic categories (identified via a lightweight part-of-speech classifier over the vocabulary). For structural tokens, we strictly reuse the original distribution $p_\theta(y \mid v, x)$. This selective triggering drastically reduces the additional forward passes, cutting the computational overhead from 100% to under 20% while preserving the core hallucination-mitigation benefits.

### 3.3.3 Adaptive Plausibility Masking
Unconstrained contrastive decoding can disrupt semantic coherence when logits of essential functional tokens are arbitrarily penalized. Furthermore, simple thresholding based on the maximum probability of the original distribution fails because hallucinated words often possess the highest confidence. 

To address this logical gap, we implement an **Adaptive Plausibility Mask** based on *relative visual dependence* rather than absolute confidence. We restrict the contrastive operation to a candidate set $\mathcal{V}_{\text{valid}}$, defined as:
$$
\begin{align*}
\mathcal{V}_{\text{valid}}(y_{<t}) &= \big\{y_t\in\mathcal{V}: p_\theta(y_t \mid v, x, y_{<t}) \ge \gamma \cdot p_\theta(y_t \mid v', x, y_{<t})\big\}, \\
p_{\text{VCD}}(y_t \mid v, v', x) &= 0, \quad \text{if } y_t \notin \mathcal{V}_{\text{valid}}(y_{<t}), \tag{4}
\end{align*}
$$
where $\gamma \in (0, 1]$ is a margin parameter. 

This condition enforces a strict constraint: a token is only considered valid if its probability under the pristine image is reasonably supported compared to its probability under the degraded image. Highly confident hallucinated tokens, which rely entirely on language priors (and thus have higher or equal probabilities under $v'$), will be aggressively filtered out of the candidate pool. The final calibrated decoding strategy operates subject to this logical constraint:
$$
\begin{align*}
y_t &\sim \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y_t \mid v, x, y_{<t}) - \alpha\operatorname{logit}_\theta(y_t \mid v', x, y_{<t})\big], \\
&\quad \text{subject to } y_t \in \mathcal{V}_{\text{valid}}(y_{<t}). \tag{5}
\end{align*}
$$
By ensuring that contrastive penalties are only applied to tokens demonstrating basic visual relevance, this formulation guarantees syntactic stability and directly resolves the paradox of high-confidence hallucinations escaping truncation.