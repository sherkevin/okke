# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$. The model takes a textual query $x$ and a visual input $v$ as input, where $v$ provides contextual visual information to ground the generation of a relevant response $y$. The response $y$ is sampled auto-regressively from the probability distribution conditioned on the query $x$ and the visual context $v$:
$$
\begin{align*}
y_t &\sim p_\theta(y_t \mid v, x, y_{<t}), \\
&\propto \exp\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
\end{align*}
$$
where $y_t$ denotes the token at time step $t$, and $y_{<t}$ represents the sequence of generated tokens up to time step $t-1$.

In the decoding phase of LVLMs, object hallucinations fundamentally emerge from a modality imbalance: the language decoder over-relies on its intrinsic statistical biases rather than faithfully anchoring to the provided visual evidence. To mitigate this without model retraining, we must decouple the pure visual contribution from language priors during inference. 

## 3.2 In-Distribution Modality Probing via Global Context Substitution
To accurately penalize language priors, we need a "prior probe" that isolates these biases. Previous attempts using Gaussian noise or random token masking severely disrupt the continuity of positional embeddings and global attention, pushing the visual features out-of-distribution (OOD) and triggering unpredictable model behaviors.

To guarantee that the perturbed visual input remains strictly within the well-trained natural manifold of the vision encoder, we introduce **Global Context Substitution (GCS)**. Instead of arbitrary destructive masking, we construct the degraded visual context $v'$ by substituting all fine-grained spatial patch features with the vision encoder's intrinsic global summary token (e.g., the `[CLS]` token or global average pooled representation):
$$
v' = \{v_{\text{global}}, v_{\text{global}}, \dots, v_{\text{global}}\}. \tag{2}
$$
Because $v_{\text{global}}$ is a naturally occurring, valid representation optimized during the LVLM's pre-training, substituting patches with it avoids any OOD mathematical perturbation. This process completely strips away localized, fine-grained visual evidence while maintaining macroscopic semantic coherence. Conditioned on $v'$, the LVLM is deprived of spatial details, forcing it to generate high-confidence predictions based entirely on inherent language priors and dataset statistics.

## 3.3 Attention-Guided Distribution Calibration (AGDC)

### 3.3.1 Zero-Overhead Dynamic Triggering
Computing dual forward passes (for $v$ and $v'$) at every autoregressive step incurs unacceptable latency. Furthermore, previous heuristic triggers (like generation entropy) fail conceptually, as hallucinations are notoriously generated with extreme confidence (low entropy). 

To achieve genuine efficiency without logical paradoxes, we introduce an **Attention-driven Triggering** mechanism that leverages the model's internal cross-modal dynamics. At step $t$, we extract the self-attention weights from the last layer of the LLM decoder. Let $w_t^{\text{vis}}$ denote the normalized sum of attention weights the current token pays to all visual tokens. We trigger the secondary forward pass for $v'$ only when the visual attention falls below the sequence's moving average:
$$
\text{Trigger}(t) = \begin{cases} 
\text{True}, & \text{if } w_t^{\text{vis}} < \frac{1}{t-1} \sum_{i=1}^{t-1} w_i^{\text{vis}} \\
\text{False}, & \text{otherwise}
\end{cases} \tag{3}
$$
This zero-overhead check naturally identifies moments when the model starts neglecting the image and heavily relying on linguistic context (e.g., transitioning from describing visual facts to hallucinating extensions). When triggered, the calibration is applied; otherwise, the original prediction $p_\theta(y_t \mid v, x, y_{<t})$ is directly adopted.

### 3.3.2 Strict Logit Differential Constraint
To prevent the disruption of essential functional tokens and systematically eliminate high-confidence hallucinations, we must robustly identify candidates valid for visual calibration. Heuristic probability ratios (e.g., $p(v) \ge \gamma \cdot p(v')$) are mathematically flawed: an extreme hallucination purely driven by language prior might yield $p(v)=0.9$ and $p(v')=0.95$, easily surviving a threshold like $\gamma=0.5$.

We replace flawed ratio thresholds with a **Strict Logit Differential Constraint**. A token is considered visually grounded if and only if the presence of fine-grained visual evidence strictly increases its unnormalized confidence compared to the global-only probe. We define the valid vocabulary subset as:
$$
\mathcal{V}_{\text{valid}}(y_{<t}) = \big\{ y_t \in \mathcal{V} \mid \operatorname{logit}_\theta(y_t \mid v, x, y_{<t}) - \operatorname{logit}_\theta(y_t \mid v', x, y_{<t}) > 0 \big\}. \tag{4}
$$
This parameter-free constraint acts as an absolute filter. Returning to the previous paradox: for the high-confidence hallucination where $\operatorname{logit}(v') > \operatorname{logit}(v)$ due to dominant language inertia, the differential is negative. Thus, the hallucinated token is ruthlessly zeroed out, regardless of its absolute confidence level.

### 3.3.3 Distribution Calibration
For the tokens surviving the rigorous filter $\mathcal{V}_{\text{valid}}$, we calibrate the final decoding distribution by explicitly penalizing the isolated language prior:
$$
\begin{align*}
p_{\text{AGDC}}(y_t) &= \operatorname{softmax}\big( \operatorname{logit}_\theta(y_t \mid v, x, y_{<t}) - \alpha \operatorname{logit}_\theta(y_t \mid v', x, y_{<t}) \big), \\
&\quad \text{subject to } y_t \in \mathcal{V}_{\text{valid}}(y_{<t}), \tag{5}
\end{align*}
$$
where $\alpha \ge 0$ modulates the penalty strength. 

By grounding the penalty mechanism strictly inside mathematically bounded candidate spaces and triggering it natively via internal attention shifts, AGDC guarantees robust hallucination suppression. It successfully decouples cross-modal priors while eliminating the prohibitive parameter-tuning and latency overheads that paralyze conventional contrastive strategies.