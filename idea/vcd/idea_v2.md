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

In the decoding phase of LVLMs, object hallucinations often emerge as a symptom of *cross-modal misalignment*, where probabilities are erroneously allocated to tokens that contradict the presented visual input $v$. Analytical studies attribute this to a modality imbalance: 
1. The dominance of statistical biases inherent in the vision-language alignment data (e.g., superficial object co-occurrences) [1,2,19];
2. The over-reliance on language priors entrenched within the large language models (LLMs) serving as decoders [22,38,69,75].

To address this, our approach seeks to decouple the visual-driven predictions from the prior-driven biases during inference, isolating and mitigating the exact source of hallucinations without requiring model retraining.

## 3.2 Modality Probing via Controlled Visual Degradation
To accurately penalize language priors without harming visually grounded predictions, we must first isolate the effect of these priors. We introduce controlled visual degradation as a *modality probe*. By purposefully injecting uncertainty into the visual input, we degrade the visual evidence, thereby forcing the model to fall back on its internal statistical and linguistic priors. 

### 3.2.1 Controlled Visual Degradation
Rather than treating visual noise as a heuristic perturbation, we formalize it within an information-theoretic degradation process. Following the forward diffusion process [24], the degraded visual representation is modeled as:
$$
\begin{align*}
q(v_t \mid v_{t-1}) &= \mathcal{N}\big(v_t; \sqrt{1-\gamma}\,v_{t-1},\ \gamma I\big), \\
q(v_T \mid v_0) &= \prod_{t=1}^T q(v_t \mid v_{t-1}), \tag{2}
\end{align*}
$$
where $v_0$ denotes the pristine visual input and $I$ refers to an identity matrix. By incrementally injecting Gaussian noise controlled by variance $\gamma$ over $T$ steps, the spatial and semantic fidelity of the original image $v_0$ is systematically eroded. The resulting degraded image $v' = v_T$ serves as an impoverished visual context that effectively neutralizes visual grounding.

### 3.2.2 Decoupling Priors from Visual Evidence
When conditioned on the degraded input $v'$, the LVLM is deprived of reliable visual evidence. Consequently, any high-confidence predictions generated under $p_\theta(y \mid v', x)$ are disproportionately driven by the language priors and training-set statistical biases. 

Importantly, this mechanism avoids the "anti-hallucination" paradox. If an object is genuinely present (e.g., a "yellow banana"), the logit under the pristine image $v$ will be significantly higher due to the synergistic combination of visual evidence and language prior. Under the degraded image $v'$, the logit drops because the visual evidence is removed, leaving only the prior. Thus, the differential between the pristine and degraded predictions intrinsically captures the *pure visual contribution*, successfully decoupling it from the prior. 

This degradation unmasks the latent biases: tokens that maintain anomalously high probabilities under $v'$ represent hallucination-prone concepts (e.g., predicting "yellow" even when a black banana is shown, solely based on linguistic inertia).

## 3.3 Visual Contrastive Decoding (VCD)
### 3.3.1 Cross-Modal Contrastive Calibration
Building upon the decoupled distributions, we introduce **Visual Contrastive Decoding (VCD)** as a training-free, plug-and-play inference-time calibration. VCD dynamically calibrates the output distribution by penalizing token probabilities that stem primarily from language priors rather than visual evidence.

Given a textual query $x$, the pristine visual input $v$, and the degraded visual input $v'$, VCD computes a calibrated probability distribution by contrasting the pristine logits against the prior-dominated logits:
$$
p_{\text{VCD}}(y \mid v, v', x) = \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y \mid v, x) - \alpha\operatorname{logit}_\theta(y \mid v', x)\big], \tag{3}
$$
where $\alpha \geq 0$ serves as the contrastive strength parameter. 

This formulation fundamentally acts as a probability regularizer. For visually faithful tokens, $\operatorname{logit}_\theta(y \mid v, x) \gg \operatorname{logit}_\theta(y \mid v', x)$, and Equation (3) amplifies their probabilities. Conversely, for hallucinated tokens heavily reliant on priors, $\operatorname{logit}_\theta(y \mid v, x) \approx \operatorname{logit}_\theta(y \mid v', x)$, and the subtraction effectively neutralizes their dominance. While VCD requires dual forward passes during autoregressive decoding—incurring additional computational overhead—it provides a highly accessible, zero-shot mechanism to drastically reduce hallucinations in off-the-shelf LVLMs without the prohibitive costs of RLHF or retraining.

### 3.3.2 Dynamic Language Manifold Preservation
A critical caveat of unconstrained contrastive decoding is the potential disruption of fundamental linguistic structures. LLM vocabularies consist not only of visual entities but also functional tokens (e.g., prepositions, conjunctions, sub-word structures) that *should* remain invariant to visual inputs. Indiscriminately penalizing the logits of these functional tokens based on Equation (3) can push the decoding trajectory off the natural language manifold, resulting in semantic incoherence or gibberish.

To ensure stability across diverse visual inputs and prevent the method from being overly sensitive to the contrastive hyperparameter $\alpha$, we introduce a **Dynamic Language Manifold Preservation** mechanism. Instead of applying a heuristic patch, we formulate a dynamic vocabulary mask that shields high-confidence, contextually indispensable tokens from contrastive distortion:
$$
\begin{align*}
\mathcal{V}_{\text{head}}(y_{<t}) &= \big\{y_t\in\mathcal{V}: p_\theta(y_t \mid v, x, y_{<t}) \geq \beta \max_{w} p_\theta(w \mid v, x, y_{<t})\big\}, \\
p_{\text{VCD}}(y_t \mid v, v', x) &= 0, \quad \text{if } y_t \notin \mathcal{V}_{\text{head}}(y_{<t}), \tag{4}
\end{align*}
$$
where $\mathcal{V}$ is the model output vocabulary, and $\beta \in [0,1]$ dictates the dynamic truncation threshold. This adaptive constraint evaluates the intrinsic confidence of the pristine distribution.

The final calibrated decoding strategy operates subject to this dynamic constraint:
$$
\begin{align*}
y_t &\sim \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y_t \mid v, x, y_{<t}) - \alpha\operatorname{logit}_\theta(y_t \mid v', x, y_{<t})\big], \\
&\quad \text{subject to } y_t \in \mathcal{V}_{\text{head}}(y_{<t}). \tag{5}
\end{align*}
$$
By localizing the contrastive operation exclusively within the dynamic candidate set $\mathcal{V}_{\text{head}}$, VCD effectively targets semantically competitive, hallucination-prone visual entities while preserving the syntactic integrity and fluency of the generated text. This ensures robustness across varying image complexities and effectively mitigates the parameter-sensitivity issues typically associated with contrastive interventions.