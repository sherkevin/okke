# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$. The model takes a textual query $x$ and a visual input $v$, generating a response $y$ auto-regressively. The token at time step $t$, denoted as $y_t$, is sampled from the probability distribution conditioned on the prompt and the visual context:
$$
\begin{align*}
y_t &\sim p_\theta(y_t \mid v, x, y_{<t}) \\
&\propto \exp\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big). \tag{1}
\end{align*}
$$
In the decoding phase, object hallucinations predominantly arise from a modality imbalance: the language decoder over-relies on its intrinsic statistical biases and robust language priors, thereby neglecting fine-grained visual evidence. To mitigate this without computationally expensive model retraining, we must cleanly decouple visually grounded predictions from prior-driven biases during inference.

## 3.2 In-Distribution Modality Probing via Macroscopic Visual Degradation
To accurately isolate and penalize language priors, an ideal "prior probe" must obscure fine-grained visual details while strictly preserving the continuous representation manifold. Previous attempts that manipulate feature tensors (e.g., dropping patches or uniformly injecting global tokens) inherently induce out-of-distribution (OOD) anomalies. These operations disrupt the continuity of positional encodings, trigger highly distorted attention distributions within the language decoder, and often inadvertently leak strong global semantics that lead to the over-penalization of genuinely salient objects.

To guarantee that the perturbed visual input remains strictly within the natural distribution of the vision encoder, we introduce **Macroscopic Visual Degradation (MVD)** at the raw pixel level. Instead of destructive tensor surgeries, we construct the degraded visual context $v'$ by applying extreme low-pass filtering (Gaussian blur) followed by drastic down-sampling to the original image $v$:
$$
v' = \operatorname{DownSample}\big(\operatorname{GaussianBlur}(v, \sigma), r\big), \tag{2}
$$
where $\sigma$ controls the blur radius and $r$ is the spatial reduction ratio. 
Because MVD operates purely in the pixel space prior to patching and embedding, the resulting $v'$ maps to a perfectly valid, continuous sequence of visual tokens that the LVLM inherently understands. This transformation eliminates high-frequency fine-grained visual evidence (the primary source of detailed visual grounding) while retaining the macroscopic layout and color blobs. Conditioned on $v'$, the LVLM is deprived of specific object details, forcing it to fall back on its language priors and statistical biases to guess the masked entities, thereby explicitly exposing its hallucination tendencies.

## 3.3 Variance-Guided Adaptive Contrastive Decoding (VACD)

### 3.3.1 Dual-Condition Dynamic Triggering
Computing dual forward passes (for both $v$ and $v'$) at every autoregressive step incurs substantial latency. Furthermore, naive triggering mechanisms—such as applying contrastive decoding purely based on low moving-average attention—fail catastrophically in practice. Functional and structural language tokens (e.g., "is", "the", "and") naturally attend very little to the image, leading to continuous false-positive triggers and unacceptable computational overhead.

To achieve genuine efficiency, we propose a **Dual-Condition Dynamic Triggering** mechanism that orthogonally combines cross-modal attention sharpness with output distribution uncertainty. We measure the sharpness of the visual attention using the Gini coefficient of the attention weights $w_{t}^{\text{vis}}$ that the current state pays to all visual tokens:
$$
\mathcal{G}(w_{t}^{\text{vis}}) = \frac{\sum_{i=1}^N \sum_{j=1}^N |w_{t, i}^{\text{vis}} - w_{t, j}^{\text{vis}}|}{2N \sum_{i=1}^N w_{t, i}^{\text{vis}}}. \tag{3}
$$
A high Gini coefficient indicates that the model is confidently focusing on specific visual patches (visually grounded), whereas a low coefficient signifies that attention is uniformly dispersed (indicating potential hallucination or generation of non-visual functional words). 

To disentangle hallucinations from harmless functional words (both of which exhibit low visual attention), we incorporate the Shannon entropy $\mathcal{H}(p_\theta)$ of the predicted vocabulary distribution. Functional words typically exhibit extremely low entropy (near deterministic prediction due to strong grammar rules), while hallucinations during entity generation often demonstrate relatively higher entropy. Thus, the secondary forward pass for $v'$ is triggered only when:
$$
\text{Trigger}(t) = \big( \mathcal{G}(w_{t}^{\text{vis}}) < \tau_g \big) \wedge \big( \mathcal{H}(p_\theta(y_t \mid v, x, y_{<t})) > \tau_h \big), \tag{4}
$$
where $\tau_g$ and $\tau_h$ are pre-defined thresholds. This mechanism elegantly bypasses functional tokens and highly confident structural generation, cutting computational overhead by over 75% while precisely intercepting moments of visual uncertainty and hallucination.

### 3.3.2 Semantic-Adaptive Distribution Calibration
When the calibration is triggered, directly applying absolute logit subtractions or rigid thresholds inevitably disrupts language fluency. Functional sub-words and high-frequency co-occurring entities exhibit inherently random logit fluctuations under varying contexts; a strict absolute cutoff ($\operatorname{logit}(v) \le \operatorname{logit}(v')$) often brutally zeroes out grammatically essential tokens or correctly guessed background objects, fragmenting the generated response.

To robustly penalize language priors without destroying structural coherence, we introduce a **Semantic-Adaptive Distribution Calibration** via a proportional penalty rather than a hard constraint. We define a dynamic modulation factor $\alpha(y_t)$ for each token in the top-$K$ candidate vocabulary:
$$
\alpha(y_t) = \alpha_{\text{base}} \cdot \max\left(0, \frac{\operatorname{logit}_\theta(y_t \mid v', x) - \mu_{v'}}{\sigma_{v'}}\right), \tag{5}
$$
where $\mu_{v'}$ and $\sigma_{v'}$ are the mean and standard deviation of the top-$K$ logits under the degraded context $v'$. 

This token-specific $\alpha(y_t)$ acts as an adaptive prior penalizer. If a token is heavily promoted by the language prior (i.e., its logit under $v'$ is exceptionally high relative to the rest of the vocabulary), it receives a proportionally stronger penalty. Conversely, functional structural tokens, which maintain average and stable logits regardless of visual input, receive an $\alpha(y_t)$ close to zero, effectively shielding them from contrastive distortion.

The final calibrated decoding distribution is formulated as:
$$
p_{\text{VACD}}(y_t) = \operatorname{softmax}\big( \operatorname{logit}_\theta(y_t \mid v, x, y_{<t}) - \alpha(y_t) \cdot \operatorname{logit}_\theta(y_t \mid v', x, y_{<t}) \big). \tag{6}
$$
By operating exclusively within high-probability candidate pools and scaling penalties according to internal prior anomalies, VACD fundamentally resolves the structural disruption and OOD paradigms that plague traditional contrastive decoding. It ensures that genuine visual entities are enhanced while grammatically essential tokens and language fluency remain perfectly intact.