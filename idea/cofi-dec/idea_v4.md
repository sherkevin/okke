# 3 Methodology

This paper introduces **CoFi-Dec**, a training-free decoding framework designed to improve the reliability of Large Vision-Language Models (LVLMs) and mitigate visual hallucinations. We achieve this through dynamic multi-granular visual conditioning, an attention-guided hallucination trigger, and adaptive contrastive decoding. 

### Problem Setting
We consider an LVLM parameterized by $\theta$, which takes a sequence of visual tokens $v$ and a tokenized textual query $x$, and generates a textual response sequence $y$ autoregressively. The raw input image $I_0$ is processed by a vision encoder and mapped into the language embedding space. Formally, the token distribution at decoding step $t$ is:
$$
y_t \sim p_\theta(y_t \mid v, x, y_{<t}) = \text{Softmax}\big(f_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
$$
where $y_{<t}$ denotes the preceding textual context, and $f_\theta$ outputs the unnormalized logits over the entire vocabulary $\mathcal{V}$.

---

## 3.1 Dynamic Multi-Granular Visual Conditioning

To address the loss of fine-grained details without suffering from static pre-cropping assumptions or catastrophic memory overhead (e.g., maintaining multiple parallel KV-caches), CoFi-Dec generates visual features dynamically during the autoregressive process.

Given the original image $I_0$, we define two auxiliary views:
- **Coarse-grained View ($I_c$)**: A heavily Gaussian-blurred version of $I_0$. This view preserves the macro spatial layout but actively suppresses fine-grained visual evidence. It serves as a diagnostic tool to expose the model's reliance on language priors and stereotypic layout biases.
- **Dynamic Fine-grained View ($I_f^{(t)}$)**: Instead of assuming salient regions statically from the initial query, $I_f^{(t)}$ is generated on-the-fly *only* when required. At step $t$, we project the current cross-attention weights between the generated sequence $y_{<t}$ and the global visual tokens $v_0$ back to the original image space. This isolates the exact local patch the model is actively trying to describe.

To strictly minimize memory overhead, the text KV-cache for $y_{<t}$ is shared. The auxiliary visual token sequences $v_c$ and $v_f^{(t)}$ are only computed and swapped into the topmost cross-attention layers conditionally, completely avoiding the $O(3\times)$ memory explosion associated with parallel sequence caching.

---

## 3.2 Attention-Guided Hallucination Trigger

Standard distributional divergence metrics often fail to detect hallucinations when the model confidently predicts the same stereotypic token across both global and coarse views (e.g., predicting "mouse" next to a "keyboard" due to strong language bias).

To robustly trigger the correction mechanism, CoFi-Dec monitors the **Text-to-Vision Attention Dispersion**. At each decoding step $t$, we compute the standard probability distribution using the global view $v_0$:
$$
P_t^{(0)} = p_\theta(y_t \mid v_0, x, y_{<t}). \tag{2}
$$
Simultaneously, we measure the normalized cross-attention score allocated to the visual prefix versus the textual prefix. We define the visual grounding score $\mathcal{G}_t$ as the proportion of attention mass directed at $v_0$. 

If $\mathcal{G}_t \geq \gamma$ (where $\gamma$ is an empirical threshold), the model is well-grounded in the image, and we directly sample $y_t \sim P_t^{(0)}$. Conversely, if $\mathcal{G}_t < \gamma$, the model is heavily relying on the textual context $y_{<t}$ (a primary indicator of hallucination generation). Only then do we trigger the dynamic extraction of $I_f^{(t)}$ and compute the auxiliary logit distributions:
$$
\ell_t^{(c)} = f_\theta(y_t \mid v_c, x, y_{<t}), \quad \ell_t^{(f)} = f_\theta(y_t \mid v_f^{(t)}, x, y_{<t}). \tag{3}
$$

---

## 3.3 Adaptive Multi-Granular Contrastive Decoding

To integrate multi-granular evidence robustly across the discrete vocabulary without suffering from arbitrary Top-K truncation or terminology inflation, we formulate the fusion as a Multi-Granular Contrastive Decoding (CD) objective over the full vocabulary $\mathcal{V}$.

Let $\ell_t^{(0)} = f_\theta(y_t \mid v_0, x, y_{<t})$ be the global logits. We construct the fused contrastive logits as:
$$
\ell_t^{(\text{fused})} = \ell_t^{(0)} + \lambda_f \ell_t^{(f)} - \lambda_c \ell_t^{(c)}, \tag{4}
$$
where $\lambda_f$ and $\lambda_c$ control the reward for fine-grained alignment and the penalty for coarse-grained layout bias, respectively.

### Adaptive Plausibility Constraint to Preserve Fluency
A known critical flaw of naive contrastive decoding is fluency degradation: indiscriminately penalizing high-probability tokens in the coarse view ($\ell_t^{(c)}$) often suppresses necessary syntactic words (e.g., "is", "the", ","), leading to broken grammar. 

To address this, we introduce an **Adaptive Plausibility Constraint**. We construct a dynamic plausible mask $\mathcal{M}_t \in \{0, 1\}^{|\mathcal{V}|}$ based on the global distribution $P_t^{(0)}$. A token $w$ is considered plausible only if its probability is sufficiently close to the most likely token:
$$
\mathcal{M}_t(w) = \begin{cases} 
1, & \text{if } P_t^{(0)}(w) \geq \alpha \max_{v \in \mathcal{V}} P_t^{(0)}(v) \\
0, & \text{otherwise}
\end{cases} \tag{5}
$$
where $\alpha \in (0,1)$ is a strict truncation hyperparameter (e.g., $\alpha=0.1$). 

The contrastive modification is strictly confined to the plausible set. For any token $w \notin \mathcal{M}_t$, we set $\ell_t^{(\text{fused})}(w) = -\infty$. 
This continuous, full-vocabulary logit adjustment ensures that correct but poorly ranked fine-grained semantic words are not prematurely discarded by fixed Top-K sets, while the plausibility constraint rigorously protects the foundational grammatical structure of the LVLM's output. The final token $y_t$ is sampled from $\text{Softmax}(\ell_t^{(\text{fused})})$.