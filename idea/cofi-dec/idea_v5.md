# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a training-free framework designed to mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Bypassing the fatal latency and Out-of-Distribution (OOD) risks of real-time image cropping and blurring, CoFi-Dec operates entirely within the native attention mechanisms of modern Decoder-only architectures, ensuring zero additional visual encoder overhead and robust multi-granular reasoning.

### Problem Setting
We assume a standard Decoder-only LVLM parameterized by $\theta$. The model processes a high-resolution input image by extracting both macroscopic global tokens $v_{g}$ and high-resolution local patch tokens $v_{l}$. These are concatenated as a visual prefix $v = [v_{g}, v_{l}]$, followed by a textual query $x$. At decoding step $t$, the standard token distribution is generated autoregressively:
$$
P_t(y_t) = \text{Softmax}\big(f_\theta(y_t \mid [v_{g}, v_{l}], x, y_{<t})\big), \tag{1}
$$
where $y_{<t}$ denotes the previously generated textual context, and $f_\theta$ denotes the causal LLM forward pass over the vocabulary $\mathcal{V}$.

---

## 3.1 Attention-Masked Macroscopic Prior Construction

Prior methods suffer from catastrophic latency and memory explosion because they attempt to extract fine-grained details or coarse layouts by repeatedly feeding cropped or heavily blurred images through the Vision Transformer (ViT) during the generation phase. Furthermore, artificial Gaussian blurring injects severe OOD noise into the ViT, corrupting the feature space rather than isolating spatial layouts.

To eliminate these architectural violations, CoFi-Dec leverages the native multi-scale representations ($v_{g}$ and $v_{l}$) already extracted during the standard prefill phase. To isolate the model's reliance on macroscopic layout biases—which frequently induce stereotypic hallucinations (e.g., hallucinating a "mouse" purely because a "keyboard" is present globally)—we construct a **Macroscopic Prior Distribution** $P_t^{(macro)}$ purely through causal attention masking.

During the LLM's forward pass at step $t$, alongside the standard fully-attended computation, we compute an auxiliary branch where the cross-attention to the local fine-grained tokens $v_{l}$ is explicitly masked out (set to $-\infty$):
$$
P_t^{(macro)}(y_t) = \text{Softmax}\big(f_\theta(y_t \mid v_{g}, \text{Mask}(v_{l}), x, y_{<t})\big). \tag{2}
$$
By hiding the fine-grained evidence $v_{l}$ within the LLM's attention mechanism, $P_t^{(macro)}$ reflects the model's prediction driven purely by the low-resolution global view $v_{g}$ and textual priors $y_{<t}$. This approach guarantees **zero additional ViT forward passes**, operates strictly within the distribution of native features, and introduces negligible latency by batched execution within the LLM decoder.

---

## 3.2 Continuous Divergence Evaluation

Fixed attention-based triggers or predefined entropy thresholds are fundamentally fragile. Attention weights naturally drift towards textual syntax tokens during generation, causing inevitable false positives, while stereotypic hallucinations are frequently generated with extreme overconfidence (low entropy).

Instead of relying on heuristic discrete triggers, CoFi-Dec dynamically evaluates the necessity of fine-grained correction at every step $t$ using the continuous Jensen-Shannon Divergence (JSD) between the full-context distribution and the macroscopic prior:
$$
\alpha_t = \text{JSD}\left( P_t \parallel P_t^{(macro)} \right) = \frac{1}{2} D_{KL}(P_t \parallel M_t) + \frac{1}{2} D_{KL}(P_t^{(macro)} \parallel M_t), \tag{3}
$$
where $M_t = \frac{1}{2}(P_t + P_t^{(macro)})$. 

The scalar $\alpha_t \in [0, 1]$ acts as an intrinsic, parameter-free confidence calibrator. When predicting grammatical syntax or standard textual structures (e.g., "is", "the", "."), $P_t$ and $P_t^{(macro)}$ are nearly identical, resulting in $\alpha_t \approx 0$. Conversely, when the full visual evidence $v_l$ strongly contradicts the macroscopic prior $v_g$ (indicating critical fine-grained dependency or an active stereotypic hallucination), $\alpha_t$ dynamically scales up to intensify the subsequent correction mechanism.

---

## 3.3 KL-Regularized Contrastive Blending

Standard contrastive decoding simply subtracts logits and relies on arbitrary Top-K truncations (e.g., Adaptive Plausibility Constraints) to prevent the catastrophic degradation of language fluency. Such truncation inherently disrupts the vocabulary distribution and risks permanently discarding correct, long-tail semantic tokens.

To execute a robust, mathematically grounded multi-granular fusion, we formulate the decoding objective as a KL-regularized preference optimization. We aim to find an optimal distribution $\tilde{P}_t$ that maximizes the probability of tokens supported by the fine-grained full context ($P_t$), penalizes tokens driven by the macroscopic bias ($P_t^{(macro)}$), while being strictly constrained by a KL-divergence penalty to $P_t$ to guarantee syntactic fluency:
$$
\tilde{P}_t = \arg\max_{Q \in \Delta^{|\mathcal{V}|}} \mathbb{E}_{w \sim Q} \left[ \log \frac{P_t(w)}{P_t^{(macro)}(w)} \right] - \frac{1}{\beta \cdot \alpha_t} D_{KL}(Q \parallel P_t), \tag{4}
$$
where $\beta > 0$ is a global scaling factor, and $\alpha_t$ is the dynamic divergence calibrator computed in Equation 3.

This constrained optimization problem yields a closed-form analytical solution over the entire vocabulary $\mathcal{V}$:
$$
\tilde{P}_t(w) \propto P_t(w) \exp\left( \beta \cdot \alpha_t \cdot \big[ \log P_t(w) - \log P_t^{(macro)}(w) \big] \right). \tag{5}
$$
Equivalently, in the logit space, the final adjusted logit for every token $w$ is defined as:
$$
\tilde{\ell}_t(w) = \ell_t(w) + \beta \cdot \alpha_t \cdot \big( \ell_t(w) - \ell_t^{(macro)}(w) \big), \tag{6}
$$
where $\ell_t$ and $\ell_t^{(macro)}$ represent the log-probabilities of $P_t$ and $P_t^{(macro)}$.

**Self-Preserving Fluency Mechanism:** The term $\big( \ell_t(w) - \ell_t^{(macro)}(w) \big)$ serves as a natural gradient for semantic validity. 
1. For **grammatical and stop words**, the difference is negligible ($\ell_t \approx \ell_t^{(macro)}$), and $\alpha_t \approx 0$. The adjustment becomes zero, naturally preserving fluency without any explicit vocabulary masking or thresholding.
2. For **hallucinated entities** driven by global bias, $\ell_t^{(macro)}(w)$ is exceptionally high compared to $\ell_t(w)$, yielding a strong negative penalty that suppresses the hallucination.
3. For **fine-grained details** that are only visible when attending to $v_l$, $\ell_t(w) > \ell_t^{(macro)}(w)$, actively boosting the token's probability.

The next token $y_t$ is finally sampled from $\text{Softmax}(\tilde{\ell}_t)$. By entirely eliminating intermediate ViT processing, leveraging continuous JSD calibration, and deriving a closed-form KL-regularized fusion, CoFi-Dec provides an elegant, latency-friendly, and highly robust solution to multi-granular hallucination mitigation.