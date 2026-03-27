# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a training-free inference framework designed to systematically mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Acknowledging the fundamental paradigm of contrastive decoding, CoFi-Dec moves beyond heuristic logit adjustments and computationally flawed feature-space distance metrics. Instead, it formulates hallucination mitigation as a parameter-free, asymmetric distributional correction that strictly preserves linguistic fluency and mitigates text-induced modality bias.

### Problem Setting
We assume a standard generic Decoder-only LVLM parameterized by $\theta$. The model processes an input image, extracts a sequence of visual tokens $v$, prepends them to a tokenized textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big), \tag{1}
$$
where $h_t^{(vl)} \in \mathbb{R}^d$ is the top-layer hidden state of the LLM given the context $[v, x, y_{<t}]$, and $f_{head}$ is the language modeling head projecting into the vocabulary space $\mathcal{V}$.

---

## 3.1 Textual Prior Isolation

Visual hallucinations typically occur when the LVLM becomes over-reliant on the textual context $y_{<t}$, allowing strong language co-occurrence priors to override the actual visual evidence $v$. Consistent with established contrastive decoding principles, to effectively penalize this textual inertia, we must first isolate it.

Alongside the standard full-modality generation branch, we construct a parallel **Text-Only Prior Branch** that explicitly omits the visual prefix $v$ from the attention context:
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big), \tag{2}
$$
and $h_t^{(text)}$ is derived entirely from the pure textual context $[x, y_{<t}]$. The distribution $P_t^{(text)}$ isolates the model's linguistic hallucination bias—what the model guesses should logically appear next without looking at the image. While this introduces an additional autoregressive step, the KV-cache footprint is significantly reduced compared to the full-modality branch, as the lengthy sequence of visual tokens is entirely discarded.

---

## 3.2 Distributional Modality Divergence Calibrator

Prior methods often attempt to quantify the modality gap by calculating cosine distances between intermediate hidden states ($h_t^{(vl)}$ and $h_t^{(text)}$). However, this introduces a fundamental feature misalignment: the attention receptive fields, sequence lengths, and positional encoding distributions are vastly different between the two sequences, injecting severe spatial noise into the distance metric.

To ensure rigorous mathematical alignment, CoFi-Dec measures modality divergence strictly within the fully aligned probability distribution space ($|\mathcal{V}|$). We quantify the dynamic reliance on visual information using **Information Entropy Reduction**. The Shannon entropy for a given distribution $P$ is denoted as $\mathcal{H}(P) = - \sum_{w \in \mathcal{V}} P(w) \log P(w)$. 

We define the parameter-free divergence calibrator $\alpha_t$ as the relative certainty gain provided by the visual modality:
$$
\alpha_t = \text{ReLU}\left( 1 - \frac{\mathcal{H}\big(P_t^{(vl)}\big)}{\mathcal{H}\big(P_t^{(text)}\big)} \right). \tag{3}
$$
**Physical Interpretation:** $\alpha_t \in [0, 1)$ serves as an intrinsic statistical trigger. When generating standard grammatical syntax (e.g., "the", "is"), the visual evidence provides negligible information gain, causing $\mathcal{H}\big(P_t^{(vl)}\big) \approx \mathcal{H}\big(P_t^{(text)}\big)$ and intrinsically shrinking $\alpha_t \to 0$. Conversely, when predicting semantic visual entities, the visual evidence sharply reduces prediction uncertainty compared to pure textual guessing. A significant drop in entropy triggers a higher $\alpha_t$, signaling the necessity for modality intervention.

---

## 3.3 Parameter-Free Asymmetric Contrastive Decoding

Traditional Contrastive Decoding heavily relies on symmetric logit subtraction ($\ell_t^{(vl)} - \ell_t^{(text)}$) paired with artificial hyperparameters. As mathematically recognized, symmetric subtraction violently suppresses high-frequency syntactical tokens (since their text and visual logits are both high), leading to vocabulary truncation and grammatically broken generation.

To systematically mitigate textual hallucinations without disrupting linguistic integrity or relying on grid-searched hyperparameters, we propose a **Parameter-Free Asymmetric Rectification** mechanism. Instead of uniformly modifying the entire logit space, we define the adjustment term strictly for tokens where textual hallucination threatens visual fidelity. The final calibrated logit $\tilde{\ell}_t(w)$ for each token $w \in \mathcal{V}$ is formulated as:
$$
\tilde{\ell}_t(w) = \ell_t^{(vl)}(w) - \alpha_t \cdot P_t^{(text)}(w) \cdot \text{ReLU}\big( \ell_t^{(text)}(w) - \ell_t^{(vl)}(w) \big). \tag{4}
$$
This theoretically rigorous formulation neutralizes hallucinations through three self-contained constraints:

1. **Strict Asymmetric Rectification ($\text{ReLU}$):** The penalty is exclusively activated if $\ell_t^{(text)}(w) > \ell_t^{(vl)}(w)$—a condition directly defining a text-dominant hallucination. For valid visual entities and grammatical stop words where $\ell_t^{(vl)}(w) \geq \ell_t^{(text)}(w)$, the penalty is strictly zero. This unilaterally preserves the structural integrity of the logit space and entirely prevents the negative infinity degradation of valid vocabulary.
2. **Confidence-Weighted Penalty ($P_t^{(text)}$):** The penalty magnitude is scaled by the raw probability of the token under the textual prior. This ensures that the intervention heavily targets "blindly confident" hallucinations driven by strong textual biases, while ignoring benign noise in the long tail.
3. **Hyperparameter Elimination:** By integrating the intrinsic entropy ratio $\alpha_t$ and the distributional probability $P_t^{(text)}$, the formulation entirely eliminates the need for arbitrary global scaling constants (e.g., $\gamma, \lambda$), ensuring immediate, out-of-the-box adaptability across different LVLM architectures.

The final token $y_t$ is sampled securely from $\text{Softmax}(\tilde{\ell}_t)$. By isolating true text priors, strictly evaluating divergence in the aligned informational space, and utilizing an asymmetric, self-scaling penalty, CoFi-Dec provides mathematically robust multi-granular hallucination mitigation with zero arbitrary tuning.