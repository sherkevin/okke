# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a robust, training-free inference framework designed to mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Addressing the critical theoretical and computational flaws of prior contrastive paradigms—specifically the dimension mismatch between probabilities and unnormalized logits, the scale-shift across varying context lengths, and the severe latency overhead of dual-branch decoding—CoFi-Dec reformulates hallucination mitigation as an information-theoretically grounded, single-forward probability rectification.

### Problem Setting
We assume a generic Decoder-only LVLM parameterized by $\theta$. The model processes an input image into visual tokens $v$, prepends them to a textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big), \tag{1}
$$
where $h_t^{(vl)}$ is the top-layer hidden state given the full context $[v, x, y_{<t}]$, and $f_{head}$ projects it into the vocabulary $\mathcal{V}$.

---

## 3.1 Single-Forward Full-Context Prior Extraction

Visual hallucinations predominantly arise when LVLMs over-rely on the textual context $y_{<t}$. Prior methods attempt to isolate this bias by running a parallel text-only branch over the sequence. However, truncating the textual history (e.g., using a short local window) destroys long-range semantic dependencies, while running a full second forward pass doubles computational latency and memory bandwidth.

To rigorously capture long-range semantic textual priors without introducing catastrophic computational bottlenecks, CoFi-Dec employs **Single-Forward 2D Attention Masking**. Instead of dual autoregressive passes, we append a parallel query token at step $t$ and manipulate the causal attention mask. The primary query attends to the full context $[v, x, y_{<t}]$, yielding $h_t^{(vl)}$. Simultaneously, the secondary query is masked to strictly block the visual prefix $v$, attending only to the full textual context $[x, y_{<t}]$, yielding $h_t^{(text)}$. 
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big). \tag{2}
$$
This mechanism extracts the complete, long-range linguistic prior $P_t^{(text)}$ by perfectly sharing the KV-cache of the textual tokens in a single Transformer forward pass, completely eliminating the $O(N)$ double-latency overhead.

---

## 3.2 Normalized Log-Ratio Divergence

A fundamental mathematical flaw in previous contrastive decoding is the direct comparison or subtraction of unnormalized logits ($\ell_t^{(vl)}$ and $\ell_t^{(text)}$). Because the full-modality and text-only queries process vastly different effective context lengths, their unnormalized logits exhibit inherent scale shifts and partition function discrepancies. Direct arithmetic on these logits is theoretically invalid.

To ensure strict mathematical alignment, CoFi-Dec maps all representations into the normalized log-probability space, which is inherently scale-invariant and invariant to sequence length differences. We define the normalized log-probabilities as $\hat{\ell}_t^{(vl)}(w) = \log P_t^{(vl)}(w)$ and $\hat{\ell}_t^{(text)}(w) = \log P_t^{(text)}(w)$. 

The hallucination risk is quantified using the pointwise log-likelihood ratio (equivalent to the log-domain divergence):
$$
\Delta_t(w) = \hat{\ell}_t^{(text)}(w) - \hat{\ell}_t^{(vl)}(w) = \log \frac{P_t^{(text)}(w)}{P_t^{(vl)}(w)}. \tag{3}
$$
This explicitly separates true textual hallucinations from benign syntactic subword completions. For necessary subword completions, both branches yield similar normalized probabilities, driving $\Delta_t(w) \approx 0$. A true textual hallucination occurs strictly when $\Delta_t(w) > 0$, indicating that the language prior assigns a disproportionately higher likelihood to the token than the visually-grounded context does.

---

## 3.3 Uncertainty-Gated Probability Rectification

To address visually-primed hallucinations (where ambiguous visual features collude with strong textual priors) without arbitrarily mixing distinct mathematical spaces, we introduce an **Uncertainty-Gated Probability Rectification** mechanism. 

Rather than hard-coded hyperparameters, we dynamically scale the intervention based on the LVLM's internal visual uncertainty. We define the visual uncertainty gate $\beta_t \in [0, 1]$ using the normalized Shannon entropy of the full-modality distribution:
$$
\beta_t = \frac{\mathcal{H}\big(P_t^{(vl)}\big)}{\log |\mathcal{V}|} = - \frac{1}{\log |\mathcal{V}|} \sum_{w \in \mathcal{V}} P_t^{(vl)}(w) \log P_t^{(vl)}(w). \tag{4}
$$
When the visual evidence is ambiguous or degraded (high entropy), $\beta_t$ increases, signaling a higher susceptibility to both text-only and visually-primed hallucinations.

The final calibrated log-probability $\tilde{\ell}_t(w)$ is formulated purely within the log-space to guarantee dimensional consistency:
$$
\tilde{\ell}_t(w) = \hat{\ell}_t^{(vl)}(w) - \beta_t \cdot \text{ReLU}\big( \Delta_t(w) \big). \tag{5}
$$
This unified formulation provides three theoretically rigorous properties:
1. **Dimensional Consistency:** All operations are conducted strictly in the normalized log-probability space, eliminating the mathematical mismatch of mixing probabilities with arbitrary logit scales.
2. **Asymmetric Preservation:** The ReLU activation ensures that structurally valid tokens supported by the visual context ($\Delta_t(w) \le 0$) are entirely untouched, perfectly preserving grammatical fluency and subword integrity.
3. **Adaptive Visual Awareness:** By scaling the pointwise log-ratio penalty with the global visual uncertainty $\beta_t$, the model intrinsically penalizes overconfident textual priors more heavily precisely when the visual grounding is weak or ambiguous, effectively mitigating visually-primed illusions.

The final token $y_t$ is sampled securely from $\text{Softmax}(\tilde{\ell}_t)$. By leveraging single-forward context extraction and mathematically rigorous log-probability rectification, CoFi-Dec provides an efficient, hyperparameter-free solution to multimodality bias.