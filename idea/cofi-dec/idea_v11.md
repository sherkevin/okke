# 3 Methodology

This paper presents **CoFi-Dec** (Contextual Filtering Decoding), an inference-time decoding strategy designed to mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Acknowledging the fundamental trade-offs in contrastive decoding paradigms, CoFi-Dec introduces a pragmatic, empirically grounded approach that leverages sequence-level log-probability adjustments to penalize overconfident textual priors while strictly preserving syntactic fluency.

### Problem Setting
We assume a generic Decoder-only LVLM parameterized by $\theta$. The model processes an input image into a sequence of visual tokens $v$, prepends them to a textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big). \tag{1}
$$
Here, $h_t^{(vl)}$ is the top-layer hidden state given the full context $[v, x, y_{<t}]$, and $f_{head}$ is the projection into the vocabulary space $\mathcal{V}$.

---

## 3.1 Textual Prior Isolation via Prefix Caching

Visual hallucinations in LVLMs are frequently driven by an over-reliance on the textual context $y_{<t}$, where strong language co-occurrence priors override the visual evidence. Following standard contrastive decoding formulations, we isolate this textual bias by computing a parallel text-only branch:
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big), \tag{2}
$$
where $h_t^{(text)}$ is conditioned exclusively on the textual context $[x, y_{<t}]$. 

While dual-branch decoding introduces inherent computational overhead, modern inference frameworks equipped with prefix-caching mechanisms (e.g., vLLM) can efficiently share the KV-cache of the textual components $[x, y_{<t}]$ across both branches. This significantly amortizes the memory bandwidth requirements, making the extraction of the text-only prior practically feasible without modifying the underlying standard attention kernels.

---

## 3.2 Stable Log-Probability Divergence

A common practice in contrastive decoding is to directly subtract the unnormalized logits (i.e., $\ell_t^{(vl)} - \ell_t^{(text)}$). However, because the full-modality and text-only queries attend to sequences of substantially different lengths, their resulting logits often exhibit a global scale shift due to the varying partition functions in the Softmax operation. 

To provide a more stable and numerically grounded metric for comparison, CoFi-Dec performs the divergence calculation in the normalized log-probability space. We define $\hat{\ell}_t^{(vl)}(w) = \log P_t^{(vl)}(w)$ and $\hat{\ell}_t^{(text)}(w) = \log P_t^{(text)}(w)$. By utilizing normalized probabilities, we implicitly account for the scale shift, enabling a direct token-level comparison.

We quantify the hallucination risk for each token $w \in \mathcal{V}$ as the log-domain divergence:
$$
\Delta_t(w) = \hat{\ell}_t^{(text)}(w) - \hat{\ell}_t^{(vl)}(w) = \log \frac{P_t^{(text)}(w)}{P_t^{(vl)}(w)}. \tag{3}
$$
This pointwise ratio serves as a straightforward indicator of textual bias: when $\Delta_t(w) > 0$, the token is assigned a higher likelihood by the blind textual prior than by the visually-grounded context, indicating a potential semantic hallucination. Conversely, for standard syntactic completions (e.g., stop words or subword continuations), both distributions yield similar probabilities, resulting in $\Delta_t(w) \approx 0$.

---

## 3.3 Syntax-Preserving Adaptive Penalty

Instead of applying a global penalty to the entire vocabulary, which often corrupts the grammatical structure of the generated text, we apply a targeted, asymmetric penalty. Recognizing that heuristic triggers (such as global entropy) often misrepresent syntactic divergence as hallucinations, CoFi-Dec relies on a direct, parameterized relative penalization.

We formulate the calibrated log-probability $\tilde{\ell}_t(w)$ as:
$$
\tilde{\ell}_t(w) = \hat{\ell}_t^{(vl)}(w) - \alpha \cdot \text{ReLU}\big( \Delta_t(w) \big), \tag{4}
$$
where $\alpha \ge 0$ is a tunable hyperparameter that controls the penalty strength. 

This formulation offers two practical advantages for robust generation:
1. **Asymmetric Syntactic Preservation:** The $\text{ReLU}$ activation strictly isolates the penalty to tokens that are disproportionately favored by the text branch ($\Delta_t(w) > 0$). Tokens that are equally supported by both branches or enhanced by the visual context ($\Delta_t(w) \le 0$) remain entirely unpenalized. This simple truncation effectively protects natural syntactic structures and subword continuations from being artificially suppressed.
2. **Stable Scale Control:** By decoupling the penalty strength from unstable global metrics and mapping it to an explicit hyperparameter $\alpha$, the framework can be easily tuned across different LVLM families (e.g., 7B vs. 34B parameters) based on their inherent susceptibility to hallucination. 

The final next-token $y_t$ is sampled from the calibrated distribution:
$$
P_{final}(y_t) = \text{Softmax}\big(\tilde{\ell}_t / T\big), \tag{5}
$$
where $T$ is the standard sampling temperature. By integrating prefix-cached prior extraction with a stabilized log-probability penalty, CoFi-Dec achieves a pragmatic and effective balance between hallucination reduction and generation fluency.