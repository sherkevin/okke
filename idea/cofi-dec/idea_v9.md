# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a training-free inference framework designed to mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Building upon the foundation of Contrastive Decoding (CD), CoFi-Dec addresses the critical limitations of prior paradigms: the severe computational overhead of dual-branch decoding, the destructive penalization of syntactic fluency (e.g., subword completion), and the distortion of global distributions. We formulate hallucination mitigation as a computationally lightweight, syntax-preserving asymmetric logit rectification.

### Problem Setting
We assume a standard Decoder-only LVLM parameterized by $\theta$. The model processes an input image into visual tokens $v$, prepends them to a textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big), \tag{1}
$$
where $h_t^{(vl)}$ is the top-layer hidden state given the full context $[v, x, y_{<t}]$, and $f_{head}$ projects it into the vocabulary $\mathcal{V}$.

---

## 3.1 Lightweight Windowed Textual Prior

Visual hallucinations predominantly arise when LVLMs become over-reliant on the textual context $y_{<t}$, allowing local language co-occurrence priors to override visual evidence. Standard contrastive methods (e.g., VCD) isolate this bias by running a parallel text-only branch over the entire sequence $[x, y_{<t}]$. However, as the sequence length grows, this parallel autoregressive forward pass drastically inflates the KV-cache memory footprint and doubles the computational latency, rendering it impractical for real-world LVLM deployment.

To establish a computationally pragmatic prior, we observe that the linguistic inertia driving textual hallucinations (e.g., n-gram biases and co-occurrences) is heavily localized. We construct a **Windowed Textual Prior Branch** that omits the visual prefix $v$ and restricts attention strictly to a local sliding window of the most recent $K$ tokens (we empirically set $K=16$):
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big). \tag{2}
$$
Here, $h_t^{(text)}$ is conditioned solely on $y_{t-K : t-1}$. This truncation bounds the computational complexity and KV-cache footprint of the auxiliary branch to $O(K)$, virtually eliminating the memory bandwidth bottleneck of dual-branch decoding while effectively isolating the local linguistic biases that trigger hallucinations.

---

## 3.2 Syntax-Preserving Hallucination Identification

A fundamental flaw in previous adaptive CD mechanisms is the conflation of *hallucinations* with *syntactic continuity*. Prior methods often assume that low textual entropy (high confidence in the text branch) equates to hallucination. However, in autoregressive generation, near-zero entropy frequently occurs during necessary grammatical structuring or subword completions (e.g., generating "ington" after "Wash", or "ing" after "work"). Penalizing these tokens structurally destroys linguistic fluency.

To rigorously decouple semantic hallucinations from syntactic completions, we analyze the relative logit deviation. In standard subword completion, *both* the text branch and the visual branch are highly confident because the local textual context perfectly determines the next token. Therefore, the difference between their unnormalized logits is negligible ($\ell_t^{(text)} \approx \ell_t^{(vl)}$). 

A genuine textual hallucination only occurs under a specific divergence: the text branch confidently predicts a semantic entity driven by language priors ($\ell_t^{(text)}$ is high), but the visual branch, grounded in the image, lacks visual evidence to support it ($\ell_t^{(vl)}$ drops). We identify this condition cleanly in the logit space:
$$
\Delta \ell_t(w) = \ell_t^{(text)}(w) - \ell_t^{(vl)}(w). \tag{3}
$$
When $\Delta \ell_t(w) > 0$, the language prior is overpowering the visual context. When $\Delta \ell_t(w) \le 0$, the token is either a visually supported entity or a naturally aligned syntactic/subword continuation.

---

## 3.3 Asymmetric Confidence-Gated Rectification

Standard contrastive decoding directly subtracts the textual prior from the visual logits ($\ell_t^{(vl)} - \alpha \ell_t^{(text)}$). This global subtraction distorts the partition function of the underlying distribution and violently penalizes high-frequency syntactical tokens (since their $\ell_t^{(text)}$ is naturally large), forcing the use of arbitrary Top-K truncations to salvage grammatical structure. 

To systematically neutralize hallucinations while strictly preserving linguistic logic and avoiding arbitrary global hyperparameters, we propose **Asymmetric Confidence-Gated Rectification**. We formulate the calibrated logit $\tilde{\ell}_t(w)$ for each token $w \in \mathcal{V}$ as follows:
$$
\tilde{\ell}_t(w) = \ell_t^{(vl)}(w) - P_t^{(text)}(w) \cdot \text{ReLU}\big( \ell_t^{(text)}(w) - \ell_t^{(vl)}(w) \big). \tag{4}
$$
This theoretically sound formulation mitigates textual hallucination through two synergistic constraints:

1. **Strict Asymmetric Preservation ($\text{ReLU}$):** The penalty is exclusively activated if $\ell_t^{(text)}(w) > \ell_t^{(vl)}(w)$. For valid visual entities, subword completions, and grammatical stop words where the visual context aligns with or boosts the token's score ($\ell_t^{(vl)}(w) \ge \ell_t^{(text)}(w)$), the penalty is strictly zero. This unilaterally preserves the structural integrity of the generation space and prevents fluency collapse.
2. **Confidence-Gated Scaling ($P_t^{(text)}$):** By weighting the penalty with the normalized probability of the textual prior, we ensure that intervention strictly targets "blindly confident" textual hallucinations. Harmless deviations in the long tail of the vocabulary—where $\ell_t^{(text)} > \ell_t^{(vl)}$ might occur purely due to random noise but $P_t^{(text)}$ is near zero—are safely ignored.

The final token $y_t$ is sampled from $\text{Softmax}(\tilde{\ell}_t)$. By utilizing a lightweight local textual prior and enforcing a confidence-gated asymmetric penalty directly on the logits, CoFi-Dec natively differentiates between grammatical continuity and semantic hallucination. This yields a highly robust, training-free decoding strategy that corrects modality bias without distorting natural language generation.