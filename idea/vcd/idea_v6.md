# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) taking a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. The token at time step $t$, denoted as $y_t$, is sampled from the probability distribution conditioned on the prompt and the visual context:
$$
p_{\text{deep}}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big). \tag{1}
$$
During decoding, object hallucinations frequently emerge when the language decoder over-relies on its intrinsic statistical biases and robust linguistic priors, neglecting fine-grained visual evidence. Mitigating this bias requires decoupling visually grounded predictions from language-driven inertia without computationally expensive retraining.

## 3.2 Internal Prior Probing via Layer-wise Representation
To isolate language priors, previous contrastive approaches typically perturb the visual input (e.g., adding noise, dropping patches, or applying severe blurring). However, these operations invariably inject out-of-distribution (OOD) artifacts into the vision encoder. Such domain shifts inherently disrupt the model's natural continuous representation space, leading to unpredictable probability distributions rather than cleanly extracting language priors.

To address this without inducing OOD anomalies, we leverage the internal hierarchical evolution of the LLM decoder. Recent studies on Transformer interpretability indicate that lower and middle layers of LLMs primarily capture syntactic structures and high-frequency linguistic co-occurrences (i.e., language priors), while the uppermost layers are responsible for fully fusing cross-modal semantics and establishing fine-grained visual grounding. 

Consequently, we propose **Internal Prior Probing**, which extracts the prior distribution directly from an intermediate layer of the decoder during a single forward pass. Let $h_t^{(l)}$ denote the hidden state at layer $l \in \{1, 2, \dots, L\}$. We project an intermediate layer state $h_t^{(M)}$ ($M < L$) to the vocabulary space using the shared language modeling head:
$$
p_{\text{shallow}}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(W \cdot h_t^{(M)}\big). \tag{2}
$$
This shallow distribution, generated without full cross-modal reasoning, naturally exhibits a stronger dependency on inherent language biases and dataset statistics. By comparing $p_{\text{shallow}}$ with the final output distribution $p_{\text{deep}}$ (derived from $h_t^{(L)}$), we can identify tokens propelled primarily by language inertia rather than deep visual confirmation. This approach requires only one forward pass and guarantees that the visual input remains strictly pristine and in-distribution.

## 3.3 Adaptive Layer-Contrastive Decoding (ALCD)

### 3.3.1 Sub-distribution Candidate Selection
Language models inherently exhibit a Zipfian, long-tail probability distribution over the vocabulary. Directly contrasting the entire vocabulary space often introduces chaotic noise from the long-tail tokens and disrupts grammatical sub-words. Rather than using statistically flawed Z-score normalizations, we restrict the contrastive calibration to a plausible candidate set $\mathcal{V}_{\text{cand}}$. 

To avoid rigid absolute thresholds, we construct $\mathcal{V}_{\text{cand}}$ using an adaptive relative confidence mask. A token is included only if its probability under the fully reasoned deep layer is comparable to the maximum predicted probability:
$$
\mathcal{V}_{\text{cand}} = \big\{ y_t \in \mathcal{V} \mid p_{\text{deep}}(y_t) \ge \beta \cdot \max_{y \in \mathcal{V}} p_{\text{deep}}(y) \big\}, \tag{3}
$$
where $\beta \in (0, 1)$ is a margin parameter (e.g., 0.1). This step safely excludes the noisy long-tail vocabulary while preserving syntactically necessary functional tokens and highly probable semantic entities.

### 3.3.2 Log-Probability Contrastive Penalty
Traditional contrastive decoding methods often apply scaling penalties directly to raw logits. Since logits are unnormalized and span $(-\infty, +\infty)$, multiplying a negative logit with a positive penalty factor mathematically reverses the penalty into an unintended reward. 

To ensure mathematical consistency, ALCD operates strictly within the log-probability space. For every token in $\mathcal{V}_{\text{cand}}$, we measure the relative excess of the language prior by comparing the shallow log-probability with the deep log-probability. We formulate the calibrated distribution as:
$$
\log \tilde{p}_{\text{ALCD}}(y_t) = \log p_{\text{deep}}(y_t) - \alpha \cdot \max\big(0, \log p_{\text{shallow}}(y_t) - \log p_{\text{deep}}(y_t)\big), \tag{4}
$$
where $\alpha \ge 0$ is the contrastive strength parameter. 

This formulation introduces an asymmetric, monotonic penalty. If a hallucinated token is heavily driven by statistical bias, its probability in the shallow layer will exceed its probability in the deep layer ($p_{\text{shallow}} > p_{\text{deep}}$), triggering a proportional penalty. Conversely, if a visually grounded token or a grammatically necessary functional word exhibits higher confidence in the deep layer (indicating successful cross-modal fusion), the penalty term evaluates to zero. This implicitly protects the syntactic flow and genuinely grounded entities without relying on heuristic part-of-speech taggers.

### 3.3.3 Divergence-Driven Dynamic Scaling
While the $\max(0, \cdot)$ operator naturally shields grounded tokens, using a static $\alpha$ across the entire generation sequence can still lead to localized over-correction, especially during complex reasoning phases. To dynamically adjust the penalty strength without relying on fragile attention-based triggers, we utilize the Jensen-Shannon Divergence (JSD) between the deep and shallow distributions:
$$
\alpha_t = \alpha_{\text{base}} \cdot \operatorname{JSD}\big(p_{\text{deep}} \parallel p_{\text{shallow}}\big). \tag{5}
$$
The JSD inherently measures the degree of cross-layer cognitive dissonance. When generating standard grammatical structures, the intermediate and final layers quickly reach a consensus, yielding a near-zero JSD and smoothly deactivating the contrastive penalty. When hallucination risks are high, the distributions diverge as the final layer attempts to negotiate visual facts against the intermediate layer's strong language biases, naturally scaling up the calibration strength $\alpha_t$.

The final token $y_t$ is sampled from the recalibrated and normalized distribution:
$$
p_{\text{ALCD}}(y_t) = \frac{\exp\big(\log \tilde{p}_{\text{ALCD}}(y_t)\big)}{\sum_{y \in \mathcal{V}_{\text{cand}}} \exp\big(\log \tilde{p}_{\text{ALCD}}(y)\big)}. \tag{6}
$$
By operating over log-probabilities and utilizing internal layer states, ALCD establishes a mathematically sound, dynamically scalable calibration framework that requires zero additional forward passes and completely circumvents the out-of-distribution visual paradigms.