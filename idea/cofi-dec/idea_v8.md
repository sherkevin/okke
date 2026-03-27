# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a training-free inference framework designed to systematically mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Building upon the established paradigm of contrastive decoding, CoFi-Dec addresses the critical theoretical flaws of prior methods—specifically, the scale-shift misalignment of unnormalized logits, the dimension mismatch in penalty formulations, and the vulnerability to temperature scaling. We formulate hallucination mitigation as a scale-invariant, asymmetric distributional correction that strictly preserves linguistic fluency.

### Problem Setting
We assume a standard generic Decoder-only LVLM parameterized by $\theta$. The model processes an input image, extracts a sequence of visual tokens $v$, prepends them to a tokenized textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big), \tag{1}
$$
where $h_t^{(vl)} \in \mathbb{R}^d$ is the top-layer hidden state of the LLM given the context $[v, x, y_{<t}]$, and $f_{head}$ is the language modeling head projecting into the vocabulary space $\mathcal{V}$.

---

## 3.1 Textual Prior Isolation

Visual hallucinations typically occur when the LVLM becomes over-reliant on the textual context $y_{<t}$, allowing strong language co-occurrence priors to override the actual visual evidence. Consistent with established contrastive decoding principles (e.g., VCD), to effectively penalize this textual inertia, we must first isolate it.

Alongside the standard full-modality generation branch, we construct a parallel **Text-Only Prior Branch** that explicitly omits the visual prefix $v$ from the attention context:
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big), \tag{2}
$$
and $h_t^{(text)}$ is derived entirely from the pure textual context $[x, y_{<t}]$. The distribution $P_t^{(text)}$ isolates the model's linguistic hallucination bias—representing what the model "blindly" guesses should logically appear next without conditioning on the image. 

---

## 3.2 Temperature-Invariant Overconfidence Calibrator

A fundamental mechanism of textual hallucination is the model's **blind overconfidence** driven by strong language priors. When the model guesses a token purely based on textual co-occurrence, the textual distribution $P_t^{(text)}$ becomes sharply concentrated, leading to near-zero entropy. Prior entropy-ratio formulations fail catastrophically in this regime due to division-by-zero risks and inverted logic.

To rigorously capture this phenomenon, we quantify the dynamic risk of textual hallucination using **Normalized Textual Overconfidence**. To ensure this metric is completely decoupled from user-defined sampling temperatures (which otherwise drastically alter distribution sharpness), all probability distributions in CoFi-Dec are strictly computed at the base temperature $T=1$. The base Shannon entropy of the text branch is $\mathcal{H}\big(P_t^{(text)}\big) = - \sum_{w \in \mathcal{V}} P_t^{(text)}(w) \log P_t^{(text)}(w)$. 

We define the temperature-invariant calibrator $\alpha_t$:
$$
\alpha_t = 1 - \frac{\mathcal{H}\big(P_t^{(text)}\big)}{\log |\mathcal{V}|}. \tag{3}
$$
**Physical Interpretation:** $\alpha_t \in [0, 1]$ serves as an intrinsic, mathematically bounded trigger. When the model is predicting standard syntax with high uncertainty or relying heavily on visual cues, textual entropy is high, driving $\alpha_t \to 0$. Conversely, when the model is severely overconfident due to a strong textual bias (the primary catalyst for hallucinations), $\mathcal{H}\big(P_t^{(text)}\big)$ approaches 0, driving $\alpha_t \to 1$. This correctly activates the maximal intervention precisely when the model exhibits blind textual overconfidence.

---

## 3.3 Scale-Invariant Asymmetric Rectification

Standard contrastive decoding directly subtracts unnormalized logits ($\ell_t^{(vl)} - \ell_t^{(text)}$). However, because the textual and visual branches process fundamentally different context lengths and attention receptive fields, their unnormalized logits exhibit natural scale shifts. Directly subtracting or mixing unnormalized logits with probabilities creates a severe dimensional mismatch and mathematical instability.

To systematically mitigate textual hallucinations with rigorous mathematical alignment, CoFi-Dec operates entirely within the normalized log-probability space. Let the normalized log-probabilities be $\hat{\ell}_t^{(vl)}(w) = \log P_t^{(vl)}(w)$ and $\hat{\ell}_t^{(text)}(w) = \log P_t^{(text)}(w)$. Because both are derived from proper probability distributions, they are strictly comparable regardless of sequence length differences.

We formulate the calibrated logit $\tilde{\ell}_t(w)$ for each token $w \in \mathcal{V}$ as a **Scale-Invariant Asymmetric Rectification**:
$$
\tilde{\ell}_t(w) = \hat{\ell}_t^{(vl)}(w) - \alpha_t \cdot \text{ReLU}\big( \hat{\ell}_t^{(text)}(w) - \hat{\ell}_t^{(vl)}(w) \big). \tag{4}
$$
This theoretically sound formulation neutralizes hallucinations through two strictly enforced constraints:

1. **Dimensional Consistency:** By mapping all operations to the normalized log-probability space ($\hat{\ell} \le 0$), the penalty term is mathematically coherent. The subtraction $\hat{\ell}_t^{(text)} - \hat{\ell}_t^{(vl)}$ exactly represents the log-ratio $\log \frac{P_t^{(text)}}{P_t^{(vl)}}$, natively anchoring the penalty in information theory rather than heuristic logit-scale approximations.
2. **Strict Asymmetric Preservation ($\text{ReLU}$):** The penalty is exclusively activated if $\hat{\ell}_t^{(text)}(w) > \hat{\ell}_t^{(vl)}(w)$—meaning the token is assigned a higher probability by the blind textual prior than by the visually-grounded context. For valid visual entities and grammatical stop words where the visual context maintains or boosts the probability ($\hat{\ell}_t^{(vl)}(w) \geq \hat{\ell}_t^{(text)}(w)$), the penalty is strictly zero. This unilaterally preserves the structural integrity of the generation space and inherently prevents vocabulary truncation.

The final token $y_t$ is sampled securely from $\text{Softmax}(\tilde{\ell}_t / T_{sample})$, where $T_{sample}$ is the final user-defined temperature. By isolating true text priors, evaluating risk via temperature-invariant overconfidence, and utilizing a dimensionally rigorous asymmetric penalty in the log-probability space, CoFi-Dec provides highly robust hallucination mitigation without arbitrary global scaling parameters.