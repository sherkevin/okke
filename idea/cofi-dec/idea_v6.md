# 3 Methodology

This paper introduces **CoFi-Dec** (Contextual Filtering Decoding), a training-free inference framework designed to mitigate visual hallucinations in Large Vision-Language Models (LVLMs). Addressing the critical bottlenecks of prior decoding strategies—namely, extreme computational latency, hardware memory-bound constraints, architectural inflexibility, and vocabulary truncation—CoFi-Dec operates as an efficient, generalized modality-balancing mechanism.

### Problem Setting
We assume a standard generic Decoder-only LVLM parameterized by $\theta$. The model projects an input image into a sequence of visual tokens $v$, prepends them to a tokenized textual query $x$, and generates a response sequence $y$ autoregressively. At decoding step $t$, the standard full-modality token distribution is:
$$
P_t^{(vl)}(y_t) = \text{Softmax}\big(\ell_t^{(vl)}\big), \quad \text{where} \;\; \ell_t^{(vl)} = f_{head}\big(h_t^{(vl)}\big), \tag{1}
$$
where $h_t^{(vl)} \in \mathbb{R}^d$ is the top-layer hidden state of the LLM given the context $[v, x, y_{<t}]$, and $f_{head}$ is the linear language modeling head projecting into the vocabulary space $\mathcal{V}$.

---

## 3.1 Lightweight Text-Only Prior Construction

Visual hallucinations primarily stem from **modality imbalance**, where strong text-based co-occurrence biases (textual priors) override actual visual evidence. Unlike previous methods that enforce rigid architectural assumptions (e.g., separating global and local visual patches) or introduce catastrophic latency via dual full-size LLM forward passes, we directly isolate the textual inertia.

To isolate the model's pure linguistic hallucination bias, we construct a **Text-Only Prior Branch**. Alongside the standard full-modality generation, we compute a lightweight auxiliary branch that explicitly omits the visual prefix $v$:
$$
P_t^{(text)}(y_t) = \text{Softmax}\big(\ell_t^{(text)}\big), \quad \text{where} \;\; \ell_t^{(text)} = f_{head}\big(h_t^{(text)}\big), \tag{2}
$$
and $h_t^{(text)}$ is derived given only the textual context $[x, y_{<t}]$. 

**Computational Efficiency:** Unlike dual-branch visual decoding, the text-only branch drops the extensive sequence of visual tokens (which typically constitute 70%-90% of the input length in high-resolution LVLMs). Consequently, its KV-cache footprint and computational FLOPs are heavily minimized, completely bypassing the severe memory-bound and compute-bound disasters of dual full-sequence forward passes.

---

## 3.2 Hardware-Efficient Hidden-State Calibrator

Prior continuous decoding strategies dynamically weight fusion mechanisms by calculating the Jensen-Shannon Divergence (JSD) or KL-divergence across the entire vocabulary $\mathcal{V}$ at every step. For modern vocabularies ($32,000 \sim 128,000$ tokens), performing step-wise, full-vocabulary exponential and log-space normalizations introduces a severe CPU/GPU memory bandwidth bottleneck, rendering real-time streaming impossible.

To eliminate this overhead, CoFi-Dec evaluates the necessity for hallucination intervention strictly within the **dense feature space ($O(d)$ complexity)** rather than the extensive vocabulary space ($O(|\mathcal{V}|)$). We quantify the dynamic modality divergence $\alpha_t$ using the bounded cosine distance between the full-modality hidden state $h_t^{(vl)}$ and the text-only hidden state $h_t^{(text)}$:
$$
\alpha_t = \frac{1}{2} \left( 1 - \frac{h_t^{(vl)} \cdot h_t^{(text)}}{\|h_t^{(vl)}\|_2 \|h_t^{(text)}\|_2} \right). \tag{3}
$$
The calibrator $\alpha_t \in [0, 1]$ acts as an intrinsic trigger. When predicting common functional words or rigid syntax (e.g., "the", "is"), the hidden states are tightly aligned, yielding $\alpha_t \approx 0$. Conversely, when predicting semantic entities where visual evidence is paramount, the feature representations diverge, and $\alpha_t$ naturally scales up to activate robust modality balancing.

---

## 3.3 Vocabulary-Preserving Visual Advantage Modulation

Standard Contrastive Decoding (CD) techniques artificially subtract the biased logits ($\ell_t^{(vl)} - \ell_t^{(text)}$). This linear subtraction aggressively pushes the logits of valid long-tail semantic words to negative infinity, leading to vocabulary truncation, grammatically broken outputs, and a heavy reliance on arbitrary Top-K/Top-p heuristics to salvage fluency.

To systematically mitigate textual hallucinations without disrupting linguistic integrity, we propose a soft **Visual Advantage Modulation (VAM)**. We first define the Visual Advantage Score $S_t(w)$ for each token $w \in \mathcal{V}$, utilizing the standard logistic sigmoid function $\sigma$:
$$
S_t(w) = \sigma\left( \ell_t^{(vl)}(w) - \ell_t^{(text)}(w) \right). \tag{4}
$$
$S_t(w) \in (0,1)$ naturally bounds the contribution of the visual modality. If a token is driven primarily by the text prior (a likely hallucination), its text logit matches or exceeds its visual logit, pushing $S_t(w)$ toward $0$. If it is genuinely grounded in the image, $S_t(w) \to 1$.

We then softly modulate the full-modality distribution by applying this bounded score as an exponential decay penalty, governed by our hidden-state calibrator $\alpha_t$:
$$
\tilde{P}_t(w) \propto P_t^{(vl)}(w) \cdot \big[ S_t(w) \big]^{\gamma \cdot \alpha_t}, \tag{5}
$$
where $\gamma > 0$ is an overarching scaling hyperparameter. In practice, this optimization is computed highly efficiently in the logit space:
$$
\tilde{\ell}_t(w) = \ell_t^{(vl)}(w) + \gamma \cdot \alpha_t \cdot \log \sigma\left( \ell_t^{(vl)}(w) - \ell_t^{(text)}(w) \right). \tag{6}
$$

**Fluency and Long-Tail Preservation:** Because $\log \sigma(\cdot)$ provides a smooth, asymptotically bounded penalty rather than unbounded linear subtraction, CoFi-Dec strictly preserves the structural integrity of the original logit space. Correct but low-probability tail tokens are never abruptly truncated to $-\infty$. The next token $y_t$ is sampled from $\text{Softmax}(\tilde{\ell}_t)$. By isolating true text priors, operating strictly in the low-dimensional hidden space for divergence calculation, and applying a structurally safe non-linear modulation, CoFi-Dec effectively neutralizes hallucination without sacrificing inference speed or generative fluency.