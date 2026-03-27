# 3 Method
We first formulate the generation procedure of multimodal large language models (MLLMs) to facilitate understanding of our proposed method. Following this, we introduce the **Cross-Modal Over-Trust Penalty** and the **Retrospective Attention Allocation Strategy**, specifically redesigned to guarantee cross-modal alignment, mathematical stability, and realistic computational efficiency.

---

## 3.1 Formulation of MLLMs Generation
The generation pipeline of MLLMs can be divided into three stages: input formulation, model forward pass, and decoding.

### Input Formulation
MLLM inputs consist of image and text modalities. Visual tokens are extracted from raw images via a vision encoder and projected into the LLM input space through a cross-modality mapping module. Let the visual tokens be denoted as:
\[
\boldsymbol{x}_v = \{x_0, x_1, \dots, x_{N-1}\}
\]
where \(N\) is the length of the visual tokens. The text input (system prompt and user instructions) is tokenized as:
\[
\boldsymbol{x}_p = \{x_N, x_{N+1}, \dots, x_{N+M-1}\}
\]
Image and text tokens are concatenated into the full input sequence context:
\[
\boldsymbol{x}_{<t} = \{x_i\}_{i=0}^{t-1},\quad t \ge N+M
\]

### Model Forward and Decoding
MLLMs are autoregressively trained with a causal attention mask. The model predicts the next token based on preceding context:
\[
\boldsymbol{h}_t = \text{MLLM}(\boldsymbol{x}_{<t})
\tag{1}
\]
A vocabulary head \(W \in \mathbb{R}^{|\mathcal{X}| \times d}\) projects the final-layer hidden states to logits for next-token prediction:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left[W \boldsymbol{h}_t\right]_{x_t},\quad x_t \in \mathcal{X}
\tag{2}
\]
where \(\mathcal{X}\) is the vocabulary set. To ensure broad applicability, our method is strictly decoupled from specific decoding algorithms, natively supporting standard Greedy Search and sampling strategies (e.g., Top-\(p\), Top-\(k\)) without requiring computationally expensive beam search.

---

## 3.2 Cross-Modal Over-Trust Logit Penalty
Visual-text misalignment is the fundamental root of MLLM hallucinations. Hallucinations occur when the model over-trusts local textual context while neglecting the grounding visual tokens. To address this, we propose an adaptive logit penalty driven by cross-modal attention dynamics, with rigorous length normalization and head selection.

### Modality-Aware Attention Aggregation
Different attention heads in LLMs exhibit functional specialization (e.g., syntactic heads, induction heads). Simple mean pooling obliterates these distinct representation spaces. Instead, we dynamically select a subset of visually-grounded heads \(\mathcal{H}_{v}\). A head \(h\) is included in \(\mathcal{H}_{v}\) if its attention allocated to the visual prefix exceeds a uniform prior (\(N/t\)):
\[
\mathcal{H}_{v} = \left\{ h \;\bigg|\; \sum_{j=0}^{N-1} \omega_{t,j}^{(h)} > \frac{N}{t} \right\}
\tag{3}
\]
where \(\omega_{t,j}^{(h)}\) is the attention weight from the current step \(t\) to token \(j\) at head \(h\). We aggregate the attention weights specifically over \(\mathcal{H}_{v}\):
\[
\bar{\omega}_{t,j} = \frac{1}{|\mathcal{H}_{v}|} \sum_{h \in \mathcal{H}_{v}} \omega_{t,j}^{(h)}
\tag{4}
\]

### Length-Normalized Knowledge Aggregation
As the text sequence grows during decoding, the cumulative attention on text naturally increases due to the Softmax normalization property. To prevent systemic bias and false hallucination alarms in long generations, we introduce strict length normalization. The normalized visual grounding score \(\hat{v}_t\) and textual reliance score \(\hat{u}_t\) are formulated as:
\[
\hat{v}_t = \frac{1}{N} \sum_{j=0}^{N-1} \bar{\omega}_{t,j}, \quad \hat{u}_t = \frac{1}{t-N} \sum_{j=N}^{t-1} \bar{\omega}_{t,j}
\tag{5}
\]
We compute the relative textual over-trust score \(\phi_t\) using a stabilized logarithmic ratio:
\[
\phi_t = \max\left(0,\, \log(1 + \hat{u}_t) - \log(1 + \hat{v}_t)\right)
\tag{6}
\]
This formulation ensures mathematical stability and fair comparison regardless of the dynamically changing sequence length.

### Adaptive Logit Calibration
Direct subtraction on logits risks destroying the probability distribution and causing grammatical collapse. Instead, we introduce an adaptive calibration mechanism. The penalty is proportionally allocated based on the semantic similarity between the candidate token \(x_t\) and the heavily attended textual context. We define the similarity measure \(\Delta(x_t)\) using the model's own vocabulary embedding matrix \(W\):
\[
\Delta(x_t) = \operatorname{Softmax}\!\left( W \!\sum_{j=N}^{t-1} \bar{\omega}_{t,j} \boldsymbol{h}_j \right)_{x_t}
\tag{7}
\]
This operation entirely reuses existing cached hidden states \(\boldsymbol{h}_j\) without extra forward passes. The calibrated logits are then formulated as:
\[
\tilde{P}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left[W \boldsymbol{h}_t - \alpha_t \cdot \phi_t \cdot \Delta(x_t) \right]
\tag{8}
\]
where \(\alpha_t = \alpha_0 \cdot \exp(-\lambda \cdot \phi_t)\) is a self-decaying factor that caps the maximum penalty, protecting the syntactic integrity of the generated sentence while gently suppressing hallucination priors.

---

## 3.3 Retrospective Attention Allocation Strategy
While Eq. (8) suppresses step-wise over-trust, hallucinations can also manifest through a gradual shift in attention. Previous rollback-based strategies discard KV-Caches, which violates the deterministic latency constraints of real-world streaming applications. We resolve this by performing retrospective probability reallocation directly at the current decoding step.

### Retrospective State Tracking
We maintain a lightweight state of attention concentration over the past \(l\) generated tokens. The persistent anchor set \(\mathcal{C}_t\) tracks the textual tokens that hijack maximum attention within the sliding window \([t-l, t-1]\):
\[
\mathcal{C}_t = \left\{\,c \,\bigg|\, c = \arg\max_{N \le j < z} \bar{\omega}_{z,j},\; z \in [t-l, t-1]\,\right\}
\tag{9}
\]
A "hallucination anchor" \(s\) is identified if its occurrence frequency \(f_s\) in \(\mathcal{C}_t\) exceeds a deterministic threshold \(\tau = \lfloor 0.5 \times l \rfloor\).

### O(1) Probability Redistribution
To avoid the absurd computational cost of computing a hypothetical next-token distribution via an extra forward pass, we directly reuse the historically cached probability distribution computed at the step when anchor \(s\) was originally generated. Let \(P_{hist} = P(\cdot \mid \boldsymbol{x}_{<s})\) be this cached distribution, which captures the linguistic prior at the exact moment the anchor was introduced. 

When a hallucination anchor \(s\) is detected, physical rollback is strictly avoided. Instead, we perform a probability redistribution operation at the current step \(t\) to dynamically discount the lingering influence of the anchor's linguistic prior:
\[
P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \frac{\tilde{P}(x_t \mid \boldsymbol{x}_{<t}) \cdot \left(1 - \gamma \cdot P_{hist}(x_t)\right)}{\sum_{x \in \mathcal{X}} \tilde{P}(x \mid \boldsymbol{x}_{<t}) \cdot \left(1 - \gamma \cdot P_{hist}(x)\right)}
\tag{10}
\]
where \(\gamma \in (0, 1)\) is a constant redistribution coefficient applied only when the anchor frequency condition is met. 

By leveraging the previously computed and cached \(P_{hist}\), our method introduces zero extra matrix multiplications or neural network forward passes. This strict \(O(1)\) probability redistribution guarantees complete preservation of the KV-Cache, ensuring deterministic inference latency while robustly mitigating cross-modal misalignment.