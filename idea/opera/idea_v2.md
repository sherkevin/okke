# 3 Method
We first formulate the generation procedure of multimodal large language models (MLLMs) to facilitate understanding of our proposed method. Following this, we introduce the **Cross-Modal Over-Trust Penalty** and the **Retrospective Attention Allocation Strategy**, specifically redesigned to guarantee cross-modal alignment, mathematical stability, and deterministic inference latency.

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
A vocabulary head \(H\) projects the final-layer hidden states to logits for next-token prediction:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{SoftMax}\!\left[H(\boldsymbol{h}_t)\right]_{x_t},\quad x_t \in \mathcal{X}
\tag{2}
\]
where \(\mathcal{X}\) is the vocabulary set. To ensure broad applicability and preserve deterministic inference latency, our method is entirely decoupled from specific decoding algorithms, inherently supporting standard Greedy Search and low-latency sampling strategies (e.g., Top-\(p\), Top-\(k\)) without requiring computationally expensive beam search or candidate maintenance.

---

## 3.2 Cross-Modal Over-Trust Logit Penalty
Visual-text misalignment is the fundamental root of MLLM hallucinations. Hallucinations occur when the model over-trusts local textual context (e.g., hallucinating objects based on linguistic priors) while neglecting the grounding visual tokens. To address this, we propose an adaptive logit penalty driven by cross-modal attention dynamics.

### Cross-Modal vs. Textual Attention
Let \(\omega_{t,j}^{(h)}\) be the causal self-attention weight from the current decoding step \(t\) to a preceding token \(j\) at attention head \(h\). To preserve the distinct semantic representations of different heads, we aggregate the attention weights via mean pooling across all \(H_{num}\) heads:
\[
\bar{\omega}_{t,j} = \frac{1}{H_{num}} \sum_{h=1}^{H_{num}} \omega_{t,j}^{(h)}
\tag{3}
\]
We divide the context into visual grounding tokens (\(j < N\)) and textual tokens (\(N \le j < t\)). The visual grounding score \(v_t\) and textual reliance score \(u_t\) are formulated as:
\[
v_t = \sum_{j=0}^{N-1} \bar{\omega}_{t,j}, \quad u_t = \sum_{j=N}^{t-1} \bar{\omega}_{t,j}
\tag{4}
\]

### Stable Knowledge Aggregation
To model the hysteresis of knowledge aggregation without suffering from the numerical instability and gradient vanishing caused by consecutive multiplications, we compute the relative textual over-trust score \(\phi_t\) using a stabilized logarithmic ratio:
\[
\phi_t = \max\left(0,\, \log(1 + u_t) - \log(1 + v_t)\right)
\tag{5}
\]
This formulation guarantees numerical stability regardless of the sequence length. A high \(\phi_t\) indicates that the generation is severing its dependence on the image and degenerating into an unconditional text completion, which is the exact precursor to hallucination.

### Penalized Logit
We apply \(\phi_t\) to dynamically penalize the logit distributions. To prevent disrupting correct syntactic formations (e.g., necessary grammar tokens), the penalty is specifically allocated to the subset of tokens strongly correlated with the over-trusted textual context, yielding the penalized logits:
\[
\tilde{P}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left[H(\boldsymbol{h}_t) - \alpha \cdot \phi_t \cdot \Delta(x_t) \right]
\tag{6}
\]
where \(\alpha\) is a dynamic scaling factor, and \(\Delta(x_t)\) measures the projection similarity between the current candidate token and the highly-attended past text tokens, avoiding indiscriminate penalization of standard long-range semantic dependencies.

---

## 3.3 Retrospective Attention Allocation Strategy
While the penalty in Eq. (6) suppresses step-wise over-trust, hallucinations can also manifest through a gradual shift in attention across multiple steps. Previous rollback-based strategies severely disrupt the deterministic latency of LLMs by discarding KV-Caches and forcing re-computation, making them unsuitable for real-world streaming applications.

We resolve this by proposing the **Retrospective Attention Allocation Strategy**, which achieves the same corrective effect via lookahead probability reallocation rather than physical sequence rollback.

### Retrospective State Tracking
Instead of rewinding the decoding tree, we maintain a lightweight, sliding state of attention concentration over the past \(l\) generated tokens. We define the persistent anchor set \(\mathcal{C}_t\) as the textual tokens that consistently hijack maximum attention over the recent window \([t-l, t-1]\):
\[
\mathcal{C}_t = \left\{\,c \,\bigg|\, c = \arg\max_{N \le j < z} \bar{\omega}_{z,j},\; z \in [t-l, t-1]\,\right\}
\tag{7}
\]
We identify a "hallucination anchor" \(s\) if a specific textual token appears in \(\mathcal{C}_t\) with a frequency exceeding a dynamic concentration threshold.

### Dynamic Probability Allocation
When a hallucination anchor \(s\) is retrospectively detected, physical rollback is strictly avoided. Instead, we dynamically allocate a contrastive divergence term directly into the current generation step \(t\). Let \(P_{anchor}\) be the hypothetical next-token distribution if conditioned heavily on the anchor \(s\). We adjust the final decoding distribution to allocate probability mass away from the anchor's linguistic prior and back toward visual alignment:
\[
P_{final}(x_t \mid \boldsymbol{x}_{<t}) \propto \tilde{P}(x_t \mid \boldsymbol{x}_{<t}) \times \left(1 - \gamma \cdot P_{anchor}(x_t \mid \boldsymbol{x}_{\le s})\right)
\tag{8}
\]
where \(\gamma \in (0, 1)\) is the allocation coefficient triggered only when the anchor is detected. 

By restructuring the retrospection as an \(O(1)\) probability reallocation operation at the current decoding step, our method eliminates the need for heuristic parameters like maximum rollback limits or candidate overlap thresholds. This preserves the KV-Cache integrity and ensures strictly deterministic latency while intrinsically resolving the cross-modal misalignment.