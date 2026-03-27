# 3 Method
We first formulate the generation procedure of multimodal large language models (MLLMs) to facilitate understanding of our proposed method. Following this, we introduce the **Cross-Modal Over-Trust Penalty** and the **Semantic Repulsion Strategy**, fundamentally redesigned to align with the internal mechanisms of LLMs, ensuring rigorous mathematical scaling, robustness against attention sinks, and genuine $O(1)$ memory efficiency without requiring historical distribution caching.

---

## 3.1 Formulation of MLLMs Generation
The generation pipeline of MLLMs can be divided into three stages: input formulation, model forward pass, and decoding.

### Input Formulation
MLLM inputs consist of image and text modalities. Visual tokens are extracted from raw images via a vision encoder and projected into the LLM input space. Let the visual tokens be denoted as:
\[
\boldsymbol{x}_v = \{x_0, x_1, \dots, x_{N-1}\}
\]
where $N$ is the length of the visual tokens. The text input (system prompt and user instructions) is tokenized as:
\[
\boldsymbol{x}_p = \{x_N, x_{N+1}, \dots, x_{N+M-1}\}
\]
Image and text tokens are concatenated into the full input sequence context:
\[
\boldsymbol{x}_{<t} = \{x_i\}_{i=0}^{t-1},\quad t \ge N+M
\]

### Model Forward and Decoding
MLLMs are autoregressively trained with a causal attention mask. The model computes the final-layer hidden state $\boldsymbol{h}_t$, which intrinsically encodes the *next-token prediction* representations rather than the semantics of token $t$ itself:
\[
\boldsymbol{h}_t = \text{MLLM}(\boldsymbol{x}_{<t})
\tag{1}
\]
A vocabulary unembedding head $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ projects $\boldsymbol{h}_t$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t
\tag{2}
\]
where $\mathcal{X}$ is the vocabulary set. Our method operates directly on the unnormalized logit space, intrinsically supporting all standard low-latency sampling strategies (e.g., Top-$p$, Top-$k$) and Greedy Search.

---

## 3.2 Sink-Masked Cross-Modal Over-Trust Penalty
Visual-text misalignment occurs when the model over-trusts local textual context while neglecting the grounding visual tokens. However, directly aggregating raw attention weights introduces massive false positives due to "Attention Sinks" (e.g., `<s>`, initial prompt tokens, and punctuation) and fails to account for the inherent information density differences between dense continuous visual tokens and highly compressed discrete textual tokens.

### Sink-Masked Attention Aggregation
To extract genuine semantic dependencies, we explicitly filter out the attention sink tokens. Let $\mathcal{S}$ be the index set of structural sink tokens (the first $K$ tokens and all punctuation marks). We define the sink-masked attention weight $\tilde{\omega}_{t,j}^{(h)}$ at head $h$:
\[
\tilde{\omega}_{t,j}^{(h)} = \begin{cases} 
\frac{\omega_{t,j}^{(h)}}{\sum_{m \notin \mathcal{S}} \omega_{t,m}^{(h)}}, & \text{if } j \notin \mathcal{S} \\
0, & \text{if } j \in \mathcal{S}
\end{cases}
\tag{3}
\]
We aggregate the attention across a specifically selected set of visually-grounded heads $\mathcal{H}_v$. To avoid direct numerical comparison between modalities with distinct information densities, $\mathcal{H}_v$ is selected based on the historical variance of visual attention within each head, capturing heads that actively shift their focus to visual elements rather than static sink-like heads. The aggregated semantic attention is:
\[
\bar{\omega}_{t,j} = \frac{1}{|\mathcal{H}_{v}|} \sum_{h \in \mathcal{H}_{v}} \tilde{\omega}_{t,j}^{(h)}
\tag{4}
\]

### Temporal Visual Shift Detection
Instead of directly comparing visual and textual attention masses (which suffer from density mismatch and length biases), we detect hallucinations by monitoring the *temporal shift* in visual attention. Let $v_t = \sum_{j=0}^{N-1} \bar{\omega}_{t,j}$ be the visual attention mass at step $t$. We compute the over-trust penalty scalar $\phi_t$ as the drop in visual grounding relative to its moving average $\bar{v}_{<t}$:
\[
\phi_t = \max\left(0,\, \bar{v}_{<t} - v_t\right)
\tag{5}
\]
This formulation is strictly immune to length increases and modality density imbalances, firing only when the model exhibits a sudden detachment from the visual context.

### Logit-Scale Semantic Calibration
To safely penalize the generation without destroying the overall probability distribution, the penalty must be subtracted in the *unnormalized logit space* ($-\infty, \infty$) with a mathematically matched scale. Furthermore, recognizing that final-layer hidden states $\boldsymbol{h}_j$ represent next-token predictions, we instead utilize the static input word embedding matrix $E \in \mathbb{R}^{|\mathcal{X}| \times d}$ to represent the semantic meaning of the historical context.

We construct a textual context vector $\boldsymbol{c}_t$ by weighting the embeddings of past text tokens:
\[
\boldsymbol{c}_t = \sum_{j=N}^{t-1} \bar{\omega}_{t,j} \boldsymbol{e}_{x_j}
\tag{6}
\]
where $\boldsymbol{e}_{x_j}$ is the input embedding of token $x_j$. We project this context vector into the vocabulary logit space using the unembedding head $W$, yielding a semantic bias vector $\Delta \boldsymbol{z}_t = W \boldsymbol{c}_t$. The scale of $\Delta \boldsymbol{z}_t$ intrinsically matches the original logits $\boldsymbol{z}_t$. We subtract this term to suppress tokens strongly correlated with the over-attended local text:
\[
\tilde{\boldsymbol{z}}_t = \boldsymbol{z}_t - \alpha \cdot \phi_t \cdot \Delta \boldsymbol{z}_t
\tag{7}
\]
where $\alpha$ is a constant scaling hyperparameter. The calibrated probability is then $P(x_t) = \operatorname{Softmax}(\tilde{\boldsymbol{z}}_t)$.

---

## 3.3 Genuine $O(1)$ Semantic Repulsion Strategy
While Eq. (7) addresses step-wise detachment, hallucinations frequently manifest as the model becoming trapped in the semantic field of a previously generated text token (an "anchor"). Previous methods cache full historical distributions ($P_{hist}$) to penalize such anchors, which requires storing immense dense tensors (e.g., $|\mathcal{X}|$ floats per step), severely bottlenecking the KV-Cache in real-world deployments. 

We completely eliminate this memory overhead by introducing the **Semantic Repulsion Strategy**, achieving identical mitigating effects with strict $O(1)$ memory complexity.

### Sink-Masked Anchor Tracking
We maintain a lightweight sliding window $[t-l, t-1]$ to track semantic anchors. An anchor token $s$ is identified if it consistently captures the maximum sink-masked attention $\bar{\omega}$ over the recent window:
\[
s = \arg\max_{N \le j < z, \, j \notin \mathcal{S}} \bar{\omega}_{z,j} \quad \text{for a majority of } z \in [t-l, t-1]
\tag{8}
\]

### Zero-Memory Semantic Repulsion
When a hallucination anchor $s$ is identified, it indicates the model's linguistic prior is overly biased toward the semantic vicinity of $s$. Instead of loading a cached hypothetical distribution, we directly repel the current generation away from the anchor's semantic embedding $\boldsymbol{e}_{x_s}$. 

We compute the repulsion logit vector dynamically using the static unembedding head $W$:
\[
\boldsymbol{r}_t = W \boldsymbol{e}_{x_s}
\tag{9}
\]
The final decoding distribution incorporates this repulsion strictly within the unnormalized logit space:
\[
P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left( \tilde{\boldsymbol{z}}_t - \gamma \cdot \boldsymbol{r}_t \right)
\tag{10}
\]
where $\gamma$ is a repulsion coefficient applied exclusively when anchor $s$ is triggered. 

By utilizing the inherent geometric properties of the vocabulary space, this operation requires zero historical logits caching, introduces no extra neural network forward passes, and consists only of a single matrix-vector multiplication triggered dynamically. It strictly preserves the integrity of the KV-Cache and guarantees the deterministic latency essential for high-throughput MLLM serving engines.