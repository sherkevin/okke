# 3 Method

We first formulate the generation procedure of multimodal large language models (MLLMs). Following this, we introduce the **Confidence-Aware Visual Grounding Tracker** and the **Context-Orthogonal Semantic Repulsion Strategy**, which are strictly designed to align with the un-tied embedding spaces of modern LLMs. The methodology guarantees mathematical rigor, completely bypasses structural attention sinks, and achieves genuine $O(1)$ memory and computational efficiency without disrupting foundational CUDA kernels (e.g., FlashAttention).

---

## 3.1 Formulation of MLLMs Generation

The autoregressive generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input consists of visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM dimension $d$, alongside text tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context is denoted as $\boldsymbol{x}_{<t} = \{x_i\}_{i=0}^{t-1}$ for $t \ge N+M$.

### Model Forward and Decoding
At decoding step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t \in \mathbb{R}^d$, which encodes the contextualized next-token representation. A vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ projects $\boldsymbol{h}_t$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t
\tag{1}
\]
Modern LLMs (e.g., LLaMA, Qwen) do not tie the input embedding matrix $E$ with the output unembedding matrix $W$. Our method operates exclusively using the structurally correct representation spaces, manipulating the final-layer hidden state $\boldsymbol{h}_t$ and output weight vectors directly.

---

## 3.2 O(1) Confidence-Aware Visual Grounding Tracker

Visual-text misalignment occurs when the model over-trusts its linguistic prior while losing visual grounding. Previous methods rely on extracting step-wise attention matrices $O(L \times H \times t)$, which severely degrades KV-Cache throughput and inevitably captures structural "attention sinks" (e.g., punctuation). We circumvent this by introducing a constant-time $O(1)$ grounding tracker.

### Global Visual Key Pooling
Instead of dynamic attention extraction, we compute a static, aggregated visual representation before autoregressive decoding begins. Let $\boldsymbol{k}_i^v \in \mathbb{R}^d$ be the Key vectors corresponding to the visual tokens from the final layer. We define the global visual key $\bar{\boldsymbol{k}}_v$:
\[
\bar{\boldsymbol{k}}_v = \frac{1}{N} \sum_{i=0}^{N-1} \boldsymbol{k}_i^v
\tag{2}
\]
### Dynamic Visual Grounding Signal
At each decoding step $t$, we measure the visual grounding $v_t$ by computing the scaled dot-product between the current final-layer Query vector $\boldsymbol{q}_t$ and the pooled visual key $\bar{\boldsymbol{k}}_v$:
\[
v_t = \sigma\left(\frac{\boldsymbol{q}_t \cdot \bar{\boldsymbol{k}}_v}{\sqrt{d}}\right)
\tag{3}
\]
where $\sigma$ is the Sigmoid activation. This single $O(1)$ vector inner product entirely avoids attention sinks—as it solely measures affinity to visual tokens—and maintains flawless compatibility with FlashAttention since the KV-Cache remains unmodified.

### Confidence-Aware Over-Trust Detection
A drop in visual grounding is natural when MLLMs perform complex textual reasoning or summarization. Penalizing *any* visual shift forces an overly literal generation. True hallucination occurs when the model exhibits *both* a detachment from visual grounding and an over-confident reliance on linguistic priors. We quantify the linguistic predictive confidence via the normalized Shannon entropy of the step's preliminary probability distribution $P_t$:
\[
H_t = \frac{-\sum_{x \in \mathcal{X}} P_t(x) \log P_t(x)}{\log |\mathcal{X}|}
\tag{4}
\]
We define the Confidence-Aware Over-Trust factor $\phi_t$, which triggers only when low visual grounding coincides with high linguistic confidence (low entropy):
\[
\phi_t = \max\left(0,\, (1 - v_t) \cdot (1 - H_t)\right)
\tag{5}
\]
This biologically-inspired gating mechanism rigorously ensures that valid abstract reasoning is preserved while hallucination precursors are flagged.

---

## 3.3 Context-Orthogonal Semantic Repulsion

When over-trust $\phi_t$ is high, the generation is trapped in the semantic inertia of recent textual tokens. Global suppression of specific vocabulary logits destructively bans tokens context-independently (e.g., globally penalizing the word "red"). Instead, we propose a dynamic, context-orthogonal projection applied directly to the hidden state, ensuring mathematically safe probability calibration.

### EMA-based Textual Inertia Tracking
To represent the semantic inertia of the generated text, we maintain a lightweight Exponential Moving Average (EMA) of the *output unembedding representations* of recently generated tokens. Let $\boldsymbol{w}_{x_{t-1}} \in \mathbb{R}^d$ be the specific row vector from the unembedding matrix $W$ corresponding to the previously generated token $x_{t-1}$. The semantic inertia vector $\boldsymbol{c}_t$ is updated via:
\[
\boldsymbol{c}_t = \beta \boldsymbol{c}_{t-1} + (1 - \beta) \boldsymbol{w}_{x_{t-1}}
\tag{6}
\]
where $\beta \in (0, 1)$ is the momentum coefficient. Using $\boldsymbol{w}_{x_{t-1}}$ guarantees strict dimensional and spatial alignment with the LLM's predictive head, avoiding the mathematically flawed projection of input embeddings. Updating $\boldsymbol{c}_t$ requires only a single $O(1)$ vector addition per step.

### Orthogonal Repulsion Projection
To break the hallucination cycle without globally damaging the vocabulary distribution, we actively suppress the component of the next-token hidden state $\boldsymbol{h}_t$ that is purely collinear with the recent textual inertia $\boldsymbol{c}_t$. We construct the repulsed hidden state $\tilde{\boldsymbol{h}}_t$ via adaptive orthogonal projection:
\[
\tilde{\boldsymbol{h}}_t = \boldsymbol{h}_t - \alpha \cdot \phi_t \cdot \left( \frac{\boldsymbol{h}_t \cdot \boldsymbol{c}_t}{\|\boldsymbol{c}_t\|^2 + \epsilon} \right) \boldsymbol{c}_t
\tag{7}
\]
where $\alpha$ is the calibration scalar and $\epsilon$ prevents division by zero. 

### Final Logit Calibration
The calibrated hidden state $\tilde{\boldsymbol{h}}_t$ is then passed through the standard unembedding matrix to obtain the final decoding probabilities:
\[
\tilde{\boldsymbol{z}}_t = W \tilde{\boldsymbol{h}}_t, \quad P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}(\tilde{\boldsymbol{z}}_t)
\tag{8}
\]

**Theoretical Advantage:** By projecting $\boldsymbol{h}_t$ away from the textual inertia $\boldsymbol{c}_t$ proportionally to the over-trust factor $\phi_t$, we specifically degrade the probability of generating contextually repetitive or hallucinated continuations. However, if a token (e.g., "red") is strongly supported by non-collinear features in $\boldsymbol{h}_t$ (such as actual residual visual features), its projection remains largely unaffected, preserving the model's vocabulary integrity. The entire pipeline requires zero caching of historical distributions, ensuring flawless deterministic latency for high-throughput deployment.