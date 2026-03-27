# 3 Method

We formulate the autoregressive generation procedure of multimodal large language models (MLLMs) and introduce a mechanistically rigorous framework: **Late-Layer Visual Masking** and **Plausibility-Bounded Contrastive Decoding**. This methodology strictly adheres to the non-linear dynamics, deep entanglement properties, and normalization structures of modern Transformers. By abandoning flawed hidden-state approximations, intermediate residual extractions, and history-dependent heuristics, we construct an exact architectural counterfactual. Coupled with instantaneous divergence gating and adaptive logit truncation, our approach directly suppresses hallucinations while natively preserving syntactic structural integrity and logit-space stability.

---

## 3.1 Formulation of MLLMs Generation

The autoregressive generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input sequence comprises visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder, alongside the textual prefix and generated tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context at decoding step $t$ is denoted as $\boldsymbol{x}_{<t}$, with indices $V$ denoting the positions of visual tokens and $T$ denoting textual tokens.

### Base Model Forward and Decoding
At step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t^{(L)} \in \mathbb{R}^d$. Modern MLLMs universally apply a normalization layer (e.g., RMSNorm) before the predefined vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$. The unnormalized base logits $\boldsymbol{z}_t$ and base probability $P_{base}$ are computed as:
\[
\boldsymbol{z}_t = W \operatorname{RMSNorm}(\boldsymbol{h}_t^{(L)}),\quad P_{base}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right)
\tag{1}
\]
Representations at layer $L$ are deeply entangled; intermediate token states contain complex mixtures of textual and visual semantics fused through preceding Multi-Layer Perceptrons (MLPs). Extracting isolated intermediate vectors or bypassing the final MLP and normalization layers yields out-of-distribution representational noise. Our intervention explicitly respects this architecture by operating entirely through the unmodified forward pass mechanics.

---

## 3.2 Exact Architectural Counterfactual

To quantify the model's reliance on visual context for the current prediction, we construct a mathematically exact counterfactual state $\boldsymbol{h}_t^{prior}$. Instead of linearly subtracting intermediate attention vectors—which erroneously ignores the non-linear translation performed by MLPs—we apply an attention mask exclusively at the final Transformer layer $L$.

At layer $L$, the standard attention output $\boldsymbol{o}_t^{(L)}$ is an interpolation of values from all tokens. We isolate the ungrounded textual reasoning path by dynamically zeroing out the attention weights directed to the visual tokens ($i \in V$) and renormalizing the weights over the textual context ($j \in T$):
\[
\hat{A}_{t,j}^{(L)} = \frac{A_{t,j}^{(L)}}{\sum_{k \in T} A_{t,k}^{(L)}}, \quad \forall j \in T
\tag{2}
\]
The counterfactual attention output $\hat{\boldsymbol{o}}_t^{(L)}$ strictly attends to the textual prefix:
\[
\hat{\boldsymbol{o}}_t^{(L)} = W_O^{(L)} \sum_{j \in T} \hat{A}_{t,j}^{(L)} W_V^{(L)} \boldsymbol{h}_j^{(L-1)}
\tag{3}
\]
Crucially, this counterfactual signal is then passed through the unmodified, non-linear final MLP sub-layer to yield the counterfactual hidden state:
\[
\boldsymbol{h}_t^{prior} = \boldsymbol{h}_t^{(L-1)} + \hat{\boldsymbol{o}}_t^{(L)} + \operatorname{MLP}^{(L)}\left( \boldsymbol{h}_t^{(L-1)} + \hat{\boldsymbol{o}}_t^{(L)} \right)
\tag{4}
\]
By projecting $\boldsymbol{h}_t^{prior}$ through the exact same RMSNorm and unembedding matrix, we derive the purely linguistic prior logits $\boldsymbol{z}_t^{prior}$:
\[
\boldsymbol{z}_t^{prior} = W \operatorname{RMSNorm}(\boldsymbol{h}_t^{prior}),\quad P_{prior}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t^{prior}\right)
\tag{5}
\]
This guarantees that $\boldsymbol{z}_t^{prior}$ remains perfectly within the pre-trained data manifold, providing a rigorously formulated linguistic prior without mathematical approximations.

---

## 3.3 Instantaneous Divergence-Gated Syntactic Filtering

Previous methods rely on history-dependent moving averages of visual attention to preserve syntactic tokens, which are highly brittle and susceptible to prefix bias. We replace this with an instantaneous, history-agnostic syntactic filter based on the information-theoretic divergence between $P_{base}$ and $P_{prior}$.

When predicting functional words, punctuation, or inherent syntactic structures, the token is governed almost entirely by local textual context. For such tokens, removing visual attention at layer $L$ yields negligible changes in the output distribution ($P_{base} \approx P_{prior}$). Conversely, for semantic visual entities, the divergence is substantial. We quantify this token-level visual reliance using the symmetric Jensen-Shannon Divergence (JSD):
\[
D_{JS}(P_{base} \parallel P_{prior}) = \frac{1}{2} D_{KL}\left(P_{base} \parallel M\right) + \frac{1}{2} D_{KL}\left(P_{prior} \parallel M\right)
\tag{6}
\]
where $M = \frac{1}{2}(P_{base} + P_{prior})$. The dynamic syntactic filter $\Phi_t \in [0, 1)$ is formulated as:
\[
\Phi_t = 1 - \exp\left( -\kappa \cdot D_{JS}(P_{base} \parallel P_{prior}) \right)
\tag{7}
\]
where $\kappa$ is a constant scaling factor. For structural tokens, $D_{JS} \to 0$, causing $\Phi_t \to 0$, naturally protecting the sequence's syntactic integrity without heuristic assumptions.

---

## 3.4 Plausibility-Bounded Contrastive Decoding

Uncalibrated linear additions in the logit space frequently trigger logit saturation and probability peaking, destroying generation quality. To safely steer the distribution, we enforce an Adaptive Plausibility Constraint during contrastive decoding, restricting the intervention strictly to tokens the model already considers contextually plausible.

We define a dynamic candidate set $\mathcal{V}_{head}$ comprising tokens whose base probability exceeds a dynamic fraction $\rho$ of the maximum predicted probability:
\[
\mathcal{V}_{head} = \left\{ x \in \mathcal{X} \mathrel{\Big|} P_{base}(x \mid \boldsymbol{x}_{<t}) \ge \rho \max_{v \in \mathcal{X}} P_{base}(v \mid \boldsymbol{x}_{<t}) \right\}
\tag{8}
\]
We construct a truncation mask $\mathcal{M}_t \in \mathbb{R}^{|\mathcal{X}|}$ where $\mathcal{M}_t(x) = 0$ if $x \in \mathcal{V}_{head}$, and $-\infty$ otherwise. The final calibrated logits $\tilde{\boldsymbol{z}}_t$ are formulated by applying the divergence-gated contrastive penalty exclusively within this bounded subspace:
\[
\tilde{\boldsymbol{z}}_t = \boldsymbol{z}_t + \alpha \cdot \Phi_t \cdot \left( \boldsymbol{z}_t - \boldsymbol{z}_t^{prior} \right) + \mathcal{M}_t
\tag{9}
\]
where $\alpha$ regulates the steering strength. The final generation probability is:
\[
P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\tilde{\boldsymbol{z}}_t\right)
\tag{10}
\]
By isolating the visual counterfactual through exact architectural pathways, gating the intervention via instantaneous JS divergence, and bounding logit modifications within a plausible subspace, this framework dynamically natively suppresses visual hallucinations while maintaining absolute strictness regarding Transformer topologies and distribution constraints.