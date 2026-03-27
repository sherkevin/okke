# 3 Method

We formulate the autoregressive generation procedure of multimodal large language models (MLLMs) and introduce a transparent, mechanistically grounded intervention framework: **Visual Residual Extraction** and **Direct Logit Steering**. This methodology strictly adheres to the non-linear dynamics and normalization structures of modern Transformers. By abandoning out-of-distribution hidden-state arithmetic and pseudo-counterfactual approximations, we leverage Direct Logit Attribution (DLA) to translate isolated visual attention residuals directly into vocabulary-space biases. Coupled with an adaptive baseline filter to account for structural attention tails, our approach suppresses hallucinations and preserves syntactic fluency without relying on hardcoded layer heuristics.

---

## 3.1 Formulation of MLLMs Generation

The autoregressive generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input sequence comprises visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM representation dimension $d$, alongside the textual prefix and generated tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context at decoding step $t$ is denoted as $\boldsymbol{x}_{<t}$.

### Model Forward and Decoding
At step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t^{(L)} \in \mathbb{R}^d$. Modern MLLMs (e.g., LLaMA-based architectures) universally apply a normalization layer (e.g., RMSNorm) before the predefined vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$. The unnormalized logits $\boldsymbol{z}_t$ are computed as:
\[
P_{base}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \operatorname{RMSNorm}(\boldsymbol{h}_t^{(L)})
\tag{1}
\]
Ignoring the normalization layer during representation manipulation leads to severe mathematical inconsistencies. Our intervention is designed to strictly respect this architectural constraint by operating directly in the logit space.

---

## 3.2 Mechanistic Visual Residual Extraction

To accurately measure the visual dependence of the current prediction, we analyze the model through the lens of Mechanistic Interpretability. The Transformer builds representations via a residual stream, where each layer $l$ updates the hidden state via attention and feed-forward (MLP) sub-layers:
\[
\boldsymbol{h}_t^{(l)} = \boldsymbol{h}_t^{(l-1)} + \Delta \boldsymbol{h}_{t, Attn}^{(l)} + \Delta \boldsymbol{h}_{t, MLP}^{(l)}
\tag{2}
\]
While MLP layers introduce deep non-linear feature entanglement, the attention mechanism explicitly routes information linearly. We can isolate the direct attention update originating purely from the visual tokens ($V$). Let $A_{t,i}^{(l)}$ be the self-attention weight from the current text token $t$ to context token $i$ at layer $l$, and $W_O^{(l)}, W_V^{(l)}$ be the projection matrices. The isolated visual update is:
\[
\Delta \boldsymbol{v}_t^{(l)} = W_O^{(l)} \sum_{i \in V} A_{t,i}^{(l)} W_V^{(l)} \boldsymbol{h}_i^{(l-1)}
\tag{3}
\]
Instead of relying on a hardcoded, static layer threshold (e.g., $L/2$) which fails to generalize across different architectures or image complexities, we aggregate these visual residuals across all layers dynamically. We define a layer-wise visual salience weight $m_t^{(l)} = \sum_{i \in V} A_{t,i}^{(l)}$, representing the total visual attention mass at layer $l$. The aggregated visual residual direction $\boldsymbol{v}_t$ is computed as a weighted sum:
\[
\boldsymbol{v}_t = \sum_{l=1}^{L} \frac{m_t^{(l)}}{\sum_{k=1}^L m_t^{(k)} + \epsilon} \Delta \boldsymbol{v}_t^{(l)}
\tag{4}
\]
This extracts a robust, dynamically routed visual direction vector while avoiding assumptions about static layer semantics.

---

## 3.3 Direct Visual Logit Attribution

Previous approaches attempt to subtract visual residuals from the final hidden state to form a "linguistic prior". However, this fundamentally ignores the non-linear processing of MLP layers, yielding an out-of-distribution noise vector that violates the pre-trained manifold. Furthermore, the linear distribution law does not hold through the RMSNorm layer.

To rigorously apply the extracted visual evidence $\boldsymbol{v}_t$, we utilize Direct Logit Attribution (DLA), projecting the isolated residual direction into the vocabulary space using the linear approximation of the final normalization layer. Let $\sigma_t$ be the computed variance scaling factor of the final state $\boldsymbol{h}_t^{(L)}$ during the RMSNorm operation:
\[
\sigma_t = \sqrt{\frac{1}{d} \|\boldsymbol{h}_t^{(L)}\|^2 + \epsilon}
\tag{5}
\]
The direct vocabulary-space contribution of the visual tokens, $\boldsymbol{z}_t^{vis}$, is explicitly decoded by scaling the visual residual and mapping it through $W$:
\[
\boldsymbol{z}_t^{vis} = W \left( \frac{\boldsymbol{v}_t}{\sigma_t} \right)
\tag{6}
\]
This mathematically translates the accumulated visual evidence into a direct logit bias without corrupting the intermediate hidden states or violating the non-linear architectural topology.

---

## 3.4 Adaptive Steering and Syntactic Filtering

With the visual logit contribution accurately mapped, we steer the final predictive distribution to promote visually grounded entities while penalizing ungrounded hallucinations. The calibrated logits $\tilde{\boldsymbol{z}}_t$ are formulated as a targeted bias addition:
\[
\tilde{\boldsymbol{z}}_t = \boldsymbol{z}_t + \alpha \cdot \Phi_t \cdot \boldsymbol{z}_t^{vis}
\tag{7}
\]
where $\alpha$ is a tunable hyperparameter controlling the empirical steering strength.

**Syntactic Preservation via Adaptive Baseline Thresholding:**
To ensure that functional words, punctuation, and inherent syntactic structures are not distorted, we must account for the "attention tail"—the structural phenomenon where the model allocates a non-zero baseline attention mass to visual prefixes even when predicting pure syntax. Thus, the relative visual magnitude $\gamma_t = \frac{\|\boldsymbol{v}_t\|}{\|\boldsymbol{h}_t^{(L)}\|}$ is never strictly zero.

We establish a dynamic syntactic filter $\Phi_t$ by continuously tracking the moving average of $\gamma$ over the generated sequence, denoted as $\bar{\gamma}_{<t}$. This moving average effectively quantifies the baseline structural attention tail:
\[
\Phi_t = \max\left(0,\, \gamma_t - \bar{\gamma}_{<t}\right)
\tag{8}
\]
When generating syntactic or structural tokens, the visual residual magnitude naturally falls near or below the moving average baseline ($\gamma_t \approx \bar{\gamma}_{<t}$), causing $\Phi_t$ to evaluate to zero and seamlessly leaving the base logits $\boldsymbol{z}_t$ unmodified. The steering mechanism engages exclusively ($\Phi_t > 0$) for semantic entity tokens that actively route significant information from the visual context, safely injecting the visual bias $\boldsymbol{z}_t^{vis}$ to enforce grounded generation.