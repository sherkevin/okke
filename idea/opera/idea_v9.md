# 3 Method

We formulate the autoregressive generation procedure of multimodal large language models (MLLMs) and introduce a theoretically grounded intervention framework: **Residual Visual Tracing** and **Counterfactual Contrastive Decoding**. This methodology completely discards arbitrary logit fusion, naive layer averaging, and heuristic attention thresholding. By leveraging the linear additive nature of Transformer residual streams to isolate visual semantics, our approach contrasts the full predictive distribution with an ungrounded linguistic prior. This natively suppresses hallucinations while naturally preserving the structural integrity of syntactic tokens through causal representation dynamics, avoiding unsupported theoretical claims or static hyperparameter reliance.

---

## 3.1 Formulation of MLLMs Generation

The autoregressive generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input sequence comprises visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM representation dimension $d$, alongside textual prefix and generated tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context at decoding step $t$ is denoted as $\boldsymbol{x}_{<t}$.

### Model Forward and Decoding
At step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t^{(L)} \in \mathbb{R}^d$, which encodes the contextualized next-token representation. A predefined vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ maps $\boldsymbol{h}_t^{(L)}$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P_{base}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t^{(L)}
\tag{1}
\]
Crucially, in causal LLMs, $\boldsymbol{h}_t^{(L)}$ is structurally optimized to predict the *subsequent* token $x_t$. Directly projecting intermediate visual representations into $W$ violates the causal alignment of the unembedding space. Therefore, our method operates exclusively on the predictive text states $\boldsymbol{h}_t^{(L)}$ to ensure manifold consistency.

---

## 3.2 Residual Visual Tracing

To isolate the visual dependence of the current prediction without falling into the trap of "Attention Sinks" (which disproportionately affect lower layers and initial structural tokens), we analyze the model through the lens of Mechanistic Interpretability. Modern Transformers build representations via a linear additive residual stream.

At each layer $l$, the hidden state is updated via attention and feed-forward sub-layers:
\[
\boldsymbol{h}_t^{(l)} = \boldsymbol{h}_t^{(l-1)} + \Delta \boldsymbol{h}_{t, Attn}^{(l)} + \Delta \boldsymbol{h}_{t, MLP}^{(l)}
\tag{2}
\]
The attention output can be explicitly decomposed into contributions from visual tokens ($V$) and textual tokens ($T$). Let $A_{t,i}^{(l)}$ be the self-attention weight from the current text token $t$ to context token $i$ at layer $l$, and $W_O^{(l)}, W_V^{(l)}$ be the standard attention projection matrices. The isolated visual update is:
\[
\Delta \boldsymbol{v}_t^{(l)} = W_O^{(l)} \sum_{i \in V} A_{t,i}^{(l)} W_V^{(l)} \boldsymbol{h}_i^{(l-1)}
\tag{3}
\]
Since lower layers predominantly process local textual syntax and are heavily biased by structural attention sinks, while higher layers route abstract cross-modal semantics, we trace the accumulated visual contribution strictly across the deep semantic layers $l \in [L_{sem}, L]$ (e.g., $L_{sem} = L/2$):
\[
\boldsymbol{v}_t = \sum_{l=L_{sem}}^{L} \Delta \boldsymbol{v}_t^{(l)}
\tag{4}
\]
This mathematically isolates the aggregated visual evidence $\boldsymbol{v}_t$ that causally influences the final predictive state $\boldsymbol{h}_t^{(L)}$, completely bypassing historical temporal averaging and lower-layer sink noise.

---

## 3.3 Counterfactual Prior Projection

Hallucinations in MLLMs primarily occur when the model ignores visual evidence and defaults to its parametric linguistic prior (the "snowballing" effect). To detect and penalize this, we construct a counterfactual hidden state that simulates what the model would predict if the visual evidence were removed from the immediate semantic reasoning steps.

Given the linearity of the residual stream, we approximate the ungrounded linguistic prior state $\boldsymbol{h}_t^{prior}$ by subtracting the accumulated visual contribution from the final predictive state:
\[
\boldsymbol{h}_t^{prior} = \boldsymbol{h}_t^{(L)} - \boldsymbol{v}_t
\tag{5}
\]
We then project this counterfactual state through the unmodified unembedding matrix $W$ to obtain the ungrounded prior logits $\boldsymbol{z}_t^{prior}$:
\[
\boldsymbol{z}_t^{prior} = W \boldsymbol{h}_t^{prior}, \quad P_{prior}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t^{prior}\right)
\tag{6}
\]
This operation is strictly compliant with the causal modeling architecture. Both $\boldsymbol{h}_t^{(L)}$ and $\boldsymbol{h}_t^{prior}$ exist in the valid predictive subspace of $W$, circumventing any risk of catastrophic logit collapse associated with heterogeneous visual-to-text mapping.

---

## 3.4 Adaptive Contrastive Intervention

With the base predictive distribution $P_{base}$ and the counterfactual linguistic prior $P_{prior}$ cleanly separated, we perform contrastive decoding to penalize tokens that rely disproportionately on textual priors rather than visual evidence.

The calibrated logits $\tilde{\boldsymbol{z}}_t$ are formulated as:
\[
\tilde{\boldsymbol{z}}_t = \boldsymbol{z}_t + \alpha \cdot \gamma_t \cdot \left( \boldsymbol{z}_t - \boldsymbol{z}_t^{prior} \right)
\tag{7}
\]
where $\alpha$ is a scaling hyperparameter. To ensure empirical stability and preserve structural syntactic generation, we introduce a dynamic scaling factor $\gamma_t$ based on the relative magnitude of the visual contribution:
\[
\gamma_t = \frac{\|\boldsymbol{v}_t\|}{\|\boldsymbol{h}_t^{(L)}\|}
\tag{8}
\]

**Mechanism of Syntactic Preservation:** 
When predicting functional tokens, punctuation, or inherent syntactic structures, the predictive state is heavily dominated by the textual context. Consequently, the attention routed to visual tokens in the deep semantic layers approaches zero, resulting in $\|\boldsymbol{v}_t\| \approx 0$ and $\gamma_t \approx 0$. The intervention term naturally vanishes ($\tilde{\boldsymbol{z}}_t \approx \boldsymbol{z}_t$), leaving syntactic fluency entirely intact. Conversely, when generating visually grounded entities, $\|\boldsymbol{v}_t\|$ is substantial. If the model attempts to hallucinate an entity supported only by the language prior ($\boldsymbol{z}_t^{prior} \approx \boldsymbol{z}_t$), the contrastive penalty demotes it, forcefully shifting the probability mass toward entities uniquely derived from the visual evidence $\boldsymbol{v}_t$.