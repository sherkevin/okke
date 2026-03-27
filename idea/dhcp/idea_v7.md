# 3 Method

## 3.1 First-Order Visual Attribution in Residual Streams
We first formally define the cross-modal interaction mechanisms within large vision-language models (LVLMs). An input image $x_i$ is mapped by a visual encoder and projector into a sequence of $N$ visual tokens. These are prepended to the text embeddings and fed into the LVLM, which generates a sequence of $M$ tokens autoregressively.

Standard attention weights $\mathcal{A}_{m,n,l,h}$ fundamentally fail to represent actual information flow because they isolate the routing probabilities from the Value ($W_V$) and Output ($W_O$) transformations. To rigorously quantify multimodal feature attribution, we must explicitly model the information accumulation within the Transformer's residual stream. 

In a standard LVLM, the hidden state of the $m$-th token at the final layer $L$ is an accumulation of both Attention and Feed-Forward (MLP) residual updates: $h_m^{(L)} = h_m^{(0)} + \sum_{l=1}^L \left( \Delta h_{m, \text{Attn}}^{(l)} + \Delta h_{m, \text{MLP}}^{(l)} \right)$. To isolate the specific visual routing mechanism, we define the additive contribution of the $n$-th visual patch at layer $l$ to the attention-based residual update $\Delta h_{m, \text{Attn}}^{(l)}$ as:
\[
c_{m,n,l} = \sum_{h=1}^H \mathcal{A}_{m,n,l,h} \cdot \left( v_{n, l-1} W_V^{l,h} W_O^{l,h} \right) \tag{1}
\]
where $v_{n, l-1}$ is the visual hidden state preceding layer $l$. 

By leveraging the standard KV-cache mechanism in Decoder-only architectures, the transformed value vectors $z_{n,l,h} = v_{n, l-1} W_V^{l,h} W_O^{l,h}$ corresponding to the visual prompt remain static during autoregressive text generation. Consequently, $z_{n,l,h}$ can be strictly pre-computed once. During generation, extracting the exact visual contribution $c_{m,n,l}$ reduces to an efficient $\mathcal{O}(N \cdot D)$ linear scaling per layer using the dynamic attention weights, seamlessly scaling to high-resolution images without introducing dynamic matrix multiplication bottlenecks.

---

## 3.2 LayerNorm-Scaled Direct Logit Attribution
A core challenge in hallucination detection is tracing how intermediate visual contributions directly influence the final vocabulary distribution, while accounting for the non-linear transformations inherent to the architecture. Prior methods attempting to compute Direct Logit Attribution (DLA) mathematically fail by ignoring the non-linear scaling of the final LayerNorm (or RMSNorm) preceding the language modeling head.

To accurately quantify the influence of a visual patch on the generated token, we approximate the marginal contribution of $c_{m,n,l}$ through the final normalization layer using a first-order Taylor expansion. Given $\text{RMS}(h_m^{(L)})$, the scaling factor is approximately $\gamma / \text{RMS}(h_m^{(L)})$, where $\gamma$ is the learned weight of the normalization layer. The normalized visual contribution is thus defined as:
\[
\tilde{c}_{m,n,l} = \frac{c_{m,n,l} \odot \gamma}{\text{RMS}(h_m^{(L)})} \tag{2}
\]
We then project this scaled contribution onto the target token's unembedding vector $W_\text{unbed}[y_m]$ to derive the **Polarity-Aware Logit Attribution**:
\[
S_{m,n,l} = \tilde{c}_{m,n,l} \cdot W_\text{unbed}[y_m] \tag{3}
\]
Crucially, this formulation inherently resolves two major artifacts without relying on heuristic masking or arbitrary truncations:
1. **Natural Sink Suppression:** "Attention Sinks" (e.g., boundary patches) consistently absorb high attention mass to satisfy the Softmax sum-to-one property, but they lack specific semantic vectors. Consequently, their projection onto specific semantic vocabulary logits via $W_\text{unbed}$ naturally evaluates to near-zero, systematically filtering out query-agnostic biases.
2. **Inhibitory Signals Preservation:** By retaining the raw scalar value without ReLU truncation, negative values of $S_{m,n,l}$ explicitly quantify how much a specific visual patch *suppressed* the generation of token $y_m$. These inhibitory signals are highly indicative of conflict between visual evidence and generated text (a strong predictor of hallucination).

To assess the spatial focus of the positive visual evidence, we extract the excitatory distribution over the spatial dimension $N$ by filtering $S_{m,n,l}^+ = \max(0, S_{m,n,l})$ and computing its spatial entropy:
\[
E_{m,l} = -\sum_{n=1}^N P_{m,n,l}^+ \log P_{m,n,l}^+, \quad \text{where } P_{m,n,l}^+ = \frac{S_{m,n,l}^+}{\sum_j S_{m,j,l}^+} \tag{4}
\]

---

## 3.3 Training-Free Hallucination Evaluator
Prior prediction modules (such as CNNs, GRUs, or Cross-Layer Transformers) applied to the layer sequence inevitably suffer from redundant sequence modeling, parameter bloat, and severe overfitting on limited hallucination datasets, fundamentally ignoring that the final layer already encapsulates the full causal history of the residual stream.

To guarantee mathematical robustness and semantic integrity, we introduce a **Training-Free Hallucination Evaluator**. We abandon trainable neural predictors and instead formulate a zero-shot metric based directly on our layer-wise DLA statistics. For a generated token $m$, hallucination fundamentally stems from two observable phenomena in the attribution space:
1. **Insufficient Visual Grounding:** The cumulative unnormalized visual logit contribution across all layers, $V_m = \sum_{l=1}^L \sum_{n=1}^N S_{m,n,l}$, is low or negative, indicating the token is driven primarily by textual priors or MLP hallucinations rather than visual evidence.
2. **Semantic Defocusing:** The visual evidence is highly dispersed across the image rather than grounded in specific semantic objects, reflected by a high average spatial entropy $\bar{E}_m = \frac{1}{L} \sum_{l=1}^L E_{m,l}$.

We construct a unified token-level hallucination score $\mathcal{H}_m \in [0,1]$ that acts as a continuous probability of hallucination:
\[
\mathcal{H}_m = \sigma\left( \beta \cdot \bar{E}_m - \lambda \cdot V_m \right) \tag{5}
\]
where $\sigma$ is the sigmoid function, and $\beta, \lambda$ are standard scaling hyper-parameters to map the distribution to $[0,1]$. 

By substituting black-box neural predictors with a fully interpretable, mathematically grounded aggregation of LayerNorm-scaled logit contributions, this approach completely eliminates the risk of overfitting, preserves explicit semantic directions in the high-dimensional space, and yields a lightweight, plug-and-play hallucination detector applicable to any standard Decoder-only LVLM without requiring any fine-tuning.