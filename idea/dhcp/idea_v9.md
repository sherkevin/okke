# 3 Method

## 3.1 Mechanistic Decomposition of Contextual Routing
We first formally define the information routing mechanisms within large vision-language models (LVLMs). An input image $x_i$ is mapped into visual tokens, prepended to the text prompt, and fed into the LVLM to generate a sequence of $M$ tokens autoregressively. Let $\mathcal{V}$ and $\mathcal{T}$ denote the sets of indices corresponding to the visual and textual input sequences, respectively.

In standard Transformer architectures, token representations become heavily mixed in deep layers due to dense cross-attention and Feed-Forward Network (MLP) processing. To trace the origin of hallucinations, we avoid the naive assumption that distinct modules solely process distinct modalities. Instead, we adopt a mechanistic framework that isolates the specific *routing sources* within the attention mechanisms. 

The final hidden state $h_m^{(L)}$ for the $m$-th generated token is the sum of the initial embedding and all intermediate residual updates:
\[
h_m^{(L)} = h_m^{(0)} + \sum_{l=1}^L \Delta h_{m, \text{Attn}}^{(l)} + \sum_{l=1}^L \Delta h_{m, \text{MLP}}^{(l)} \tag{1}
\]
While the MLP acts as a shared non-linear fusion module and parametric memory, the attention mechanism explicitly selectively routes information from preceding token positions. We partition the attention update $\Delta h_{m, \text{Attn}}^{(l)}$ into the visually-routed contribution ($U_{m,l}^{\text{vis}}$) and the textually-routed contribution ($U_{m,l}^{\text{txt}}$):
\[
U_{m,l}^{\text{vis}} = \sum_{n \in \mathcal{V}} \sum_{h=1}^H \mathcal{A}_{m,n,l,h} \cdot \left( v_{n, l-1} W_V^{l,h} W_O^{l,h} \right) \tag{2}
\]
\[
U_{m,l}^{\text{txt}} = \sum_{k \in \mathcal{T}} \sum_{h=1}^H \mathcal{A}_{m,k,l,h} \cdot \left( t_{k, l-1} W_V^{l,h} W_O^{l,h} \right) \tag{3}
\]
To maintain semantic integrity, we explicitly exclude known structural "Attention Sinks" (e.g., `<bos>`, `<pad>`, and fixed boundary tokens) from both sets via an index mask. Efficiently computing these aggregated updates requires only standard matrix-vector operations, easily scaling to high-resolution visual inputs without architectural modifications.

---

## 3.2 Direct Visual Anchoring in Residual Streams
While multi-step logical reasoning heavily utilizes indirect pathways (where one layer's output is non-linearly transformed by subsequent MLPs), mechanistic interpretability has empirically established that the generation of concrete visual entities strongly depends on the *direct pathway*—the mechanism by which intermediate features bypass subsequent layers via the residual connection to directly influence the final output logit.

To rigorously evaluate whether a generated token is explicitly grounded in the visual input, we isolate this direct pathway. Accounting for the non-linear scaling of the final RMSNorm via a first-order Taylor expansion over the normalization parameters $\gamma$, we define the Direct Visual Anchoring (DVA) to the generated token $y_m$ as:
\[
\text{DVA}(y_m) = \sum_{l=1}^L \left( \frac{U_{m,l}^{\text{vis}} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) \cdot W_\text{unbed}[y_m] \tag{4}
\]
Simultaneously, we quantify the Direct Textual Anchoring (DTA) to measure the token's reliance on the text prompt context:
\[
\text{DTA}(y_m) = \sum_{l=1}^L \left( \frac{U_{m,l}^{\text{txt}} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) \cdot W_\text{unbed}[y_m] \tag{5}
\]
By strictly comparing the routed sources rather than misattributing all MLP updates to "language priors", this formulation reliably captures semantic polarity. A highly positive $\text{DVA}(y_m)$ indicates strong, direct extraction of visual evidence to support the token. Conversely, a negative $\text{DVA}(y_m)$ explicitly quantifies an inhibitory signal, revealing that the visually-routed features actively suppressed the generation of $y_m$.

---

## 3.3 Continuous Hallucination Risk Assessment
Visual hallucinations in LVLMs frequently occur when the model generates visually descriptive tokens driven predominantly by textual context or internal semantic drifts, while the direct visual evidence is either absent or actively contradictory.

Rather than formulating a rigid binary classifier that requires tuning arbitrary thresholds, we introduce a continuous Hallucination Risk Score $\mathcal{R}_m \in [0, 1]$. This metric captures the degree of multi-modal conflict—specifically, when a token relies heavily on textual routing but lacks corresponding direct visual support:
\[
\mathcal{R}_m = \sigma\left( \frac{\text{DTA}(y_m) - \text{DVA}(y_m)}{|\text{DTA}(y_m)| + |\text{DVA}(y_m)| + \epsilon} \right) \tag{6}
\]
where $\sigma(\cdot)$ is the standard sigmoid function to normalize the output into a probability-like continuous range, and $\epsilon$ ensures numerical stability.

This relative risk formulation naturally mitigates the conflation of hallucination with valid language reasoning. If a token logically follows from both text and vision (both DTA and DVA are high), the difference is negligible, resulting in a low risk score. However, if the text strongly promotes a token ($\text{DTA}(y_m) \gg 0$) while the direct visual pathway actively inhibits it ($\text{DVA}(y_m) < 0$), $\mathcal{R}_m$ aggressively approaches 1. This continuous metric serves as a robust indicator of generation uncertainty, which can be directly employed for downstream evaluation or adaptive thresholding in contrastive decoding interventions, without requiring any trainable parameters or external hallucination annotations.