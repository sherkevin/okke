# 3 Method

## 3.1 Causal Isolation of Multimodal Contributions via Contrastive States
We first formally define the multimodal generation process within large vision-language models (LVLMs). An input image $x_v$ is tokenized and prepended to the text prompt $x_t$, fed into the LVLM to generate a sequence of tokens $Y = \{y_1, y_2, \dots, y_M\}$ autoregressively. 

Prior mechanistic interpretability methods severely suffer from the *feature entanglement* problem. In deep Transformer layers, dense cross-attention and Feed-Forward Networks (MLPs) deeply mix visual and textual representations. Naively partitioning attention updates based on input indices explicitly ignores the fact that deep "visual tokens" are heavily contextualized with text. Furthermore, MLPs process nearly two-thirds of the model's parameters and act as critical associative memories for cross-modal alignment. Excluding MLP updates from the attribution process fundamentally truncates the actual causal pathways of the model.

To rigorously trace the origin of hallucinations without violating the non-linear dynamics of LVLMs, we abandon brittle intermediate-layer linear approximations. Instead, we formulate the visual attribution as a precise **Causal Intervention**. Specifically, we perform a dual-stream forward pass. The primary stream maintains the full multimodal context to compute the exact final hidden state $h_m^{(L)} = \text{LVLM}(x_v, x_t, Y_{<m})$. Concurrently, we maintain a parallel textual-only generation stream (via visual masking or utilizing a visual-blank prompt) to capture the unconditioned textual hidden state $\tilde{h}_m^{(L)} = \text{LVLM}(\emptyset, x_t, Y_{<m})$.

By isolating the representations at the final layer $L$, the strictly rigorous causal visual contribution to the $m$-th token's hidden state is the exact difference:
\[
\Delta h_{m}^{\text{vis}} = h_m^{(L)} - \tilde{h}_m^{(L)} \tag{1}
\]
This system-level causal difference inherently encapsulates all non-linear transformations, deep feature entanglement, and the extensive parametric mapping performed by MLPs across all layers, completely overcoming the theoretical fatal flaws of partial linear decomposition.

---

## 3.2 Strict Output Space Attribution
A well-documented phenomenon in multimodal architectures is that visual features must undergo deep non-linear transformations before they are semantically aligned with the discrete natural language vocabulary space. Forcing intermediate visual attention updates to project directly onto the unembedding matrix $W_\text{unbed}$ via the "direct pathway" yields mathematically meaningless noise, as those intermediate features have not yet been mapped to the logit space.

Because our causal visual contribution $\Delta h_{m}^{\text{vis}}$ is derived exclusively at the *final* layer—after all essential multimodal alignment has concluded—we can strictly and accurately project it into the output vocabulary space. Let $\text{RMS}(\cdot)$ denote the final normalization layer with learned parameter $\gamma$. The final multimodal logit vector $Z_m$ and the text-prior logit vector $\tilde{Z}_m$ are computed as:
\[
Z_m = \left( \frac{h_m^{(L)} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) W_\text{unbed}, \quad \tilde{Z}_m = \left( \frac{\tilde{h}_m^{(L)} \odot \gamma}{\text{RMS}(\tilde{h}_m^{(L)})} \right) W_\text{unbed} \tag{2}
\]
We define the **Causal Visual Anchoring (CVA)** for the generated token $y_m$ strictly as the exact logit difference driven by the visual input:
\[
\text{CVA}(y_m) = Z_m[y_m] - \tilde{Z}_m[y_m] \tag{3}
\]
This formulation perfectly captures the semantic polarity of the visual influence without arbitrary Taylor expansion approximations. A positive $\text{CVA}(y_m)$ guarantees that the visual features collectively promoted the token. Crucially, a negative $\text{CVA}(y_m)$ indicates explicit inhibition, providing mathematically sound proof that the visual context actively conflicted with and suppressed the text-driven generation of $y_m$.

---

## 3.3 Semantic-Aware Hallucination Risk Assessment
Visual hallucinations occur when the model asserts concrete, descriptive entities driven entirely by textual priors or semantic drift, contradicting the actual visual evidence. However, naive attribution metrics fail on structural language components: functional/syntactic tokens (e.g., "the", "is", "a") are naturally driven by the textual context. A mathematically sound evaluator must isolate semantic entities without relying on external Part-of-Speech taggers or threshold-heavy heuristics.

To systematically filter out syntactic noise and accurately quantify hallucination risks, we introduce a **Semantic-Aware Hallucination Risk Score** $\mathcal{R}_m$. We first quantify the inherent text-prior reliance of token $y_m$ using the softmax probability from the text-only stream:
\[
P_{\text{txt}}(y_m) = \frac{\exp(\tilde{Z}_m[y_m])}{\sum_{j} \exp(\tilde{Z}_m[j])} \tag{4}
\]
Functional tokens inherently exhibit high $P_{\text{txt}}(y_m)$ regardless of visual input, whereas specific visual entities exhibit high uncertainty in the absence of an image. Thus, we define the semantic importance weight as $w_m = 1 - P_{\text{txt}}(y_m)$, which dynamically heavily penalizes the risk contribution of stop-words and grammar structures.

The token-level hallucination risk is then formulated as the degree to which a semantically critical token is actively inhibited by the visual context relative to its text-prior generation:
\[
\mathcal{R}_m = w_m \cdot \text{ReLU}\left( \frac{\tilde{Z}_m[y_m] - Z_m[y_m]}{\max(|Z_m[y_m]|, \epsilon)} \right) \tag{5}
\]
where $\epsilon$ ensures numerical stability (e.g., $\epsilon=1$).

This metric is highly interpretable and avoids arbitrary bounding functions (e.g., Sigmoid) that inappropriately map neutral states to high risk values. If a token is supported by the image ($Z_m \ge \tilde{Z}_m$), the ReLU completely zeroes out the risk ($\mathcal{R}_m = 0$). If a semantic token ($w_m \approx 1$) is strongly pushed by the textual prior but explicitly suppressed by the visual pathways ($Z_m \ll \tilde{Z}_m$), $\mathcal{R}_m$ yields a substantial positive penalty. This parameter-free formulation dynamically isolates visually-unsupported semantic fabrications, providing a theoretically robust signal for hallucination detection across diverse multimodal contexts.