# 3 Method

## 3.1 Mechanistic Decomposition of Residual Streams
We first formally define the information routing mechanisms within large vision-language models (LVLMs). An input image $x_i$ is mapped into $N$ visual tokens, prepended to the text embeddings, and fed into the LVLM to generate $M$ tokens autoregressively.

Standard interpretability methods fail to isolate visual influence because they ignore the fundamental role of Feed-Forward Networks (MLPs) as parametric key-value memories, which often dominate token generation and inject strong language priors. To rigorously trace hallucinations, we adopt a mechanistic interpretability framework that explicitly decomposes the Transformer's residual stream into distinct modal and functional contributions.

In a standard LVLM, the final hidden state $h_m^{(L)}$ for the $m$-th token is the exact sum of the initial embedding and all intermediate residual updates from both Attention and MLP sub-layers:
\[
h_m^{(L)} = h_m^{(0)} + \sum_{l=1}^L \Delta h_{m, \text{Attn}}^{(l)} + \sum_{l=1}^L \Delta h_{m, \text{MLP}}^{(l)} \tag{1}
\]
The attention update $\Delta h_{m, \text{Attn}}^{(l)}$ routes information from the context sequence. We partition this update into the visual contribution ($U_{m,l}^{\text{vis}}$) and the textual/system contribution ($U_{m,l}^{\text{txt}}$). Specifically, the aggregated visual update at layer $l$ is:
\[
U_{m,l}^{\text{vis}} = \sum_{n \in \mathcal{V}} \sum_{h=1}^H \mathcal{A}_{m,n,l,h} \cdot \left( v_{n, l-1} W_V^{l,h} W_O^{l,h} \right) \tag{2}
\]
where $\mathcal{V}$ is the set of valid visual token indices. Crucially, we explicitly exclude known structural "Attention Sinks" (e.g., `<bos>`, `<pad>`, and image boundary tokens) from $\mathcal{V}$ via an explicit index mask, completely avoiding dangerous assumptions about the orthogonality of sink features in the semantic space. 

By aggregating the visual contribution *before* vocabulary projection, the computation reduces from $\mathcal{O}(N \cdot D)$ to exactly $\mathcal{O}(D)$ per layer during inference, definitively resolving the computational bottleneck for high-resolution images.

---

## 3.2 Contrastive Direct Logit Attribution
The Transformer architecture routes these decomposed residual vectors through a final normalization layer and an unembedding matrix $W_\text{unbed}$. While intermediate layers perform complex non-linear computations, the *direct pathway* (how a specific layer's output bypasses subsequent layers via the residual connection to directly influence the final logit) is a well-established linear approximation in mechanistic interpretability. 

To account for the final non-linear RMSNorm scaling while quantifying this direct causal effect, we apply a first-order Taylor expansion over the final normalization parameters $\gamma$. The Direct Logit Attribution (DLA) of the aggregated visual stream to the final generated token $y_m$ is computed as:
\[
\text{DLA}_{\text{vis}}(y_m) = \sum_{l=1}^L \left( \frac{U_{m,l}^{\text{vis}} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) \cdot W_\text{unbed}[y_m] \tag{3}
\]
Simultaneously, we quantify the direct contribution of the model's internal parametric memory (the MLP layers) to the same token, which represents the language prior:
\[
\text{DLA}_{\text{MLP}}(y_m) = \sum_{l=1}^L \left( \frac{\Delta h_{m, \text{MLP}}^{(l)} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) \cdot W_\text{unbed}[y_m] \tag{4}
\]
Unlike spatial entropy metrics—which incorrectly penalize valid global visual concepts (e.g., "weather", "crowd") that naturally distribute attention broadly—this token-level DLA framework explicitly measures *semantic polarity*. A negative $\text{DLA}_{\text{vis}}(y_m)$ provides a direct inhibitory signal, mathematically indicating that the visual evidence inherently suppressed the generation of token $y_m$, whereas a highly positive $\text{DLA}_{\text{MLP}}(y_m)$ indicates strong reliance on language priors.

---

## 3.3 Parameter-Free Hallucination Evaluator
Visual hallucinations in LVLMs primarily manifest when the model generates visually concrete tokens driven entirely by textual priors (MLP memorization) while explicitly ignoring or contradicting the actual visual context.

To eliminate the over-fitting risks associated with heuristic hyper-parameters ($\beta, \lambda$) or trainable linear probes, we introduce a strictly parameter-free, contrastive metric. We define the token-level Hallucination Confidence $\mathcal{H}_m$ as the degree to which language priors override contradictory or absent visual evidence:
\[
\mathcal{H}_m = \frac{\max\left(0, \text{DLA}_{\text{MLP}}(y_m) - \text{DLA}_{\text{vis}}(y_m)\right)}{|\text{DLA}_{\text{MLP}}(y_m)| + |\text{DLA}_{\text{vis}}(y_m)| + \epsilon} \tag{5}
\]
where $\epsilon$ is a standard numerical stability constant. 

This formulation naturally resolves the conflation of "valid language reasoning" and "hallucination". If the model relies on common sense to answer a question, but the visual evidence is supportive or neutral ($\text{DLA}_{\text{vis}}(y_m) \ge 0$), the penalty is minimal. However, if the MLP strongly pushes for a token while the visual direct pathway actively inhibits it ($\text{DLA}_{\text{vis}}(y_m) < 0$), the numerator maximizes, accurately flagging severe multimodal conflict. 

By contrasting direct visual routing against MLP memorization at the logit level, this method functions as a mathematically rigorous, zero-shot hallucination detector that respects the non-linear dynamics of the residual stream without requiring any external training data or threshold tuning.