# 3 Method

## 3.1 Token-Level Cross-modal Attention Extraction
We first formulate the cross-modal attention mechanisms within large vision-language models (LVLMs). Let $x_i$ and $x_t$ denote the input image and text query. The visual encoder extracts features mapped by a projector into a sequence of $N$ visual tokens. These are prepended to the text embeddings and fed into the LVLM, which generates a sequence of $M$ tokens autoregressively. Advanced architectures (e.g., Qwen2.5-VL) employ a dynamic visual sequence length $N$ conditioned on the varying image resolution.

Let $\mathcal{A}_{m,n,l,h}(x_i, x_t)$ denote the attention weight from the $m$-th generated textual token to the $n$-th visual token at the $l$-th layer and $h$-th attention head. Because hallucinations are fine-grained phenomena occurring at specific tokens or spans rather than uniformly across the entire response, aggregating attention across the generation dimension $M$ would severely destroy the causal alignment between the generated concepts and their visual evidence. To preserve this strict alignment and maintain the underlying probability distribution properties established by the Softmax function, we extract the intact attention tensor for *each* generated token $m$:
\[
\mathcal{A}_m \in \mathbb{R}^{N \times L \times H} \tag{1}
\]
By analyzing $\mathcal{A}_m$ sequentially, we shift the paradigm from coarse-grained response-level evaluation to fine-grained, token-level hallucination detection, ensuring exact spatial-temporal causality.

---

## 3.2 Quantitative Distributional Discrepancy & Sink-Filtering
To rigorously investigate attention patterns, we must first account for the "Attention Sinks" phenomenon—a well-documented LLM mechanism where massive attention weight is consistently allocated to semantically meaningless tokens (e.g., the `<bos>` token, punctuation, or fixed image boundary patches) to preserve attention scores. Treating these benign, pervasive high-activation sinks as hallucinations is mathematically and theoretically flawed.

We first apply a heuristic sink-filtering mask to exclude known non-semantic visual patches, resulting in a normalized attention distribution $\hat{\mathcal{A}}_{m, :, l, h}$ over the valid visual tokens. To quantitatively capture hallucination characteristics, we measure two distinct distributional statistics for each layer $l$ and head $h$:
1. **Attention Entropy (Defocusing)**: Measures the dispersion of attention across visual patches. High entropy indicates that the model fails to ground the textual token in specific visual evidence.
\[
E_{m,l,h} = - \sum_{n} \hat{\mathcal{A}}_{m,n,l,h} \log \hat{\mathcal{A}}_{m,n,l,h} \tag{2}
\]
2. **Maximum Activation (Spurious Concentration)**: Measures the peak attention weight. A structurally correct but semantically flawed activation (e.g., confidently attending to the wrong patch).
\[
S_{m,l,h} = \max_{n} \hat{\mathcal{A}}_{m,n,l,h} \tag{3}
\]

Statistical analysis across truthful and hallucinatory tokens reveals a profound spatio-hierarchical discrepancy. Truthful tokens exhibit distinct low-entropy, high-activation signatures in deep layers. Conversely, hallucinations quantifiably manifest either as an extreme entropy spike (attention uniformly diffusing across the background) or a misaligned high-activation spike (confidently generating a non-existent concept based on a spurious local patch).

---

## 3.3 DHCP: Distribution-aware Hierarchical Cross-modal Predictor
Previous methods often rely on simplistic flattening or complex self-attention over raw attention weights, which either destroys the structural hierarchy of LLM layers or over-engineers the problem, leading to severe overfitting. Furthermore, global max-pooling operations inherently eliminate the crucial "defocusing/diffusion" patterns described above. To resolve this, we propose the **Distribution-aware Hierarchical Cross-modal Predictor (DHCP)**, which learns directly from the dimension-invariant statistical metrics rather than the raw, variable-length spatial tokens.

For each generated token $m$, we compute a comprehensive set of distribution statistics (Entropy, Maximum, Variance, and Top-K mass) over the visual dimension $N$ for every layer $l$ and head $h$. This transforms the raw attention tensor $\mathcal{A}_m$ into a robust, resolution-agnostic feature map:
\[
\Phi_m \in \mathbb{R}^{L \times (H \cdot C)} \tag{4}
\]
where $C$ is the number of statistical metrics. Because these metrics are computed over the probability distributions, $\Phi_m$ is intrinsically immune to the dynamic variation of $N$, seamlessly supporting arbitrary image resolutions without rigid cropping or pooling.

To respect the hierarchical nature of LLMs—where shallow layers capture low-level correlations and deep layers form complex semantic alignments—we treat $\Phi_m$ as a sequential trace along the depth of the network. We employ a lightweight 1D-Convolutional network along the layer dimension $L$ to capture layer-wise transitions and dependencies, followed by an MLP layer:
\[
Z_m = \text{1D-CNN}(\Phi_m), \quad p_m = \sigma(\text{MLP}(\text{Flatten}(Z_m))) \tag{5}
\]
where $p_m \in [0, 1]$ is the predicted probability of token $m$ being a hallucination. 

By predicting hallucinations at the **token level** using dimension-agnostic distribution statistics, DHCP elegantly resolves the contradiction between dynamic resolutions and architectural rigidity. It avoids the destructive nature of global pooling, preserves the true probability attributes of the attention mechanism, and provides granular, actionable hallucination localization even in extensively long and complex descriptive generations.