# 3 Method

## 3.1 Vector-Norm Information Flow Extraction
We first define the cross-modal interaction within large vision-language models (LVLMs). Let $x_i$ and $x_t$ denote the input image and text query. The visual encoder extracts features that are projected into a sequence of $N$ visual tokens, where $N$ dynamically adapts to the image resolution. These tokens are prepended to the text embeddings and fed into the LVLM, generating a sequence of $M$ tokens autoregressively.

Standard attention weights $\mathcal{A}_{m,n,l,h}$ and heuristic Attention Rollout are inadequate for analyzing modern LVLMs (e.g., Qwen2.5-VL), as they ignore the feature transformations of Value matrices ($W_V$), output projections ($W_O$), and the accumulation within the residual stream. To accurately quantify the actual contribution of visual tokens to the generated text, we adopt a **Vector-Norm based Information Flow** approach. 

For the $m$-th generated token at layer $l$, the true magnitude of information transferred from the $n$-th visual patch $v_{n, l-1}$ via the $h$-th attention head is computed as the L2-norm of the transformed representation:
\[
I_{m,n,l,h} = \left\| \mathcal{A}_{m,n,l,h} \cdot \left( v_{n, l-1} W_{V}^{l,h} W_{O}^{l,h} \right) \right\|_2 \tag{1}
\]
By summing across all heads, we obtain the unified layer-wise visual contribution $I_{m,n,l} = \sum_h I_{m,n,l,h}$. This vector-norm metric naturally captures the non-linear transformations inherent to deep Transformers. To avoid the prohibitive $\mathcal{O}(M \cdot N \cdot L)$ memory footprint of storing full generation traces, $I_{m,n,l}$ is computed on-the-fly during the forward pass of token $m$ and immediately aggregated, ensuring scalable inference for long texts and high-resolution images.

---

## 3.2 Natural Sink Suppression and Expected Semantic Grounding
A fundamental challenge in attention analysis is the "Attention Sinks" phenomenon, where LVLMs allocate massive attention probabilities to semantically devoid patches (e.g., the `<bos>` token). Unlike raw attention probabilities, the vector-norm contribution $I_{m,n,l}$ naturally suppresses these sinks, because sink tokens typically act as resting states with small or constant transformed vector norms. We normalize these norms to form a robust, sink-resilient visual attribution distribution:
\[
\hat{\mathcal{P}}_{m,n,l} = \frac{I_{m,n,l}}{\sum_{j=1}^N I_{m,j,l}} \tag{2}
\]

To identify hallucinations driven by misaligned activations—where the model attends to a localized but semantically incorrect region—we must measure the semantic alignment between the visual evidence and the generated token $y_m$. Directly computing cosine similarity between intermediate hidden states is theoretically flawed because shallow features and deep linguistic representations do not reside in the same metric space. 

Instead, we apply the Logit Lens technique. We project the intermediate visual hidden state $v_{n,l}$ directly into the language model's vocabulary space using the pretrained LM head ($W_\text{unbed}$) and final LayerNorm ($\text{LN}_f$). The explicit semantic grounding score $S_{m,n,l}$ is the probability mass assigned to the generated token $y_m$ by the visual patch $n$:
\[
S_{m,n,l} = \text{Softmax}\left( W_\text{unbed} \left( \text{LN}_f(v_{n,l}) \right) \right)_{y_m} \tag{3}
\]
To ensure the process remains fully differentiable and robust to noise, we avoid hard `argmax` truncations. We compute the **Expected Semantic Grounding** $\mathbb{E}[G_{m,l}]$ by weighting the vocabulary projections with our information flow distribution:
\[
\mathbb{E}[G_{m,l}] = \sum_{n=1}^N \hat{\mathcal{P}}_{m,n,l} \cdot S_{m,n,l} \tag{4}
\]
Simultaneously, we compute the distribution entropy $E_{m,l} = -\sum_n \hat{\mathcal{P}}_{m,n,l} \log \hat{\mathcal{P}}_{m,n,l}$. This combination objectively captures both "Defocusing" (high entropy) and "Spurious Concentration" (low expected grounding despite localized high activation) without resorting to arbitrary manual thresholds.

---

## 3.3 DHCP: Depth-aware High-dimensional Cross-modal Predictor
Compressing the intermediate states of massive LVLMs into a few scalar statistics leads to severe feature collapse. Furthermore, modeling network depth via 1D-CNNs or GRUs incorrectly assumes local translation invariance or strict Markovian transitions, ignoring the fact that Transformers utilize a residual stream where each layer reads from and writes to a global representation.

To preserve the rich representational capacity and the non-Markovian nature of the residual stream, we propose the **Depth-aware High-dimensional Cross-modal Predictor (DHCP)**. For each layer $l$, rather than collapsing the spatial dimension into scalars, we extract the attribution-weighted visual context vector $C_{m,l} \in \mathbb{R}^D$:
\[
C_{m,l} = \sum_{n=1}^N \hat{\mathcal{P}}_{m,n,l} \cdot v_{n,l} \tag{5}
\]
where $D$ is the hidden dimension of the LVLM. The comprehensive layer feature $F_{m,l} \in \mathbb{R}^{D+2}$ is constructed by concatenating the high-dimensional context $C_{m,l}$ with the explicit scalars $E_{m,l}$ and $\mathbb{E}[G_{m,l}]$. 

To model the complex, non-sequential interactions across the network depth $L$, we treat the sequence of layer representations $\Phi_m = [F_{m,1}, F_{m,2}, \dots, F_{m,L}]$ as an unordered set of residual updates. We apply a lightweight **Cross-Layer Self-Attention** mechanism:
\[
H_m = \text{TransformerEncoder}(\Phi_m + P_{depth}) \tag{6}
\]
where $P_{depth}$ are learnable depth embeddings. A standard attention pooling over the layer dimension produces the final token-level representation, which is passed through an MLP to predict the hallucination probability $p_m \in [0, 1]$. 

By utilizing vector-norm information flows, exact vocabulary-space projections, and a high-dimensional cross-layer attention mechanism, DHCP maintains strict mathematical integrity. It effectively maps resolution-agnostic visual states to fine-grained token-level hallucination predictions without suffering from feature collapse or violating the internal architecture of modern LLMs.