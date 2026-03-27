# 3 Method

## 3.1 Cross-modal Attention in LVLM
We first define the cross-modal attention mechanisms within large vision-language models (LVLMs). Let $x_i$ and $x_t$ denote the input image and text query, respectively. The visual encoder extracts visual features $f_\text{V}(x_i)$, which are mapped by a projector $f_\text{P}$ into a sequence of visual tokens of length $N$. These $N$ visual tokens are prepended to the text embeddings and fed into the large language model $f_\text{LLM}$, which generates a sequence of $M$ tokens autoregressively.

Let $\mathcal{A}_{m,n,l,h}(x_i, x_t)$ denote the attention weight from the $m$-th generated textual token to the $n$-th visual token at the $l$-th layer and $h$-th attention head. Hallucinations in LVLMs often occur sporadically, manifesting in a few specific generated tokens (e.g., fabricating an non-existent object) rather than uniformly across the entire response. Consequently, applying average pooling over the generation length $M$ would heavily dilute these localized abnormal signals. 

To preserve the critical anomalous activations that trigger hallucinations, we define the aggregated cross-modal attention tensor $\mathcal{A}$ by performing a **max-pooling** operation along the temporal generation dimension $M$:
\[
\mathcal{A}_{n,l,h}(x_i, x_t) = \max_{m \in \{1, \dots, M\}} \mathcal{A}_{m,n,l,h}(x_i, x_t) \tag{1}
\]
The resulting attention tensor $\mathcal{A} \in \mathbb{R}^{N \times L \times H}$ inherently encapsulates the maximal visual reliance of the entire response, effectively highlighting the exact moments where the LLM over-attends or under-attends to specific visual patches. Typical dimensions vary by architecture (e.g., LLaVA-v1.5 yields $N=576, L=32, H=32$), and notably, advanced models like Qwen2.5-VL exhibit a **dynamic $N$** depending on the input image resolution.

---

## 3.2 Spatio-Hierarchical Discrepancy in Hallucinations
To investigate whether the extracted tensor $\mathcal{A}$ exhibits discernible patterns for hallucinations, we analyze the attention distributions rather than merely comparing token indices, as different visual tokens encode varying semantic and spatial information across distinct images. 

We split the evaluated outputs into truthful responses ($\mathcal{A}_\text{T}$) and hallucinatory responses ($\mathcal{A}_\text{H}$). By examining the layer-wise and spatial distribution statistics, a profound spatio-hierarchical discrepancy emerges. In truthful generations, cross-modal attention exhibits distinct **semantic sparsity and deep-layer focus**, where the attention cleanly peaks at visual tokens corresponding to the queried objects. 

Conversely, when a hallucination occurs, the attention distribution exhibits severe abnormalities characterized by two primary patterns: (1) **Attention Defocus (High Entropy)**: the model fails to locate the relevant visual evidence, causing the attention weights to diffuse uniformly across background tokens; (2) **Spurious Activation**: the attention abruptly spikes on entirely irrelevant visual regions, leading the LLM to hallucinate semantic concepts absent from the image. These distributional anomalies are highly consistent across different LVLM families and serve as robust, model-agnostic indicators of hallucinations.

---

## 3.3 DHCP: Dynamic Hierarchical Cross-modal Pattern Detector
Relying on simplistic flattening and dense layers destroys both the spatial structure of visual tokens and the hierarchical progression across LLM layers. Furthermore, rigid network structures fail to accommodate the dynamic visual token length $N$ inherent to modern LVLMs like Qwen2.5-VL. To address these structural and dynamic challenges, we propose **DHCP** (Dynamic Hierarchical Cross-modal Pattern detector), an architecture tailored to capture the intrinsic anomalies in attention distributions.

Instead of flattening the tensor, we formulate the attention tensor $\mathcal{A} \in \mathbb{R}^{N \times L \times H}$ as a sequence of visual observations. Specifically, for each of the $N$ visual tokens, we flatten its layer and head dimensions to form a feature vector $v_n \in \mathbb{R}^{L \cdot H}$, which preserves the hierarchical trace of how the LLM processes this specific spatial patch from shallow to deep layers.

The sequence of vectors $V = \{v_1, v_2, \dots, v_N\}$ is linearly projected into a hidden dimension $D$ and injected with spatial positional encodings. We then process the sequence through a lightweight **Transformer Encoder** consisting of $K=2$ self-attention layers:
\[
Z = \text{TransformerEncoder}(V W_P + E_{pos}) \tag{2}
\]
where $W_P \in \mathbb{R}^{(L \cdot H) \times D}$ and $E_{pos}$ is the positional embedding. 

To seamlessly handle the **dynamic $N$** and prevent the model from overfitting to specific spatial biases, we apply an **Adaptive Attention Pooling** (or Global Max Pooling) over the sequence dimension $N$ to aggregate the token-wise representations into a fixed-length global pattern vector $z_{global} \in \mathbb{R}^D$:
\[
z_{global} = \max_{n \in \{1, \dots, N\}} Z_n \tag{3}
\]
Finally, a linear classifier maps $z_{global}$ to the binary hallucination prediction. 

This design inherently decouples the detection mechanism from varying image resolutions and predefined vision encoder structures, eliminating the need for ad-hoc patches like cascaded MLPs. By modeling the interactions among spatially distributed attention traces via self-attention, DHCP effectively learns the generic distributional patterns of hallucinations (e.g., spatial defocus and layer-wise spurious spikes) rather than overfitting to specific dataset content formats, ensuring robust generalization across diverse, descriptive long-text scenarios.