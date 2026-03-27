# 3 Method
## 3.1 Cross-modal Attention in LVLM
We first define the cross-modal attention $\mathcal{A}$ in large vision-language models (LVLMs). Taking InstructBLIP and LLaVA as examples—both of which use a Q-former or MLP as their projection module—we identify three core components: the CLIP visual encoder $f_\text{V}$, the projector $f_\text{P}$, and the large language model $f_\text{LLM}$.

Given an image $x_i$ and a text query $x_t$, the visual encoder extracts visual features $f_\text{V}(x_i)$. These features are then aligned with text via the projector, producing $f_\text{P}(f_\text{V}(x_i), x_t)$ formatted to match the input specification of the LLM $f_\text{LLM}$. Following common practice in many LVLMs, the text $x_t$ is also passed through the projector $f_\text{P}$. The resulting visual tokens of length $N$ are fed into the LLM, which then generates a response sequence of length $M$.

Let $\mathcal{A}_{m,n}(x_i, x_t)$ denote the attention weight assigned by the $m$-th generated token to the $n$-th image token.
Different from [46,47], we define the cross-modal attention $\mathcal{A}(x_i, x_t)$ as the **average attention** from all generated tokens to each visual token, aggregated across all LLM layers and attention heads:
\[
\mathcal{A} = \mathcal{A}(x_i, x_t) = \frac{\sum_m \mathcal{A}_{m,n}(x_i, x_t)}{M} \tag{1}
\]

The attention tensor $\mathcal{A}$ has three dimensions: $(N, L, H)$, where
- $N$: number of visual tokens,
- $L$: number of LLM layers,
- $H$: number of attention heads.

Typical dimensions across models include:
- LLaVA-v1.5-7B: $(N=576,\, L=32,\, H=32)$
- InstructBLIP-7B: $(N=32,\, L=32,\, H=32)$
- InternVL-2.5-4B: $(N=256,\, L=36,\, H=16)$

Notably, Qwen2.5-VL uses a **dynamic $N$** that depends on input resolution. For a fixed resolution such as $336\times336$, Qwen2.5-VL-7B yields $(N=144,\, L=28,\, H=28)$.

---

## 3.2 Cross-modal Attention Excels in Revealing Trace of LVLM Hallucinations
To investigate whether $\mathcal{A}$ exhibits distinguishable patterns between hallucinatory and normal outputs, we evaluated InstructBLIP-7B on images from the POPE dataset [22]. Samples were split into two groups:
- $\mathcal{A}_\text{T}$: non-hallucinatory (truthful) responses
- $\mathcal{A}_\text{H}$: hallucinatory responses

We visualized cross-modal attention for 4,000 randomly selected samples from each category. Each row corresponds to an image–question pair, each column to a visual token. We analyzed one randomly chosen LLM layer and took the maximum attention value across heads. The results are shown in Figure 2.

From the visualization, a key observation emerges:
Red highlighting reveals strong differences in cross-modal attention patterns between truthful and hallucinatory responses even when the textual answer is identical. For example, the attention value on the 8th visual token is noticeably higher in the hallucination case (Figure 2b) than in the truthful case (Figure 2a). Similar discrepancies are consistent across layers and heads in various LVLMs.

---

## 3.3 DHCP: Hallucination Detection Method
Based on the divergent attention patterns between hallucinatory and non-hallucinatory samples, we propose **DHCP**, a simple yet effective hallucination detector.

The detector takes the cross-modal attention tensor $\mathcal{A}\in\mathbb{R}^{N\times L\times H}$ as input. After flattening, the features are fed into a lightweight MLP classifier. In general, we use a **two-layer MLP** with a hidden dimension of 128 and an output dimension of 2, corresponding to non-hallucinatory and hallucinatory classes respectively.

For models with very few visual tokens (e.g., InstructBLIP with $N=32$), we use a **cascaded MLP structure**:
- The first MLP performs initial hallucination scoring.
- The second MLP further verifies only samples flagged as hallucinatory by the first.
- A sample is marked as hallucinatory only if both MLPs agree.

For other models including Qwen2.5-VL, InternVL-2.5, and LLaVA-v1.5, a single MLP is sufficient.