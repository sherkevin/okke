**Review Comments:**

The proposed method exhibits severe conceptual flaws, mathematical inconsistencies, and a profound lack of understanding of the inner workings of Large Language Models (LLMs). The claims of "mathematical stability" and "realistic computational efficiency" are completely unfounded. Below are the specific and critical defects of this work:

**1. Fundamental Misunderstanding of LLM Hidden States (Eq. 7)**
In Eq. (7), you construct a similarity measure $\Delta(x_t)$ by projecting the attention-weighted sum of historical hidden states $\boldsymbol{h}_j$ using the vocabulary head $W$. You explicitly state $\boldsymbol{h}_j$ are the "final-layer hidden states". It is a well-established fact in auto-regressive language modeling that the final-layer hidden state at step $j$ is trained to predict the *next* token ($j+1$), not to encode the semantic meaning of the token at step $j$ itself. Summing these next-token prediction representations to gauge the semantic similarity of the current context is conceptually baseless and mathematically absurd.

**2. Dimension and Scale Mismatch in Logit Calibration (Eq. 8)**
Your calibrated logit equation (Eq. 8) subtracts $\alpha_t \cdot \phi_t \cdot \Delta(x_t)$ from the original logit $W \boldsymbol{h}_t$. However, $\Delta(x_t)$ is generated via a Softmax operation (Eq. 7), meaning its values strictly lie in the interval $(0, 1)$ and sum to 1. Logits, on the other hand, are unnormalized scores in $(-\infty, \infty)$, often with magnitudes ranging in the tens or hundreds. Subtracting a bounded fraction $(0, 1)$ from an unnormalized logit will have absolutely negligible impact on the final Softmax distribution unless $\alpha_t$ is artificially inflated to extreme values. This reveals a glaring lack of rigor in designing the penalty mechanism.

**3. Ignorance of "Attention Sinks" (Eq. 3, 5, 9)**
The entire premise of your attention aggregation relies on raw attention weights. You compare attention heads against a uniform prior $N/t$ (Eq. 3) and compute length-normalized averages (Eq. 5). Furthermore, Eq. (9) tracks the token with the "maximum attention". Have the authors never heard of "Attention Sinks"? It is widely documented in LLM literature that massive amounts of attention mass are consistently dumped onto the first few tokens (e.g., `<s>`, system prompts) or punctuation marks, regardless of semantic relevance, purely to maintain Softmax properties. Your uniform prior and max-pooling strategies will overwhelmingly flag these sink tokens as "over-trusted" or "hallucination anchors," leading to catastrophic false positives.

**4. Misalignment Between Multimodal Information Density**
In Eq. 3 and Eq. 5, you directly compare image token attention ($\hat{v}_t$) with text token attention ($\hat{u}_t$). Visual tokens (e.g., from CLIP/ViT) are highly redundant and continuous, whereas text tokens are highly compressed, discrete semantic units. Assuming that cross-modal alignment implies equal or uniformly proportional attention mass between vastly different modality representations is naive and empirically incorrect.

**5. Flawed Hallucination Mitigation Logic (Eq. 10)**
When a "hallucination anchor" $s$ is detected, you penalize the current distribution using $P_{hist}$, which is the next-token probability distribution cached at step $s$. This logic heavily penalizes the model for generating the token that was likely to follow step $s$. This functions as a bizarre, delayed *repetition penalty*, not a hallucination penalty. Hallucinations in MLLMs typically manifest as factually incorrect spans generated smoothly over multiple steps, not merely repeating a token from a previous time step. The connection between mitigating hallucination and suppressing $P_{hist}$ is entirely disconnected.

**6. False Claims of Zero Overhead and Deployment Feasibility**
You loudly claim $O(1)$ computation and zero extra forward passes in Section 3.3. However, caching the full probability distribution $P_{hist}$ (size $|\mathcal{X}|$) for every single generated token introduces a staggering memory overhead. For a standard vocabulary of 100K and 1000 generated tokens, you are storing hundreds of megabytes of dense float tensors *per sequence*. In a real-world serving engine (e.g., vLLM, TensorRT-LLM), memory (specifically KV-cache) is the absolute bottleneck. Proposing to cache full logits/distributions at every step demonstrates a complete disconnect from "deterministic latency constraints" and "real-world streaming applications."

**Conclusion:**
The proposed method is built on flawed assumptions about LLM internals, features mathematically mismatched equations, ignores critical LLM characteristics like attention sinks, and proposes a memory-heavy caching mechanism while falsely claiming efficiency. The paper requires a complete overhaul from the ground up.

**Score: 1 / 5 (Strong Reject)**