# 3 Method

## 3.1 Modality-Aware Residual Stream Decomposition

To systematically address multimodal hallucinations without triggering Out-of-Distribution (OOD) artifacts or incurring prohibitive computational overhead, we must directly quantify the explicit contribution of the visual modality to the model's final prediction. Rather than manipulating inputs or projecting unaligned intermediate layers—which inherently generates high-entropy noise and semantic mismatch—we propose an intrinsic, zero-intervention mechanism: **Modality-Aware Residual Stream Decomposition (MRSD)**.

In autoregressive LVLMs, the generation of the $t$-th token is governed by the final hidden state $h_t^{(L)}$. The transformer architecture builds this state additively across layers: $h_t^{(L)} = h_t^{(0)} + \sum_{l=1}^L \Delta h_t^{(l)}$. During the self-attention mechanism at layer $l$, the token dynamically attends to the combined prefix comprising the visual tokens $\mathcal{V}$ (derived from the image) and the textual context $\mathcal{T}$ (system prompt and generated text prefix). 

We mathematically partition the layer-wise attention output into modality-specific spatial contributions. Let $A_{t, j}^{(l, h)}$ denote the attention weight from the current token $t$ to a preceding token $j$ at head $h$, and $v_j^{(l, h)}$ the corresponding value vector. Because the self-attention update is intrinsically linear with respect to the value vectors, we rigidly decouple the accumulated state into a strictly visual-sourced component $x_{t, \mathcal{V}}$ and a strictly linguistic-sourced component $x_{t, \mathcal{T}}$:

$$
x_{t, \mathcal{V}} = \sum_{l=1}^L \sum_{h=1}^H W_O^{(h)} \sum_{j \in \mathcal{V}} A_{t, j}^{(l, h)} v_j^{(l, h)} \tag{1}
$$
$$
x_{t, \mathcal{T}} = h_t^{(0)} + \sum_{l=1}^L \text{MLP}^{(l)}(h_t^{(l-1)}) + \sum_{l=1}^L \sum_{h=1}^H W_O^{(h)} \sum_{j \in \mathcal{T}} A_{t, j}^{(l, h)} v_j^{(l, h)} \tag{2}
$$

By tracking these accumulators natively during the standard forward pass, we precisely isolate the exact feature mass $x_{t, \mathcal{V}}$ explicitly derived from the visual embeddings. This ensures $100\%$ distribution fidelity (zero OOD shift) while completely eliminating the necessity for arbitrary layer selection or heuristic approximations.

## 3.2 Direct Modality Logit Attribution

With the residual stream rigorously decomposed ($h_t^{(L)} = x_{t, \mathcal{V}} + x_{t, \mathcal{T}}$), we project these components into the vocabulary space to evaluate modality-specific confidence. To maintain perfect alignment with the final language modeling (LM) head $W$ without violating the final Layer Normalization ($\text{LN}_f$), we employ a first-order Taylor approximation of $\text{LN}_f$ around the full context state $h_t^{(L)}$. 

Since $\text{LN}_f$ involves non-linear mean subtraction and variance division, we compute its Jacobian matrix $J_t = \frac{\partial \text{LN}_f(h_t^{(L)})}{\partial h_t^{(L)}}$. The exact logit attribution for the visual modality $Z_{t, \mathcal{V}}$ and the linguistic modality $Z_{t, \mathcal{T}}$ for candidate token $y_t$ is computed via:
$$
Z_{t, \mathcal{V}} = W \cdot (J_t \, x_{t, \mathcal{V}}) \tag{3}
$$
$$
Z_{t, \mathcal{T}} = W \cdot \big(\text{LN}_f(h_t^{(L)}) - J_t \, x_{t, \mathcal{V}}\big) \tag{4}
$$

This mathematically sound projection explicitly defines $Z_{t, \mathcal{V}}(y_t)$ as the direct causal evidence provided by the image for generating token $y_t$, while $Z_{t, \mathcal{T}}(y_t)$ represents the generative inertia driven purely by the language prior. Because we strictly utilize the final aligned projection matrix $W$ on properly normalized vectors, this completely resolves the semantic gap and OOD noise problems associated with intermediate-layer projections.

We define the token-level **Visual Grounding Differential (VGD)** as the margin between the visual and textual logit attributions:
$$
\text{VGD}(y_t) = Z_{t, \mathcal{V}}(y_t) - Z_{t, \mathcal{T}}(y_t) \tag{5}
$$
A significantly negative $\text{VGD}$ directly exposes a ungrounded hallucination: the model generated the token overwhelmingly based on linguistic inertia ($Z_{t, \mathcal{T}}$) despite a severe lack of actual visual evidence ($Z_{t, \mathcal{V}}$).

## 3.3 Dynamic Concept-Level Hallucination Indexing

Standard sequence-level aggregation inevitably suffers from signal dilution, where a few catastrophic hallucinated entities are mathematically drowned out by high-frequency, safe grammatical stop-words. To effectively aggregate the sequence hallucination risk without arbitrary hyperparameters (e.g., ad-hoc Top-$p\%$ thresholds) or flawed external NLP parsers, we utilize an attention-weighted pooling mechanism strictly constrained by the model's internal structural logic.

Structural syntactic stop-words naturally exhibit extremely low direct attention to visual tokens, while semantic visual concepts inherently trigger higher image-directed attention. We compute the structural visual weight $w_t = \sum_{j \in \mathcal{V}} A_{t, j}^{(L)}$, representing the final layer's raw attention allocation to the image. 

We define the parameter-free **Visual Hallucination Severity (VHS)** for the sequence $Y = \{y_1, \dots, y_m\}$ as the weighted expectation of the negative grounding margins:
$$
\text{VHS}(Y) = \frac{\sum_{t=1}^m w_t \cdot P(y_t) \cdot \max\big(0, \, Z_{t, \mathcal{T}}(y_t) - Z_{t, \mathcal{V}}(y_t)\big)}{\sum_{t=1}^m w_t \cdot P(y_t)} \tag{6}
$$

By weighting the deficit by the final generative confidence $P(y_t)$ and the structural visual weight $w_t$, VHS inherently isolates and penalizes scenarios where the model outputs an image-relevant concept with high confidence, yet the explicit logit attribution reveals that the confidence was falsely synthesized by the text prior rather than the visual stream.

## 3.4 Active Mitigation via Modality-Guided Decoding

Because the residual stream accumulators $x_{t, \mathcal{V}}$ and $x_{t, \mathcal{T}}$ are dynamically tracked natively during the standard forward pass self-attention operations, our method requires absolutely no multi-pass inference, no computationally disastrous layer searches, and strictly maintains $\mathcal{O}(1)$ vocabulary matrix projections per generation step.

This structural efficiency enables robust, real-time active mitigation. During autoregressive decoding, rather than passively sampling from the standard uncalibrated logits $Z_t$, we introduce **Modality-Guided Decoding (MGD)**. We dynamically recalibrate the final logits by explicitly penalizing ungrounded linguistic momentum:
$$
\tilde{Z}_t(y_t) = Z_t(y_t) + \gamma \cdot \text{VGD}(y_t) \tag{7}
$$
where $\gamma > 0$ is a static calibration scalar controlling grounding strictness. The next token is formally sampled via $y_t \sim \text{Softmax}(\tilde{Z}_t(y_t))$. 

By directly shifting the probability mass towards tokens with positive Visual Grounding Differentials ($\text{VGD} > 0$), MGD systematically suppresses text-prior hallucinations at their mathematical root. It forces the LVLM to articulate only entities that possess rigorous, mathematically traceable causal origins within the visual feature space, effectively mitigating multimodal hallucinations "on the fly" without requiring any model retraining or parameter updates.