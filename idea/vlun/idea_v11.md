# 3 Method

## 3.1 Counterfactual Visual Marginalization via Expected Priors

To systematically address multimodal hallucinations, we must rigorously isolate the explicit contribution of the visual modality to the model’s final prediction. Previous intrinsic decomposition attempts fail because they algebraically split residual streams or rely on macroscopic Taylor approximations, which fundamentally ignore the deep non-linear cross-modal entanglement occurring within the Multi-Layer Perceptrons (MLPs). Once visual and textual features interact in the first transformer layer, their representations become indivisibly fused in the hidden states.

Instead of mathematically fragile algebraic splitting, we propose a rigorous causal intervention: **Counterfactual Visual Marginalization (CVM)**. To isolate the causal effect of specific visual semantics without triggering the severe Out-of-Distribution (OOD) activation shifts caused by zero-tensors or Gaussian noise, we marginalize out the visual information by replacing the specific image $V$ with an uninformative, yet statistically valid, visual prior $V_\emptyset$. 

We define $V_\emptyset$ as the dataset-level expectation of visual token embeddings: $V_\emptyset = \mathbb{E}_{V \in \mathcal{D}}[V]$. By substituting the explicit image with this mean visual feature, we provide the Large Vision-Language Model (LVLM) with a mathematically stable input that perfectly preserves the expected mean and variance for all LayerNorm operations and attention query-key distributions. This cleanly neutralizes OOD artifacts while containing zero specific semantic evidence about the current image. 

During the autoregressive generation of the $t$-th token, we compute two separate forward passes—efficiently batched and sharing the identical textual KV-cache to avoid unrolling attention mechanisms or requiring custom CUDA kernels:
1.  **Full Multimodal State:** Computes the standard logits $Z_{\text{full}}(y_t)$ and probability $P_{\text{full}}(y_t)$ conditioned on the actual image $V$ and text prefix $T$.
2.  **Linguistic Prior State:** Computes the counterfactual logits $Z_{\text{prior}}(y_t)$ and probability $P_{\text{prior}}(y_t)$ conditioned on the uninformative prior $V_\emptyset$ and text prefix $T$.

This causal intervention produces two distribution sets where the internal MLP non-linearities are naturally resolved, directly yielding the true multimodal prediction and the text-prior prediction.

## 3.2 Causal Visual Grounding Differential

By isolating the full and counterfactual states at the final language modeling head, we bypass the need for any mathematically flawed intermediate layer projections or Jacobian approximations. We directly compute the logit attribution of the actual visual evidence.

We define the **Visual Grounding Differential (VGD)** for each candidate token $y_t$ in the vocabulary as the difference between the fully conditioned logits and the prior-driven logits:
$$
\text{VGD}(y_t) = Z_{\text{full}}(y_t) - Z_{\text{prior}}(y_t) \tag{1}
$$

This formulation elegantly captures the exact causal contribution of the image. 
*   If $\text{VGD}(y_t) > 0$, the presence of the specific image actively increased the model's confidence in $y_t$, indicating strong visual grounding.
*   If $\text{VGD}(y_t) \approx 0$, the token is entirely driven by the language context. This naturally occurs for syntactic stop-words (e.g., "the", "is") and sub-word fragments, which are dictated by local linguistic grammar regardless of visual input.
*   If $\text{VGD}(y_t) \ll 0$, the specific image actually *contradicts* or provides less evidence for the token than the generic prior. When coupled with a high $P_{\text{full}}(y_t)$, this strictly identifies a severe hallucination: the model is confidently generating an entity driven by pure linguistic inertia, which the actual visual evidence does not support.

## 3.3 Grammar-Preserving Hallucination Evaluation

To evaluate sequence-level hallucination severity without relying on logically contradictory internal attention weights (which naturally collapse to zero for pure linguistic hallucinations) or external NLP parsers, we rely strictly on the causal differential. 

Because syntactic tokens inherently yield $\text{VGD} \approx 0$, they mathematically filter themselves out of the hallucination penalty. We define the **Causal Hallucination Index (CHI)** for a generated sequence $Y = \{y_1, \dots, y_m\}$ by taking the expected magnitude of the negative grounding differential, weighted by the generative confidence of the full model:
$$
\text{CHI}(Y) = \frac{1}{m} \sum_{t=1}^m P_{\text{full}}(y_t) \cdot \big|\min\big(0, \, \text{VGD}(y_t)\big)\big| \tag{2}
$$

By exclusively pooling the negative $\text{VGD}$ values ($\min(0, \cdot)$), CHI ignores visually grounded concepts. By weighting the penalty by $P_{\text{full}}(y_t)$, the index heavily penalizes tokens where the model is highly confident *despite* a severe lack of causal visual grounding. This provides a robust, attention-free, and parser-free continuous metric for multimodal hallucination density.

## 3.4 Active Mitigation via Plausibility-Constrained Decoding

The CVM framework inherently supports active hallucination mitigation via Contrastive Decoding. However, standard unconstrained logit subtraction heavily degrades grammatical coherence, frequently leading to repetitive or broken text when prior logits dominate.

To ensure generative integrity, we introduce **Plausibility-Constrained Modality Decoding (PCMD)**. At generation step $t$, we strictly limit the contrastive intervention to the semantically plausible candidate set. We define the adaptive plausibility subset $\mathcal{V}_{\text{head}}$ as the smallest set of top tokens whose cumulative probability under $P_{\text{full}}$ exceeds a dynamic nucleus threshold $\beta$ (e.g., 0.9):
$$
\sum_{y \in \mathcal{V}_{\text{head}}} P_{\text{full}}(y) \ge \beta \tag{3}
$$

We then dynamically recalibrate the logits by penalizing tokens based on their lack of visual grounding, but *only* if they are within the plausible head, leaving the tail distribution completely intact to preserve base grammatical structures:
$$
\tilde{Z}_t(y_t) = 
\begin{cases} 
Z_{\text{full}}(y_t) + \alpha \cdot \text{VGD}(y_t) & \text{if } y_t \in \mathcal{V}_{\text{head}} \\ 
Z_{\text{full}}(y_t) & \text{otherwise} 
\end{cases} \tag{4}
$$
where $\alpha \ge 0$ is the contrastive intervention strength. The next token is formally sampled via $y_t \sim \text{Softmax}(\tilde{Z}_t(y_t))$. 

By shifting the probability mass towards tokens with positive Visual Grounding Differentials strictly within the structurally coherent nucleus, PCMD systematically suppresses highly confident linguistic hallucinations at their mathematical root while mathematically bounding the risk of grammatical degradation. Utilizing standard KV-cache sharing for the textual prefix $\mathcal{T}$, this active intervention operates with minimal computational overhead, achieving robust real-time hallucination mitigation without model retraining.