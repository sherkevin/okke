# 4 Methodology
In this section, we first introduce the task formulation and then provide a comprehensive description of our method, including its theoretical foundations and practical implementation.

## 4.1 Problem Formulation
Large Vision-Language Models (LVLMs) aim to generate reasonable text responses to multi-modal inputs by integrating visual and textual information. The standard pipeline extracts visual features via a visual encoder, maps them into the linguistic semantic space through a projection layer for cross-modal fusion and alignment, and then decodes the fused representations using an autoregressive language model to produce the final response.

Formally, given an input image \(I\) and a corresponding question or text prompt \(Q\), the generated answer sequence \(Y\) is modeled as:
\[
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, Q, y_{<t}),
\tag{1}
\]
where \(y_{<t}\) denotes the token sequence before the current token \(y_t\) at step \(t\), and \(\theta\) denotes the parameters of the LVLM.

Based on our preliminary analysis, we observe that incorporating object categories \(C\) and their corresponding bounding boxes \(B\) can improve generation quality. We therefore reformulate the generation process as:
\[
p(Y) = \prod_{t=1}^{T} p_\theta(y_t \mid I, D, Q, y_{<t}),
\tag{2}
\]
where \(D = [C(I); B(I)]\).

In the object hallucination recognition task, the input question \(Q\) is designed to verify the existence of objects. The task goal is to generate the correct answer \(Y\) by judging whether a given object exists in the image \(I\).

## 4.2 PATCH
We propose **PATCH**, a parameter-efficient fine-tuning strategy that mitigates object hallucinations in LVLMs by introducing trainable virtual tokens to leverage auxiliary object-level information. The overall architecture (using MiniGPT-v2 as an example) is shown in Figure 1.

Inspired by prior work [49], we insert a set of \(n\) virtual tokens
\[
\mathcal{T} = [t_1, t_2, \dots, t_n]
\]
between the image features and the object detection information \(D\). These token embeddings are optimized during training, with parameters \(\delta \in \mathbb{R}^{n\times d}\), where \(d\) is the embedding dimension of the LVLM.

The generation process of the LVLM augmented with virtual tokens is formulated as:
\[
p(Y) = \prod_{t=1}^{T} p_{\delta,\theta}(y_t \mid I, [t_1,t_2,\dots,t_n], D, Q, y_{<t}).
\tag{3}
\]

To reduce computational overhead, all original LVLM parameters \(\theta\) are **frozen** during fine-tuning; only the virtual token parameters \(\delta\) are updated. For example, with 20 additional virtual tokens, only \(20 \times 4096 = 0.08\mathrm{M}\) parameters are trainable, accounting for merely **0.0012%** of the total model parameters. This design ensures high computational efficiency while maintaining strong optimization ability for alleviating object hallucinations. Comprehensive experimental analysis is provided in Section 5.4.

During inference, we extend the model vocabulary with special tokens such as `[ref1]`, `[ref2]`, …, `[refn]`, whose embeddings are initialized using the fine-tuned virtual token embeddings. This makes PATCH a **plug-and-play** method that can be dynamically adjusted according to application requirements.

Specifically:
- When object-level information is included in user input, virtual tokens are inserted before detection results to help reduce object hallucinations.
- When no extra detection information is provided, the LVLM can revert to its standard pipeline without PATCH.

PATCH strengthens visual understanding by optimizing the alignment between visual features and textual semantics while preserving the model’s original capabilities. This flexibility is highly valuable in real-world deployment, where LVLMs are applied to diverse downstream tasks. By narrowing the representation gap between modalities, PATCH improves cross-modal feature fusion, especially for tasks that benefit from enriched detection prompts, thus reducing object hallucinations and boosting overall LVLM performance.