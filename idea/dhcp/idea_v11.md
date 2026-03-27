# 3 Method

## 3.1 Counterfactual Formulation of Multimodal Dependency
We first formally define the autoregressive generation process in large vision-language models (LVLMs). Given an input image $x_v$ and a text prompt $x_t$, the model generates a token sequence $Y = \{y_1, \dots, y_M\}$. 

Prior interpretability efforts attempt to linearly decompose intermediate residual streams into "visual" and "textual" components. However, this fundamentally ignores the *covariate shift* inherent in deep Transformer architectures: dense cross-attention and highly non-linear Feed-Forward Networks (MLPs) inextricably entangle multimodal features. A naive subtraction of hidden states (e.g., $f(x_v, x_t) - f(\emptyset, x_t)$) does not yield "pure visual features," because the absence of visual tokens dynamically cascades through the non-linear activations, fundamentally altering the contextualization of the text tokens themselves.

To rigorously evaluate visual grounding without violating the non-linear dynamics of LVLMs, we shift the paradigm from feature-space decomposition to **Counterfactual Output-Space Divergence**. Instead of falsely claiming isolation of intermediate modalities, we measure the end-to-end predictive dependency. We maintain a primary multimodal stream yielding the exact final hidden state $h_m^{(L)} = \text{LVLM}(x_v, x_t, Y_{<m})$, and a parallel counterfactual text-only stream yielding $\tilde{h}_m^{(L)} = \text{LVLM}(\emptyset, x_t, Y_{<m})$. By strictly deferring the comparison to the final predictive distributions, we fully absorb all non-linear transformations, feature entanglement, and MLP memorization into a mathematically consistent evaluation framework.

---

## 3.2 Strict Output-Space Calibration
Because the non-linear shift caused by the counterfactual intervention invalidates direct algebraic operations (e.g., subtraction) in the intermediate hidden spaces, we must map these states to the final vocabulary space before performing any comparative analysis. 

Let $\text{RMS}(\cdot)$ denote the final normalization layer with learned scale parameter $\gamma$. The hidden states are non-linearly normalized and projected via the unembedding matrix $W_\text{unbed}$ to obtain the full multimodal logit vector $Z_m$ and the counterfactual text-prior logit vector $\tilde{Z}_m$:
\[
Z_m = \left( \frac{h_m^{(L)} \odot \gamma}{\text{RMS}(h_m^{(L)})} \right) W_\text{unbed}, \quad \tilde{Z}_m = \left( \frac{\tilde{h}_m^{(L)} \odot \gamma}{\text{RMS}(\tilde{h}_m^{(L)})} \right) W_\text{unbed} \tag{1}
\]
To avoid the mathematical instability of unbounded logit comparisons—where raw logits can cross zero and cause meaningless oscillations in magnitude-based metrics—we map the logits to the strict probability simplex via the softmax function:
\[
P_{\text{mm}}(y_m) = \frac{\exp(Z_m[y_m])}{\sum_{j} \exp(Z_m[j])}, \quad P_{\text{txt}}(y_m) = \frac{\exp(\tilde{Z}_m[y_m])}{\sum_{j} \exp(\tilde{Z}_m[j])} \tag{2}
\]
Here, $P_{\text{mm}}(y_m)$ represents the token's generation confidence conditioned on the full multimodal context, while $P_{\text{txt}}(y_m)$ explicitly quantifies the model's inherent linguistic bias and memorized text-prior for the exact same sequential context.

---

## 3.3 Probabilistic Hallucination Risk Assessment
Visual hallucinations specifically manifest when an LVLM generates concrete, descriptive tokens driven overwhelmingly by the textual prior, directly contradicting or lacking support from the actual visual evidence. Previous heuristic metrics drastically fail here: if a hallucination is severely driven by text, weighting it by the inverse of the text-prior probability inherently masks the most dangerous hallucinations. Furthermore, naive metrics incorrectly flag functional and syntactic tokens (e.g., "the", "is", "a"), which are naturally text-driven but do not constitute visual fabrications.

To systematically and logically resolve this, we propose the **Probabilistic Visual Discount (PVD)** as the hallucination risk score $\mathcal{R}_m$. A genuine hallucination occurs when the text prior strongly predicts a token, but the introduction of the visual context actively *discounts* or *suppresses* that probability. We define the token-level hallucination risk strictly as the relative predictive degradation caused by the visual input:
\[
\mathcal{R}_m = \max\left(0, \frac{P_{\text{txt}}(y_m) - P_{\text{mm}}(y_m)}{P_{\text{txt}}(y_m)} \right) = \max\left(0, 1 - \frac{P_{\text{mm}}(y_m)}{P_{\text{txt}}(y_m)} \right) \tag{3}
\]

This elegant, parameter-free probabilistic formulation possesses several mathematically robust properties:
1. **Natural Syntactic Filtering:** Functional and stop words are visually invariant; their probabilities remain fundamentally stable regardless of the image content ($P_{\text{mm}} \approx P_{\text{txt}}$). Consequently, their risk score naturally converges to zero without requiring arbitrary external weights or Part-of-Speech taggers.
2. **Accurate Prior Penalization:** If a severe hallucination is driven by a massive language prior ($P_{\text{txt}} \to 1$), but the actual visual evidence fails to support it (causing an explicit drop to $P_{\text{mm}}$ during the multimodal forward pass), the risk score sharply peaks. This correctly identifies the hallucination precisely because of its ungrounded reliance on text.
3. **Bounded Stability:** The metric is strictly bounded within $[0, 1)$ and completely eliminates the numerical instability associated with raw logit division and arbitrary epsilon heuristics. 

By grounding the evaluation in counterfactual probability divergence, this method provides a rigorous, computationally stable, and theoretically sound diagnostic signal for hallucination detection.