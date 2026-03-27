# 3 Method

## 3.1 Intrinsic Visual-Linguistic Decoupling via Layer Dynamics

To systematically expose "highly confident hallucinations" driven by strong linguistic priors in Large Vision-Language Models (LVLMs), we must decouple the visual dependency from the language model's inherent statistical inertia. However, introducing explicitly corrupted inputs (e.g., noise, zero-tensors) or forcibly masking attention mechanisms inherently provokes severe out-of-distribution (OOD) activation shifts, producing unreliable proxy distributions.

Instead of intervening at the input or architectural level, we introduce an intrinsic decoupling mechanism rooted in the well-documented layer-wise dynamics of transformer-based LLMs. Recent probing studies demonstrate that during multimodal generation, the intermediate layers of an LVLM predominantly focus on aligning and grounding multi-modal visual features, while the deepest layers (closer to the output head) are heavily dominated by the pre-trained linguistic priors and auto-regressive syntactic generation.

Exploiting this architectural property, we can measure multimodal dependency via a single forward pass, eliminating all OOD risks. For a given image $V$ and prompt $T$, let the model generate the $t$-th token based on the prefix $Y_{<t}$. We project the hidden states from an intermediate "vision-heavy" layer $k$ and the final "prior-heavy" layer $L$ through the LM head to obtain two distinct predictive distributions over the vocabulary:
$$
P_{\text{vision}}(y_t) = \text{Softmax}\big(W \cdot h_t^{(k)}\big)
$$
$$
P_{\text{final}}(y_t) = \text{Softmax}\big(W \cdot h_t^{(L)}\big)
$$
where $W$ is the vocabulary projection matrix, and $h_t^{(k)}$ is the hidden state at layer $k$. Here, $P_{\text{vision}}$ represents the prediction highly conditioned on the raw, uncorrupted visual features, whereas $P_{\text{final}}$ represents the final prediction heavily modulated by the model's textual context and structural prior.

## 3.2 Sub-word Agnostic Hallucination Margin

Existing token-level evaluation methods often struggle with the sub-word tokenization barrier (e.g., relying on flawed word-level POS taggers like NLTK to filter syntactic stop-words) or suffer from severe numerical instability when employing probability ratios on long-tail tokens. Furthermore, naive uncertainty metrics frequently over-penalize "commonsense"—situations where the text is highly predictable from prior knowledge but is simultaneously supported by the visual evidence.

To resolve these logical and engineering flaws, we define a mathematically stable, token-level **Prior-Driven Hallucination Margin (PHM)** utilizing the log-probability differential (logit margin) between the final state and the visual state:
$$
\text{PHM}(y_t) = \log P_{\text{final}}(y_t) - \log P_{\text{vision}}(y_t)
$$

This formulation elegantly resolves three core challenges intrinsically:
1. **Sub-word and Stop-word Robustness:** Syntactic stop-words (e.g., "the", "is") and sub-word fragments are driven purely by the base grammar logic shared across all layers. Consequently, their probabilities in both $P_{\text{vision}}$ and $P_{\text{final}}$ are nearly identical, pushing their $\text{PHM}$ close to zero. This naturally filters out non-semantic tokens without requiring any external NLP parsers.
2. **Commonsense Calibration:** If a generated concept is a common prior (e.g., "airplane in the sky") *and* visually present, the intermediate layer $k$ (which actively perceives the visual airplane) will assign it a high probability, matching the final layer's prior. The resulting $\text{PHM}$ will be low, correctly recognizing the factual grounding and preventing false-positive hallucination alerts.
3. **Isolating True Hallucinations:** A "highly confident hallucination" occurs strictly when $P_{\text{final}}$ is very high (driven by prior momentum) but $P_{\text{vision}}$ is notably low (the intermediate layer found no actual visual evidence). In this scenario, $\text{PHM}$ yields a significant positive magnitude, directly quantifying the degree to which the language prior "hijacked" the absent visual grounding.

## 3.3 Sequence-Level Hallucination Index Evaluation

To aggregate the step-wise dynamics into a coherent response-level metric without violating the Markovian dependency of autoregressive generation (i.e., erroneously summing marginal probabilities), we calculate the statistically rigorous expected divergence across the generated sequence $Y = \{y_1, y_2, \dots, y_m\}$.

We define the **Commonsense-Calibrated Hallucination Index (CCHI)** by taking the expected value of the positive hallucination margins, weighted by the model's final predictive confidence $P_{\text{final}}(y_t)$ to penalize *highly confident* ungrounded generations, normalized over the sequence length:
$$
\text{CCHI}(Y) = \frac{1}{m} \sum_{t=1}^m P_{\text{final}}(y_t) \cdot \max\big(0, \, \text{PHM}(y_t)\big)
$$

The operation $\max(0, \cdot)$ strictly isolates tokens where the language prior overrode the visual evidence, ignoring tokens that were heavily visually grounded ($\text{PHM} \le 0$). 

A higher $\text{CCHI}$ implies that a significant portion of the model's generation confidence was synthesized by internal linguistic inertia despite a lack of intermediate visual support. Crucially, because this methodology evaluates hidden states within a single forward autoregressive pass, it inherently operates at an optimal $\mathcal{O}(L)$ time complexity with zero additional memory overhead from dual-pass encoding or OOD contrastive generation. 

For empirical evaluation, hallucination detection is framed as a continuous scoring task. To comprehensively assess the robustness of $\text{CCHI}$ against severely imbalanced multimodal benchmark distributions, we employ threshold-agnostic metrics, strictly benchmarking via the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC).