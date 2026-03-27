# 3 Method

## 3.1 Dynamic Layer Alignment for Intrinsic Decoupling

To systematically expose "highly confident hallucinations" driven by strong linguistic priors in Large Vision-Language Models (LVLMs), we must decouple visual dependency from the language model's inherent statistical inertia. Introducing explicitly corrupted inputs or forcibly masking attention mechanisms provokes severe out-of-distribution (OOD) activation shifts. Conversely, directly projecting unaligned intermediate hidden states through the final language modeling (LM) head creates artificial internal OOD noise due to the absence of appropriate normalization and feature disentanglement.

To achieve mathematically rigorous intrinsic decoupling, we leverage the natural layer-wise evolution of representations in transformer-based LVLMs. Intermediate layers actively align and ground multi-modal visual features, while the deepest layers converge toward pre-trained linguistic priors. To safely project intermediate states into the vocabulary space without distribution shift, we apply the model's final Layer Normalization ($\text{LN}_{f}$) before the LM head projection matrix $W$. For any layer $l$ and generation step $t$, the calibrated predictive distribution is:
$$
P^{(l)}(y_t) = \text{Softmax}\big(W \cdot \text{LN}_{f}(h_t^{(l)})\big) \tag{1}
$$
Furthermore, relying on a static, empirically chosen intermediate layer is overly simplistic, as feature fusion dynamics vary drastically across different models, prompts, and visual complexities. We introduce a **Dynamic Layer Selection** mechanism. Given a candidate set of intermediate layers $\mathcal{K}$ (e.g., the middle layers of the transformer), we dynamically identify the specific "vision-anchored" layer $\hat{k}$ at each step $t$ where the divergence between the evolving visual grounding and the final linguistic prior is maximized. We quantify this using the Jensen-Shannon Divergence (JSD):
$$
\hat{k} = \arg\max_{l \in \mathcal{K}} \text{JSD}\big(P^{(l)}(Y) \parallel P^{(L)}(Y)\big) \tag{2}
$$
where $P^{(L)}$ is the final predictive distribution at layer $L$. We then define the dynamic visual state as $P_{\text{vision}}(y_t) = P^{(\hat{k})}(y_t)$ and the prior-dominated final state as $P_{\text{final}}(y_t) = P^{(L)}(y_t)$. This dynamic mechanism flawlessly adapts to varying fusion depths without arbitrary hyperparameters.

## 3.2 Sub-word Agnostic Hallucination Differential

Existing token-level evaluation methods struggle with sub-word tokenization barriers or suffer from severe numerical instability when employing probability ratios on long-tail tokens. To resolve these mathematical flaws, we define a highly stable, token-level **Prior-Driven Hallucination Differential (PHD)** utilizing the strictly bounded log-probability differential between the final state and the dynamic visual state:
$$
\text{PHD}(y_t) = \log P_{\text{final}}(y_t) - \log P_{\text{vision}}(y_t) \tag{3}
$$
This formulation naturally resolves three fundamental challenges:
1. **Inherent Stop-word Robustness:** Syntactic stop-words and sub-word fragments converge exceptionally early in transformer layers due to deeply ingrained base grammar. Consequently, their representations stabilize well before layer $\hat{k}$, meaning $P_{\text{vision}}$ and $P_{\text{final}}$ are nearly identical. This drives $\text{PHD}$ to zero, naturally filtering out non-semantic tokens without relying on fragile, external NLP parsers.
2. **Commonsense Calibration:** If a generated concept is a common prior (e.g., "airplane in the sky") *and* visually present, the dynamic intermediate layer $\hat{k}$ will confidently predict it, closely matching the final layer's prior. The resulting $\text{PHD}$ remains heavily suppressed, correctly recognizing the factual grounding and eliminating false-positive hallucination penalties on valid commonsense.
3. **Isolating True Hallucinations:** A severe hallucination occurs precisely when $P_{\text{final}}$ is exceptionally high (driven by prior linguistic momentum) but $P_{\text{vision}}$ is notably low. In this exact scenario, $\text{PHD}$ yields a massive positive magnitude, directly quantifying the extent to which the language prior hijacked the generation process in the absence of visual evidence.

## 3.3 Anti-Dilution Sequence Indexing

A critical flaw in standard sequence-level metrics is signal dilution: averaging token-level scores across the entire sequence $m$ mathematically drowns out sparse, highly confident hallucinations (which typically occur on a few critical entity tokens) amidst a sea of safe grammatical structures. 

To aggregate step-wise dynamics without violating Markovian autoregressive constraints or suffering from signal dilution, we define the **Commonsense-Calibrated Hallucination Index (CCHI)** by strictly pooling the differential over the subset of tokens that actively exhibit hallucination risks. Let $\mathcal{H}_t = \max\big(0, \, \text{PHD}(y_t)\big)$ represent the rectified hallucination magnitude. We dynamically extract the set of at-risk tokens $\mathcal{S}_{p}$ representing the top-$p\%$ of tokens in the sequence with the highest $\mathcal{H}_t$ values. The sequence-level index is calculated as the expected confidence-weighted differential over this critical subset:
$$
\text{CCHI}(Y) = \frac{1}{|\mathcal{S}_{p}|} \sum_{y_t \in \mathcal{S}_{p}} P_{\text{final}}(y_t) \cdot \mathcal{H}_t \tag{4}
$$
By weighting the differential by $P_{\text{final}}(y_t)$ and strictly aggregating over $\mathcal{S}_{p}$, CCHI heavily penalizes ungrounded generations only when the model is extremely confident, effectively resolving the signal dilution problem for long-context generations.

## 3.4 Active Mitigation via Contrastive Layer Decoding

Beyond serving as a post-hoc evaluation metric, our framework is inherently designed for active, real-time hallucination intervention during autoregressive generation. Because the dynamic intermediate representations $h_t^{(l)}$ are computed natively during the standard forward pass, we achieve $\mathcal{O}(L)$ time complexity with zero multi-pass overhead.

We seamlessly transition our metric into a **Contrastive Layer Decoding (CLD)** strategy. At generation step $t$, rather than passively sampling from $P_{\text{final}}$, we dynamically penalize the output logits using the calculated hallucination differential:
$$
\mathcal{F}(y_t) = \log P_{\text{final}}(y_t) - \alpha \cdot \max\big(0, \text{PHD}(y_t)\big) \tag{5}
$$
where $\alpha$ is a scaling factor controlling intervention strength. The next token is then sampled via $y_t \sim \text{Softmax}(\mathcal{F}(y_t))$. This active intervention mathematically suppresses highly confident linguistic hallucinations on the fly, forcing the LVLM to pivot toward tokens strictly grounded by the intermediate visual representations, directly mitigating multi-modal hallucinations at inference time without requiring any model retraining or parameter updates.