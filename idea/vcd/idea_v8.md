# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) taking a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. At time step $t$, the token $y_t$ is sampled from the conditional probability distribution:
$$
p_{\text{full}}(y_t) \equiv p_{\theta}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{full}}(y_t)\big). \tag{1}
$$
During this step-by-step generation, object hallucinations frequently occur when the autoregressive process exhibits an over-reliance on the marginal language prior $p_{\theta}(y_t \mid x, y_{<t})$. Instead of genuinely conditioning on the visual evidence $v$, the model defaults to high-frequency linguistic co-occurrences or plausible but incorrect entity completions. Mitigating this issue requires systematically separating and re-weighting the visual signals against these text-centric inertia.

## 3.2 Strict In-Distribution Prior Extraction via Parallel Decoding
Previous methods attempt to extract the language prior by manipulating the model’s internal mechanisms—such as masking visual tokens in the KV-cache, perturbing the input image, or contrasting intermediate transformer layers. However, these architectural interventions inherently disrupt the natural flow of cross-modal self-attention and map the hidden states into an unaligned, out-of-distribution (OOD) manifold.

To extract a mathematically rigorous language prior without inducing any OOD artifacts, we adopt a straightforward, system-level **Parallel Batched Decoding** strategy. Alongside the primary multimodal generation stream, we maintain an independent, vision-free text sequence in the same batch. The prior distribution is simply computed as:
$$
p_{\text{text}}(y_t) \equiv p_{\theta}(y_t \mid x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{text}}(y_t)\big). \tag{2}
$$
Since this vision-free forward pass mirrors the standard pure-text inference phase seen extensively during the model's pre-training (e.g., text-only instruction tuning), $p_{\text{text}}$ is strictly in-distribution. While maintaining an additional sequence introduces supplementary computational FLOPS, treating both streams as parallel instances within a standard batch perfectly preserves the contiguous memory layout. This approach naturally leverages hardware-accelerated kernels (e.g., FlashAttention) without the memory fragmentation and customized kernel overheads associated with complex layer-wise masking techniques.

## 3.3 Information-Theoretic Modality Calibration (ITMC)

### 3.3.1 Pointwise Mutual Information Guidance over Full Vocabulary Space
Many contrastive decoding variants formulate intuitive logit subtractions that mathematically mirror Classifier-Free Guidance (CFG). To provide a rigorous theoretical grounding for modality calibration, we frame the decoding process through the lens of Pointwise Mutual Information (PMI). The pointwise mutual information between the generated token $y_t$ and the visual context $v$ given the textual history is defined as:
$$
\operatorname{PMI}(y_t ; v \mid x, y_{<t}) = \log \frac{p_{\text{full}}(y_t)}{p_{\text{text}}(y_t)}. \tag{3}
$$
A high positive PMI indicates that the visual context actively promotes the token, reflecting genuine visual grounding. Conversely, a negative PMI indicates that the token is primarily driven by the language prior and contradicts the visual evidence—a strong indicator of hallucination. 

We recalibrate the base logits by explicitly injecting this information-theoretic metric:
$$
\tilde{\operatorname{logit}}(y_t) = \operatorname{logit}_{\text{full}}(y_t) + \alpha_t \cdot \operatorname{PMI}(y_t ; v \mid x, y_{<t}), \tag{4}
$$
where $\alpha_t$ controls the guidance strength.

Crucially, **we apply Eq. (4) across the entire vocabulary space prior to any heuristic truncation.** Previous methods inherently flaw the contrastive process by pre-filtering candidates (e.g., via Top-$p$ or Top-$k$) based on $p_{\text{full}}$. If a genuinely correct visual entity has its probability suppressed by an overpowering, erroneous language prior, it might fall outside the initial truncation threshold and be permanently discarded. By operating over the complete vocabulary, the PMI formulation inherently resuscitates visually grounded tokens that were suppressed by language inertia while penalizing highly probable textual hallucinations.

### 3.3.2 Consistency-Aware Dynamic Modulator
Applying a static guidance strength $\alpha$ uniformly across all generation steps can disrupt syntactic fluency. During the generation of grammatical structures, punctuation, or functional completions, the model's predictions are highly deterministic regardless of the visual input. Under these circumstances, $p_{\text{full}}$ and $p_{\text{text}}$ are nearly identical, and aggressively modifying the logits can induce numerical instability or logical breakdowns.

To safely preserve grammatical coherence, we introduce a **Consistency-Aware Dynamic Modulator**. Rather than relying on the single-point entropy of the prior—which cannot differentiate between confident hallucinations and confident syntax—we measure the macroscopic distribution divergence using the Kullback-Leibler (KL) Divergence between the multimodal and text-only distributions over the full vocabulary $\mathcal{V}$:
$$
D_{\text{KL}}(p_{\text{full}} \parallel p_{\text{text}}) = \sum_{y \in \mathcal{V}} p_{\text{full}}(y) \log \frac{p_{\text{full}}(y)}{p_{\text{text}}(y)}. \tag{5}
$$
The KL divergence precisely captures the necessity of visual intervention. When the divergence is near zero (e.g., predicting "is", "the", or common phrase completions), the visual input has negligible influence on the decision, indicating that syntax and prior knowledge are sufficient. When the divergence is high, a critical visual-semantic routing is occurring, warranting stronger calibration. We dynamically scale $\alpha_t$ as:
$$
\alpha_t = \alpha_{\text{base}} \cdot \tanh\big( \lambda \cdot D_{\text{KL}}(p_{\text{full}} \parallel p_{\text{text}}) \big), \tag{6}
$$
where $\lambda$ is a scaling sensitivity hyperparameter. The $\tanh$ function ensures that the guidance strength smoothly transitions to $0$ during syntactical predictions—leaving the natural linguistic flow perfectly unmodified—while asymptotically saturating at $\alpha_{\text{base}}$ during moments of high cross-modal disagreement.

### 3.3.3 Final Decoding Strategy
After the full-vocabulary ITMC recalibration, the logits are projected back into a normalized probability distribution:
$$
\tilde{p}(y_t) = \operatorname{softmax}\big(\tilde{\operatorname{logit}}(y_t)\big). \tag{7}
$$
Only at this final stage do we apply standard Nucleus Sampling (Top-$p$) and Top-$k$ truncation on $\tilde{p}(y_t)$ to eliminate chaotic long-tail noise before sampling the definitive token. This ordered pipeline ensures that the modality alignment is fully realized before vocabulary pruning, mathematically resolving both the representation misalignment and the truncation traps that plague previous contrastive decoding frameworks.