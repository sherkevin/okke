# 3 Method

## 3.1 Contrastive Visual-Textual Grounding

To systematically expose "highly confident hallucinations" driven by strong linguistic priors and spurious statistical correlations in Large Vision-Language Models (LVLMs), we transition from fragile input-level perturbations to an intrinsic, representation-level contrastive framework. We posit that a factually grounded response must exhibit a measurable predictive dependency on the visual modality. If an LVLM produces an answer purely based on language priors or object co-occurrence shortcuts, its predictive distribution will remain largely unchanged even when the visual evidence is withdrawn.

To quantify this multimodal dependency without introducing Out-of-Distribution (OOD) artifacts or relying on flawed attention-based heuristics, we employ a contrastive decoding paradigm. 

For a given image $V$ and prompt $T$, we define the **Contextual Grounded State** as the standard LVLM forward pass conditioned on both modalities $(V, T)$. Concurrently, to measure the inherent linguistic bias embedded in the LLM's parametric memory, we establish a **Vision-Deprived Baseline State** by conditioning the model strictly on the textual prompt and a universally aligned blank visual placeholder $V_{\emptyset}$ (e.g., a standard zero-tensor image or the default uninformative padding tokens used during the LVLM's pre-training). 

By contrasting the output distributions of the grounded state $(V, T)$ against the baseline prior state $(V_{\emptyset}, T)$, we can directly isolate the true step-wise contribution of the visual modality to the generated sequence, bypassing arbitrary spatial masking, heuristic attention filtering, and physical pixel corruptions entirely.

## 3.2 Step-wise Visual Contribution Estimation

Existing uncertainty estimations relying on external Natural Language Inference (NLI) matrices suffer from prohibitive $\mathcal{O}(N^2)$ computational overhead. Instead, we propose a proxy-free, single-pass estimator leveraging the token-level logit dynamics of the LVLM itself.

Let $Y = \{y_1, y_2, \dots, y_m\}$ be the generated token sequence for the factual input $(V, T)$. At each autoregressive generation step $t$, the LVLM outputs a predictive distribution $P(y_t \mid Y_{<t}, V, T)$. Simultaneously, using the identical generated prefix $Y_{<t}$ (effectively forcing the evaluation of the immediate visual dependency for the exact generated trajectory), we compute the vision-deprived distribution $P(y_t \mid Y_{<t}, V_{\emptyset}, T)$.

We define the **Step-wise Visual Contribution (SVC)** using the log-probability margin between the grounded state and the prior state:
$$
\text{SVC}_t = \max \Big( 0, \log P(y_t \mid Y_{<t}, V, T) - \log P(y_t \mid Y_{<t}, V_{\emptyset}, T) \Big). \tag{1}
$$
A high $\text{SVC}_t$ mathematically ensures that the generation of token $y_t$ is heavily supported by the visual input. Conversely, if $\text{SVC}_t$ is near zero, the token $y_t$ is generated independently of the image, relying entirely on the text prefix $Y_{<t}$ and the model's internal linguistic priors—a primary symptom of hallucination.

To obtain a stable sequence-level metric, we aggregate the token-level contributions. Crucially, to specifically target *highly confident hallucinations*, we explicitly weight the aggregation by the model's predictive confidence for each token, $w_t = P(y_t \mid Y_{<t}, V, T)$. This upweights tokens the model is highly certain about, ensuring that confident but ungrounded tokens are penalized the most:
$$
\text{SVC}_{\text{seq}} = \frac{\sum_{t=1}^m w_t \cdot \text{SVC}_t}{\sum_{t=1}^m w_t}. \tag{2}
$$
This token-level contrastive divergence eliminates the need for expensive external semantic clustering or autoregressive counterfactual rollouts, reducing computational complexity to $\mathcal{O}(1)$ relative to the sample size, rendering it highly tractable for dense VQA deployments.

## 3.3 Prior-Calibrated Hallucination Scoring

The fundamental flaw in traditional entropy-based hallucination detection is the assumption that hallucinations always manifest as uncertainty (high entropy). In reality, LVLMs frequently produce "highly confident hallucinations" due to deeply ingrained statistical shortcuts. Our contrastive framework resolves this by explicitly uncoupling model confidence from visual grounding.

A response is factually grounded if it possesses both high predictive confidence and a high visual contribution. A highly confident hallucination occurs when the model is certain of its output (high $P(y_t)$) but the output is entirely driven by textual priors (low $\text{SVC}_t$).

To avoid the instability of unnormalized log-probability subtractions across varying sequence lengths and vocabularies, we formulate a strictly bounded **Prior-Calibrated Hallucination Score ($\mathcal{H}$)**. We map the step-wise visual contribution to a decay factor using an exponential function, where a lack of visual grounding ($\text{SVC}_t \approx 0$) results in a penalty approaching $1$:
$$
\mathcal{H}(Y) = \frac{\sum_{t=1}^m P(y_t \mid Y_{<t}, V, T) \cdot \exp(-\beta \cdot \text{SVC}_t)}{\sum_{t=1}^m P(y_t \mid Y_{<t}, V, T)}, \tag{3}
$$
where $\beta$ is a positive scaling hyperparameter. The score $\mathcal{H}(Y) \in (0, 1]$ fundamentally represents the proportion of the model's generation confidence that is *not* supported by the visual modality. A high $\mathcal{H}$ score implies the sequence is a highly confident hallucination driven by unimodal priors.

Hallucination detection is framed as a binary classification task where an output is flagged if $\mathcal{H}(Y) \ge \gamma$. Because this metric is normalized and bounded, $\gamma$ is highly stable across diverse tasks. To provide a comprehensive, threshold-agnostic evaluation against severely imbalanced hallucination datasets, we utilize the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC) as the primary benchmarking metrics, strictly outperforming naive accuracy or standard entropy-based proxies.