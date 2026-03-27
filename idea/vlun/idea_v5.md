# 3 Method

## 3.1 Causal-Aware Cross-Modal Intervention

To systematically expose "highly confident hallucinations" driven by strong linguistic priors and spurious statistical correlations in Large Vision-Language Models (LVLMs), we abandon superficial input-level perturbations (e.g., global noise injection or lexical substitutions) that fail to break robust biases. Instead, we formulate hallucination detection through the lens of causal inference. We posit that a factual, grounded response must exhibit a strong causal dependency on task-relevant visual semantics. If an LVLM produces an answer purely based on language priors or object co-occurrence shortcuts, the causal contribution of the visual modality will be abnormally low.

To quantify this, we design a Causal-Aware Cross-Modal Intervention mechanism that operates directly within the model's feature space, bypassing the artifact-inducing physical pixel manipulations (e.g., ringing artifacts from frequency modification) and bypassing coarse-grained alignment filters (e.g., CLIP).

### Task-Critical Visual Feature Masking
To isolate the causal effect of the visual modality on the generated output, we construct a factual counterfactual. For a given image $V$ and prompt $T$, the LVLM's visual encoder and projector extract a sequence of visual tokens $X_V \in \mathbb{R}^{L \times d}$. 

During the initial forward pass, we extract the cross-attention weights $\alpha \in \mathbb{R}^L$ from the last layer of the LLM decoder concerning the visual tokens. These weights indicate the spatial features the model attends to for its prediction. We define a binary intervention mask $M \in \{0, 1\}^L$, where $M_k = 1$ if $\alpha_k$ is within the top-$p\%$ of attention scores (representing task-critical fine-grained semantics), and $0$ otherwise. 

We perform a counterfactual intervention by masking these critical visual features:
$$
\tilde{X}_V = X_V \odot (1 - M) + \bar{X}_V \odot M, \tag{1}
$$
where $\bar{X}_V$ is the mean pooling vector of all non-critical background tokens. This context-preserving feature ablation creates a counterfactual state $\tilde{V}$ where the specific fine-grained visual evidence required to answer $T$ is mathematically removed, without introducing out-of-distribution (OOD) visual artifacts.

### Prior-Eliciting Textual Intervention
Concurrently, to measure the inherent linguistic bias embedded in the LLM's parametric memory, we construct a purely text-driven counterfactual $\tilde{T}$. Rather than rule-based paraphrasing, $\tilde{T}$ is designed as a context-free semantic probe (e.g., stripping the specific question context to a generalized prior format, or employing a blank visual token sequence $X_{\emptyset}$). This establishes the model's baseline output distribution when deprived of any conditional visual stimuli.

## 3.2 Generative Causal Effect Estimation

Existing uncertainty estimations relying on external Natural Language Inference (NLI) matrices and graph clustering suffer from prohibitive $\mathcal{O}(N^2)$ computational overhead and vulnerability to cascading graph failures. We propose a highly efficient, proxy-free estimator leveraging the token-level logit dynamics of the LVLM itself, requiring only a constant number of forward passes.

Let $Y = \{y_1, y_2, \dots, y_m\}$ be the generated token sequence for the factual input $(V, T)$. We estimate the Visual Causal Contribution (VCC) by analyzing the divergence in the predictive probability distribution under the factual and counterfactual conditions. 

At each generation step $t$, the LVLM outputs a probability distribution $P(y_t \mid Y_{<t}, V, T)$ over the vocabulary. The Total Direct Effect (TDE) of the critical visual features on the generated response is quantified by the Kullback-Leibler (KL) divergence between the factual distribution and the visually-ablated counterfactual distribution:
$$
\text{VCC}_t = D_{\text{KL}} \Big( P(\cdot \mid Y_{<t}, V, T) \parallel P(\cdot \mid Y_{<t}, \tilde{V}, T) \Big). \tag{2}
$$
A high $\text{VCC}_t$ indicates that the generation of token $y_t$ is strictly grounded in the precise visual evidence. Conversely, if $\text{VCC}_t$ is near zero, the model is generating $y_t$ independently of the visual input—a primary symptom of hallucination via strong unimodal priors.

To obtain a sequence-level metric, we aggregate the token-level causal effects, weighted by the normalized self-information (surprisal) of each token to diminish the influence of common stop-words:
$$
\text{VCC}_{\text{seq}} = \frac{\sum_{t=1}^m w_t \cdot \text{VCC}_t}{\sum_{t=1}^m w_t}, \quad \text{where} \; w_t = -\log P(y_t \mid Y_{<t}, V, T). \tag{3}
$$
This token-level divergence computation eliminates the need for expensive external NLI evaluation, reducing the computational complexity strictly to $\mathcal{O}(1)$ relative to the sample size, rendering it highly tractable for real-time dense VQA deployments.

## 3.3 Prior-Calibrated Hallucination Detection

The fundamental flaw in traditional entropy-based hallucination detection is the assumption that hallucinations always manifest as uncertainty (high entropy). In reality, LVLMs frequently produce "highly confident hallucinations" due to deeply ingrained statistical shortcuts. Our causal framework systematically resolves this paradox.

A response is factually grounded if it possesses both high predictive confidence and a high visual causal effect. A highly confident hallucination occurs when the model is certain of its output (high confidence) but the output is causally disconnected from the visual evidence (low VCC).

We formulate a Prior-Calibrated Hallucination Score ($\mathcal{H}$) for the generated sequence $Y$:
$$
\mathcal{H}(Y) = \left( \frac{1}{m} \sum_{t=1}^m \log P(y_t \mid Y_{<t}, V, T) \right) - \lambda \cdot \text{VCC}_{\text{seq}}, \tag{4}
$$
where $\lambda$ is a balancing coefficient. A high $\mathcal{H}$ score implies that the model's generation is driven almost entirely by internal linguistic confidence rather than multi-modal grounding.

Hallucination detection is thus framed as a binary classification task where an output is flagged if $\mathcal{H}(Y) \ge \gamma$. To provide a robust, threshold-agnostic evaluation of our causal-driven framework against severely imbalanced hallucination datasets, we utilize the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC) as the primary benchmarking metrics, comprehensively outperforming rudimentary accuracy or standard entropy-based proxies.