# 3 Method

## 3.1 Architecture-Level Visual Decoupling

To systematically expose "highly confident hallucinations" driven by strong linguistic priors and statistical shortcuts in Large Vision-Language Models (LVLMs), we must isolate the true causal contribution of the visual modality. However, naive contrastive methods that feed Out-of-Distribution (OOD) placeholder images (e.g., zero-tensors or pure noise) fundamentally corrupt the continuous latent manifold of the model. Such inputs trigger erratic activation shifts, yielding unreliable output distributions that reflect OOD noise rather than a clean linguistic prior.

To quantify multimodal dependency without introducing input-level OOD artifacts, we propose an intrinsic, **Architecture-Level Visual Decoupling** mechanism. For a given image $V$ and prompt $T$, the standard **Contextual Grounded State** executes a standard forward pass, computing the full attention matrix across the joint visual-textual sequence $(X_V, X_T)$. 

Concurrently, to establish the **Vision-Deprived Prior State**, we directly manipulate the causal attention mask within the LLM decoder. During the autoregressive generation of the $t$-th token, we strictly mask out the cross-attention from the current query token to the visual sequence $X_V$, while perfectly preserving the self-attention flow over the textual prompt and the generated prefix $Y_{<t}$. This architectural intervention mathematically forces the model to rely solely on its parametric linguistic memory and the textual context, establishing a pristine, in-distribution textual prior baseline without corrupting the input space.

## 3.2 Entity-Centric Visual Dependency Estimation

In standard autoregressive decoding, the probability of generating syntactic stop-words (e.g., "the", "is", ".") is inherently dominated by language grammar, independent of visual stimuli. Aggregating logit margins across all tokens inevitably buries the critical visual dependency of semantic concepts under the statistical weight of these highly confident stop-words. Furthermore, if a text prefix heavily leaks the context, the model can guess the next token without visual input, confounding standard contrastive decoding.

To resolve stop-word domination and context leakage, we introduce an **Entity-Centric Visual Dependency Estimation**. We apply a lightweight Part-of-Speech (POS) filter (e.g., via NLTK) on the generated sequence $Y = \{y_1, y_2, \dots, y_m\}$ to extract the set of critical content tokens $\mathcal{E}$ (comprising nouns, verbs, and adjectives), representing the factual claims of the response.

For each semantic token $y_t \in \mathcal{E}$, we compute its generative probability under the grounded state, $P_{V}(y_t) = P(y_t \mid Y_{<t}, V, T)$, and its probability under the architecture-decoupled prior state, $P_{\emptyset}(y_t) = P(y_t \mid Y_{<t}, \text{Mask}(V), T)$. 

We formalize the **Visual Information Ratio (VIR)** for a generated entity as:
$$
\text{VIR}_t = \frac{P_{V}(y_t)}{P_{\emptyset}(y_t) + \epsilon}, \tag{1}
$$
where $\epsilon$ is a minimal smoothing factor. The VIR explicitly quantifies the magnitude of visual reliance. A high $\text{VIR}_t$ indicates that the visual evidence was indispensable for predicting the entity. Conversely, if $\text{VIR}_t \approx 1$ (or lower), the generation of $y_t$ is entirely dictated by the textual prefix $Y_{<t}$ and parametric language priors, directly exposing a hallucination independent of the actual image content.

## 3.3 Parameter-Free Hallucination Severity Index

The core paradox of LVLM hallucinations is that models are often highly confident in their hallucinated outputs due to strong prior inertia. A truly grounded response requires both high generative confidence and high visual dependency. 

To eliminate empirical scaling hyper-parameters and fragile exponential decays, we formulate a strictly bounded, parameter-free **Hallucination Severity Index (HSI)**. For a generated sequence $Y$, the HSI evaluates the proportion of the model's confidence on semantic entities that is derived purely from the blind linguistic prior:

$$
\text{HSI}(Y) = \frac{\sum_{y_t \in \mathcal{E}} P_{\emptyset}(y_t)}{\sum_{y_t \in \mathcal{E}} \max \big(P_{V}(y_t), P_{\emptyset}(y_t)\big)}. \tag{2}
$$

The index $\text{HSI}(Y) \in [0, 1]$ fundamentally represents the hallucination risk. An $\text{HSI}$ approaching $1$ mathematically guarantees that the model's high confidence in its generated entities is entirely ungrounded in the visual modality (a highly confident hallucination). An $\text{HSI}$ approaching $0$ signifies that the structural semantics are completely dictated by the visual evidence.

By restricting the analysis strictly to the extracted entity set $\mathcal{E}$ and utilizing a normalized ratio, this index is intrinsically robust to varying sequence lengths and vocabulary sizes. Furthermore, our dual-pass attention-masking paradigm strictly requires $\mathcal{O}(L)$ forward steps (where $L$ is the sequence length). This linear time complexity eliminates the prohibitive $\mathcal{O}(N^2)$ scaling bottleneck of external NLI-graph clustering algorithms, ensuring high tractability.

Hallucination detection is executed as a binary classification based on the continuous $\text{HSI}$ score. To provide a rigorous, threshold-agnostic benchmark against severely imbalanced hallucination datasets, we strictly utilize the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC), removing the necessity for ad-hoc threshold tuning.