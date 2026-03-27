# 3 Method

## 3.1 Content-Preserving Synergistic Perturbation

To robustly elicit the epistemic uncertainty of Large Vision-Language Models (LVLMs) without corrupting task-critical semantics, we propose a content-preserving synergistic perturbation strategy. Unlike approaches that rely on destructive spatial masking or architecture-specific assumptions, our method employs universal, non-destructive transformations to strictly maintain both focal and contextual semantic integrity.

### Lossless Visual Perturbation
Methods utilizing spatial Gaussian blurring or saliency-guided masking inevitably risk destroying crucial contextual cues in the background (e.g., distant traffic signs or secondary objects) that are vital for complex Visual Question Answering (VQA). To address this, we adopt a purely content-preserving visual degradation paradigm. 

Given the original image $I$, we generate a perturbed image $I_i$ by sampling from a distribution of global photometric shifts (e.g., variations in brightness, contrast, and saturation) and non-destructive affine transformations (e.g., slight scaling with padding):
$$
I_i = \phi_{\text{affine}}(\phi_{\text{photo}}(I, \delta_i), \theta_i), \tag{1}
$$
where $\delta_i$ and $\theta_i$ are transformation parameters sampled from a bounded multivariate uniform distribution. This guarantees that all fine-grained spatial semantics—regardless of their salience—remain globally intact while shifting the pixel-level distribution to probe the model's visual robustness.

### Lightweight Lexical Paraphrasing
To eliminate the high latency, inherent semantic drift, and high computational costs associated with LLM-based temperature sampling, we utilize a highly efficient, rule-based linguistic augmentation engine. 

For a given textual prompt $T$, we apply deterministic syntactic restructuring and constrained synonym substitution via an off-the-shelf natural language toolkit (e.g., WordNet-guided adjective/adverb swapping that strictly preserves core noun entities and verbs):
$$
T_i = \phi_{\text{lex}}(T), \tag{2}
$$
where $\phi_{\text{lex}}$ generates syntactically diverse but strictly semantically isomorphic queries in CPU-bound micro-seconds, completely avoiding generative LLM overhead.

### Correlated Multi-modal Synergy
To construct a truly synergistic perturbation space rather than relying on independent Cartesian product sampling, we dynamically correlate the severity of visual and textual perturbations. Let $s_i \in (0, 1]$ denote a unified perturbation severity factor. We sample paired multi-modal inputs where both $\delta_i, \theta_i$ and the syntactic complexity of $T_i$ are parameterized by $s_i$:
$$
\mathcal{S} = \{\langle I_i, T_i \rangle \mid s_i \sim \mathcal{U}(0, \sigma_{\max}), i \in [1, N]\}, \tag{3}
$$
This coupled interaction ensures that the LVLM is evaluated across a synchronized manifold of cross-modal degradation, effectively destabilizing fragile spurious correlations without violating the ground-truth semantics.

## 3.2 Parallelized Multi-modal Uncertainty Estimation

To quantify the LVLM's internal uncertainty while strictly maintaining computational efficiency for real-world deployment, we abandon serial evaluation paradigms (e.g., step-by-step early stopping) that severely disrupt modern GPU batching capabilities.

For the input ensemble $\mathcal{S}$ of size $N$, the LVLM performs a single parallelized batch inference to generate a set of responses $\mathcal{Y} = \{y_1, y_2, \dots, y_N\}$. To evaluate semantic equivalence without the massive overhead of pairwise Natural Language Inference (NLI) matrices, we deploy a lightweight dense sentence encoder (e.g., MiniLM) to project the responses into a high-dimensional continuous semantic space, yielding embeddings $E \in \mathbb{R}^{N \times d}$.

We then apply a highly efficient density-based clustering algorithm (e.g., DBSCAN) over $E$ using cosine distance to partition the responses into distinct semantic equivalence classes $\{c_k\}_{k=1}^{N_c}$. The epistemic uncertainty is formalized as the Shannon entropy of this cluster distribution:
$$
U_{\text{LVLM}} = -\sum_{k=1}^{N_c} p(c_k)\log p(c_k), \tag{4}
$$
where $p(c_k) = \frac{|c_k|}{N}$. By leveraging fully vectorized batch generation and fast embedding-based clustering, the time complexity of uncertainty estimation is reduced from $\mathcal{O}(N^2)$ LLM inference calls to $\mathcal{O}(N)$ lightweight forward passes, ensuring seamless integration into production pipelines.

## 3.3 Robust Hallucination Detection and Evaluation Metrics

The derived uncertainty score $U_{\text{LVLM}}$ serves as a quantifiable indicator of hallucination. A critical challenge in LVLMs is the phenomenon of "highly confident hallucinations," where models confidently output identical incorrect answers due to strong unimodal language priors. By subjecting the model to our correlated synergistic perturbations (Section 3.1), we actively disrupt these fragile unimodal priors. Responses grounded in true multi-modal comprehension remain invariant (yielding low entropy), whereas responses hallucinated from spurious correlations exhibit semantic divergence under coupled noise (yielding high entropy).

We formulate hallucination detection as a threshold-based binary classification task, where an output is flagged as hallucinatory if $U_{\text{LVLM}} \ge \gamma$. 

Recognizing that hallucination detection is inherently a severely imbalanced classification problem, we discard standard Accuracy, which heavily biases towards the majority class. Instead, we evaluate detection performance using robust metrics suited for imbalanced distributions. Based on the counts of True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$), we define Precision and Recall:
$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}. \tag{5}
$$
The primary threshold-dependent evaluation metric is the F1-score:
$$
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}. \tag{6}
$$
Furthermore, to provide a comprehensive, threshold-agnostic assessment of our uncertainty estimation reliability, we report the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC), ensuring rigorous benchmarking of the hallucination detection capability.