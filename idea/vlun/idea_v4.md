# 3 Method

## 3.1 Manifold-Constrained Synergistic Perturbation

To robustly elicit the epistemic uncertainty of Large Vision-Language Models (LVLMs) without corrupting task-critical fine-grained semantics or introducing out-of-distribution (OOD) artifacts, we propose a manifold-constrained synergistic perturbation strategy. This approach directly operates on the intrinsic visual-textual manifold, strictly preserving logical equivalence while probing the model's multi-modal grounding robustness.

### Phase-Preserving Frequency Perturbation
Conventional visual augmentations (e.g., spatial blurring, photometric shifts) either destroy fine-grained spatial cues or fail to adequately probe deep visual comprehension. To address this, we introduce a phase-preserving frequency-domain perturbation. 

In natural images, the phase spectrum encapsulates structural semantics (e.g., contours, boundaries, and critical fine-grained details), while the amplitude spectrum governs low-level visual statistics (e.g., textures, illumination). Given an image $I$, we compute its Fourier transform $\mathcal{F}(I) = \mathcal{A}(I) \cdot e^{j\mathcal{P}(I)}$, where $\mathcal{A}$ and $\mathcal{P}$ denote the amplitude and phase spectra, respectively. We generate the perturbed image $I_i$ by injecting constrained stochastic noise into the amplitude spectrum while keeping the phase strictly invariant:
$$
\mathcal{A}_i = \mathcal{A}(I) \odot (1 + \mathcal{N}(0, \sigma^2)),
$$
$$
I_i = \mathcal{F}^{-1}(\mathcal{A}_i \cdot e^{j\mathcal{P}(I)}). \tag{1}
$$
This frequency-domain paradigm completely preserves the geometric and factual semantics required for dense Visual Question Answering (VQA) while rigorously testing the LVLM's robustness against low-level statistical variations.

### Distribution-Aligned Textual Augmentation
Rule-based lexical substitutions often generate rigid, OOD sentences that trigger artificial uncertainty merely due to distribution shifts. To maintain the natural language distribution of the LVLM's pre-training data, we adopt a distribution-aligned prompting strategy. 

Instead of destructive paraphrasing, we curate a comprehensive set of task-equivalent instruction templates rigorously validated for the VQA domain. For a given original prompt $T$, the augmented query $T_i$ is uniformly sampled from this isomorphic template set (e.g., transitioning from direct interrogatives to polite descriptive requests). This ensures zero semantic drift, avoids OOD triggers, and incurs strictly zero generative latency.

### Multi-modal Manifold Filtering
Rather than arbitrarily coupling perturbation intensities, we construct a synergistic perturbation space bounded by actual cross-modal semantic consistency. The true synergy lies in ensuring that the combined perturbed inputs still reside on the original semantic manifold. 

We generate a candidate pool of multi-modal pairs $\langle I_i, T_j \rangle$ and utilize a lightweight pre-trained vision-language aligner (e.g., CLIP) to extract their cosine similarity $S(\cdot, \cdot)$. We retain only the pairs that satisfy a strict manifold constraint:
$$
\mathcal{S} = \left\{ \langle I_i, T_j \rangle \mid \left| S(I_i, T_j) - S(I, T) \right| \le \tau \right\}, \tag{2}
$$
where $\tau$ is a tight tolerance margin. This manifold-constrained sampling guarantees that the synergistic inputs present diverse sensory stimuli to the LVLM while strictly anchoring the underlying ground-truth multi-modal semantics.

## 3.2 Entailment-Graph Uncertainty Estimation

Standard dense embedding spaces (e.g., MiniLM with cosine similarity) often fail to distinguish fine-grained factual contradictions (e.g., "a red car" vs. "a blue car"), leading to fatal false-negative clustering of hallucinated responses. To accurately capture epistemic uncertainty rooted in factual divergence, we propose an entailment-graph-based estimation framework.

For the input ensemble $\mathcal{S}$ of size $N$, the LVLM performs a parallelized batch inference to generate responses $\mathcal{Y} = \{y_1, y_2, \dots, y_N\}$. To evaluate rigorous logical equivalence, we deploy a lightweight, highly optimized Natural Language Inference (NLI) model (e.g., DeBERTa-V3-small) that excels at fine-grained contradiction detection. 

We construct a directed entailment graph $\mathcal{G} = (\mathcal{Y}, \mathcal{E})$, where a directed edge $e_{u,v}$ exists if and only if the NLI model predicts that response $y_u$ strictly entails $y_v$. We then identify the Strongly Connected Components (SCCs) of $\mathcal{G}$, which mathematically correspond to classes of bidirectional entailment (i.e., strict semantic equivalence). These SCCs form our distinct semantic clusters $\{c_k\}_{k=1}^{N_c}$.

The epistemic uncertainty is precisely quantified via the Shannon entropy of this logically verified cluster distribution:
$$
U_{\text{LVLM}} = -\sum_{k=1}^{N_c} p(c_k)\log p(c_k), \tag{3}
$$
where $p(c_k) = \frac{|c_k|}{N}$. By leveraging logical entailment rather than spatial distance, this method correctly identifies fine-grained factual hallucinations as distinct clusters (increasing entropy) while maintaining $\mathcal{O}(N^2)$ tractability through the use of lightweight NLI forward passes on small $N$.

## 3.3 Hallucination Detection Formulation

The derived multi-modal uncertainty score $U_{\text{LVLM}}$ acts as a direct proxy for identifying hallucination. When confronted with manifold-constrained synergistic perturbations, an LVLM exhibiting genuine comprehension will produce logically consistent responses across all inputs, yielding a highly concentrated entailment graph (low $U_{\text{LVLM}}$). Conversely, "highly confident hallucinations" driven by fragile unimodal priors or spurious correlations will fracture under perturbation, resulting in divergent factual statements and high semantic entropy.

We formulate hallucination detection as a binary classification task, flagging an output as hallucinatory if $U_{\text{LVLM}} \ge \gamma$. Recognizing that hallucination detection datasets typically exhibit severe class imbalance, we evaluate our framework using threshold-dependent metrics (Precision, Recall, F1-score) and comprehensive threshold-agnostic metrics, specifically the Area Under the Receiver Operating Characteristic Curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC), ensuring robust and standardized benchmarking.