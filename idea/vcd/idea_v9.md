# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by $\theta$ taking a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. At time step $t$, the token $y_t$ is sampled from the conditional probability distribution:
$$
p_{\text{full}}(y_t) \equiv p_{\theta}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_{\text{full}}(y_t)\big). \tag{1}
$$
Object hallucinations during this process are widely attributed to the language decoder's over-reliance on its marginal textual prior. When the language inertia overwhelms the visual signals, the model tends to output highly plausible linguistic co-occurrences that contradict the actual visual context. Contrastive Decoding (CD) and Classifier-Free Guidance (CFG) address this by penalizing the language prior during generation.

## 3.2 Confidence-Aware Textual Prior Extraction
To extract the language prior, previous approaches often maintain an auxiliary vision-free text sequence in parallel, feeding the generated multi-modal prefix $y_{<t}$ into a pure-text stream to obtain $p_{\text{text}}(y_t \mid x, y_{<t})$. However, as accurately pointed out by critical analyses in the field, this "forced teaching" mechanism inherently suffers from a severe out-of-distribution (OOD) paradox. When $y_{<t}$ contains strong visually-dependent entities (e.g., describing a specific complex scene), feeding this prefix into a purely text-conditioned stream confronts the text model with incomprehensible, unaligned context. This forces the text stream to collapse into an OOD state, yielding chaotic prior distributions.

To mathematically rigorously address this, we introduce **Confidence-Aware Prior Extraction**. We acknowledge that the text-only stream should not be universally trusted across all generation steps. Instead of blindly extracting the prior at every step, we monitor the intrinsic Shannon entropy of the textual distribution to detect OOD collapse:
$$
\mathcal{H}(p_{\text{text}}) = - \sum_{y \in \mathcal{V}} p_{\text{text}}(y) \log p_{\text{text}}(y). \tag{2}
$$
A low entropy indicates that the textual stream confidently understands the prefix $y_{<t}$ driven by syntax or strong language habits (e.g., common idioms or fixed transitions). In contrast, an abnormally high $\mathcal{H}(p_{\text{text}})$ signifies that the prefix is heavily anchored in visual semantics that the text-only model cannot extrapolate, marking an OOD collapse.

Furthermore, to mitigate the strictly doubled KV-cache and computational FLOPs overhead associated with parallel decoding, we adopt an **On-Demand Triggering** strategy. The auxiliary text-stream forward pass is only executed when $p_{\text{full}}$ exhibits uncertainty (e.g., the confidence of the top-1 token falls below a safe threshold $\gamma$). For highly deterministic steps where the primary LVLM is extremely confident, the prior extraction is skipped, effectively reducing the computational overhead to a negligible fraction compared to standard full-sequence dual-decoding.

## 3.3 Adaptive Contrastive Modality Decoding (ACMD)

### 3.3.1 Formulating the Base Contrastive Objective
Instead of mathematically obfuscating the mechanism, we adopt the standard, well-established logit-level contrastive formulation equivalent to CFG/CD. The fundamental objective is to recalibrate the logits by subtracting the language prior bias:
$$
\tilde{\operatorname{logit}}(y_t) = \operatorname{logit}_{\text{full}}(y_t) - \alpha_t \cdot \operatorname{logit}_{\text{text}}(y_t), \tag{3}
$$
where $\alpha_t \ge 0$ modulates the penalty strength. Following standard engineering practices in modern language generation frameworks, this logit arithmetic is executed strictly before any non-linear transformations (e.g., Softmax) and before applying Nucleus Sampling (Top-$p$) or Top-$k$ truncation. This avoids the trivial truncation trap where visually grounded tokens are prematurely discarded before the contrastive penalty can rescue them.

### 3.3.2 OOD-Gated Dynamic Modulator
The core limitation of standard contrastive decoding in LVLMs is the assumption that the language prior is uniformly valid. Based on our OOD analysis in Section 3.2, applying Eq. (3) when the text stream has collapsed into an OOD state will inject random noise into the logits, destroying the generation quality. Furthermore, relying on divergence metrics (e.g., KL divergence between $p_{\text{full}}$ and $p_{\text{text}}$) as penalty scalers is fundamentally flawed, because an extreme KL divergence is precisely the symptom of the text stream's OOD collapse, not a signal for visual intervention.

To resolve this, we propose an **OOD-Gated Dynamic Modulator**. The penalty strength $\alpha_t$ should be maximized *only* when the text stream is highly confident (low entropy, no OOD collapse) but its prediction contradicts the multi-modal reasoning. We formulate the adaptive penalty weight as:
$$
\alpha_t = \alpha_{\text{base}} \cdot \underbrace{\exp\left( -\frac{\mathcal{H}(p_{\text{text}})}{\tau} \right)}_{\text{OOD Gate}} \cdot \underbrace{\mathbb{I} \big( \arg\max(p_{\text{text}}) \neq \arg\max(p_{\text{full}}) \big)}_{\text{Conflict Trigger}}, \tag{4}
$$
where $\tau$ is a temperature factor adjusting the sensitivity to entropy.

The logic of this modulation is strictly constrained:
1. **OOD Collapse Phase**: If the text stream cannot comprehend the multi-modal prefix, $\mathcal{H}(p_{\text{text}})$ spikes. The OOD Gate rapidly decays towards zero, safely deactivating the contrastive penalty and completely trusting the multi-modal stream.
2. **Syntactic Consensus Phase**: If both streams are confident and agree on the next token (e.g., generating punctuation or syntax), the Conflict Trigger evaluates to 0, preventing unnecessary perturbations to the fluent linguistic structure.
3. **Hallucination Risk Phase**: If the text stream is highly confident (low entropy) but predicts a different token than the full model, it indicates a strong language habit attempting to hijack the visually grounded reasoning. Here, $\alpha_t$ approaches $\alpha_{\text{base}}$, applying targeted suppression to the pure-text inertia.

The final token $y_t$ is sampled from the recalibrated distribution:
$$
p_{\text{ACMD}}(y_t) = \operatorname{softmax}\big(\tilde{\operatorname{logit}}(y_t)\big), \tag{5}
$$
followed by standard truncation strategies (Top-$p$/Top-$k$). By explicitly recognizing and gating the OOD collapse inherent to auto-regressive multi-modal extraction, ACMD establishes a robust, theoretically coherent, and computationally efficient contrastive paradigm.