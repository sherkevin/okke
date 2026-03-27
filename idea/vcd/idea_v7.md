# 3 Method

## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) taking a textual query $x$ and a visual input $v$ to generate a response $y$ auto-regressively. The token at time step $t$, denoted as $y_t$, is sampled from the probability distribution conditioned on the prompt and the visual context:
$$
p_{\theta}(y_t \mid v, x, y_{<t}) = \operatorname{softmax}\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big). \tag{1}
$$
In prevailing LVLMs (e.g., LLaVA), visual features are converted into tokens and concatenated with the text sequence, meaning cross-modal fusion occurs densely from the very first Transformer layer. Object hallucinations predominantly arise when the language decoder's intrinsic statistical priors overpower the visually grounded signals during this autoregressive process.

## 3.2 Terminal Modality Decoupling (TMD) via KV-Cache Masking
Previous contrastive approaches attempt to isolate language priors either by perturbing the visual input (which induces severe out-of-distribution domain shifts) or by contrasting intermediate layers (which fundamentally ignores that LVLMs fuse vision and text from layer 1, meaning intermediate layers are already multimodal). 

To extract the pure language prior without inducing OOD anomalies or requiring full redundant forward passes, we introduce **Terminal Modality Decoupling (TMD)**. TMD operates exclusively at the final Transformer layer $L$. During the forward pass at step $t$, the standard final hidden state $h_t$ is computed using the full Key-Value (KV) cache of all preceding visual and textual tokens. 

Simultaneously, we compute an unconditioned text-only state $\hat{h}_t$ by applying a modality-specific attention mask in layer $L$. Specifically, we mask out the attention weights from the current query token to all visual tokens $\mathcal{I}_{\text{vis}}$ in the KV cache:
$$
\hat{A}_{t, j} = 
\begin{cases} 
-\infty, & \text{if } j \in \mathcal{I}_{\text{vis}} \\
A_{t, j}, & \text{otherwise}
\end{cases} \tag{2}
$$
where $A_{t, j}$ is the pre-softmax attention score in the final layer. This produces $\hat{h}_t$, which forces the final token prediction to rely entirely on the accumulated textual context ($x$ and $y_{<t}$) without direct access to the visual evidence at the critical classification step. Because layers $1$ to $L-1$ remain unmodified, the hidden representations remain strictly within the model's natural continuous manifold. 

The text-only state $\hat{h}_t$ is then projected to the vocabulary space to yield the language-prior logits:
$$
\operatorname{logit}_{\text{text}}(y_t) = W \cdot \hat{h}_t. \tag{3}
$$
This requires only one additional forward pass of a *single* Transformer block (layer $L$) and the LM head, completely bypassing the massive computational overhead of dual full-model forward passes.

## 3.3 Visual Advantage Decoding (VAD)

### 3.3.1 Robust Candidate Sub-space Construction
Language models naturally exhibit a Zipfian distribution. Applying contrastive operations across the entire vocabulary space amplifies noise from long-tail tokens. To construct a mathematically robust candidate set $\mathcal{V}_{\text{cand}}$ without relying on fragile heuristic thresholds, we employ standard Nucleus Sampling (Top-$p$) combined with Top-$k$ truncation on the full unconditioned distribution $p_\theta$:
$$
\mathcal{V}_{\text{cand}} = \big\{ y \in \operatorname{Top}_k(\mathcal{V}) \mid \sum_{y' \in \mathcal{V}_{\text{cand}}} p_\theta(y') \le p \big\}, \tag{4}
$$
where $p$ is the cumulative probability threshold (e.g., 0.9). This standard strategy guarantees the retention of syntactically necessary functional tokens and highly probable entities while safely discarding chaotic long-tail noise.

### 3.3.2 Modality-Advantage Calibration
To correctly penalize language-induced hallucinations without logical contradictions, we introduce the **Visual Advantage** $\Delta(y_t)$, defined as the direct logit differential between the fully grounded representation and the text-only representation:
$$
\Delta(y_t) = \operatorname{logit}_{\text{full}}(y_t) - \operatorname{logit}_{\text{text}}(y_t). \tag{5}
$$
A hallucinated token heavily driven by language inertia will exhibit a high $\operatorname{logit}_{\text{text}}$, resulting in a negative or near-zero visual advantage. Conversely, a genuinely grounded visual entity will receive a significant positive boost from the full visual context, yielding a strongly positive $\Delta(y_t)$.

We formulate the calibrated decoding logit as:
$$
\tilde{\operatorname{logit}}_{\text{VAD}}(y_t) = \operatorname{logit}_{\text{full}}(y_t) + \alpha_t \cdot \Delta(y_t), \quad \forall y_t \in \mathcal{V}_{\text{cand}} \tag{6}
$$
Unlike methods that use rigid $\max(0, \cdot)$ operators (which fail to penalize hallucinations that compound over time), Eq. (6) provides a continuous, bidirectional calibration. It explicitly rewards tokens that rely on vision (positive $\Delta$) and strictly suppresses tokens that are pure language guesses (negative $\Delta$), naturally preserving grammatical structures which typically have $\Delta \approx 0$.

### 3.3.3 Prior-Entropy Dynamic Scaling
Scaling the penalty using divergence metrics (like JSD) is logically flawed, as a high divergence often occurs precisely when the deep layer successfully corrects a language prior using visual evidence—penalizing this would destroy correct cross-modal reasoning. 

Instead, the risk of hallucination is fundamentally tied to the *confidence* of the language prior itself. We dynamically scale the calibration strength $\alpha_t$ based on the Shannon entropy $\mathcal{H}(p_{\text{text}})$ of the text-only distribution:
$$
\alpha_t = \alpha_{\text{base}} \cdot \exp\left( - \frac{\mathcal{H}(p_{\text{text}})}{\tau} \right), \tag{7}
$$
where $\tau$ is a temperature hyperparameter. When the language prior is highly confident in predicting the next token (low entropy), the risk of hallucination driven by textual inertia is maximized, naturally increasing $\alpha_t$ to enforce visual grounding. When the language prior is uncertain (high entropy), the model inherently relies more on visual signals, and $\alpha_t$ scales down gracefully. 

The final token $y_t$ is sampled from the normalized calibrated distribution:
$$
p_{\text{VAD}}(y_t) = \operatorname{softmax}\big(\tilde{\operatorname{logit}}_{\text{VAD}}(y_t)\big). \tag{8}
$$