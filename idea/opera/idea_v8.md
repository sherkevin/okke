# 3 Method

We formulate the autoregressive generation procedure of multimodal large language models (MLLMs) and introduce the **Sink-Calibrated Visual Routing** and **Native Vocabulary-Space Anchoring** strategies. This methodology fundamentally discards arbitrary hidden-state mixing and fragile cross-layer heuristics. By mapping visual evidence into the native vocabulary space *a priori* and dynamically calibrating distributions using sink-resilient semantic routing, our approach strictly preserves the un-tied manifold topologies of modern LLMs, dynamically suppresses ungrounded entities, and mathematically guarantees the integrity of syntactic token generation without static hyperparameter reliance.

---

## 3.1 Formulation of MLLMs Generation

The generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The sequence comprises visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM representation dimension $d$, alongside text tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context at decoding step $t$ is denoted as $\boldsymbol{x}_{<t}$.

### Model Forward and Decoding
At step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t^{(L)} \in \mathbb{R}^d$. A predefined vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ maps $\boldsymbol{h}_t^{(L)}$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P_{LLM}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t^{(L)}
\tag{1}
\]
Modern LLMs strictly separate the input embedding space from the output unembedding hyperplanes $W$. To avoid catastrophic logit collapse caused by feeding pure visual aggregated features into the text-specialized $W$ matrix, our intervention is strictly constructed within the isomorphic probability space.

---

## 3.2 Sink-Calibrated Dynamic Visual Routing

During autoregressive generation, intermediate self-attention layers perform complex semantic routing. However, directly utilizing raw attention weights is heavily distorted by "Attention Sinks"—where models allocate massive attention to initial tokens (including the visual prefix) merely to maintain softmax distributional stability, particularly during the generation of functional or syntactic words.

To extract genuine semantic visual dependence while bypassing hardcoded layer selection, we propose a sink-calibrated, layer-agnostic routing mechanism. Let $A_{t,i}^{(l)}$ be the pre-computed self-attention weight from the current text token $t$ to visual token $i$ at layer $l \in \{1, \dots, L\}$. 

We continuously track the historical attention sink prior for each visual token at each layer using a running average over the generated sequence:
\[
S_{i}^{(l)} = \frac{1}{t} \sum_{k=0}^{t-1} A_{k,i}^{(l)}
\tag{2}
\]
The active, semantic visual routing score $\hat{A}_{t,i}^{(l)}$ isolates genuine feature retrieval by subtracting the structural sink baseline:
\[
\hat{A}_{t,i}^{(l)} = \max\left(0,\, A_{t,i}^{(l)} - S_{i}^{(l)}\right)
\tag{3}
\]
We aggregate these active routing signals across all intermediate layers to construct a highly robust, dynamic visual alignment distribution $\boldsymbol{w}_t \in \mathbb{R}^N$:
\[
w_{t,i} = \frac{\sum_{l=1}^{L} \hat{A}_{t,i}^{(l)}}{\sum_{j=0}^{N-1} \sum_{l=1}^{L} \hat{A}_{t,j}^{(l)} + \epsilon}
\tag{4}
\]
where $\epsilon$ prevents zero-division. This calculation relies entirely on passively read, pre-computed attention matrices, introducing strictly $O(N)$ sequence-length-independent overhead.

---

## 3.3 Native Vocabulary-Space Anchoring

Hallucination occurs when the predicted linguistic distribution heavily deviates from the underlying visual evidence. Previous methods attempt to measure this deviation via cosine distance in the final hidden space—a theoretically flawed assumption, as the final layer is structurally optimized for vocabulary hyperplane projection, not contrastive text-image alignment.

To rigorously compute visual grounding, we project the final-layer visual hidden states into the native vocabulary space *a priori*. During the initial pre-fill stage (executed only once), we map each final-layer visual token representation $\boldsymbol{h}_i^{(L)}$ through the unmodified unembedding matrix $W$:
\[
P_i^v = \operatorname{Softmax}\left( W \boldsymbol{h}_i^{(L)} \right)
\tag{5}
\]
This mathematically valid operation explicitly decodes the intrinsic semantic concepts (e.g., nouns, attributes) captured by each visual patch into the precise predefined hyperplanes of the vocabulary space. 

At any decoding step $t$, the context-aware **Visual Grounding Prior** $P_t^{ground}$ is dynamically formulated as the expectation of these visual concept distributions, weighted by the sink-calibrated routing $\boldsymbol{w}_t$:
\[
P_t^{ground} = \sum_{i=0}^{N-1} w_{t,i} P_i^v
\tag{6}
\]
By aggregating in the probability space rather than the hidden state space, we completely circumvent the topological distortion of the $W$ manifold.

---

## 3.4 Adaptive Distributional Calibration

With the mathematically sound visual grounding prior $P_t^{ground}$ established, we dynamically calibrate the final predictive distribution to suppress ungrounded entity hallucinations without degrading structural syntax.

### Entropy-Gated Semantic Intervention
Functional tokens and punctuation naturally lack semantic visual alignment. For such tokens, the visual patches do not form a consensus, leading to a high-entropy, uniformly dispersed $P_t^{ground}$. Conversely, when genuine visual entities are retrieved, $P_t^{ground}$ exhibits sharp predictive spikes (low entropy).

We quantify the semantic density of the visual prior using normalized Shannon entropy to derive a dynamic interpolation parameter $\lambda_t$:
\[
\lambda_t = \max\left(0,\, 1 - \frac{-\sum_{x \in \mathcal{X}} P_t^{ground}(x) \log P_t^{ground}(x)}{\log |\mathcal{X}|}\right)
\tag{7}
\]
The final decoding probability is calibrated via convex interpolation:
\[
P_{final}(x_t \mid \boldsymbol{x}_{<t}) = (1 - \lambda_t) P_{LLM}(x_t) + \lambda_t P_t^{ground}(x_t)
\tag{8}
\]

**Theoretical Guarantee for Syntactic Preservation:** 
This formulation guarantees zero interference with functional/syntactic tokens without relying on static hyperparameter tuning. When generating functional tokens, the sink-calibration (Eq. 3) suppresses structural attention, and the entropy-gating (Eq. 7) evaluates to $\lambda_t \approx 0$, seamlessly falling back to the unmodified linguistic prior $P_{LLM}$. The calibration exclusively engages ($\lambda_t > 0$) when predicting visually grounded entities, forcing the probability mass away from linguistically hallucinated artifacts and strictly anchoring it to the visually supported textual concepts defined in $P_t^{ground}$.