# 3 Method

We first formulate the generation procedure of multimodal large language models (MLLMs). Following this, we introduce the **Dynamic Fine-Grained Visual Retrieval** and the **Visual Manifold State Projection** strategies. This methodology fundamentally discards crude pooling and structural manipulation of the unembedding space. Instead, it operates entirely within the native continuous hidden manifold to dynamically suppress ungrounded hallucinations ("hallucinated entities") while rigorously preserving the topological integrity of syntactic and functional tokens, achieving genuine sequence-length-independent $O(1)$ efficiency.

---

## 3.1 Formulation of MLLMs Generation

The autoregressive generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input consists of visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM dimension $d$, alongside text tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context is denoted as $\boldsymbol{x}_{<t} = \{x_i\}_{i=0}^{t-1}$ for $t \ge N+M$.

### Model Forward and Decoding
At decoding step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t \in \mathbb{R}^d$, which encodes the contextualized next-token representation. A vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ projects $\boldsymbol{h}_t$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t
\tag{1}
\]
Modern LLMs do not tie the input embedding matrix with the output unembedding matrix $W$. To maintain rigorous mathematical stability, our proposed calibration strictly manipulates the continuous representation $\boldsymbol{h}_t$ before the final projection, circumventing any interference with the predefined hyperplane topologies of $W$.

---

## 3.2 Dynamic Fine-Grained Visual Retrieval

Hallucinations frequently manifest as the generation of ungrounded entities (e.g., fabricating objects absent from the image). Previous attempts to anchor generation via global visual average pooling suffer from severe **Semantic Aliasing**—collapsing hundreds of spatially distinct visual features into a single vector obliterates fine-grained multi-entity representations. Conversely, extracting historical attention matrices disrupts CUDA-level optimizations (e.g., FlashAttention) and scales poorly $O(L \times H \times t)$. 

To resolve this, we propose a lightweight, step-wise retrieval mechanism confined strictly to the final layer, ensuring constant-time complexity with respect to the decoding length $t$.

Before the autoregressive generation begins, we cache the final-layer Keys $\boldsymbol{K}_v \in \mathbb{R}^{N \times d}$ and Values $\boldsymbol{V}_v \in \mathbb{R}^{N \times d}$ corresponding explicitly to the visual tokens $\boldsymbol{x}_v$. At any decoding step $t$, let $\boldsymbol{q}_t \in \mathbb{R}^d$ be the final-layer Query vector for the current token. We dynamically retrieve the fine-grained visual context $\boldsymbol{v}_t$ by computing a localized attention mechanism exclusively over the visual prefix:
\[
a_{t,i} = \operatorname{Softmax}\left( \frac{\boldsymbol{q}_t \cdot \boldsymbol{k}_i^v}{\sqrt{d}} \right), \quad i \in \{0, \dots, N-1\}
\tag{2}
\]
\[
\boldsymbol{v}_t = \sum_{i=0}^{N-1} a_{t,i} \boldsymbol{v}_i^v
\tag{3}
\]
**Computational Complexity:** This operation requires only a single vector-matrix multiplication against a fixed size $N$ (visual tokens). Thus, the time complexity is strictly $O(N)$, which effectively equates to $O(1)$ regarding the dynamically growing sequence length $t$. It completely bypasses historical attention extraction, naturally avoids structural "attention sinks" in the textual context, and preserves localized visual representations without aliasing.

---

## 3.3 Visual Manifold State Projection

With the accurate, fine-grained visual evidence $\boldsymbol{v}_t$ retrieved, the objective is to suppress ungrounded linguistic generation (hallucination) while preserving valid grammatical structures. 

Instead of imposing fragile entropy-based penalties or enforcing historical textual repulsion—which degenerates into mere repetition penalties and actively damages the generation of functional words—we introduce an adaptive **Visual Manifold Amplification**. 

### Adaptive Subspace Amplification
When the language model attempts to hallucinate an entity due to an over-reliance on parametric knowledge (linguistic prior), the predictive hidden state $\boldsymbol{h}_t$ deviates from the underlying visual manifold. To enforce visual grounding without disrupting the manifold topology, we linearly project and amplify $\boldsymbol{h}_t$ along the direction of the dynamic visual evidence $\boldsymbol{v}_t$:
\[
\boldsymbol{h}_t^\parallel = \left( \frac{\boldsymbol{h}_t \cdot \boldsymbol{v}_t}{\|\boldsymbol{v}_t\|^2 + \epsilon} \right) \boldsymbol{v}_t
\tag{4}
\]
\[
\tilde{\boldsymbol{h}}_t = \boldsymbol{h}_t + \alpha \cdot \operatorname{ReLU}\!\left( \frac{\boldsymbol{h}_t \cdot \boldsymbol{v}_t}{\|\boldsymbol{h}_t\|\|\boldsymbol{v}_t\|} \right) \cdot \boldsymbol{h}_t^\parallel
\tag{5}
\]
where $\alpha$ is a scaling hyperparameter, and $\epsilon$ prevents numerical instability. The $\operatorname{ReLU}$ gating factor explicitly ensures that amplification only occurs when the model is exhibiting at least a marginal intention to generate visual semantics, averting noisy perturbation when $\boldsymbol{h}_t$ and $\boldsymbol{v}_t$ are diametrically opposed.

### Final Logit Calibration and Syntactic Preservation
The calibrated hidden state $\tilde{\boldsymbol{h}}_t$ is normalized to preserve original variance and projected via the standard unembedding matrix $W$:
\[
\tilde{\boldsymbol{z}}_t = W \left( \frac{\tilde{\boldsymbol{h}}_t}{\|\tilde{\boldsymbol{h}}_t\|} \|\boldsymbol{h}_t\| \right), \quad P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}(\tilde{\boldsymbol{z}}_t)
\tag{6}
\]

**Theoretical Justification against Syntactic Degradation:** 
A critical flaw in heuristic-based logit penalties is their destructive impact on functional words (e.g., "is", "the", punctuation), which do not require visual grounding. Our projection strategy inherently circumvents this. In the representation space of modern LLMs, syntactic and functional tokens possess exceptionally large biases and broad reception fields in the hyperplanes of $W$. They dominate the probability distribution unconditionally when syntactically required. Because $\boldsymbol{v}_t$ lies strictly within the semantic visual subspace, amplifying $\boldsymbol{h}_t$ along $\boldsymbol{v}_t$ alters the relative competition exclusively among semantically dense entity candidates (e.g., shifting the probability mass from a hallucinated "cat" back to the visually supported "dog"). It inflicts mathematically negligible perturbations on the robust logit dominance of necessary functional tokens, achieving seamless hallucination suppression without compromising linguistic fluency.