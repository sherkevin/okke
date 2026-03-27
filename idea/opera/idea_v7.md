# 3 Method

We first formulate the autoregressive generation procedure of multimodal large language models (MLLMs). Following this, we introduce the **Cross-Layer Visual Context Extraction** and **Dependence-Gated State Steering** strategies. This methodology directly addresses the complex representational dynamics of modern LLMs. By leveraging intermediate-layer semantic cross-attention and performing strictly bounded convex state steering in the final layer, our method suppresses severe visual hallucinations while rigorously guaranteeing the preservation of syntactic and functional tokens, introducing only a marginal, sequence-length-independent static overhead.

---

## 3.1 Formulation of MLLMs Generation

The generation pipeline of MLLMs maps a joint cross-modal input into a sequence of textual tokens. 

### Input Formulation
The input comprises visual tokens $\boldsymbol{x}_v = \{x_0, \dots, x_{N-1}\}$ extracted via a vision encoder and projected into the LLM representation dimension $d$, alongside text tokens $\boldsymbol{x}_p = \{x_N, \dots, x_{N+M-1}\}$. The full sequence context at decoding step $t$ is denoted as $\boldsymbol{x}_{<t} = \{x_i\}_{i=0}^{t-1}$ for $t \ge N+M$.

### Model Forward and Decoding
At step $t$, the LLM computes the final-layer hidden state $\boldsymbol{h}_t^{(L)} \in \mathbb{R}^d$, which encodes the contextualized next-token representation. A predefined vocabulary unembedding matrix $W \in \mathbb{R}^{|\mathcal{X}| \times d}$ maps $\boldsymbol{h}_t^{(L)}$ to the unnormalized logits $\boldsymbol{z}_t$:
\[
P(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}\!\left(\boldsymbol{z}_t\right),\quad \boldsymbol{z}_t = W \boldsymbol{h}_t^{(L)}
\tag{1}
\]
Modern LLMs strictly separate the input embedding space from the output unembedding hyperplanes $W$. To maintain mathematical stability, our intervention strictly calibrates the continuous hidden state $\boldsymbol{h}_t^{(L)}$ prior to the vocabulary projection, circumventing arbitrary logit-level manipulation.

---

## 3.2 Cross-Layer Visual Context Extraction

Hallucinations often emerge when models over-rely on linguistic priors, generating ungrounded entities. To ground the generation without falling into "Semantic Aliasing" (caused by global visual pooling) or disrupting the primary $O(t)$ generation throughput with expensive dynamic attention recalculations, we propose a cross-layer extraction mechanism.

In modern LLMs, intermediate layers are highly specialized for deep cross-modal semantic fusion, whereas the final layers are structurally dedicated to vocabulary mapping. Relying on final-layer Queries for cross-modal retrieval yields noisy alignment. Therefore, we extract the fine-grained visual attention distribution intrinsically computed by the model at a designated intermediate semantic layer $l_{sem}$ (e.g., $L/2$).

Let $a_{t,i}^{(l_{sem})}$ be the pre-computed self-attention weight from the current token $t$ to the visual token $i$ at layer $l_{sem}$. We passively read these weights—introducing zero extra $O(t^2)$ self-attention computations. We define the intrinsic **Visual Dependence Score** $\gamma_t$ of the current token as the total attention mass allocated to the visual prefix:
\[
\gamma_t = \sum_{i=0}^{N-1} a_{t,i}^{(l_{sem})}
\tag{2}
\]
To project this semantic visual focus into the final unembedding space without dimensional or manifold mismatch, we construct the target visual context vector $\boldsymbol{v}_t$ by pooling the *final-layer* representations of the visual tokens $\boldsymbol{h}_i^{(L)}$, weighted by the *intermediate-layer* semantic attention:
\[
\boldsymbol{v}_t = \frac{1}{\gamma_t + \epsilon} \sum_{i=0}^{N-1} a_{t,i}^{(l_{sem})} \boldsymbol{h}_i^{(L)}
\tag{3}
\]
where $\epsilon$ prevents division by zero. This cross-layer mechanism operates with a strictly sequence-length-independent complexity of $O(N)$ vector additions, accurately localizing visual evidence in the mathematically correct final-layer space.

---

## 3.3 Dependence-Gated State Steering

With the target visual context $\boldsymbol{v}_t$ accurately defined, our objective is to steer the predictive hidden state $\boldsymbol{h}_t^{(L)}$ towards visual grounding when the model hallucinates, while enforcing absolute preservation of structural and functional linguistics.

### Deviation-Proportional Amplification
Hallucination fundamentally occurs when the model ignores existing visual evidence in favor of parametric linguistic priors. In the hidden space, this manifests as the final predictive state $\boldsymbol{h}_t^{(L)}$ deviating structurally from the supporting visual context $\boldsymbol{v}_t$. We quantify this semantic deviation $D_t$ using cosine distance:
\[
D_t = 1 - \frac{\boldsymbol{h}_t^{(L)} \cdot \boldsymbol{v}_t}{\|\boldsymbol{h}_t^{(L)}\| \|\boldsymbol{v}_t\|}
\tag{4}
\]
By definition, $D_t \in [0, 2]$. Critically, when a severe hallucination occurs (e.g., the linguistic prior generates features entirely orthogonal or opposite to the visual evidence), $D_t$ approaches its maximum. This provides a robust continuous metric that peaks exactly when visual correction is most urgently required.

### Syntactic-Preserving Convex Calibration
To correct the state without destroying the manifold topology, we perform a convex interpolation between the original predictive state and the visual context vector. We define the dynamic steering intensity $\lambda_t$:
\[
\lambda_t = \min\left(1,\, \alpha \cdot \gamma_t \cdot D_t \right)
\tag{5}
\]
where $\alpha$ is a constant scaling hyperparameter. The calibrated hidden state is derived via:
\[
\tilde{\boldsymbol{h}}_t = (1 - \lambda_t) \boldsymbol{h}_t^{(L)} + \lambda_t \boldsymbol{v}_t
\tag{6}
\]
Finally, $\tilde{\boldsymbol{h}}_t$ is normalized to preserve variance and projected to the vocabulary logits:
\[
\tilde{\boldsymbol{z}}_t = W \left( \frac{\tilde{\boldsymbol{h}}_t}{\|\tilde{\boldsymbol{h}}_t\|} \|\boldsymbol{h}_t^{(L)}\| \right), \quad P_{final}(x_t \mid \boldsymbol{x}_{<t}) = \operatorname{Softmax}(\tilde{\boldsymbol{z}}_t)
\tag{7}
\]

**Theoretical Guarantee for Syntactic Preservation:** 
The preservation of functional words (e.g., "the", "is", punctuation) is mathematically guaranteed by the structural design of $\lambda_t$. For syntactic tokens, the generation is driven almost entirely by the textual prefix; hence, the intrinsic visual attention mass at the intermediate fusion layer approaches zero ($\gamma_t \approx 0$). Consequently, $\lambda_t \rightarrow 0$, leaving $\tilde{\boldsymbol{h}}_t = \boldsymbol{h}_t^{(L)}$. The steering mechanism activates exclusively for semantically dense entity tokens that naturally possess high visual dependence ($\gamma_t \gg 0$) but exhibit high semantic deviation ($D_t \gg 0$). This ensures zero perturbation to grammatical fluency while forcefully shifting the probability mass back to visually supported entities during hallucination triggers.