# 3 Method

## 3.1 Residual-Aware Visual Contribution Approximation
We first formally define the cross-modal interaction within large vision-language models (LVLMs). An input image $x_i$ is mapped by a visual encoder and projector into a sequence of $N$ visual tokens (where $N$ is dynamically conditioned on image resolution). These tokens are prepended to the text embeddings and fed into the LVLM, which generates a sequence of $M$ tokens autoregressively.

Standard attention weights $\mathcal{A}_{m,n,l,h}$ fundamentally fail to represent actual information flow because they isolate the routing probabilities from the Value ($W_V$) and Output ($W_O$) transformations. Building upon vector-norm interpretability methods established in text-only LLMs (e.g., Kobayashi et al.), we adapt this framework to quantify multimodal feature attribution by explicitly modeling the accumulation within the Transformer's residual stream.

The hidden state of the $m$-th token at the final layer $L$ is a linear accumulation of residual updates: $h_m^{(L)} = h_m^{(0)} + \sum_{l=1}^L \Delta h_m^{(l)}$. The specific additive contribution of the $n$-th visual patch at layer $l$ to the residual update $\Delta h_m^{(l)}$ is defined as:
\[
c_{m,n,l} = \sum_{h=1}^H \mathcal{A}_{m,n,l,h} \cdot \left( v_{n, l-1} W_V^{l,h} W_O^{l,h} \right) \tag{1}
\]
where $v_{n, l-1}$ is the visual hidden state preceding layer $l$. 

Crucially, to eliminate the $\mathcal{O}(M \cdot N \cdot L)$ computational bottleneck during autoregressive generation, we exploit the static nature of the visual projections. The transformed value vectors $z_{n,l,h} = v_{n, l-1} W_V^{l,h} W_O^{l,h}$ are entirely independent of the generated textual tokens and are strictly **pre-computed** once during the initial prompt-encoding phase. During the autoregressive generation of token $m$, computing the visual contribution $c_{m,n,l}$ reduces to a simple scalar-vector scaling by the dynamic attention weights $\mathcal{A}_{m,n,l,h}$. This mathematically drops the dynamic inference complexity from cubic matrix multiplications to a highly efficient $\mathcal{O}(N \cdot D)$ linear scaling per layer, making it fully scalable for high-resolution images and long-context generation.

---

## 3.2 Query-Agnostic Sink Calibration and Direct Logit Attribution
A critical challenge in multimodal attribution is the "Attention Sinks" phenomenon, where specific visual tokens (e.g., boundary patches) consistently absorb massive attention probabilities to act as bias terms, irrespective of semantic relevance. Since these sinks function as query-agnostic biases, we introduce **Query-Agnostic Sink Calibration**. We compute a baseline contribution map $B_{n,l}$ by passing a standardized dummy query (e.g., `<pad>` tokens) alongside the image. The calibrated, query-specific visual contribution is obtained by subtracting this semantic-free baseline:
\[
\hat{c}_{m,n,l} = \text{ReLU}\left( c_{m,n,l} - B_{n,l} \right) \tag{2}
\]
The magnitude of this calibrated vector, $\| \hat{c}_{m,n,l} \|_2$, represents the pure semantic information flow from visual token $n$. We normalize these magnitudes to form a valid, sink-resilient attribution distribution $\hat{\mathcal{P}}_{m,n,l}$ over the spatial dimension $N$.

To measure whether this visual attribution correctly grounds the generated concept, we avoid the mathematical fallacy of projecting unaligned, shallow visual features into the language vocabulary (i.e., Logit Lens). Instead, we employ **Direct Logit Attribution (DLA)**. Due to the linear additive property of the residual stream, the direct influence of the specific residual update $\hat{c}_{m,n,l}$ on the final unnormalized output logit of the generated token $y_m$ can be exactly quantified by projecting it onto the target token's unembedding vector $W_\text{unbed}[y_m]$:
\[
S_{m,n,l} = \hat{c}_{m,n,l} \cdot W_\text{unbed}[y_m] \tag{3}
\]
$S_{m,n,l}$ mathematically defines how much the visual patch $n$ at layer $l$ strictly increased (or decreased) the probability of generating token $y_m$. We compute the **Expected Logit Attribution** across the visual context:
\[
\mathbb{E}[S_{m,l}] = \sum_{n=1}^N \hat{\mathcal{P}}_{m,n,l} \cdot S_{m,n,l} \tag{4}
\]
Paired with the spatial entropy of the attribution distribution ($E_{m,l} = -\sum_n \hat{\mathcal{P}}_{m,n,l} \log \hat{\mathcal{P}}_{m,n,l}$), this mathematically robust framework natively captures hallucinations caused by both dispersed attention (high entropy) and localized but semantically detrimental activations (negative or low DLA).

---

## 3.3 DHCP: Causal Bottleneck Cross-layer Predictor
Prior architectures typically suffer from either extreme feature collapse (compressing high-dimensional layer updates into a few scalars) or severe parameter bloat and overfitting (concatenating massive $D$-dimensional vectors directly into standard Transformer Encoders). Furthermore, treating layer sequences as unordered sets violates the fundamental forward-dependency of deep networks.

To rigorously model the non-Markovian, sequentially accumulated nature of the residual stream without inducing parameter explosion, we propose the **Causal Bottleneck Cross-layer Predictor**. For each layer $l$, we first compute the attribution-weighted visual context update $U_{m,l} = \sum_{n=1}^N \hat{\mathcal{P}}_{m,n,l} \cdot \hat{c}_{m,n,l} \in \mathbb{R}^D$. To definitively prevent overfitting on high-dimensional representations, we project $U_{m,l}$ through a frozen, orthogonally initialized random projection matrix $W_\text{bottle} \in \mathbb{R}^{D \times d}$ (where $d \ll D$, e.g., $d=64$), yielding a compact residual update proxy $\tilde{U}_{m,l}$. 

The unified layer feature $F_{m,l} \in \mathbb{R}^{d+2}$ is formed by concatenating $\tilde{U}_{m,l}$ with the computed scalars $E_{m,l}$ and $\mathbb{E}[S_{m,l}]$. This forms an ordered sequence across network depth: $\Phi_m = [F_{m,1}, F_{m,2}, \dots, F_{m,L}]$.

To strictly enforce the directional information flow inherent to Transformer architectures—where layer $l$ can only read from layers $k \le l$—we process $\Phi_m$ using a lightweight **Causal Transformer Decoder**:
\[
H_m = \text{CausalTransformer}(\Phi_m + P_\text{depth}) \tag{5}
\]
The causal attention mask explicitly guarantees that the sequential dependency of the residual stream is respected, mimicking the true forward pass dynamics. The hidden state of the final layer $L$ perfectly encapsulates the cumulative multimodal alignment history and is passed through a simple MLP to output the token-level hallucination probability $p_m \in [0, 1]$. By leveraging pre-computed static projections, mathematically exact Direct Logit Attribution, and causally masked bottleneck sequence modeling, DHCP achieves scalable, theoretically rigorous, and highly accurate hallucination detection.