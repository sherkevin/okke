# 3 Method

## 3.1 Token-Level Causal Attribution Extraction
We first formulate the cross-modal interaction mechanisms within large vision-language models (LVLMs). Let $x_i$ and $x_t$ denote the input image and text query. The visual encoder extracts features mapped by a projector into a sequence of $N$ visual tokens, where $N$ is dynamic conditioned on the varying image resolution (e.g., in Qwen2.5-VL). These are prepended to the text embeddings and fed into the LVLM, generating a sequence of $M$ tokens autoregressively.

Relying on raw attention weights $\mathcal{A}_{m,n,l,h}$ is insufficient for evaluating feature attribution, as information mixing in deep Transformers obscures the exact causal contributions. Furthermore, explicitly storing the full tensor $\mathcal{A}_m \in \mathbb{R}^{N \times L \times H}$ for all generated tokens incurs an intolerable $\mathcal{O}(M \cdot N \cdot L \cdot H)$ memory overhead. 

To resolve both the theoretical inaccuracy of raw attention and the computational bottleneck, we employ an **on-the-fly Layer-wise Attention Rollout** mechanism. During the forward pass of the $m$-th generated token, we trace the causal information flow from the textual token to the visual tokens across the network depth. By aggregating across attention heads and computing the layer-wise rollout $\mathcal{R}_{m, n, l}$, we quantify the true causal contribution of the $n$-th visual patch at the $l$-th layer. The rollout vector $\mathcal{R}_{m,:,l} \in [0, 1]^N$ is computed sequentially during generation and immediately discarded from the computational graph once the token-level statistical metrics are extracted, ensuring a strict $\mathcal{O}(N \cdot L)$ memory footprint per step.

---

## 3.2 Dynamic Sink Calibration and Semantic Grounding
A fundamental challenge in attention analysis is the "Attention Sinks" phenomenon, where LVLMs consistently allocate massive attention mass to semantically devoid patches (e.g., image boundaries or the `<bos>` token) to preserve internal Softmax scores. Applying static or heuristic masking to these sinks arbitrarily truncates the probability space and destroys the distribution's mathematical integrity.

Instead, we introduce **Dynamic Sink Calibration**. Since attention sinks naturally exhibit zero-variance, high-activation behaviors across the temporal generation axis, we maintain a moving average of the causal attribution over the previously generated tokens: $\mathcal{B}_{n, l} = \frac{1}{m-1} \sum_{t=1}^{m-1} \mathcal{R}_{t,n,l}$. We then compute the relative causal divergence, effectively calibrating the rollout distribution without hard masking:
\[
\hat{\mathcal{R}}_{m,n,l} = \text{Softmax}(\alpha (\mathcal{R}_{m,n,l} - \mathcal{B}_{n,l})) \tag{1}
\]
where $\alpha$ is a scaling temperature. 

To address the "misaligned high-activation" pattern—where the model confidently attends to the *wrong* visual patch—we must preserve spatial semantics without relying on fixed spatial indices. We introduce the **Attribution-guided Semantic Grounding Score** $G_{m,l}$. Let $v_{n, l}$ denote the hidden visual representation of the $n$-th patch at layer $l$, and $h_{m,l}$ denote the hidden representation of the generated textual token. We identify the most causally relevant visual patch index $n^* = \arg\max_n \hat{\mathcal{R}}_{m,n,l}$, and measure its semantic alignment with the generated concept via cosine similarity:
\[
G_{m,l} = \text{cos\_sim}(v_{n^*, l}, h_{m,l}) \tag{2}
\]
By pairing the semantic grounding score $G_{m,l}$ with the distributional entropy of $\hat{\mathcal{R}}_{m,:,l}$, we successfully capture both "Attention Defocusing" (high entropy) and "Spurious Concentration" (high activation but low semantic grounding $G_{m,l}$) without collapsing critical spatial-semantic correlations.

---

## 3.3 DHCP: Dynamic Hierarchical Causal Predictor
Previous methods employing 1D-CNNs over the layer dimension operate under the flawed assumption of local translation invariance. In Transformers, shallow layers (low-level visual correlation) and deep layers (high-level semantic alignment) serve distinctly different functions; sliding a convolutional kernel across them destroys this explicit hierarchy. 

To rigorously model this layer-wise functional transition, we propose the **Dynamic Hierarchical Causal Predictor (DHCP)**. For each generated token $m$ at layer $l$, we construct a compact, resolution-agnostic feature vector $f_{m,l} \in \mathbb{R}^3$ consisting of: (1) the calibrated attribution entropy $E_{m,l}$, (2) the maximum calibrated attribution weight $S_{m,l} = \max_n \hat{\mathcal{R}}_{m,n,l}$, and (3) the semantic grounding score $G_{m,l}$. 

This yields a sequential layer-wise trace $\Phi_m = [f_{m,1}, f_{m,2}, \dots, f_{m,L}] \in \mathbb{R}^{L \times 3}$. To respect the directional progression from shallow to deep layers without assuming translation invariance, we process $\Phi_m$ using a Gated Recurrent Unit (GRU):
\[
h_{m,l} = \text{GRU}(f_{m,l}, h_{m, l-1}) \tag{3}
\]
The final hidden state $h_{m,L}$, which elegantly encapsulates the full hierarchical evolution of causal visual grounding, is fed into an MLP to yield the final hallucination probability:
\[
p_m = \sigma(\text{MLP}(h_{m,L})) \tag{4}
\]
DHCP completely circumvents the memory bottlenecks of full attention storage, dynamically calibrates structural attention sinks, and rigorously evaluates token-level hallucinations by assessing both causal distribution entropy and explicit semantic grounding. This architecture scales seamlessly to arbitrary image resolutions and extensive long-text generations.