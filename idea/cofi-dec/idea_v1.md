# 3 Methodology
This paper introduces **CoFi-Dec**, a novel training-free framework that improves the reliability of Large Vision-Language Model (LVLM) responses via recursive refinement with feedback from a text-to-image generative model, as illustrated in Figure 2.

### Problem Setting
We consider an LVLM parameterized by \(\theta\), which takes a visual input \(v\) and a textual query \(x\), and generates a coherent, relevant textual response sequence \(y\) autoregressively.
The image \(v\) is encoded by a vision encoder and projected into visual tokens via a vision-language alignment module (e.g., Q-Former [28] or linear projection [33]), which maps visual features into the language embedding space. These visual tokens and the tokenized query are fed into the language encoder to guide generation.

Formally, the token distribution at each step \(t\) is:
\[
y_t \sim p_\theta(y_t \mid v, x, y_{<t}) \propto \exp\big(f_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
\]
where \(y_t\) is the token at step \(t\), \(y_{<t} = [y_0,\dots,y_{t-1}]\) denotes the preceding tokens, and \(f_\theta\) outputs unnormalized logits over vocabulary \(\mathcal{V}\). Generation proceeds until an end-of-sequence token is produced, yielding the final response \(y = [y_0,\dots,y_T]\).

---

## 3.1 Generative Feedback with Multi-Granular Conditioning
Hallucination remains a critical issue for LVLMs, especially in fine-grained visual grounding. Existing decoding methods that use auxiliary synthetic visual signals typically rely on single-scale inputs, which may miss key visual details needed to resolve ambiguity and ensure semantic consistency.

Inspired by human multi-scale visual perception—coarse global scanning followed by fine-detail inspection—we propose a decoding framework that integrates **coarse-grained, fine-grained, and original visual cues**. By constructing a hierarchical multi-resolution context and embedding it into a generative feedback loop, the model achieves more robust and faithful visual grounding.

### Hierarchical Visual Decomposition
Given an input image \(I_0 \in \mathbb{R}^{H\times W\times 3}\) and textual prompt \(T\), we decompose the image into patches at two granularities:
- **Coarse-grained views** \(I_c = \{I_c^1,\dots,I_c^n\}\):
  Uniform non-overlapping low-resolution patches that retain global spatial structure while discarding fine details.
- **Fine-grained views** \(I_f = \{I_f^1,\dots,I_f^m\}\):
  High-resolution crops on salient or uncertain regions, derived from attention maps or region proposals, capturing local discriminative features.

The full multi-resolution visual input is:
\[
\mathcal{I} = I_0 \cup I_c \cup I_f. \tag{2}
\]

The initial response is generated using only the original image:
\[
R_0 = \text{LVLM}(I_0, T). \tag{3}
\]

We then generate responses under coarse- and fine-grained visual conditions separately:
\[
R_c = \text{LVLM}(I_c, T), \tag{4}
\]
\[
R_f = \text{LVLM}(I_f, T). \tag{5}
\]

For self-verification, a text-to-image generative model \(G\) (e.g., Stable Diffusion) synthesizes pseudo-images reflecting the model’s internal beliefs:
\[
v_c = G(R_c),\quad v_f = G(R_f). \tag{6}
\]

---

## 3.2 Self-Correcting Decoding with Generative Feedback
Previous multi-granular responses are static and cannot dynamically resolve semantic inconsistencies across views. To address this, we propose a **token-level self-correcting decoding strategy** that fuses predictive signals across granularities at each generation step. This allows adaptive weighting of global structure, local details, and raw image evidence for more faithful outputs.

Let \(v, v_c, v_f\) denote visual embeddings from the original image, synthesized coarse view, and synthesized fine view respectively. For a given prompt \(x\) and history \(y_{<t}\), the model outputs three conditional distributions:
\[
\begin{aligned}
p_\theta(y_t \mid v, x, y_{<t}) &= \text{Softmax}\big[f_\theta(y_t \mid v, x, y_{<t})\big], \\
p_\theta(y_t \mid v_c, x, y_{<t}) &= \text{Softmax}\big[f_\theta(y_t \mid v_c, x, y_{<t})\big], \\
p_\theta(y_t \mid v_f, x, y_{<t}) &= \text{Softmax}\big[f_\theta(y_t \mid v_f, x, y_{<t})\big].
\end{aligned} \tag{7}
\]

### Wasserstein Barycenter Fusion
To unify these distributions while preserving semantic geometry, we fuse them using the **Wasserstein barycenter**, which finds a central distribution that minimizes overall transport cost to the source distributions.

Denote the three distributions at step \(t\) as:
\[
P_t^{(v)} = p_\theta(y_t \mid v, x, y_{<t}),\quad
P_t^{(c)} = p_\theta(y_t \mid v_c, x, y_{<t}),\quad
P_t^{(f)} = p_\theta(y_t \mid v_f, x, y_{<t}).
\]

The fused distribution is obtained by solving:
\[
P_t^{(\text{fused})} = \arg\min_{P\in\Delta^{|\mathcal{V}|}} \Big( W(P,P_t^{(v)}) + W(P,P_t^{(c)}) + W(P,P_t^{(f)}) \Big), \tag{8}
\]
where \(W(P,Q)\) is the Wasserstein distance between distributions \(P\) and \(Q\), and \(\Delta^{|\mathcal{V}|}\) is the probability simplex.

This barycenter distribution is then used for token selection. It adaptively balances global context from \(v_c\), fine-grained alignment from \(v_f\), and grounding from the original image \(v\), enabling consistent and robust multi-granular visual reasoning.
