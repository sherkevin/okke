# 3 Method
We first formulate the generation procedure of multimodal large language models (MLLMs) to facilitate understanding of our proposed **OPERA** method, then introduce the **Over-Trust Logit Penalty** and **Retrospection-Allocation Strategy** in detail.

---

## 3.1 Formulation of MLLMs Generation
The generation pipeline of MLLMs can be divided into three stages: input formulation, model forward pass, and decoding.

### Input Formulation
MLLM inputs consist of images and text. Visual tokens are extracted from raw images via a vision encoder, then projected into the LLM input space through a cross-modality mapping module.
Let visual tokens be denoted as:
\[
\boldsymbol{x}_v = \{x_0, x_1, \dots, x_{N-1}\}
\]
where \(N\) is the fixed length of visual tokens in most models.
The text input is tokenized as:
\[
\boldsymbol{x}_p = \{x_N, x_{N+1}, \dots, x_{N+M-1}\}
\]
Image and text tokens are concatenated into a full input sequence:
\[
\{x_i\}_{t=0}^{T-1},\quad T = N+M.
\]

### Model Forward
MLLMs are autoregressively trained with a causal attention mask. Each token predicts the next token based on preceding context:
\[
\boldsymbol{h} = \text{MLLM}(\boldsymbol{x}_i),\quad
\boldsymbol{h} = \{h_0, h_1, \dots, h_{T-1}\}
\tag{1}
\]
where \(\boldsymbol{h}\) denotes the hidden states of the last layer.

A vocabulary head \(H\) projects hidden states to logits for next-token prediction:
\[
p(x_t \mid x_{<t}) = \operatorname{SoftMax}\!\left[H(h_t)\right]_{x_t},\quad x_t \in \mathcal{X}
\tag{2}
\]
where \(x_{<t}\) denotes \(\{x_i\}_{i=0}^{t-1}\) and \(\mathcal{X}\) is the vocabulary set.

### Decoding
Common decoding strategies include greedy search, beam search, DoLa, etc. The decoded token is appended to the input sequence for the next step until the end token is generated.

Our OPERA is built upon **beam search** with beam size \(N_{\text{beam}}\). At each step, each candidate sequence expands \(N_{\text{beam}}\) tokens via top-\(N_{\text{beam}}\) probabilities, and the sequence with the highest beam score is finally selected.

---

## 3.2 Over-Trust Logit Penalty
Hallucinations are highly correlated with knowledge aggregation patterns in attention, which exhibit hysteresis: they only become observable after several tokens are generated, by which time hallucination may already have occurred.

To address this, we propose an **Over-Trust Logit Penalty**, an accumulated penalty term integrated into the beam score to suppress hallucination-prone candidates.

### Local Window Attention
Let \(\{\omega_{t-1,j}\}_{j=0}^{t-1}\) be the causal self-attention weights for next-token prediction, where
\[
\omega = \operatorname{SoftMax}\!\left(\frac{QK^\top}{\sqrt{D}}\right).
\]

We define a local attention window focusing only on generated tokens:
\[
W_{t-1}^k = \{w_i\}_{i=t-k}^{t-1},\quad
w_i = \{\omega_{i,j}\}_{j=t-k}^{i}
\tag{3}
\]
where \(k\) is the window size. The window satisfies \(t-k \ge N+M\) to exclude image and prompt tokens. We use the **maximum attention weight across heads** and renormalize.

### Scaled Window and Column-Wise Score
We scale the attention values and zero out the upper triangle:
\[
\widetilde{W}_{t-1}^k = \{w_i\}_{i=t-k}^{t-1},\quad
w_i = \{\sigma\omega_{i,j}\}_{j=t-k}^{t-1}
\tag{4}
\]
where \(\sigma\) is a scaling factor and \(\{\omega_{i,j}\}_{j=i+1}^{t-1}=0\).

We compute column-wise products and take the maximum value to measure knowledge aggregation:
\[
\phi(\omega_{<t}) = \prod_{i=c}^{t-1}\sigma\omega_{i,c},\quad
c = \arg\max_{t-k\le j\le t-1}\prod_{i=j}^{t-1}\sigma\omega_{i,j}
\tag{5}
\]

### Penalized Logit
To ensure efficiency and rationality, we restrict candidates to a set \(\mathcal{Y}\) of size \(N_{\text{can}} \cdot N_{\text{beam}}\). The penalized prediction becomes:
\[
p(x_t \mid x_{<t}) = \operatorname{Softmax}\!\left[H(h_t) - \alpha\phi(\omega_{\le t})\right]_{x_t},\quad
x_t \in \mathcal{Y}
\tag{6}
\]
where \(\alpha\) controls penalty strength.

---

## 3.3 Retrospection-Allocation Strategy
In extreme cases, all candidates may be penalized and hallucination remains unavoidable. This arises because early tokens over-rely on summary tokens, which the penalty cannot fully correct.

We therefore propose a **Retrospection-Allocation Strategy** that rolls back decoding to reselect tokens before hallucination propagates.

### Location Overlap Detection
From Eq. (5), we obtain the coordinate set of maximum column-wise scores for recent \(l\) tokens:
\[
\mathcal{C} = \left\{\,c \,\bigg|\,
c = \arg\max_{t-k\le j\le z}\prod_{i=j}^{z}\sigma\omega_{i,j},\;
z \in [t-l, t-1]\,\right\}
\tag{7}
\]
where \(l > r\) (we set \(l=k\) by default).

We compute the overlap count of the modal location:
\[
N_{\text{overlap}} = \sum_{c\in\mathcal{C}} \mathbb{1}_{c=s},\quad
s = \operatorname{Mode}(\mathcal{C})
\tag{8}
\]
where \(\mathbb{1}\) is the indicator function.

### Retrospection Execution
If \(N_{\text{overlap}} \ge r\), we identify \(s\) as the summary token and **roll back** the decoding process to \(\{x_0,\dots,x_s\}\). We then resample the next token from \(\mathcal{Y} \setminus \{x_{s+1}\}\).

To ensure stability:
- The rollback position \(s\) must be **non-decreasing**.
- A maximum rollback limit \(\beta\) is set; if exceeded, we roll back further to \(x_{s-1}\).