# 3 Method
## 3.1 Decoding of Vision-Language Models
We consider a large vision-language model (LVLM) parameterized by θ. The model takes a textual query \(x\) and a visual input \(v\) as input, where \(v\) provides contextual visual information to assist the model in generating a relevant response \(y\) to the textual query. The response \(y\) is sampled auto-regressively from the probability distribution conditioned on the query \(x\) and the visual context \(v\). Mathematically, this can be formulated as:
\[
\begin{align*}
y_t &\sim p_\theta(y_t \mid v, x, y_{<t}), \\
&\propto \exp\big(\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})\big), \tag{1}
\end{align*}
\]
where \(y_t\) denotes the token at time step \(t\), and \(y_{<t}\) represents the sequence of generated tokens up to time step \(t-1\).

In the decoding phase of LVLMs, object hallucinations often emerge when probabilities are erroneously allocated to tokens that do not align with the presented visual input \(v\). Previous studies have identified two primary causes of this problem:
1. Statistical biases inherent in training data (e.g., prevalent but superficial object correlations) [1,2,19];
2. Over-reliance on language priors embedded within the powerful large language models (LLMs) used as decoders [22,38,69,75].

Our approach to mitigate object hallucinations first amplifies these undesirable behaviors with vague inputs and subsequently contrasts with them in the decoding process.

## 3.2 Visual Uncertainty Amplifies Hallucinations
The fidelity of visual input is pivotal for LVLMs to accurately encode visual features and generate outputs faithfully. Yet, the introduction of uncertainty in visual inputs can tilt the equilibrium. This section provides a comprehensive analysis to validate the assumption that increased visual uncertainty amplifies language priors and statistical biases in LVLMs, thus exacerbating object hallucination.

### 3.2.1 Introduction of Visual Uncertainty
In this paper, we adopt the most elementary method—applying a Gaussian noise mask to the original image—to introduce visual uncertainty. Although straightforward, this method provides an initial benchmark to estimate the baseline effects of visual uncertainty on model outputs. Following the forward diffusion process in image generation [24], the distorted image is modeled as:
\[
\begin{align*}
q(v_t \mid v_{t-1}) &= \mathcal{N}\big(v_t; \sqrt{1-\gamma}\,v_{t-1},\ \gamma I\big), \\
q(v_T \mid v_0) &= \prod_{t=1}^T q(v_t \mid v_{t-1}), \tag{2}
\end{align*}
\]
where \(v_0\) denotes the original visual input (i.e., the original image) and \(I\) refers to an identity matrix. We incrementally add a small amount of Gaussian noise for \(T\) steps, producing a sequence of distorted images \(v_1,\dots,v_T\). The original image \(v_0\) gradually loses distinguishable features as step \(t\) increases, where the noise amount in each step is controlled by \(\gamma\). Eventually, as \(T\rightarrow\infty\), visual uncertainty reaches its maximum and \(v_T\) becomes indistinguishable from Gaussian noise.

### 3.2.2 Visual Uncertainty Amplifies Language Priors
Figure 2 illustrates that visual uncertainty can compel LVLMs to overlook visual evidence and overly exploit language priors for decision-making. Given an image featuring a black banana among other colorful fruits, LVLMs favor more conventional banana colors such as “yellow” and “green” as visual uncertainty increases. The ground-truth color “black” diminishes in log-probability \(\log p(y\mid x,v')\) as distortion escalates, making LVLMs over-reliant on language priors from LLM pre-training that typically associate bananas with yellow or green.

This tendency is not entirely unexpected, as LLMs are designed to predict next-word probabilities based on vast textual corpora. When confronted with ambiguous visual stimuli, an LVLM may misinterpret these conventional text-based predictions as a “safety net”. While generally useful, these priors can introduce biases or assumptions inconsistent with actual visual content, especially when visual input lacks clarity.

### 3.2.3 Visual Uncertainty Amplifies Statistical Bias
Most vision-language pretraining datasets are predominantly constructed based on MSCOCO [40], which inherently suffers from unbalanced object distributions and biased object correlations. Previous works [38,77] show that LVLMs trained on such data may inherit these statistical biases and generate descriptions with hallucinated objects.

To further examine the hypothesis that visual uncertainty amplifies statistical biases from pretraining, we designed two targeted experiments to verify:
1. Whether LVLMs hallucinate frequent objects more with distorted visual inputs;
2. Whether LVLMs are more prone to hallucinate objects that frequently co-occur with ground-truth objects in the image under distorted visual inputs.

Figure 3 shows a clear tendency: LVLMs are more likely to hallucinate frequent and co-occurring objects, due to imbalanced object distributions and spurious object correlations inherited from training data.

## 3.3 Visual Contrastive Decoding
### 3.3.1 Contrasting the Predictions
Our observations reveal that visual uncertainty not only amplifies reliance on language priors but also makes LVLMs more susceptible to superficial object correlations in pretraining datasets, leading to more severe hallucinations. In light of this, we introduce **Visual Contrastive Decoding (VCD)**.

VCD counteracts statistical biases and language priors in LVLMs by contrasting model outputs generated from original and distorted visual inputs. It requires no additional training or external pretrained models, making it cost-effective and efficient.

Specifically, given a textual query \(x\) and a visual input \(v\), the model generates two distinct output distributions: one conditioned on the original \(v\), and the other on the distorted visual input \(v'\) (obtained by adding Gaussian noise to \(v\)). A new contrastive probability distribution is computed by exploiting the difference between the two distributions:
\[
p_{\text{VCD}}(y \mid v, v', x) = \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y \mid v, x)
-\alpha\operatorname{logit}_\theta(y \mid v', x)\big], \tag{3}
\]
where larger \(\alpha\) indicates stronger amplification of the difference between the two distributions (\(\alpha=0\) reduces to regular decoding). From the adjusted distribution \(p_{\text{VCD}}\), various sampling strategies can be applied, such as nucleus sampling [25] and beam search [15].

Essentially, VCD acts as a corrective mechanism that reduces hallucinations by contrasting against a hallucination-prone distribution. It can also be interpreted as a contrastive ensemble that differentiates logits of \(p_\theta(y\mid v,x)\) and \(p_\theta(y\mid v',x)\). This idea echoes contrastive objectives in image generation—for example, classifier-free diffusion models [23] estimate diffusion noise using \((1+\alpha)\epsilon_\theta(x,c)-\alpha\epsilon_\theta(x)\), where \(c\) is a control factor. In text generation, contrastive decoding has also been explored for more faithful generation [37,41,52,56].

### 3.3.2 Adaptive Plausibility Constraints
A potential challenge arises from Equation (3): VCD penalizes the entire output behavior influenced by distorted visual inputs, which is not universally appropriate. The distribution under distorted inputs may still preserve basic linguistic norms and common-sense reasoning. Indiscriminate penalization could wrongly suppress valid outputs and encourage implausible tokens.

To address this issue, we implement an adaptive plausibility constraint contingent on the model’s confidence under the original visual input, following Li et al. [37]:
\[
\begin{align*}
\mathcal{V}_{\text{head}}(y_{<t}) &= \big\{y_t\in\mathcal{V}:
p_\theta(y_t \mid v, x, y_{<t}) \geq \beta\max_{w}p_\theta(w \mid v, x, y_{<t})\big\}, \\
p_{\text{VCD}}(y_t \mid v, v', x) &= 0,\quad \text{if } y_t\notin\mathcal{V}_{\text{head}}(y_{<t}), \tag{4}
\end{align*}
\]
where \(\mathcal{V}\) is the model output vocabulary, and \(\beta\in[0,1]\) is a hyperparameter controlling truncation of the next-token distribution. Larger \(\beta\) leads to more aggressive truncation, retaining only high-probability tokens.

Combining visual contrastive decoding and adaptive plausibility constraints, the full formulation is:
\[
\begin{align*}
y_t &\sim \operatorname{softmax}\big[(1+\alpha)\operatorname{logit}_\theta(y_t \mid v, x, y_{<t})
-\alpha\operatorname{logit}_\theta(y_t \mid v', x, y_{<t})\big], \\
&\quad \text{subject to } y_t\in\mathcal{V}_{\text{head}}(y_{<t}). \tag{5}
\end{align*}
\]

Incorporating adaptive plausibility constraints refines the contrastive distribution and strengthens confidence in straightforward decisions. When the model is highly confident in outputs from original inputs, the candidate token pool is streamlined—often retaining only one high-probability token. This effectively mitigates potential adverse effects of VCD, preventing implausible token generation and preserving the integrity of generated content.
