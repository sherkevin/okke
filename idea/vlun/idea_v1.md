# 3 Method
## 3.1 Semantic-equivalent Perturbation
### Visual Prompts
We perturb the original input image multiple times using 2D Gaussian blurring with adjustable radii to control intensity from clear to heavily blurred:
\[
I_i = \phi_{\text{vis}}(I, r_i),
\tag{1}
\]
where \(I\) is the original image, \(r_i\) denotes the Gaussian blur radius of the \(i\)-th perturbation (\(r_i < r_j\) for \(i < j\)), \(\phi_{\text{vis}}\) is the blur operation, and \(I_i\) is the perturbed visual prompt. The perturbation runs from \(i=1\) to \(N\) (total perturbation times).

Gaussian blur is a semantic-equivalent transformation that preserves full image content, structure, object relationships, spatial layout, and visual attributes (color, shape) without adding or removing objects.

### Textual Prompts
For textual prompts, we perform semantic-preserving paraphrasing via a pre-trained LLM, adjusting wording, grammar, and narrative style without changing core meaning. We vary the LLM temperature to control alteration intensity:
\[
T_i = \phi_{\text{text}}(T, \tau_i),
\tag{2}
\]
where \(T\) is the original question, \(\tau_i\) is the temperature for the \(i\)-th paraphrase (\(\tau_i < \tau_j\) for \(i < j\)), \(\phi_{\text{text}}\) denotes the LLM, and \(T_i\) is the perturbed textual prompt.

### Combination of Perturbed Prompts
We align visual and textual perturbations by intensity level: blur radius for vision and LLM temperature for text. The final paired perturbed inputs are:
\[
\{\langle I_i, T_i\rangle \mid i = 1,2,\dots,N\}.
\]

### Discussion
Semantic-equivalent perturbation ensures response fluctuations reflect genuine LVLM uncertainty, rather than semantic changes in prompts. Image blurring is inspired by human visual perception (e.g., nearsighted vision): stable outputs under varying blur indicate low uncertainty, while inconsistent responses imply high uncertainty.

## 3.2 Uncertainty Estimation
We quantify LVLM uncertainty using the semantic entropy of answers under perturbed prompts. A pre-trained LLM evaluates semantic entailment between generated answers and groups them into semantic clusters \(\{c_i\}_{i=1}^{N_c}\) (\(N_c\le N\)).

Uncertainty is computed as the entropy of the cluster distribution:
\[
U_{\text{LVLM}} = -\sum_{i=1}^{N_C} p(c_i)\log p(c_i),
\tag{3}
\]
where \(p(c_i)\) is the probability of the \(i\)-th semantic cluster.

### Discussion
Direct confidence scoring by LVLMs suffers from over-confidence. Our perturbation-based estimation avoids this issue. Compared with vanilla semantic entropy, VL-Uncertainty uses controlled multi-modal semantic-equivalent perturbations with graded intensity, enabling finer-grained uncertainty measurement for vision-language tasks.

## 3.3 LVLM Hallucination Detection
The continuous uncertainty score indicates the severity of hallucination. We set a threshold to classify outputs: answers with uncertainty above the threshold are labeled as hallucinatory, others as non-hallucinatory.

Detection accuracy is calculated as:
\[
\frac{\text{FN-N} + \text{NT-P}}{\text{NTotal}},
\]
where FN-N denotes true hallucinations correctly detected, NT-P denotes non-hallucinations correctly classified, and NTotal is the total number of samples.