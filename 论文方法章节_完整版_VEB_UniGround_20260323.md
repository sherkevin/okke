# 4. Methodology

## 4.1 Problem Setup and Design Goal

Let `m` denote a host MLLM, `I` an input image, and `Q` a text query. The host model produces an autoregressive answer `Y = (y_1, \dots, y_T)` through conditional next-token distributions

$$
P_t^{(m)}(\cdot) = p_m(\cdot \mid I, Q, y_{<t}).
$$

Our goal is not to retrain the host model, construct an internal textual counterfactual branch, or rewrite the host model's native attention dynamics. Instead, we seek an **external decode-time controller** that can be attached to heterogeneous MLLM families while preserving a strict universality boundary. Concretely, the learned module must not consume model-native hidden states, must not emit corrections in model-specific lexical geometry, and must not rely on learned host-specific attachment parameters.

The central failure mode we target is **prior-dominated answering**. In hallucination-prone settings, especially object-existence probing, the host model may produce an answer through a shortcut

$$
(Q \rightarrow Z \rightarrow Y),
$$

where `Z` denotes question-triggered prior or world-knowledge bias, instead of through a visually grounded path

$$
(I, Q \rightarrow E \rightarrow Y),
$$

where `E` is question-relevant visual evidence. The purpose of our method is therefore to impose an **external visual evidence bottleneck** over the final decision, so that the host prior cannot dominate the answer without being checked against object-conditioned evidence. We do not claim that this produces a perfect internal separation between prior and evidence inside the host network; rather, it constrains the externally visible answer decision through an independent evidence path.

This design yields two requirements. First, the learned component must remain **universal**: it should operate only in a standardized external semantic space built from frozen public encoders and deterministic interface rules. Second, the controller must remain **task-aware at the interface level**: each downstream task should be translated into a universal evidence query without retraining the learned scorer for each host or benchmark.

## 4.2 Universality Boundary: Universal Scorer, Task Adapter, Decision Layer

We decompose the method into three layers:

1. **Universal evidence scorer `\Psi_{univ}`.**
   This is the only learned component. It operates on frozen external image, region, and text embeddings and predicts generic evidence signals such as support, contradiction, and abstention.

2. **Task adapter `A_{task}`.**
   This is deterministic and task-specific. It maps a downstream task instance into a universal evidence query. For example, in object-existence probing it extracts the queried object and constructs the competing hypotheses "present" and "absent".

3. **Decision layer `D_{task}`.**
   This is also deterministic and lightweight. It converts generic evidence scores into task-specific decode-time control, such as binary yes/no steering.

This decomposition is essential. The learned module must remain universal; benchmark-specific behavior must not be absorbed into the learned scorer itself. In our formulation, `\Psi_{univ}` is not a `POPE`-specific classifier, not a host-coupled answer head, and not an internal prior extractor. It is a reusable external evidence interpreter. Downstream tasks differ only in how they **query** this scorer and how they **consume** its outputs.

## 4.3 Universal Observation and Action Spaces

To preserve portability across hosts, the method is defined over model-agnostic observation and action spaces.

### Universal observation space

For image `I`, task-conditioned query text `u`, hypothesis text `h`, and a sparse region bank `\mathcal{R}(I)=\{r_i\}_{i=1}^N`, we use frozen public encoders `E_v` and `E_t` to obtain

$$
z_I = E_v(I), \qquad z_u = E_t(u), \qquad z_h = E_t(h), \qquad Z_R = \{E_v(r_i)\}_{i=1}^N.
$$

The observation passed to the universal scorer is thus

$$
\mathcal{O}_{univ}(I,u,h) = \{z_I, z_u, z_h, Z_R\}.
$$

All host-dependent quantities, such as hidden states, attention maps, and `lm_head` geometry, are excluded from the learned path.

### Universal action space

Instead of emitting host-specific vocabulary corrections directly, the learned scorer predicts three generic evidence dimensions:

$$
\Psi_{univ}(\mathcal{O}_{univ}) = (e_{sup}, e_{con}, e_{abs}),
$$

where:

- `e_{sup}` measures visual support for the queried hypothesis,
- `e_{con}` measures contradiction from visual evidence,
- `e_{abs}` measures abstention or uncertainty.

These three outputs define a universal action space because they are semantically meaningful across tasks and hosts. Different tasks consume them differently, but the learned semantics themselves remain unchanged.

## 4.4 Task Adapter: From Query to Object-Centered Hypotheses

The key methodological change relative to generic token-local control is that we do not treat the unfinished text prefix as the primary object of grounding. Instead, we first identify the **semantic object of verification**.

Given a question `Q`, the task adapter extracts a queried object

$$
o = h(Q),
$$

where `h(\cdot)` is a deterministic parser or rule-based extractor. The adapter then constructs a pair of competing hypotheses:

$$
H_{yes}(o), \qquad H_{no}(o).
$$

For object-existence probing, a concrete instantiation is:

$$
H_{yes}(o) = \text{"a photo containing } o\text{"}, \qquad
H_{no}(o) = \text{"a photo without } o\text{"}.
$$

This translation step is critical because it separates **task semantics** from **scorer semantics**. The downstream task provides the object and the hypothesis format, while the universal scorer remains agnostic to whether the task is `POPE`, caption grounding, constrained VQA, or another object-centered benchmark. This separation also prevents the method from collapsing into a benchmark-specific verifier head disguised as a universal module.

## 4.5 Object-Conditioned Evidence Bottleneck

After constructing the object-centered query, the controller retrieves sparse visual evidence that is specifically relevant to that object. Let

$$
E = \mathrm{Retrieve}(I, o)
$$

denote the retrieved evidence set. In practice, `E` may be instantiated through frozen detector proposals, region crops, or a fixed region bank encoded by a frozen public vision encoder. The retrieval is **object-conditioned**: the queried object acts as the retrieval key, and only the evidence awakened by that object is allowed to influence the final decision.

This step implements the evidence bottleneck. Rather than allowing the host model to answer from global language prior or unrelated image content, the controller restricts itself to evidence explicitly activated by the queried object. The method therefore reasons through

$$
(I, Q) \rightarrow o \rightarrow E \rightarrow \{H_{yes}(o), H_{no}(o)\} \rightarrow Y,
$$

instead of directly through

$$
Q \rightarrow Z \rightarrow Y.
$$

The essential intuition is simple: the object query should first wake up the relevant image evidence, and only then should the system decide whether the answer should move toward "yes" or "no".

## 4.6 Universal Evidence Scoring

Given retrieved evidence `E`, the task-conditioned query representation `u`, and a hypothesis `h`, the universal scorer produces

$$
\Psi_{univ}(E,u,h) = \big(e_{sup}(h), e_{con}(h), e_{abs}(h)\big).
$$

For binary object verification, we evaluate both hypotheses:

$$
\big(e_{sup}^+, e_{con}^+, e_{abs}^+\big)
= \Psi_{univ}(E, u, H_{yes}(o)),
$$

$$
\big(e_{sup}^-, e_{con}^-, e_{abs}^-\big)
= \Psi_{univ}(E, u, H_{no}(o)).
$$

We then summarize each hypothesis into an evidence score

$$
s(h) = (1 - e_{abs}(h))\big(e_{sup}(h) - \lambda_{con} e_{con}(h)\big),
$$

where `\lambda_{con} > 0` balances contradiction against support. The resulting binary evidence gap is

$$
g_{evid} = s\!\big(H_{yes}(o)\big) - s\!\big(H_{no}(o)\big).
$$

This definition preserves the universality boundary. The scorer does not know anything about host logits, answer tokens, or benchmark labels. It only evaluates whether the visual evidence supports or contradicts a hypothesis.

## 4.7 Answer-Step Decision Control

The evidence scorer alone does not produce the final answer. Its role is to constrain the host model's answer at the decision step.

For binary-answer tasks, let `\ell_{yes}` and `\ell_{no}` denote the host logits of the answer labels at the answer step. We define the host preference gap as

$$
g_{host} = \ell_{yes} - \ell_{no}.
$$

The decision layer then composes host preference and external evidence:

$$
g_{final} = g_{host} + \Delta(g_{evid}, c_{evid}),
$$

where `c_{evid}` is an evidence-confidence term derived from support and abstention, and `\Delta(\cdot)` is a bounded control function. Importantly, the decision layer is defined over the designated answer labels themselves, rather than over an arbitrary semantic frontier that happens to expose binary tokens. The final binary choice is

$$
\hat y =
\begin{cases}
\text{yes}, & g_{final} > 0,\\
\text{no}, & g_{final} \le 0.
\end{cases}
$$

Two properties are important.

First, intervention occurs **only at the answer decision stage**. We do not define the core method as a generic modification over arbitrary frontier tokens, nor do we require the relevant answer labels to be discovered through an incidental top-`K` semantic frontier. The control target is the binary answer interval itself.

Second, the decision rule is **asymmetric** when appropriate. In hallucination-prone settings where recall is the main failure mode, the controller may be more willing to repair unsupported host "no" answers than to suppress already well-supported host "yes" answers. This asymmetry is a property of the decision layer, not of the universal scorer.

## 4.8 Risk-Aware Activation and Support-Aware Backoff

The controller should not intervene at every decode step, and it should not enforce hard flips when evidence is weak. We therefore use a two-part safety design.

### Risk-aware activation

The controller is eligible to act only when the model is at the answer step of a verification-relevant query and the decision is genuinely at risk. This differs from generic entropy-only gating: the relevant question is not merely whether the host is uncertain, but whether a visually relevant answer decision is being made under possible prior conflict.

### Support-aware backoff

Even when the controller is active, intervention is reduced or suppressed if evidence is weak. Let `c_{evid}` denote evidence confidence. If

$$
c_{evid} < \tau_{conf}
$$

or the abstention signal is too high, the controller backs off and leaves the host logits unchanged. Thus abstention is not an auxiliary diagnostic; it is part of the decision rule itself.

This mechanism prevents the universal scorer from over-correcting under ambiguous evidence and preserves the host model's native behavior when the external evidence cannot justify intervention. In this sense, the main safety principle is not an increasingly complex penalty design, but a conservative rule: unsupported evidence should not override the host.

## 4.9 Parameter-Free Attachment and Universality

The universality claim depends on a strict separation between learned semantic judgment and deterministic host attachment.

The learned path includes only:

- frozen public image/text encoders,
- retrieved visual evidence,
- task-conditioned hypothesis text,
- the shared scorer `\Psi_{univ}`.

The host-dependent path is limited to deterministic interface logic, such as:

- prompt formatting,
- answer-label identification,
- optional tokenizer-aware label mapping,
- non-learned retrieval and bridge rules.

No learned per-host adapter is introduced anywhere in the path from observation to control. Therefore, the same checkpoint of `\Psi_{univ}` can in principle be reused across heterogeneous MLLM families, while task-specific behavior is expressed only through the deterministic adapter and decision layer. This is the precise sense in which the method is universal: the learned content lives in external evidence scoring, not in host-specific attachment or benchmark-specific answer classification.

## 4.10 Scope of the Claim

Our claim is intentionally precise. We do **not** claim that the method fully removes prior dependence inside the host model, nor that it establishes a perfect causal isolation between evidence and answer generation. We also do not claim `zero OOD`, exact internal prior isolation, mathematically exact recovery of hidden causal factors, or that a universal scorer can replace task-specific supervision for every downstream benchmark.

What we do claim is the following:

> A single learned external scorer, defined in a shared frozen semantic space and queried through deterministic task adapters, can impose an object-conditioned visual evidence bottleneck over the host model's final decision, thereby reducing prior-dominated hallucination without model-specific retraining.

Under this view, the scientific contribution of the method is not merely a better benchmark score. It is a clean decomposition of multimodal hallucination control into universal evidence scoring, deterministic task translation, and lightweight decode-time decision correction.

## 4.11 Instantiation for POPE

Although the framework is task-general, this paper's primary empirical instantiation is `POPE`, where the output space is binary and the hallucination signal is tightly coupled to object-existence bias.

For a `POPE` question, the adapter:

1. extracts the queried object `o`,
2. constructs `H_{yes}(o)` and `H_{no}(o)`,
3. retrieves object-conditioned evidence `E`,
4. evaluates the two hypotheses with `\Psi_{univ}`,
5. uses the resulting evidence gap to steer the final yes/no answer.

This instantiation is particularly suitable for exposing the difference between a true evidence bottleneck and a question-prior shortcut, since `POPE random`, `POPE popular`, and `POPE adversarial` probe increasing levels of prior conflict. A method that truly follows the proposed principle should therefore improve not only on generic random probes, but also under popularity bias and adversarial prior pressure. Conversely, a method that only works when the correct binary answer is already trivially exposed by the host distribution should not be regarded as a faithful realization of the present framework.
