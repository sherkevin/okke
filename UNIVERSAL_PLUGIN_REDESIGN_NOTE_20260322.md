# Universal Plugin Redesign Note

## Goal

Define a hallucination-reduction pipeline that is:

- decode-time
- train once, run anywhere
- hot-swappable across heterogeneous MLLMs
- supported by a single learned lightweight module rather than per-model retraining

## 1. Theory Boundary: Why the Current `Phi_calib` Cannot Be Universal

The current `Phi_calib` path is fundamentally **model-coupled**, not universal.

### 1.1 Coupling Source A: Output-space entanglement

`Phi_calib` is trained to map visual states into the target model's lexical space:

$$
\Phi_{calib}: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_t}.
$$

This output space is not a universal semantic space. It is defined by the active MLLM's:

- hidden size
- `lm_head.weight` geometry
- tokenizer-induced vocabulary partition
- training distribution and representation manifold

If two MLLMs have different `lm_head` rows or different tokenizers, the same projected vector does not correspond to the same lexical semantics. Therefore a single learned checkpoint cannot be honestly reused across arbitrary models in this formulation.

### 1.2 Coupling Source B: Visual-state provenance mismatch

The current path consumes model-internal visual hidden states such as cached visual-token positions from the language-model trunk. These states are not standardized across models:

- the capture layer differs
- the multimodal fusion mechanism differs
- the hidden semantics differ
- some families expose image tokens, others expose query tokens or mixed fusion states

So even before lexical projection, the input representation is already architecture-specific.

### 1.3 Coupling Source C: Vocab-anchored protection logic

Current VASM operates over model token IDs and continuation rules. That means the protection layer itself depends on:

- tokenizer family
- subword continuation conventions
- vocabulary inventory

This is acceptable for an internal baseline, but it breaks the strict claim that the learned module plus control logic are model-agnostic.

## 2. Universality Requirement

A learned module can be strictly universal only if both its **observation space** and **action space** are model-agnostic.

### 2.1 Required observation space

The learned module must read only from standardized inputs such as:

- pixels
- frozen public image embeddings
- frozen public text embeddings
- decoded text strings
- optional external region proposals

It must not depend on:

- model-native hidden states
- model-native `lm_head` rows
- family-specific adapter internals

### 2.2 Required action space

The learned module must emit signals in a standardized control space such as:

- candidate-span support
- contradiction
- abstention
- confidence or uncertainty

It must not emit model-specific vocab logits directly.

## 3. New Core Design

We therefore redefine the main method around a universal sidecar plugin:

### 3.1 Universal learned core: `Psi_univ`

`Psi_univ` is the only learned module. It operates in a frozen external semantic space and produces:

- `support(span | image, prefix)`
- `contradiction(span | image, prefix)`
- `abstain(span | image, prefix)`

### 3.2 Parameter-free model interface

The active MLLM is touched only through a non-learned interface:

1. read current Top-`M` next-token candidates
2. decode them into candidate strings or short spans
3. score those spans with `Psi_univ`
4. map span scores back to current-token logits using deterministic prefix attribution

This interface can be tokenizer-dependent, but it must be parameter-free.

## 4. Universal Observation and Action Spaces

### 4.1 Observation space

Define:

$$
\mathcal{O}_{univ} = \{E_v(I), E_t(s), E_t(p), R(I)\},
$$

where:

- `I` is the image
- `s` is a candidate string/span
- `p` is the prompt plus decoded prefix
- `E_v` is a frozen public vision encoder
- `E_t` is a frozen public text encoder
- `R(I)` is optional region-level evidence from a frozen detector/segmenter

### 4.2 Action space

Define:

$$
\mathcal{A}_{univ} = \{a_{support}, a_{contradict}, a_{abstain}\},
$$

with one action tuple per candidate span. These actions are converted into logits bias by deterministic rules, not by learned model-specific adapters.

## 5. Inference Pipeline

1. Run the base MLLM normally and read Top-`M` next-token candidates.
2. Expand token prefixes into short candidate spans.
3. Apply a string-side structural gate to exclude function words, punctuation, and active continuation fragments.
4. Encode image, prefix, and candidate spans with frozen external encoders.
5. Use `Psi_univ` to estimate support, contradiction, and abstention.
6. Convert those scores into bounded next-token logit biases for the currently available prefixes.
7. Leave all other model internals untouched.

## 6. What Changes in the Paper

The paper should stop implying that universality can emerge from model-native lexical alignment. The correct stronger claim is:

> Strict universality is achievable only when the learned plugin is moved out of model-native hidden/logit geometry and into a shared external semantic evidence space, with a parameter-free tokenizer bridge handling runtime attachment.

This lets the paper make a much cleaner comparison:

- `TLRA_internal`: model-coupled internal path with `Phi_calib`
- `TLRA_univ`: strict universal sidecar path with `Psi_univ`

## 7. New Failure Boundaries

The universal route introduces different, more honest boundaries:

- Top-`M` ceiling remains
- candidate-span expansion may miss the intended entity
- prefix ambiguity may split evidence across several tokenizations
- frozen public encoders may miss fine OCR or domain-specific semantics
- abstention calibration may be critical when image evidence is weak

These are acceptable because they are architecture-independent and therefore scientifically compatible with a universal claim.

## 8. Final Design Judgment

The correct route is not to force `Phi_calib` into a false universality story. The correct route is to:

1. keep `Phi_calib` as a strong internal, model-coupled baseline
2. make `Psi_univ` the main learned universal plugin
3. make the tokenizer bridge explicit, deterministic, and parameter-free
4. move the paper's central novelty from "internal alignment" to "universal sidecar evidence routing"
