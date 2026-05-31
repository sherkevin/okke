# Nana Banana Figure Prompts

## Use Policy

- `Figure 4` and `Figure 5` should remain code-drawn because they present evidence tied to real benchmark cases and real design diagnostics.
- `nana banana` is best reserved for conceptual figures whose purpose is explanation or visual framing rather than direct empirical evidence.

## Recommended External Prompt

### Figure 1 Teaser Refresh

Use this only if you want to replace the current teaser with a more polished conceptual visual while keeping the quantitative and qualitative evidence figures untouched.

**Prompt**

Create a publication-quality ACM Multimedia paper teaser figure for a method named `CHORD` that mitigates hallucination in multimodal large language models. The figure should be a clean scientific infographic, not a poster, and must fit a single-column research paper figure. Use a restrained academic palette: slate blue, muted green, warm gray, and soft red accents. No dark background.

Layout:
- Two-panel horizontal comparison.
- Left panel title: `Regular decoding`.
- Right panel title: `CHORD decoding`.

Left panel content:
- Show an image-question pair at the far left.
- Show a decoding trajectory that first appears reasonable, then drifts toward a text-dominated answer.
- Visually communicate `premature commitment`, `text prior drift`, and `weak visual grounding`.
- Use 1 concise equation-free annotation near the drift, such as `locally plausible, globally risky`.

Right panel content:
- Show the same image-question pair.
- Show a gated admission pipeline with three compact modules: `Past`, `Current`, `Future`.
- `Past` means rollback-based protection.
- `Current` means object-resonant grounding.
- `Future` means short-horizon branch arbitration.
- The selected continuation should visibly remain aligned with the image evidence.
- Use one concise annotation like `support now + stability next`.

Style constraints:
- No fake paragraphs.
- No tiny illegible text.
- No glossy corporate UI style.
- No decorative 3D effects.
- Use vector-like shapes, elegant arrows, minimal shadows, strong spacing, and crisp labels.
- The final figure must look like a serious conference paper figure, not marketing art.

Negative prompt:
- avoid crowded layout
- avoid handwritten style
- avoid heavy gradients
- avoid futuristic neon colors
- avoid fake equations
- avoid lorem ipsum
- avoid extra panels
- avoid cartoon aesthetics
