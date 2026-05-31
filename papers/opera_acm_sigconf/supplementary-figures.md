# CHORD Supplementary Figure Prompts

This file contains only the optional supplementary figures that could reasonably be AI-assisted with `nanabanana`.

Important:

- These figures are optional.
- All benchmark plots, ablations, and qualitative case grids should still be produced from real measurements or manually assembled from real benchmark cases.
- Keep the visual language aligned with the main paper: white background, thin lines, restrained colors, publication-style composition.
- Do not use decorative poster aesthetics, cartoon elements, or oversized panel numbering.

Use this negative prompt block at the end of every prompt:

`Do not generate fantasy art, painterly style, cinematic lighting, cartoon characters, anime styling, 3D glossy interface elements, handwritten text, illegible labels, cluttered infographic decoration, scientific nonsense equations, colorful poster aesthetics, or oversized panel numbers. Keep it clean, academic, publication-style, vector-like, and readable after downscaling.`

## Prompt S1

### Target

Optional `Figure S1`: layer aggregation intuition figure.

### Purpose

This figure is meant to visually explain why CHORD uses the `last-4 decoder blocks` for attention aggregation. It should read as a supplementary conceptual explainer, not as quantitative evidence.

### Prompt

Create a publication-style scientific supplementary figure for an ACM Multimedia paper on multimodal decoding.

The figure should compare three decoder-layer attention views over the same image-query pair:

1. `middle-4`
2. `last-1`
3. `last-4 (used by CHORD)`

Layout:

- Single-row or clean three-column layout.
- Each panel shows the same underlying image thumbnail.
- Over each image, place a translucent attention overlay.
- `middle-4` should look too diffuse and spread across several irrelevant regions.
- `last-1` should look sharp but brittle, with unstable focus or over-concentration.
- `last-4 (used by CHORD)` should look stable, query-relevant, and semantically specific.

Embedded labels should be short only:

- `middle-4`
- `last-1`
- `last-4 (used)`
- `diffuse`
- `brittle`
- `stable object-resonant support`

Visual style:

- White background.
- Clean academic vector-like figure.
- Thin separators.
- No heavy gradients.
- Muted red for diffuse or unstable attention.
- Muted orange for brittle final-layer focus.
- Muted green for stable last-4 support.

Add one concise side annotation or bottom note:

`Late-stage aggregation preserves semantic specificity while reducing single-step volatility.`

Do not include large paragraphs, giant numbers, or any decorative infographic elements.

## Prompt S2

### Target

Optional `Figure S2`: failure taxonomy schematic.

### Purpose

This figure summarizes the main failure categories of CHORD in a compact, reviewer-friendly way.

### Prompt

Create a publication-style scientific schematic for the supplementary material of an ACM Multimedia paper.

The figure should present a clean three-block failure taxonomy for a multimodal hallucination mitigation method named `CHORD`.

The three failure categories are:

1. `Missed anchors`
2. `Weak candidate set`
3. `Continuation ambiguity`

Layout:

- Horizontal three-panel layout or triangular layout with a small central title.
- Each block contains a simple abstract icon and a one-line explanation.
- Use abstract scientific motifs only, not realistic scenes.

Desired semantics:

- `Missed anchors`: the object proposer fails to capture the relevant visual entity, so current support becomes weak.
- `Weak candidate set`: the top-k candidates are all linguistically plausible but similarly ungrounded, leaving little room for reranking.
- `Continuation ambiguity`: early admission looks reasonable, but the downstream continuation remains semantically underspecified.

Embedded text should be short:

- `Missed anchors`
- `Weak candidate set`
- `Continuation ambiguity`
- `low current support`
- `poor branch diversity`
- `unstable continuation`

Visual style:

- White background.
- Clean publication-style vector figure.
- Muted red and gray for failure states.
- Small restrained green accents only if needed to indicate CHORD's decision filter.
- No character art, no glossy UI, no posters, no busy textures.

The figure should feel like a compact scientific taxonomy, not a marketing infographic.
