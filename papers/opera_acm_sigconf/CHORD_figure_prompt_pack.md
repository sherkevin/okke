# CHORD Figure Prompt Pack

This document is a production-facing prompt pack for generating the core CHORD paper figures with `nanabanana`. It is optimized for an ACM MM main-paper submission under an 8-page main-text budget.

The goal is not to create decorative AI art. The goal is to generate structured, publication-style scientific figures that can be lightly post-edited into camera-ready assets.

## Global Art Direction

### Scientific Intent

The core main-paper figures should tell one coherent story:

1. Hallucination is a trajectory-level failure, not only a local token error.
2. CHORD prevents collapse through a chronological admission bottleneck.
3. The method is understandable as a concrete system, not a vague concept.
4. The gain is meaningful while the extra inference cost remains controlled.

### Visual Language

- Background: pure white or near-white.
- Text: dark charcoal or black.
- Neutral structural elements: muted blue.
- Failure / hallucination / collapse: muted red.
- Success / grounded / approved: muted green.
- Future rollout / forecasting emphasis: restrained orange.
- Optional gray for nonessential scaffolding.

### Style Constraints

- Publication-style system figure, not poster art.
- Clean vector-like composition.
- Thin lines, precise arrows, restrained shadows.
- Minimal gradients, if any.
- No 3D rendering, no glossy UI, no cartoon icons.
- No busy textures.
- High contrast and high legibility after ACM two-column downscaling.

### Typography Guidance

- Use short embedded labels only.
- Prefer 1-4 word labels.
- Avoid long sentences inside the figure.
- Use consistent casing.
- Avoid tiny text blocks.

### Shared Embedded Label Style

Preferred labels across figures:

- `Premature commitment`
- `Structural temporal collapse`
- `Chronological admission bottleneck`
- `Past`
- `Present`
- `Future`
- `Rollback`
- `Anchor support`
- `Oracle rollout`
- `Shared KV`
- `B&B pruning`
- `Approved`
- `Rejected`
- `Grounded output`
- `Text-dominated drift`

### Chrono Easter Egg Guidance

The `CHORD` name can carry a subtle resonance-and-harmony metaphor, but it must be expressed as original abstract visual language rather than any recognizable external character or franchise reference.

Safe motifs:

- a three-branch timeline that reconverges into one selected path
- a small rewind arrow paired with a short forward probe
- faint token afterimages suggesting short-horizon timeline probing
- a compact chrono gate icon near `Past`, `Present`, `Future`
- a tiny pulse-ring or time-rift motif around the admission bottleneck

Unsafe motifs:

- recognizable game-character silhouettes
- distinctive weapons, hairstyles, costumes, or facial designs
- any explicit mention of `League of Legends`, `Ekko character`, `champion`, or similar IP cues
- any visual element that reads like fan art or franchise homage

The metaphor should read first as rigorous object resonance and temporal arbitration, and only secondarily as a subtle naming wink for readers who notice it.

### Layout And Numbering Warnings

- Do not use oversized panel numbers such as giant `1`, `2`, `3`.
- Do not place circular number badges, comic-style step markers, or presentation-slide numbering in the main artwork.
- If panel distinction is necessary, use spacing, divider lines, or very small corner labels only.
- Avoid anything that makes the figure look like a slide deck, poster board, or tutorial infographic.

### Nanabanana Negative Prompt Block

Use this block at the end of every prompt unless otherwise noted:

`Do not generate fantasy art, painterly style, cinematic lighting, cartoon characters, anime styling, 3D glossy interface elements, handwritten text, illegible labels, cluttered infographic decoration, stock-photo marketing style, scientific nonsense equations, colorful poster aesthetics, or oversized panel numbers such as giant 1/2/3 badges. Keep it clean, academic, publication-style, vector-like, and readable after downscaling.`

### Post-Generation Cleanup Expectation

Assume the first-pass image may require light manual cleanup in PowerPoint, Figma, Illustrator, or Inkscape:

- Replace any slightly malformed text.
- Align panel spacing.
- Normalize line weights.
- Replace example numbers or placeholders with final experimental values.

## Main-Paper Figure Policy

Use the following main-paper lineup for the ACM MM submission:

1. `Figure 1`: single-column teaser.
2. `Figure 2`: double-column method overview.
3. `Figure 3`: quantitative latency-performance figure based on real experimental numbers.
4. `Figure 4`: composite mechanism-plus-qualitative figure based on real experimental traces and real cases.

Additional rules:

- `Figure 1` and `Figure 2` are concept figures and should now be treated as visually locked unless a serious paper-level inconsistency appears.
- `Figure 3` and `Figure 4` should be finalized from real measured data, not purely imagined visual content.
- Any hyperparameter sensitivity plot, extra rollout-depth analysis, or supplementary ablation figure should move to the appendix / supplementary material rather than the 8-page main paper.
- Do not introduce a new `Figure 5` into the main paper unless a later revision removes another figure.
- References are outside the 8-page main-text budget; main-text figures should be planned accordingly.

## Figure 1

### Filename Suggestion

`fig1_teaser_problem_method.png`

### Scientific Purpose

Show that hallucination is a structural temporal collapse and position CHORD as a chronological admission bottleneck that preserves grounded generation with limited extra cost.

### Recommended Layout

Wide horizontal three-panel figure, roughly:

- Left panel: 34% width
- Middle panel: 32% width
- Right panel: 34% width

Strong left-to-right causal flow. The middle panel should act as the hinge of the full figure.

### Panel Specification

#### Left Panel: Failure Mechanism

Show a compact visual story of a baseline MLLM trajectory:

- An input image thumbnail on top-left.
- A short prompt under or beside it.
- A token sequence strip or mini decoding chain moving left to right.
- One token marked as `Premature commitment`.
- After that token, show the path bending away from the visual side toward a text-only continuation zone.
- A tiny inset line or attention bar indicating visual grounding dropping while text dominance rises.

The feeling should be: a plausible token entered the prefix too early and then hijacked the rest of the response.

Allowed embedded labels:

- `Baseline decoding`
- `Premature commitment`
- `Visual grounding`
- `Text-dominated drift`
- `Structural temporal collapse`

#### Middle Panel: CHORD Gate

This is the core conceptual panel.

Show a gate or checkpoint metaphor, but in a technical systems-paper style, not a fantasy gate:

- A candidate token stream enters.
- Three compact modules or checkpoints labeled `Past`, `Present`, and `Future`.
- Each module should have a subtle icon:
  - `Past`: self-attention matrix / rollback warning
  - `Present`: image with anchor boxes
  - `Future`: short rollout branches over 3 steps
- Risky branch gets rejected in red.
- Safe branch gets approved in green.

Allowed embedded labels:

- `Chronological admission bottleneck`
- `Past`
- `Present`
- `Future`
- `Rejected`
- `Approved`

#### Right Panel: Grounded Outcome

Show the positive result:

- Same or similar image thumbnail context.
- Short grounded output text block.
- Small visual-attention cue remaining linked to image anchors.
- Optional tiny inset scatter showing `Greedy`, `OPERA`, `CHORD` with CHORD near a favorable corner. Keep this tiny and simple.

Allowed embedded labels:

- `Grounded output`
- `Sustained visual support`
- `Limited extra cost`

### Color Semantics

- Red: risky token, collapse path, hallucinated continuation.
- Green: approved safe path, grounded output.
- Blue: system structure, neutral modules, arrows.
- Orange: future rollout emphasis.
- Gray: de-emphasized baseline scaffolding.

### Composition Rules

- The figure should remain readable without any caption.
- The middle panel must visually dominate, but only slightly.
- The left and right panels should look like before/after states around the gate.
- Do not use realistic human faces unless necessary; a generic scene thumbnail is safer.
- Introduce a subtle chrono motif through abstract timeline branching, rewind arrows, or faint afterimages, but never through recognizable character references.
- Do not use giant panel numerals, circular step badges, or large decorative `1`, `2`, `3` markers.

### Final Prompt For Figure 1

Use this version when the model keeps hallucinating sentences, leaking instructions, or producing slide-like numbering. This prompt is intentionally strict and should be copied as-is.

```text
Generate a publication-ready teaser figure for an ACM Multimedia paper. The content should follow a left-versus-right comparison exactly like a polished conference teaser: left side shows Regular Decoding failure, right side shows CHORD Decoding success. Prioritize composition, scientific logic, visual hierarchy, and top-tier paper aesthetics over exact text rendering. The figure must look like a polished scientific paper figure, not a slide, not a poster, not a tutorial diagram, and not concept art.

Use a pure white background, vector-like geometry, thin but confident arrows, consistent module borders, restrained academic colors, generous whitespace, balanced alignment, and crisp visual hierarchy. The goal is not a simple sketch. The goal is a refined top-conference teaser figure with strong design discipline, subtle depth, clean spacing, and the feeling of a final camera-ready systems paper asset.

Important instruction: if exact text rendering is difficult, prefer short clean labels or simple label placeholders rather than long malformed text. Do not generate explanatory sentences, helper phrases, or copied prompt language. The figure should remain correct and understandable even if text is later manually replaced.

Required high-level layout:
- Use a wide horizontal two-part comparison layout.
- Left half: `Regular Decoding`
- Right half: `CHORD Decoding`
- Separate the two halves with one elegant thin vertical divider and generous whitespace.
- Do not use panel letters, panel numbers, badges, bullets, or slide-style section markers.

Target visual story:
1. Left half: regular decoding makes a premature semantic commitment and collapses into a bad continuation driven by language inertia.
2. Right half: CHORD screens candidate tokens through three equal checks, Past, Present, and Future, then selects a grounded continuation.

Left half specification:
- Show a small image thumbnail of a park bench scene near the far left.
- Show a short question or prompt cue near the image only if text renders cleanly; otherwise omit it.
- Show a compact token trajectory moving horizontally.
- Highlight one token as the risky commitment point.
- From that point, show two bad drift directions with red or dark red trajectories to convey failure modes.
- One drift should suggest a hallucinated or text-prior-driven continuation.
- Keep the left half visually simple, with only a few objects and no dense text.
- If text is rendered, keep it very short, such as `Regular Decoding`, `Premature commitment`, and one or two short drift labels.
- Do not include long phrases, pseudo-sentences, or helper text.

Right half specification:
- Title it `CHORD Decoding` if text quality permits; otherwise a short clean title placeholder is acceptable.
- Show several candidate tokens entering a compact but elegant gate-like bottleneck.
- Past, Present, and Future must appear as three equally important sibling checks with the same visual hierarchy. None of the three may appear downstream of another. They must be arranged as one balanced grouped unit, either three stacked modules or three side-by-side modules of equal size.
- Past should use a tiny grayscale attention-map cue plus a rollback-like arrow.
- Present should use a tiny image crop with anchor boxes.
- Future should use a small rollout-tree or branching probe motif.
- The three checks should visually feel like a coordinated triad.
- One branch should be visually rejected in muted red.
- One branch should be visually accepted in muted green.
- After the bottleneck, show a grounded output cue, such as a concise grounded caption or visually grounded object phrase leading to a correct image-linked outcome.
- Include a subtle abstract chrono motif using faint afterimages, reconverging traces, or restrained time-like echoes only. It must feel original and abstract, not like any recognizable game or franchise reference.

Style requirements for top-tier conference quality:
- Make the right half slightly more visually prominent than the left half, but still balanced.
- Use refined line weights and clean rounded rectangles.
- Use subtle soft shadows or depth only if very restrained and publication-appropriate.
- Avoid flat childish infographic styling.
- Avoid clip-art feeling.
- Avoid overdecorated gradients.
- Avoid oversized icons.
- Make the whole figure feel like a polished ACM MM / CVPR / NeurIPS systems-paper teaser, with calm sophistication and strong editorial composition.

Color rules:
- Dark charcoal text.
- Muted blue for neutral structure and modules.
- Muted red for bad branches and failure drift.
- Muted green for accepted grounded branch.
- A very small amount of muted orange only for future-rollout emphasis if needed.

Hard constraints:
- No visible panel letters or panel numbers.
- No helper phrases copied from the prompt.
- No pseudo-caption blocks.
- No glossy UI.
- No poster styling.
- No decorative fantasy gate.
- No recognizable franchise or character reference.
- No residual broken characters after labels.
- Do not turn this into a full pipeline figure with MLLM, aggregate score, admitted token, or lower optimization tier.

The final result should feel like a strong conference teaser: concise, elegant, conceptually sharp, and visually polished. It must communicate the difference between naive decoding and CHORD at a glance, even if some text is later manually replaced.
```

## Figure 2

### Filename Suggestion

`fig2_method_overview.png`

### Scientific Purpose

Explain the CHORD inference-time algorithm clearly enough that readers can understand the method before reading the formal equations.

### Recommended Layout

Two-tier double-column figure:

- Upper tier: algorithm flow
- Lower tier: efficiency strip

Suggested height balance:

- Upper tier: 68%
- Lower tier: 32%

### Upper Tier Specification

Show the full inference-time pipeline:

1. Input image and text query.
2. Base MLLM produces top-k candidate tokens.
3. Each candidate flows into three modules:
   - `Past`
   - `Present`
   - `Future`
4. Scores are aggregated.
5. Best candidate is admitted.

Use concise visual encodings for each module:

- `Past`: tiny self-attention heatmap plus rollback arrow.
- `Present`: image patch with two or three anchor boxes and token-to-region weighting.
- `Future`: one candidate appended to the prefix, then short rollout over steps `t+1`, `t+2`, `t+3`.

Allowed embedded labels:

- `Image`
- `Query`
- `Top-k candidates`
- `Past`
- `Rollback`
- `Present`
- `Anchor support`
- `Future`
- `Oracle rollout`
- `Aggregate score`
- `Admit token`

### Lower Tier Specification

This strip should visualize why the method is still practical:

- A shared prefix trunk feeding multiple candidate branches.
- Candidate branches share KV cache.
- Some branches stop early after failure at step 1 or step 2.
- One branch continues to the end and survives.

Allowed embedded labels:

- `Shared prefix`
- `Shared KV`
- `Batched rollout`
- `Early stop`
- `B&B pruning`

### Composition Rules

- Use rectangles and arrows, not organic illustrations.
- Keep the upper tier mechanically precise.
- Lower tier should look like an engineering optimization inset, not a separate figure.
- Make sure `Future` is visually the most novel module.
- Add a subtle abstract chrono motif around the `Future` and admission components, such as afterimages, reconverging timelines, or a compact rewind-forward glyph.
- Do not use large step numerals, oversized numbered badges, or slide-like `1`, `2`, `3` decorations.

### Final Prompt For Figure 2

Use this version when the model fails to produce the true two-tier method figure or keeps writing long broken phrases. This prompt is intentionally strict and should be copied as-is.

```text
Generate a publication-ready method overview figure for an ACM Multimedia paper. Prioritize structure, module hierarchy, scientific logic, and visual clarity over exact text rendering. The image must look like a polished top-tier systems-paper diagram, not a slide, not a poster, not a tutorial graphic. Use a pure white background, vector-like geometry, precise alignment, refined spacing, consistent rounded module boxes, elegant thin arrows, restrained colors, subtle depth only if extremely light, and minimal text.

Important instruction: if exact text rendering is difficult, prefer short clean labels or simple label placeholders rather than long malformed text. Do not generate explanatory sentences, helper phrases, or copied prompt language. The figure should remain correct and understandable even if text is later manually replaced.

Target visual story:
1. The figure should show one continuous inference-time pipeline from left to right.
2. The pipeline starts from image plus query, then model, then top-k candidate generation.
3. The candidates are broadcast into a central tri-module evaluation stage with three parallel sibling paths: Past, Present, and Future.
4. These three paths evaluate the same candidate token in parallel, not in serial order.
5. The Future path visibly unfolds into short rollout branches with branch pruning, but it remains one of the three parallel scoring signals rather than a downstream stage after Past or Present.
6. Shared prefix / shared KV / batched rollout / early stop should be integrated into the same global pipeline story, not split into a disconnected second figure.
7. The three scores are fused into one aggregate score, then the pipeline ends with token admission and grounded continuation.

Create exactly one continuous full-pipeline figure, not a two-tier split figure.

Critical role separation rule:
- This must remain a method overview figure, not a teaser.
- It must explicitly include the model block, candidate-token stage, a parallel tri-module evaluation stage, rollout/pruning mechanism inside the Future signal, aggregate-score stage, and admitted-token stage.
- Do not divide it into three teaser-style story panels.
- Do not make grounded-output storytelling the main focus.
- Do not present the figure as a high-level comparison against baselines.
- Make this figure feel more technical, more specific, and more mechanistic than Figure 1.
- Do not render it as a top-half / bottom-half composition with two independent subfigures.

Main pipeline:
- On the left, show a small image thumbnail and a very subtle empty speech/query bubble with no text inside; if the bubble looks too much like a placeholder, omit it entirely.
- Feed them into a central block representing the model.
- The model outputs several compact token blocks grouped as top-k candidates.
- Make it visually clear that these candidate tokens are evaluated before the final token is chosen.
- Route them into one central tri-module evaluation region containing Past, Present, and Future.
- Past, Present, and Future must be shown as three equally important parallel modules with the same visual hierarchy.
- None of the three may appear sequentially downstream of another.
- The candidate-token block should fan out or broadcast into all three modules simultaneously.
- The three module outputs should reconverge only at the aggregate-scoring block.
- Arrange them as sibling modules in one balanced grouped unit, preferably as one horizontal row of equal-size boxes or three clearly parallel lanes with matched input alignment and matched output alignment.
- The figure must never imply the order `Past -> Present -> Future`.
- Past must visually unfold its internal logic using a compact mini-diagram rather than mostly text: show a small attention heatmap, an emphasized problematic column or hotspot, and a rollback arrow or rewind cue that clearly suggests historical attention diagnosis and correction.
- Present must visually unfold its internal logic using a compact mini-diagram rather than mostly text: show a cropped image with multiple anchor boxes, token-to-region link cues, and a sense of query-conditioned grounding support.
- Future must visually unfold its internal logic using a compact mini-diagram rather than mostly text: show a short-horizon rollout tree or probe structure with several possible short continuations, visually suggesting future-trajectory testing and branch filtering.
- The Future region should visually contain the rollout-and-pruning mechanism inside its own module box, explaining how the Future score is computed for the same candidate token.
- Shared prefix, shared KV, batched rollout, branch-and-bound pruning, and early stop should be shown as compact internal structures inside the Future module rather than as a separate external inset.
- Branch-and-bound pruning and early stop should appear as pruning events along compact rollout branches embedded inside the Future box.
- The outputs of Past, Present, and Future, including the future rollout-derived signal, converge into one aggregate-scoring block.
- Then show one final admitted-token block.
- Future may be slightly emphasized using color only, but it must remain structurally parallel to Past and Present rather than becoming a downstream stage.
- If text is rendered, keep it minimal and limited to: `MLLM`, `Top-k candidate tokens`, `Past`, `Present`, `Future`, `Aggregate score`, `Admitted token`.
- Avoid duplicate labels when possible. Prefer one clean occurrence of each major label.
- In particular, render `Top-k candidate tokens` only once and `Admitted token` only once.
- Do not repeat `Top-k candidate tokens` above and below the same token stack.
- Do not repeat `Admitted token` to the right of the same output token.
- The enlarged central tri-module region should be the visual focus of the entire figure.
- The `Past`, `Present`, and `Future` modules should look crisp, balanced, and publication-grade rather than like rough sketch boxes.
- The three modules must be visually large enough that each one reads as a real mechanism panel rather than a small label box.
- The aggregate-score block should be clearly downstream of the three modules, but visually smaller than the combined three-module unit.
- Each of the three modules should include richer internal visual detail, not just a single icon.
- The three modules should each feel self-contained and information-rich, with small internal substructure, like mini-diagrams embedded inside the larger pipeline.
- Avoid simplistic empty boxes with only one icon and one word.
- Prefer visual explanation over verbal explanation. A reviewer should understand the internal mechanism of each module at a glance without reading sentences inside the boxes.
- Keep module names visible, but minimize any additional internal text. Internal arrows, highlights, small nodes, box overlays, and attention patterns should do most of the explanatory work.
- Batched rollout, shared KV, early stop, and B&B pruning should be integrated as local substructures inside the Future module.
- Do not place the optimization logic in a separate lower band, detached inset, or overlapping overlay that reads like another figure.
- Avoid giant green success arrows or overextended branches.
- Keep branch spacing, line weights, and pruning marks crisp and publication-grade.

Chrono motif rule:
- A very subtle abstract time-selection cue is allowed, such as faint afterimages or reconverging branch traces.
- Do not write any chrono explanation as text.
- Do not include any recognizable character or franchise-like symbol.

Color rules:
- Dark charcoal text.
- Muted blue for neutral modules and structure.
- Muted orange only for Future emphasis.
- Muted red for pruned branches.
- Muted green for the surviving path.

Hard constraints:
- No visible panel letters or panel numbers.
- No helper phrases copied from the prompt.
- No long sentences.
- No duplicate large labels unless necessary.
- No title banner.
- No legends.
- No slide-style headers.
- No heavy gradients.
- No glossy UI.
- No poster styling.
- No decorative sparkles or corner ornaments.
- No placeholder words such as `model`, `query`, `token`, `candidate-token group`, `aggregate-score`, `admitted-token`, or other control-language variants outside the approved short labels.
- No giant green success arrow dominating the Future module.
- Do not rely on dense text inside modules to explain the algorithm.
- Do not place sentence-level explanations inside the `Past`, `Present`, or `Future` boxes.

The final result should immediately read as one continuous inference-time pipeline: top-k candidate tokens are broadcast into three parallel evaluation signals, Past, Present, and Future; the Future signal is computed through compact rollout branches inside the Future module with shared-KV-aware efficient reasoning; and the final token is admitted only after these parallel signals are fused by aggregate scoring. It should feel like a polished ACM MM / CVPR / NeurIPS method figure rather than a generic AI-generated infographic. The composition must succeed even if all text is later manually replaced.
```

### Modular Generation Strategy For Figure 2

If end-to-end generation keeps producing shallow or noisy results, generate `Figure 2` as separate assets and manually compose them. This is recommended when you want richer internal structure inside each module.

For this modular strategy, `Figure 2` should be a detailed continuous pipeline figure, not a teaser, not a side-by-side comparison, and not a top-half / bottom-half split figure. Its structure should be clearly different from `Figure 1`.

The overall layout should feel like a true systems-method diagram:

- one continuous left-to-right inference pipeline
- a central tri-module evaluation core with three clearly parallel lanes
- the three core modules expanded as large detailed internal cards
- rollout/pruning logic absorbed into the Future card rather than detached below or overlaid outside
- stronger mechanical flow and less storytelling
- denser internal mechanism detail than Figure 1, while still visually clean

Generate these assets separately:

1. `figure2_skeleton`
2. `figure2_past_module`
3. `figure2_present_module`
4. `figure2_future_module`

All assets should share the same visual language:

- white background
- vector-like geometry
- rounded rectangles
- thin dark arrows
- muted blue neutral structure
- muted orange for Future emphasis
- muted red for pruned branches
- muted green for accepted/surviving branches
- minimal text
- identical line weight family

### Figure 2 Skeleton Prompt

Use this to generate the large composition skeleton only. This asset should define the layout and spatial relationships, not the module internals.

```text
Generate a publication-ready layout skeleton for an ACM Multimedia method figure. White background, vector-like geometry, precise alignment, refined spacing, thin arrows, rounded module boxes, and minimal text. This is only the composition skeleton, not the detailed final figure.

Create a single continuous left-to-right pipeline layout that is clearly different from a teaser figure.

Pipeline stages:
- far left: input image area and subtle query cue
- next: model block
- next: top-k candidate-token block
- next: a grouped tri-module region containing three large equal-size module containers labeled only `Past`, `Present`, and `Future`
- the candidate-token block should visually fan out into all three module containers at once
- the three module containers should be drawn as parallel siblings with matched input alignment and matched output alignment as much as possible
- the `Future` container should reserve richer internal space for compact rollout branches and pruning cues, but its outer size should remain aligned with `Past` and `Present`
- after the three lanes, one reconvergence point or fusion block should collect all three outputs before the aggregate-score block
- next: aggregate-score block
- next: admitted-token output block

The grouped tri-module region should feel like the central scientific core of the figure. It should be substantially larger than the model block and substantially larger than the aggregate-score block. Each of the three module containers should be large enough to hold a real internal mini-diagram and should not read as a small label box.

Do not reserve or draw any detached lower-right or overlapping rollout inset. The rollout detail must live inside the Future module itself. Leave enough internal space inside the Future module for:
- shared-prefix / shared-KV cues near rollout branch emergence
- batched rollout indication
- early-stop / pruning markers on short branches
- one moderate surviving branch that does not dominate the module

This skeleton must feel like a detailed continuous pipeline figure for a top-tier paper, not a comparison panel, not a high-level overview cartoon, and not two independent stacked subfigures. The skeleton must clearly communicate parallel candidate evaluation rather than serial stage ordering. Do not include detailed icons inside the three modules. Do not include large text beyond the minimal labels already named.
```

### Figure 2 Past Module Prompt

Use this to generate a self-contained `Past` mini-diagram asset that can be inserted into the Past box.

```text
Generate a compact publication-style mini-diagram for the `Past` module of a multimodal decoding method. White background, vector-like geometry, crisp edges, refined spacing, minimal text.

This mini-diagram should visually explain historical-attention diagnosis and rollback without relying on sentence text.

Include:
- a compact attention heatmap
- one emphasized problematic column or hotspot suggesting over-trust in previous context
- a rollback or rewind arrow
- a subtle sense that the system is checking whether the current candidate is being corrupted by historical textual inertia
- a small before/after or flagged/mitigated visual contrast inside the same card, if possible
- enough internal detail that the module feels like a real mechanism diagram, not a single icon

If any text appears, keep it minimal and limited to `Past` and optionally `Rollback`. Do not add sentence-level explanations.

The module should feel self-contained, information-rich, and mechanism-oriented, like a small systems-paper inset rather than a generic icon.
```

### Figure 2 Present Module Prompt

Use this to generate a self-contained `Present` mini-diagram asset that can be inserted into the Present box.

```text
Generate a compact publication-style mini-diagram for the `Present` module of a multimodal decoding method. White background, vector-like geometry, crisp edges, refined spacing, minimal text.

This mini-diagram should visually explain query-conditioned grounding support without relying on sentence text.

Include:
- a cropped image region
- multiple anchor boxes with varied sizes
- token-to-region links or small attention connectors
- a clear sense that the current candidate is being checked against query-relevant visual support
- layered or weighted visual emphasis so some regions read as more relevant than others
- enough internal detail that the module feels like a real grounding mechanism rather than a thumbnail with boxes

If any text appears, keep it minimal and limited to `Present` and optionally `Anchor support`. Do not add sentence-level explanations.

The module should feel concrete, visual, and grounded, with richer internal structure than a simple image thumbnail.
```

### Figure 2 Future Module Prompt

Use this to generate a self-contained `Future` mini-diagram asset that can be inserted into the Future box.

```text
Generate a publication-style mini-diagram for the `Future` module of a multimodal decoding method. White background, vector-like geometry, crisp edges, refined spacing, minimal text. This module should be richer and slightly denser internally than Past and Present, but its outer box should still match them in size and hierarchy.

This mini-diagram should visually explain short-horizon oracle rollout and branch filtering without relying on sentence text. It should read as the internal computation of the Future score for the same candidate token, not as a later stage after Past and Present.

Include:
- a compact rollout tree or branching probe structure
- several short continuation branches
- visual cues that some branches are less desirable and one branch is preferred
- a strong sense of future-trajectory testing rather than a generic decision tree
- enough internal detail that the rollout looks like a mechanistic probing process, not a decorative branching icon
- one compact cue suggesting pruning or branch rejection during lookahead
- one compact cue for shared prefix or shared KV reuse near the branch origin
- all of this detail must remain inside the Future module box, with no detached external inset

Use muted orange as the dominant accent for this module. If any text appears, keep it minimal and limited to `Future` and optionally `Oracle rollout`. Do not add sentence-level explanations.

The module should feel like the most novel component in the triad while still matching the same visual family as Past and Present. It must still look like one parallel sibling in the tri-module evaluation set, not like a module plus a second attached figure.
```

### Figure 2 Assembly Guidance

When manually merging the assets:

- Start from `figure2_skeleton`.
- Insert `figure2_past_module`, `figure2_present_module`, and `figure2_future_module` into the three sibling boxes in the central tri-module region.
- Keep the central tri-module region visually dominant.
- Ensure the three module boxes remain the same outer size even if their internal mini-diagrams differ.
- If one module looks visually denser than the others, reduce its internal contrast or scale slightly so the triad remains balanced.
- Do not attach any extra detached or overlapping rollout box below the tri-module region.
- Make the Future box internally richer instead of externally larger.
- Replace or normalize labels manually if needed after composition.
- Preserve one strong left-to-right pipeline read across the full composition.
- Make sure the composition clearly shows `candidate -> parallel Past/Present/Future -> fused aggregate score`, not `candidate -> Past -> Present -> Future`.
- Make sure the resulting figure no longer resembles a teaser, side-by-side comparison, or stacked top/bottom figure.
- The final composed figure should feel denser, more technical, and more explanatory than Figure 1.

## Figure 3

### Filename Suggestion

`fig3_latency_performance.png`

### Scientific Purpose

Show that naive prospective rollout is expensive, but CHORD becomes practical after shared-KV batching and branch-and-bound pruning.

### Recommended Layout

Single-panel academic scatter plot with light annotation. This figure should not look like a marketing infographic. It should look like a polished chart ready for a paper.

### Required Elements

- x-axis placeholder for latency or decoding cost
- y-axis placeholder for performance or POPE F1
- labeled points:
  - `Greedy`
  - `VCD`
  - `OPERA`
  - `Naive CHORD`
  - `CHORD + shared-KV`
  - `Full CHORD`
- A short arrowed progression from `Naive CHORD` to `CHORD + shared-KV` to `Full CHORD`
- Optional faint dashed Pareto guide

### Axis Guidance

Keep labels generic enough that you can replace them later:

- x-axis: `Latency per token` or `Relative inference cost`
- y-axis: `POPE F1` or `Grounding performance`

### Color Semantics

- Greedy: neutral gray
- Baselines like VCD and OPERA: blue or slate
- Naive CHORD: orange-red
- Optimized CHORD variants: green gradient or blue-green progression

### Composition Rules

- Use generous white space.
- Keep point labels horizontal and readable.
- Leave room near points for final numerical post-editing.
- No unnecessary legends if direct labeling works.

### Data You Must Provide Before Final Drawing

For the final `Figure 3`, collect one consistent measurement setup and provide:

- main model name, such as `LLaVA-1.5 7B`
- hardware description, such as GPU type and precision
- x-axis definition, such as `latency per generated token (ms)` or `relative decoding cost`
- y-axis definition, such as `POPE Adversarial F1`
- one point each for `Greedy`, `VCD`, `OPERA`, `Naive CHORD`, `CHORD + shared-KV`, and `Full CHORD`
- whether latency includes preprocessing or only autoregressive decoding
- the exact evaluation split used for the y-axis

Recommended note for the caption or plotting log:

- all methods should be measured on the same hardware, same batch setting, and same prompt/eval subset
- if `Naive CHORD` is too slow to run on the full benchmark, it can be measured on the same representative subset and explicitly marked as such in internal notes

### Full Nanabanana Prompt

```text
Create a polished academic scatter plot for a machine learning paper on a white background. The figure should communicate a latency-versus-performance tradeoff for decoding methods. It must look like a publication-ready chart, not a business infographic.

Use a single-panel scatter plot with direct point labels. The x-axis is a placeholder for latency or relative inference cost. The y-axis is a placeholder for performance such as POPE F1. Include labeled points: “Greedy”, “VCD”, “OPERA”, “Naive CHORD”, “CHORD + shared-KV”, and “Full CHORD”.

Make “Naive CHORD” clearly high-cost and high-performance relative to basic methods. Show an arrowed optimization progression from “Naive CHORD” to “CHORD + shared-KV” to “Full CHORD”, where the optimized variants move toward a more favorable tradeoff region. Optionally include a subtle Pareto-style guide or small annotation emphasizing that optimization makes prospective decoding practical.

Use restrained academic colors: neutral gray for Greedy, muted blue/slate for baseline methods, orange-red for Naive CHORD, and green or blue-green for optimized CHORD variants. Use thin axes, clean typography, direct labeling, and plenty of whitespace. Keep everything easy to edit later with final numbers.

Do not generate fantasy art, painterly style, cinematic lighting, cartoon characters, anime styling, 3D glossy interface elements, handwritten text, illegible labels, cluttered infographic decoration, stock-photo marketing style, scientific nonsense equations, or colorful poster aesthetics. Keep it clean, academic, publication-style, vector-like, and readable after downscaling.
```

### Fallback Prompt

```text
Create a simple academic scatter plot with six labeled points on a white background: Greedy, VCD, OPERA, Naive CHORD, CHORD + shared-KV, Full CHORD. Show an optimization arrow from Naive CHORD toward Full CHORD. Use restrained colors, direct labels, thin axes, and publication-style formatting only.
```

## Figure 4

### Filename Suggestion

`fig4_qualitative_mechanism.png`

### Scientific Purpose

Tie the mechanistic trajectory claim to real visual examples by showing that CHORD rejects locally plausible but trajectory-risky continuations before collapse.

### Recommended Layout

Composite figure with:

- Left block: 42% width
- Right block: 58% width

The left side is analytical and chart-based. The right side is example-based.

### Left Block: Trajectory Mechanism

Show a line chart across decoding steps:

- x-axis: decoding step
- y-axis: visual grounding ratio or visual support
- curves:
  - `Baseline`
  - `OPERA`
  - `CHORD`

Desired qualitative shape:

- `Baseline`: noticeable drop after a risky token, then sustained collapse
- `OPERA`: partial stabilization but still unstable
- `CHORD`: higher and more stable grounding trajectory

Add one annotation:

- `Collapse point`

Optional second annotation:

- `Oracle rejection`

### Right Block: Two Compact Cases

Each case should contain:

- Small image thumbnail
- Visible anchor boxes on query-relevant regions
- Short prompt
- Baseline output with hallucinated phrase highlighted in red
- CHORD output with corrected or grounded phrase highlighted in green

Keep the text snippets short enough to stay readable.

Allowed embedded labels:

- `Case 1`
- `Case 2`
- `Prompt`
- `Baseline`
- `CHORD`
- `Anchor boxes`
- `Hallucinated`
- `Grounded`

### Composition Rules

- The chart must stay readable and not be squeezed by the qualitative side.
- The cases should look like compact evidence cards, not social-media tiles.
- Anchor boxes must be visible but not overly thick.
- Avoid using too many colors beyond the shared paper palette.
- If a chrono motif appears, keep it extremely subtle and abstract, for example faint afterimages or a tiny rewind-forward glyph near the chart annotation.
- Do not use large case numbers, giant badges, or oversized `1`, `2`, `3` labels.

### Data You Must Provide Before Final Drawing

For the final `Figure 4`, collect both mechanism traces and qualitative cases:

- Left chart data:
  - decoding-step x values
  - one grounding-ratio curve for `Baseline`
  - one grounding-ratio curve for `OPERA`
  - one grounding-ratio curve for `CHORD`
  - the step of the risky token or collapse point
  - optional step where CHORD rejects a risky branch
- Right qualitative cases, for each case:
  - image path
  - prompt text
  - baseline output with the hallucinated span identified
  - optional OPERA output if you want three-way comparison
  - CHORD output with the grounded span identified
  - anchor boxes, either as coordinates or as already annotated images

Important methodological note:

- if the left chart uses a `visual grounding ratio`, define it in the paper as an aggregated signal from the monitored decoder blocks, preferably head-averaged and layer-aggregated over the last few decoder layers, rather than claiming a single fixed layer unless experiments explicitly justify that choice
- this layer-aggregation detail should be explained in the methodology text and caption, not by complicating `Figure 4` with extra layer diagrams

### Full Nanabanana Prompt

```text
Create a composite publication-style scientific figure for a multimodal AI paper. The figure has two main parts on a white background and should look like a top-tier paper figure rather than an infographic poster.

On the left, create a clean line chart showing visual grounding ratio across decoding steps. Include three curves labeled “Baseline”, “OPERA”, and “CHORD”. The Baseline curve should show a sharp drop after a risky token and remain low, indicating structural collapse. The OPERA curve should be partially stabilized but still less robust. The CHORD curve should remain higher and more stable. Add a concise annotation such as “Collapse point” and optionally “Oracle rejection”. Keep the chart academic, simple, and easy to read.

On the right, create two compact qualitative evidence cards labeled “Case 1” and “Case 2”. For each case, include a small image thumbnail, visible anchor boxes around query-relevant regions, a short prompt, a short baseline output with a hallucinated phrase highlighted in muted red, and a short CHORD output with grounded correction highlighted in muted green. Keep text snippets short and legible. The qualitative side should feel like compact evidence, not decorative demo panels.

Use the same visual language as a scientific system paper: dark charcoal text, muted blue for neutral structure, muted red for hallucination, muted green for grounded correction, and restrained orange only if needed for forecast-related annotations. A very subtle abstract chrono motif is allowed, such as faint afterimages or a tiny reconverging timeline mark, but it must remain original and non-referential. Maintain balanced spacing, thin lines, and strong readability after downscaling to an ACM two-column layout. Do not use giant case numbers, circular badges, or oversized 1/2/3 markers.

Do not generate fantasy art, painterly style, cinematic lighting, cartoon characters, anime styling, 3D glossy interface elements, handwritten text, illegible labels, cluttered infographic decoration, stock-photo marketing style, scientific nonsense equations, colorful poster aesthetics, recognizable franchise character references, or oversized panel numbers such as giant 1/2/3 badges. Keep it clean, academic, publication-style, vector-like, and readable after downscaling.
```

### Fallback Prompt

```text
Create a simple two-part academic figure on a white background. Left: a line chart comparing Baseline, OPERA, and CHORD visual grounding over decoding steps, with one collapse annotation. Right: two compact example cards with image thumbnails, anchor boxes, prompt, baseline output in red-highlighted error, and CHORD output in green-highlighted correction. Clean paper style only, no artistic styling.
```

## Table Layout Guidance

These should be authored directly in LaTeX, not generated as artistic figures.

## Table 1

### Role

Flagship main-results table for POPE. This is the highest-priority table in the paper.

### Recommended Structure

- Double-column table.
- Rows:
  - `Greedy`
  - `DoLa`
  - `VCD`
  - `OPERA`
  - `CHORD`
- Columns grouped by split:
  - `Random`
  - `Popular`
  - `Adversarial`
- Within each split, prefer:
  - either `F1` only
  - or `Acc` and `F1`

### Strong Recommendation

Do not keep `Accuracy + Precision + F1` all together in the main paper if width becomes a problem. `Precision` is the first metric to move to supplement.

### Caption Intent

Emphasize that CHORD improves over static decode-time baselines and that gains are strongest in adversarial settings where language priors are most misleading.

## Table 2

### Role

Compact broader-evaluation table proving both:

- reduced hallucination in long-form generation
- no severe alignment tax on general multimodal capability

### Recommended Structure

Single-column compact table with two blocks.

Block A: CHAIR

- `CHAIR_S`
- `CHAIR_I`
- `Recall`
- `Length`

Block B: General capability

- `MME`
- `MMBench`
- `HallusionBench`

### Rows

- `Greedy`
- `VCD`
- `OPERA`
- `CHORD`

If width becomes tight, use only the most relevant baselines.

### Caption Intent

State that CHORD reduces hallucination in caption generation while preserving general multimodal competence, supporting the no-alignment-tax claim.

## Table 3

### Role

Compact ablation proving that the full Past-Present-Future design matters.

### Recommended Structure

Single-column table.

Rows:

- `Baseline`
- `+ Past`
- `+ Past + Present`
- `Full CHORD`

Columns:

- `POPE F1`
- `CHAIR_S`
- optional `Latency`

If space allows, add one more grounding-related column only if it clearly supports the value of `Present`.

### Caption Intent

Stress that `Past` helps but is insufficient, `Present` improves grounding support, and `Future` provides the decisive gain that differentiates CHORD from retrospective-only methods.

## Suggested Handoff Workflow

1. Generate `Figure 1` first to establish the visual language.
2. Reuse that same palette and panel spacing for `Figure 2` and `Figure 4`.
3. Finalize `Figure 3` only after the experimental axis definition, hardware note, and point ordering are stable.
4. Finalize `Figure 4` only after the grounding traces and qualitative cases are frozen.
5. Post-edit all figure text manually if the model produces typographic artifacts.

## Minimal Message To Send With Any Prompt

If you want a short wrapper before each main prompt, prepend this:

```text
Please generate a publication-style scientific figure for an ACM Multimedia paper. Prioritize clarity, vector-like structure, clean white background, legible text, and paper-ready composition over artistic flair.
```

## Final Sanity Checklist

Before accepting any generated figure, verify:

- Can I read it when scaled to two-column paper size?
- Does it communicate one main claim without the caption?
- Is the color semantics consistent with the other CHORD figures?
- Does it look like a systems-paper figure rather than AI marketing art?
- Can I fix remaining issues with light manual editing instead of redrawing from scratch?
