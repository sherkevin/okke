# CHORD Rebuttal Demand, 2026-05-29

Canonical review source: `reviews_20260528.md` only. Earlier simulated or practice reviews must not guide the rebuttal.

## Format Demand

Current verified boundary:

- ACM MM 2026 public pages do not state that the rebuttal is limited to a one-page PDF.
- ACM MM 2026 public pages state that rebuttal is submitted through OpenReview.
- The rebuttal must maintain anonymity.
- The rebuttal cannot include links to external material such as code, videos, or externally hosted evidence.
- OpenReview's default rebuttal form is a Markdown text field with `maxLength: 2500`, but venues can override this.
- Therefore, the logged-in ACMMM 2026 OpenReview rebuttal form and any official author email are the controlling sources for final length, field type, and per-paper/per-reviewer response mode.

Operational drafting demand until the logged-in form is checked:

1. Do not assume a one-page PDF response.
2. Draft as strict short OpenReview text, targeting <=2500 characters unless the actual form shows a different limit.
3. Do not include external links.
4. Do not include author-identifying information.
5. Do not rely on a revised PDF, new supplement, GitHub, Drive, arXiv update, or other external upload.
6. Refer only to submitted paper/supplement content and newly measured evidence that can be stated directly in text.
7. Use compact inline evidence tables only if the actual OpenReview field preserves readable Markdown/plain text.

Sources checked:

- ACM MM 2026 Call for Technical Papers: https://2026.acmmm.org/site/cfp-guidelines.html
- ACM MM 2026 Author Instructions: https://2026.acmmm.org/site/author-instructions.html
- ACM MM 2026 Important Dates: https://2026.acmmm.org/site/important-dates.html
- OpenReview Rebuttal Stage documentation: https://docs.openreview.net/reference/stages/rebuttal-stage
- OpenReview Default Rebuttal Form documentation: https://docs.openreview.net/reference/default-forms/default-rebuttal-form

Evidence boundary: unauthenticated OpenReview confirms the ACMMM 2026 venue and that `Rebuttal` is configured as the rebuttal name, but the submission-specific rebuttal invitation/form is private. The logged-in form must be checked before final submission.

## Content Demand

The rebuttal should be evidence-first and should not read like a broad debate. The highest-value response modules are:

1. Future rollout mechanism: Full vs Past+Current flip rate, corrected flips, harmful flips, and when the future term helps.
2. Detector attribution: no/uniform anchors, random anchors, same-anchor non-CHORD control if feasible, and proposer failure cases.
3. Efficiency: detector proposal time, decode ITL, total answer latency, peak VRAM, and batch-size setting.
4. Hyperparameters: compact `k` and `m` sweep with quality and latency.
5. Related work: ONLY, VHD/VHR, HALC positioning, with direct empirical comparison only if reproducible under matched settings.
6. Presentation: Figure 2 cleanup commitment and simpler pipeline explanation.

Claim discipline:

- Say "training-free for the base MLLM" or "detector-assisted training-free decoding"; do not imply no external pretrained module.
- Treat attention as an operational grounding signal, not a causal explanation of model decisions.
- Preserve the two-regime interpretation: Past+Current is latency-oriented; Full CHORD is quality-oriented for open-ended or continuation-sensitive generation.
- Do not claim detector-independent gains, future-term superiority, k/m robustness, or end-to-end overhead unless measured evidence supports it.
