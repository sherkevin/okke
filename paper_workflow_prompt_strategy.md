# Paper Workflow Prompt Strategy

## Active API Settings
- API URL: `https://new.lemonapi.site/v1`
- Model: `[L]gemini-3.1-pro-preview`

## Reviewer Strategy
- Every reviewer call uses a fresh request context
- The reviewer is instructed to act like an ACM MM chief reviewer
- Output includes an explicit `Score: x/5` line and concrete figure/table/experiment suggestions
- The reviewer must separate what should be kept from what must be fixed
- The reviewer should focus on methodology and experimental design instead of typography or formatting
- The reviewer only receives title + Abstract + Introduction + Methodology + Experiments + Conclusion/Limitations
- The reviewer does not receive related work, appendices, or references, which saves tokens and keeps attention on the core paper
- The reviewer is explicitly told that `DoLa`, `VCD`, and `OPERA` are competitive baselines, not evidence for the paper's problem definition
- The reviewer evaluates the paper through three evidence chains:
  - hallucination reduction
  - structure/reasoning preservation
  - local-evidence value
- The reviewer also checks whether the paper stays within the active benchmark contract:
  - image-centric core claim
  - bounded secondary video pilot
  - no OCR-heavy overclaim unless the intervention is actually active there

## Scientist Strategy
- The scientist performs targeted upgrades on top of the latest draft
- The scientist must preserve valid innovation and strong structure already present in the paper
- Unsupported claims must be softened instead of defended with fabricated evidence
- Planned but unrun experiments must be clearly marked as planned work
- The scientist should turn reviewer criticism into a stronger experiment outline for future execution
- The scientist is explicitly forbidden from reviving the invalid `Pooling Paradox`-as-title-eye framing
- The scientist must preserve the active benchmark core:
  - decode-time / logits-space intervention
  - token-local visual evidence
  - VASM-based structure preservation
  - `TLRA_zero` / `TLRA_calib` fairness boundary
- The scientist must not let OCR-heavy motivation, PEFT identity debates, or over-detailed engineering mechanisms take over the main paper narrative
- The scientist should keep video as a bounded secondary pilot unless there is clear evidence that temporal locality materially strengthens the paper
- The scientist emits:
  - the full revised paper
  - a structured revision log
  - a compact scientist memory block for future rounds

## Context Policy
- Reviewer: no memory, always fresh
- Scientist: cumulative memory saved in `Scientist_Memory.md`
- All paper and review versions are immutable once written
