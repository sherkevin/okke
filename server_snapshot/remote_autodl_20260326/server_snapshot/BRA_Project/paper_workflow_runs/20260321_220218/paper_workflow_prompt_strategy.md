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
- The reviewer does not receive the references section, which saves tokens and keeps attention on the core paper

## Scientist Strategy
- The scientist performs targeted upgrades on top of the latest draft
- The scientist must preserve valid innovation and strong structure already present in the paper
- Unsupported claims must be softened instead of defended with fabricated evidence
- Planned but unrun experiments must be clearly marked as planned work
- The scientist should turn reviewer criticism into a stronger experiment outline for future execution
- The scientist emits:
  - the full revised paper
  - a structured revision log
  - a compact scientist memory block for future rounds

## Context Policy
- Reviewer: no memory, always fresh
- Scientist: cumulative memory saved in `Scientist_Memory.md`
- All paper and review versions are immutable once written
