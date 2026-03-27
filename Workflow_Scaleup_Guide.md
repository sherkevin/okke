# Workflow Scale-Up Guide

## Current State
- Pilot model: `[L]gemini-3.1-pro-preview`
- Reviewer policy: always use a fresh request context
- Scientist policy: carry forward accumulated scientist memory between rounds

## Recommended Next Loop
1. Feed the current benchmark-based latest paper to a fresh reviewer and create the matching `Review_Strict_Vx.md`
2. Use `Review_Strict_Vx.md` plus `Scientist_Memory.md` to generate `论文大纲_V(x+1).md`
3. Keep every version and never overwrite prior outputs
4. Stop when a fresh reviewer gives `4/5` or `5/5`, or after 100 rounds

## What To Optimize First
- Tone down unsupported absolute claims
- Preserve the positive core: token-local logits intervention + VASM + fair zero-shot/calibrated split
- Tighten the method-to-evaluation loop with explicit ablations and planned experiments
- Avoid reviving invalid baseline-framing arguments
- Add limitations and failure cases before the reviewer asks for them
- If the loop flatlines around `3/5`, reset from the current benchmark instead of continuing a drifted branch

## Operating Notes
- Re-run the same script with a larger `--max-version` target when you want to continue the loop
- Keep the API key outside the code and provide it via environment variable
- Review `Scientist_Memory.md` occasionally to prevent local overfitting to one reviewer style
