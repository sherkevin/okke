# IFCB Phase-2 Handoff for InstructBLIP (2026-03-24)

## Current Phase-1 Status

- Default `run_eval_pipeline.py` path now accepts `--method ifcb`.
- Phase 1 is intentionally wired only for `llava-v1.5-7b`.
- The new IFCB core lives in `ifcb/processor.py`.
- `POPE` is implemented as an explicit binary commitment step (`yes` / `no`) with one-token decode.
- `CHAIR` is implemented as iterative top-k semantic-token control on the caption path.

## Why InstructBLIP Is Not Turned On Yet

The current IFCB implementation depends on a LLaVA-specific assumption about how a single `<image>` placeholder is expanded into a longer multimodal sequence inside the language model. That assumption is encoded in `build_modal_masks()` and is sufficient for the LLaVA path that is now running on GPU1. InstructBLIP uses query tokens and a different host serialization, so blindly reusing the same mask reconstruction would make the visual-participation term `G_t` unreliable.

## Exact Delta Required for InstructBLIP

1. Add an InstructBLIP-specific controller instead of reusing `LLaVAIFCBProcessor` directly.
2. Replace the LLaVA image-token mask reconstruction with a query-token-aware mask builder.
3. Use the existing `InstructBLIPAdapter` in `bra_operator_multi.py` only to identify host structure, not to route IFCB through BRA.
4. Verify the final norm and LM head accessors against InstructBLIP's `language_model` path before enabling decode-time control.
5. Rebuild the `POPE` binary answer-token set with the InstructBLIP tokenizer and confirm the one-token `yes/no` surface is stable.
6. Re-run the `CHAIR` semantic-token filter on InstructBLIP tokenization, because its wordpiece boundaries can differ from LLaVA/Vicuna.

## Suggested Implementation Steps

1. Split `LLaVAIFCBProcessor` into a host-agnostic base class plus host-specific mask builders.
2. Add `build_instructblip_modal_masks()` that marks visual query tokens instead of LLaVA-expanded image spans.
3. Add `InstructBLIPIFCBProcessor` that reuses the same risk computation:
   `R_t = U_t (1 - Pi_t) (1 - G_t)`.
4. Extend `run_eval_pipeline.py` model gating from `llava-v1.5-7b` only to `{"llava-v1.5-7b", "instructblip-7b"}` after the new processor passes tests.
5. Add at least three InstructBLIP-specific tests:
   - modal mask reconstruction for query-token layouts
   - `POPE` binary answer routing
   - one `CHAIR` semantic-token filtering case

## Activation Criterion

Do not enable `ifcb` for `instructblip-7b` until:

- modal masks are validated on a real forward pass,
- `POPE` one-token answer control runs without fallback errors,
- and at least one `CHAIR` slice finishes with non-empty captions and zero runtime errors.
