# TLRA Benchmark V1 Smoke - 2026-03-22

## What changed

Aligned the current BRA codebase to `论文大纲_标杆V1.md` as the latest methodology source.

Key implementation updates:

- added first-class `TLRA` aliases on top of the existing BRA core:
  - `tlra_zero`
  - `tlra_calib`
  - `tlra_full`
  - `tlra_meanpool`
  - `tlra_max`
  - `tlra_no_vasm`
  - `tlra_v1_like`
- upgraded `bra_vasm.py` to better match the paper's deterministic VASM requirement:
  - root-token lookup is now used when building the vocabulary table
  - continuation subwords inherit the root token's gamma when available
- fixed a real multi-GPU bug in `BRAVisionExtractor.extract_vision_features()` where visual-token masks could index hidden states from the wrong device under `device_map="auto"`
- added `tlra_semantic_validity_pilot.py` to implement Stage 0:
  - patch-level lexical overlap on a held-out annotated image subset
  - reports Top-1 / Top-5 / Top-10 image overlap
  - reports patch-level overlap
  - reports candidate-window overlap as a lightweight out-of-candidate risk signal

## Smoke results

### Stage 0 semantic-validity pilot

Remote artifact:

- `logs/minitest/tlra_semantic_validity_tlra_zero_20260322_012117.json`

Local mirror:

- `experiment_logs/remote_mirror/tlra_semantic_validity_tlra_zero_20260322_012117.json`

Observed on `qwen3-vl-2b`, `tlra_zero`, `n_samples=3`:

- `top1_overlap = 1.0000`
- `top5_overlap = 1.0000`
- `top10_overlap = 1.0000`
- `patch_top1_overlap = 0.0480`
- `patch_top5_overlap = 0.1527`
- `patch_top10_overlap = 0.2053`
- `candidate_window_overlap = 1.0000`

Interpretation:

- the zero-shot branch is at least not collapsing into immediate near-random noise on this tiny held-out smoke subset
- the overlap signal is promising enough to keep `TLRA_zero` alive as a viability probe
- this is still only a smoke result, not a publication-grade conclusion

### Decode-time smoke

Remote artifact:

- `logs/minitest/tlra_zero_pope_20260322_012030.json`

Local mirror:

- `experiment_logs/remote_mirror/tlra_zero_pope_20260322_012030.json`

Observed on `POPE random`, `n_samples=1`:

- `accuracy = 1.0000`
- `f1 = 1.0000`
- `AGL = 87.0`
- `ITL = 36.34 ms/token`
- `tokens_per_second = 27.515`

Interpretation:

- the new `TLRA` alias path is executable end-to-end
- the updated VASM and alias changes did not break the decode pipeline

## Remaining gaps

- `Stage 0` needs a larger held-out subset before it can be used for claim contraction decisions.
- `TLRA_calib` still needs a clearly registered official projector/checkpoint path if it is to serve as the benchmark upper bound.
- The benchmark datasets are still in a mixed readiness state:
  - `POPE` runnable
  - `CHAIR` runnable, but needs a fresh rerun under the newer higher length cap
  - `FREAK` runnable
  - `DocVQA` loader path exists, but current remote data layout still needs confirmation
  - `MMBench` / `MME` / `MMMU` need to be rerun under the `TLRA` naming path
