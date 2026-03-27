# LLaVA-1.5-7B · POPE Baseline Main Table

**Model:** `llava-v1.5-7b`  
**Dataset:** POPE (COCO), `mini_test=3000` per split  
**Methods:** Base, Beam Search (5 beams), DoLa, OPERA  
**Result JSON (remote):** `/root/autodl-tmp/BRA_Project/logs/minitest/`  
**Local mirror:** `pope_baseline_fullrun_sync/llava_jsons/`  
**Matrix logs:**  
- `random`: `logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log`  
- `popular` + `adversarial`: `logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log`

---

## Table 1 — F1 (primary for POPE)

| Method | random | popular | adversarial |
|--------|--------|---------|-------------|
| Base | 0.8865 | 0.8562 | 0.8036 |
| Beam Search | 0.8877 | **0.8575** | **0.8132** |
| DoLa | **0.8878** | 0.8525 | 0.8103 |
| OPERA | 0.8850 | 0.8556 | 0.8042 |

*Best per column in **bold**.*

---

## Table 2 — Accuracy

| Method | random | popular | adversarial |
|--------|--------|---------|-------------|
| Base | 0.8857 | 0.8500 | 0.7817 |
| Beam Search | 0.8907 | **0.8563** | **0.8013** |
| DoLa | **0.8917** | 0.8517 | 0.7993 |
| OPERA | 0.8840 | 0.8493 | 0.7827 |

---

## Table 3 — Precision / Recall (optional appendix)

| Method | split | Precision | Recall |
|--------|-------|-----------|--------|
| Base | random | 0.8798 | 0.8933 |
| Beam Search | random | 0.9121 | 0.8647 |
| DoLa | random | 0.9205 | 0.8573 |
| OPERA | random | 0.8775 | 0.8927 |
| Base | popular | 0.8221 | 0.8933 |
| Beam Search | popular | 0.8505 | 0.8647 |
| DoLa | popular | 0.8477 | 0.8573 |
| OPERA | popular | 0.8215 | 0.8927 |
| Base | adversarial | 0.7302 | 0.8933 |
| Beam Search | adversarial | 0.7675 | 0.8647 |
| DoLa | adversarial | 0.7682 | 0.8573 |
| OPERA | adversarial | 0.7317 | 0.8927 |

---

## JSON file mapping (for traceability)

| split | method | remote JSON filename |
|-------|--------|----------------------|
| random | base | `base_pope_20260323_140310.json` |
| random | beam_search | `beam_search_pope_20260323_144348.json` |
| random | dola | `dola_pope_20260323_151139.json` |
| random | opera | `opera_pope_20260323_153647.json` |
| popular | base | `base_pope_20260323_155904.json` |
| popular | beam_search | `beam_search_pope_20260323_163906.json` |
| popular | dola | `dola_pope_20260323_170619.json` |
| popular | opera | `opera_pope_20260323_173110.json` |
| adversarial | base | `base_pope_20260323_175141.json` |
| adversarial | beam_search | `beam_search_pope_20260323_183055.json` |
| adversarial | dola | `dola_pope_20260323_185821.json` |
| adversarial | opera | `opera_pope_20260323_192322.json` |

Full path prefix on server: `/root/autodl-tmp/BRA_Project/logs/minitest/`

---

## One-line takeaway (for caption / text)

On **LLaVA-1.5-7B** and POPE at 3000 samples per split, **DoLa** edges **F1** on **random** (+0.0001 vs beam); **Beam Search** is strongest on **popular** and **adversarial** F1; **OPERA** does not beat the best decoding baseline on any split in this run.

Registry: see `experiment_logs/experiment_registry_latest.md` rows `pope_llava7b_*`.
