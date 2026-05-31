# Real CHORD Rebuttal Metrics, 2026-05-29

Run dir: `/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit2`

Present JSON files: 12 / 12

Boundary: rows with null values were not measured and must not be inserted into the PDF as real evidence.

## Table 1 Future Mechanism

- LLaVA POPE-Adv: {"model_task": "LLaVA POPE-Adv", "contrast": "Full-P+C", "present": true, "n": 2, "flip_rate": 0.0, "flips": 0, "corrected": 0, "harmful": 0, "metric": "POPE adversarial F1", "metric_delta": 0.0, "pc_json": "/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit2/llava_pope_adv2_pc_real.json", "full_json": "/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit2/llava_pope_adv2_full_k5_m3.json"}
- InstructBLIP POPE-Adv: {"model_task": "InstructBLIP POPE-Adv", "present": false, "evidence_boundary": "not run in this LLaVA-focused remote pass"}
- LLaVA CHAIR: {"model_task": "LLaVA CHAIR", "present": false, "evidence_boundary": "CHAIR captioning was not measured by pope_eval JSON"}
- InstructBLIP CHAIR: {"model_task": "InstructBLIP CHAIR", "present": false, "evidence_boundary": "not run in this LLaVA-focused remote pass"}

## Table 2 Detector Attribution

- OPERA / Past: F1=1.0 FP=0.0 CHAIR-S=None n=2
- Same-anchor non-CHORD: F1=1.0 FP=0.0 CHAIR-S=None n=2
- P+C with uniform anchors: F1=1.0 FP=0.0 CHAIR-S=None n=2
- P+C with random matched anchors: F1=1.0 FP=0.0 CHAIR-S=None n=2
- Past+Future, no Current: F1=1.0 FP=0.0 CHAIR-S=None n=2
- P+C real anchors: F1=1.0 FP=0.0 CHAIR-S=None n=2
- Full real anchors: F1=1.0 FP=0.0 CHAIR-S=None n=2

## Table 3 Efficiency

- OPERA / Past: total_ms=139775.8168950677 itl=12706.892445006153 vram=8.462521076202393 n=2
- Past+Current: total_ms=145602.3315967694 itl=12472.302872433582 vram=8.501898288726807 n=2
- Full CHORD: total_ms=141558.95545367897 itl=12104.723223061725 vram=8.501898288726807 n=2

## Table 4 k/m Robustness

- k=None m=0: F1=1.0 total_ms=137195.3315967694 n=2
- k=3 m=2: F1=1.0 total_ms=132942.8943581879 n=2
- k=5 m=1: F1=1.0 total_ms=139431.26649037004 n=2
- k=5 m=2: F1=1.0 total_ms=139907.26401098073 n=2
- k=5 m=3: F1=1.0 total_ms=133151.95545367897 n=2
- k=5 m=4: F1=1.0 total_ms=140068.16880125552 n=2
- k=10 m=3: F1=1.0 total_ms=132941.2304451689 n=2
