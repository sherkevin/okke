# Real CHORD Rebuttal Metrics, 2026-05-29

Run dir: `/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit8`

Present JSON files: 12 / 12

Boundary: rows with null values were not measured and must not be inserted into the PDF as real evidence.

## Table 1 Future Mechanism

- LLaVA POPE-Adv: {"model_task": "LLaVA POPE-Adv", "contrast": "Full-P+C", "present": true, "n": 8, "flip_rate": 0.0, "flips": 0, "corrected": 0, "harmful": 0, "metric": "POPE adversarial F1", "metric_delta": 0.0, "pc_json": "/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit8/llava_pope_adv8_pc_real.json", "full_json": "/media/data3/dengkw/chord_rebuttal_20260529/runs/real_rebuttal_20260530_limit8/llava_pope_adv8_full_k5_m3.json"}
- InstructBLIP POPE-Adv: {"model_task": "InstructBLIP POPE-Adv", "present": false, "evidence_boundary": "not run in this LLaVA-focused remote pass"}
- LLaVA CHAIR: {"model_task": "LLaVA CHAIR", "present": false, "evidence_boundary": "CHAIR captioning was not measured by pope_eval JSON"}
- InstructBLIP CHAIR: {"model_task": "InstructBLIP CHAIR", "present": false, "evidence_boundary": "not run in this LLaVA-focused remote pass"}

## Table 2 Detector Attribution

- OPERA / Past: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- Same-anchor non-CHORD: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- P+C with uniform anchors: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- P+C with random matched anchors: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- Past+Future, no Current: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- P+C real anchors: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8
- Full real anchors: F1=0.888888888888889 FP=0.25 CHAIR-S=None n=8

## Table 3 Efficiency

- OPERA / Past: total_ms=128540.06683430634 itl=9701.13711957029 vram=8.46364450454712 n=8
- Past+Current: total_ms=136716.08511791192 itl=10205.157367389578 vram=8.503136157989502 n=8
- Full CHORD: total_ms=129830.1351453606 itl=9685.463029838536 vram=8.503136157989502 n=8

## Table 4 k/m Robustness

- k=None m=0: F1=0.888888888888889 total_ms=135218.33511791192 n=8
- k=3 m=2: F1=0.888888888888889 total_ms=135538.7606942095 n=8
- k=5 m=1: F1=0.888888888888889 total_ms=134821.21920748614 n=8
- k=5 m=2: F1=0.888888888888889 total_ms=134737.806733232 n=8
- k=5 m=3: F1=0.888888888888889 total_ms=128332.3851453606 n=8
- k=5 m=4: F1=0.888888888888889 total_ms=134325.29870979488 n=8
- k=10 m=3: F1=0.888888888888889 total_ms=134071.02408795618 n=8
