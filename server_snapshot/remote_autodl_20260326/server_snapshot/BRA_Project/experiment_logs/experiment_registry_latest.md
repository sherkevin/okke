# Experiment Registry

Last updated: `2026-03-23`

Purpose:

- Record each experiment with a stable name.
- Record the remote log path and remote JSON result path whenever available.
- Keep one continuously updated registry instead of scattering paths across chats and ad hoc notes.

Update rule for future experiments:

- Add one row immediately when an experiment is launched.
- Fill `remote_log_path` at launch time.
- Fill `remote_result_json` when the JSON lands.
- If an experiment is cancelled or superseded, do not delete it; mark `status` accordingly.
- For all future launches, prefer creating a dedicated remote log file instead of relying only on terminal output.

Remote root used below:

- `/root/autodl-tmp/BRA_Project`

## Tracked Experiments

| experiment_name | host_model | dataset | split | status | remote_log_path | remote_result_json | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `uniground_qwen2b_pope_checkpoint_probe_20260322_224217` | `qwen3-vl-8b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-8b_pope_checkpoint_probe_20260322_224217.json` | Early probe before method tightening. |
| `uniground_qwen2b_chair_checkpoint_probe_20260322_224534` | `qwen3-vl-8b` | `CHAIR` | `default` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-8b_chair_checkpoint_probe_20260322_224534.json` | Early probe before POPE-focused redesign. |
| `uniground_qwen2b_pope_checkpoint_probe_20260323_093037` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_checkpoint_probe_20260323_093037.json` | Pre-fix checkpoint probe. |
| `uniground_qwen2b_chair_checkpoint_probe_20260323_093138` | `qwen3-vl-2b` | `CHAIR` | `default` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_chair_checkpoint_probe_20260323_093138.json` | Pre-fix checkpoint probe. |
| `uniground_qwen2b_triggerfix_hardcoded_20260323_094749` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_triggerfix_hardcoded_20260323_094749.json` | Hardcoded scorer after trigger fix. |
| `uniground_qwen2b_triggerfix_checkpoint_20260323_095014` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_triggerfix_checkpoint_20260323_095014.json` | Checkpoint scorer after trigger fix. |
| `uniground_qwen2b_aggressive_grid_20260323_095342` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_aggressive_grid_20260323_095342.json` | Grid-only aggressive control probe. |
| `uniground_qwen2b_detector_probe_20260323_095717` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_detector_probe_20260323_095717.json` | Early detector probe. |
| `uniground_qwen2b_object_hypothesis_20260323_102420` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_hypothesis_20260323_102420.json` | Object-hypothesis bridge, grid regions. |
| `uniground_qwen2b_object_hypothesis_v2_20260323_103121` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_hypothesis_v2_20260323_103121.json` | Object-hypothesis bridge after token mapping fix. |
| `uniground_qwen2b_object_detector_query_20260323_103928` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_20260323_103928.json` | First object-specific detector success signal. |
| `uniground_qwen2b_gpu1_batched_20_20260323_111805` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_batched_20260323_111805.json` | Batch crop encoding optimization check. |
| `uniground_qwen2b_gpu1_fullgpu20_20260323_112538` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_fullgpu20_20260323_112538.json` | Full-GPU 20-sample run. |
| `uniground_qwen2b_gpu1_fullgpu100_20260323_112642` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_fullgpu100_20260323_112642.json` | Full-GPU 100-sample run, batch=1. |
| `uniground_qwen2b_gpu1_fullgpu20_b8_20260323_114021` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_fullgpu20_b8_20260323_114021.json` | First batch=8 full-GPU test before padding/accounting fix. |
| `uniground_qwen2b_gpu1_fullgpu20_b8_fix_20260323_114255` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_fullgpu20_b8_fix_20260323_114255.json` | Corrected batch=8 validation run. |
| `uniground_qwen2b_gpu1_fullgpu100_b8_20260323_114352` | `qwen3-vl-2b` | `POPE` | `random` | `done` | `N/A` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-2b_pope_pope_object_detector_query_gpu1_fullgpu100_b8_20260323_114352.json` | Full-GPU batch=8 run; faster but lower F1 than batch=1. |
| `uniground_qwen8b_gpu1_fullgpu3000_random_20260323_131916` | `qwen3-vl-8b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/qwen8b_gpu1_fullgpu3000_b1.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-8b_pope_pope_object_detector_query_qwen8b_gpu1_fullgpu3000_b1_20260323_131916.json` | Main Qwen8B full run; Acc 0.9153, F1 0.9090. |
| `uniground_qwen8b_gpu1_fullgpu3000_popular_20260323_134413` | `qwen3-vl-8b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_queue/qwen8b_gpu1_popular_adversarial.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_qwen3-vl-8b_pope_pope_object_detector_query_qwen8b_gpu1_fullgpu3000_popular_b1_20260323_134413.json` | Popular split completed before host switch. |
| `uniground_qwen8b_gpu1_fullgpu3000_adversarial` | `qwen3-vl-8b` | `POPE` | `adversarial` | `stopped` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_queue/qwen8b_gpu1_popular_adversarial.log` | `N/A` | Stopped after user decided to switch away from Qwen3 host family. |
| `uniground_llava7b_gpu1_smoke_20260323_140115` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `N/A (launched before dedicated remote log was added)` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_llava-v1.5-7b_pope_pope_object_detector_query_llava7b_gpu1_smoke_20260323_140115.json` | First successful LLaVA UniGround smoke test. |
| `uniground_llava7b_gpu1_fullgpu3000_random_20260323_141655` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `N/A (launched from wrapper; no dedicated remote log file)` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe/uniground_v2_llava-v1.5-7b_pope_pope_object_detector_query_llava7b_gpu1_fullgpu3000_b1_20260323_141655.json` | Main LLaVA full run on GPU1. |

## Related Baseline / Matrix Logs

These are not all our UniGround result rows, but they are relevant experiment assets currently observed on the remote server.

| experiment_name | status | remote_log_path | remote_result_json | notes |
| --- | --- | --- | --- | --- |
| `pope_full_matrix_20260323_111111` | `running / partially completed` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `TBD / not registered here` | Qwen3VL baseline matrix on GPU0. |
| `pope_llava_matrix_20260323_134023` | `running / partially completed` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `TBD / not registered here` | LLaVA baseline matrix. |

## Notes

- Some older experiments do not have a dedicated remote log file because they were launched through an interactive wrapper that only streamed stdout back to the local terminal transcript.
- From this point onward, new experiments should always create a dedicated remote log file so this registry can stay complete.
