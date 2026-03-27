# Experiment Registry

Last updated: `2026-03-23` (theory-aligned llava fixed-slice matrix)

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
- For matrix runs, record both:
  - one parent row for the shared matrix runner/log;
  - one child row per `model + method + split` sub-experiment as soon as its JSON lands or its status changes.

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
| `llava_verifier_pope_random_20_round5_smoke` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `N/A (single direct command)` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_smoke/uniground_v2_llava-v1.5-7b_pope_llava_verifier_pope_random_20_round5_smoke_20260323_201938.json` | Post-theory-alignment smoke for evidence-confidence backoff; `Acc=0.95`, `F1=0.9474` on 20 samples. |
| `llava_pope_round5_matrix_200` | `llava-v1.5-7b` | `POPE` | `random+popular+adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_pope_round5_matrix_200_manifest.tsv` | `see child rows below` | Fixed-slice round5 matrix; `frontier` vs `verifier`, `mini_test=200`, evidence-confidence backoff. |
| `llava_frontier_pope_random_200_round5` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_frontier_pope_random_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_frontier_pope_random_200_round5_20260323_202150.json` | Acc 0.885, F1 0.8783. |
| `llava_verifier_pope_random_200_round5` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_verifier_pope_random_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_verifier_pope_random_200_round5_20260323_202315.json` | Acc 0.89, F1 0.8842. |
| `llava_frontier_pope_popular_200_round5` | `llava-v1.5-7b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_frontier_pope_popular_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_frontier_pope_popular_200_round5_20260323_202440.json` | Acc 0.86, F1 0.8557. |
| `llava_verifier_pope_popular_200_round5` | `llava-v1.5-7b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_verifier_pope_popular_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_verifier_pope_popular_200_round5_20260323_202605.json` | Acc 0.875, F1 0.8705. |
| `llava_frontier_pope_adversarial_200_round5` | `llava-v1.5-7b` | `POPE` | `adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_frontier_pope_adversarial_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_frontier_pope_adversarial_200_round5_20260323_202730.json` | Acc 0.795, F1 0.8019. |
| `llava_verifier_pope_adversarial_200_round5` | `llava-v1.5-7b` | `POPE` | `adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/llava_verifier_pope_adversarial_200_round5.log` | `/root/autodl-tmp/BRA_Project/logs/pope_verifier_round5_matrix/uniground_v2_llava-v1.5-7b_pope_llava_verifier_pope_adversarial_200_round5_20260323_202853.json` | Acc 0.825, F1 0.8276. |
| `llava_theory_align_matrix_100_20260323` | `llava-v1.5-7b` | `POPE` | `random+popular+adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/theory_align_matrix_manifest.tsv` | `see child rows below` | Post-refactor fixed-slice matrix after task-protocol runtime, explicit answer-label control, and universal training-contract alignment. |
| `llava_base_pope_random_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/base_random.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_232127.json` | Base slice for direct comparison; Acc 0.85, F1 0.8571. |
| `llava_frontier_pope_random_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/frontier_random.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_frontier_random_100_20260323_232223.json` | Old frontier path after protocol refactor; Acc 0.86, F1 0.86. |
| `llava_verifier_pope_random_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `random` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/verifier_random.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_verifier_random_100_20260323_232319.json` | Rebuilt explicit answer-label verifier; Acc 0.87, F1 0.8687. |
| `llava_base_pope_popular_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/base_popular.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_232420.json` | Base slice for direct comparison; Acc 0.87, F1 0.8738. |
| `llava_frontier_pope_popular_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/frontier_popular.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_frontier_popular_100_20260323_232515.json` | Old frontier path after protocol refactor; Acc 0.88, F1 0.8776. |
| `llava_verifier_pope_popular_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `popular` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/verifier_popular.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_verifier_popular_100_20260323_232610.json` | Rebuilt explicit answer-label verifier; Acc 0.89, F1 0.8866. |
| `llava_base_pope_adversarial_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/base_adversarial.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_232710.json` | Base slice for direct comparison; Acc 0.78, F1 0.8036. |
| `llava_frontier_pope_adversarial_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/frontier_adversarial.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_frontier_adversarial_100_20260323_232806.json` | Old frontier path after protocol refactor; Acc 0.8, F1 0.8113. |
| `llava_verifier_pope_adversarial_100_theoryalign` | `llava-v1.5-7b` | `POPE` | `adversarial` | `done` | `/root/autodl-tmp/BRA_Project/logs/theory_align_slices_20260323/verifier_adversarial.log` | `/root/autodl-tmp/BRA_Project/logs/uniground_v2/uniground_v2_llava-v1.5-7b_pope_theoryalign_verifier_adversarial_100_20260323_232901.json` | Rebuilt explicit answer-label verifier; Acc 0.83, F1 0.835. |

## Related Baseline / Matrix Logs

These are not all our UniGround result rows, but they are relevant baseline / matrix experiment assets currently observed on the remote server.

| experiment_name | status | remote_log_path | remote_result_json | notes |
| --- | --- | --- | --- | --- |
| `pope_baseline_pending_gpu0_20260323_202913` | `stopped by user` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.log` | `see manifest TSV below` | Stopped intentionally after scope correction to avoid spending GPU0 on out-of-scope models/methods. Recoverable completed child assets are listed in `experiment_logs/baseline_required_before_325_20260324.md`. |
| `pope_baseline_pending_gpu0_20260323_202913.manifest` | `partial` | same parent log | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.manifest.tsv` | Manifest currently contains 6 completed child rows: `qwen3-vl-8b/random beam_search,dola,opera` and `qwen3-vl-8b/popular base,beam_search,dola`. |
| `chair_baseline_pending_gpu0_20260324_101840` | `cancelled before launch` | `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.log` | `see manifest TSV below` | Cancelled after baseline policy was corrected from `beam_search/dola/opera` to the `3.25`-required `Base + VCD + DAMO` package. Watcher log remains for traceability. |
| `chair_baseline_pending_gpu0_20260324_101840.manifest` | `not_created` | same parent log | `/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.manifest.tsv` | Queue never launched; watcher was killed before parent dependency finished. |
| `llava_damo_pope_random_smoke_20260324_123557` | `done` | `N/A (direct smoke command)` | `/root/autodl-tmp/BRA_Project/logs/minitest/damo_pope_20260324_123557.json` | DAMO integration smoke after GitHub import; `llava-v1.5-7b`, `POPE random`, `mini_test=2`. |
| `instructblip_damo_pope_random_smoke_20260324_123619` | `done` | `N/A (direct smoke command)` | `/root/autodl-tmp/BRA_Project/logs/minitest/damo_pope_20260324_123619.json` | DAMO integration smoke after GitHub import; `instructblip-7b`, `POPE random`, `mini_test=2`. |
| `baseline_pre325_gpu0_20260324_123920` | `paused by user` | `/root/autodl-tmp/BRA_Project/logs/baseline_pre325/baseline_pre325_gpu0_20260324_123920.log` | `see manifest /root/autodl-tmp/BRA_Project/logs/baseline_pre325/baseline_pre325_gpu0_20260324_123920.manifest.tsv` | Baseline-only queue aligned to the `3.25` deadline. Paused because GPU0 was reassigned to other work; manifest currently records the completed `llava-v1.5-7b / pope / random / vcd` cell. |
| `baseline_continue_gpu0_20260324_173547` | `stopped / superseded` | `/root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173547.log` | `see manifest /root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173547.manifest.tsv` | First continuation queue launch. Stopped after launch because skip logic bug allowed a completed cell to restart; superseded by `baseline_continue_gpu0_20260324_173737`. |
| `baseline_continue_gpu0_20260324_173737` | `paused by user` | `/root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173737.log` | `see manifest /root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173737.manifest.tsv` | Continuation queue for requested `llava-v1.5-7b + instructblip-7b`, methods `base + dola + opera`. Paused because GPU0 was reassigned; partial POPE progress and completed child JSONs remain in the manifest/log. |
| `baseline_continue_gpu0_20260324_211848` | `cancelled before launch` | `/root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_211848.log` | `see manifest /root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_211848.manifest.tsv` | `MMBench`-only continuation queue for `llava-v1.5-7b + instructblip-7b`, methods `base + dola + opera`. Cancelled because GPU0 was reassigned before the watcher launched the queue. |
| `pope_full_matrix_20260323_111111` | `stopped / partially completed` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `multiple child JSONs; see rows below` | Original GPU0 baseline matrix. Stopped after user requested switching away from Qwen continuation to LLaVA-first execution. |
| `pope_qwen8b_base_random_20260323_125009` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_125009.json` | First completed child result from the Qwen matrix; `sample_count=3000`, `F1=0.9189`. |
| `pope_qwen8b_beam_search_random_20260323_111111` | `stopped` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `N/A` | Started after Qwen base completed, then manually stopped when user switched priority to LLaVA. |
| `pope_qwen8b_dola_random_20260323_111111` | `not_started` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `N/A` | Was queued in the original matrix runner but never started. |
| `pope_qwen8b_opera_random_20260323_111111` | `not_started` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_full_matrix_20260323_111111.log` | `N/A` | Was queued in the original matrix runner but never started. |
| `pope_llava_matrix_20260323_134023` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `multiple child JSONs; see rows below` | Dedicated LLaVA-only POPE matrix on GPU0 for the `random` split; all four methods completed. |
| `pope_llava7b_base_random_20260323_134023` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_140310.json` | Random split, base method. |
| `pope_llava7b_beam_search_random_20260323_134023` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/beam_search_pope_20260323_144348.json` | Random split, beam search method. |
| `pope_llava7b_dola_random_20260323_134023` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/dola_pope_20260323_151139.json` | Random split, DoLa method. |
| `pope_llava7b_opera_random_20260323_134023` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_matrix_20260323_134023.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260323_153647.json` | Random split, OPERA method. |
| `pope_llava_splits_followup_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `multiple child JSONs; see rows below` | Follow-up GPU0 queue for `popular` and `adversarial`; all eight child runs completed. |
| `pope_llava7b_base_popular_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_155904.json` | Popular split, base method. |
| `pope_llava7b_beam_search_popular_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/beam_search_pope_20260323_163906.json` | Popular split, beam search method. |
| `pope_llava7b_dola_popular_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/dola_pope_20260323_170619.json` | Popular split, DoLa method. |
| `pope_llava7b_opera_popular_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260323_173110.json` | Popular split, OPERA method. |
| `pope_llava7b_base_adversarial_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260323_175141.json` | Adversarial split, base method. |
| `pope_llava7b_beam_search_adversarial_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/beam_search_pope_20260323_183055.json` | Adversarial split, beam search method. |
| `pope_llava7b_dola_adversarial_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/dola_pope_20260323_185821.json` | Adversarial split, DoLa method. |
| `pope_llava7b_opera_adversarial_20260323_145038` | `done` | `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_llava_splits_followup_20260323_145038.log` | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260323_192322.json` | Adversarial split, OPERA method. |

## POPE Decoding Baseline — Full Matrix Status

Single source for **which cells are still missing** (model × split × method, 3000 samples):

- **`experiment_logs/pope_baseline_experiment_matrix_20260323.md`**

Summary:

| Model | POPE baseline cells done / total (4×3) |
| --- | --- |
| `llava-v1.5-7b` | **12 / 12** |
| `qwen3-vl-8b` | **1 / 12** (plus `beam_search`+`random` **stopped**, needs rerun) |
| `instructblip-7b` | **0 / 12** |
| `qwen3-vl-4b` | **out-of-scope** |
| `qwen3-vl-2b` | **out-of-scope** |

## Baseline Result Storage Convention

- Shared matrix runner logs live under:
  - `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/`
- Per-subexperiment result JSONs land under:
  - `/root/autodl-tmp/BRA_Project/logs/minitest/`
- Naming convention for result JSONs:
  - `{method}_{dataset}_{timestamp}.json`
- When a child run finishes, add or update its row in this file immediately with:
  - `experiment_name`
  - `status`
  - shared `remote_log_path`
  - exact `remote_result_json`
  - short note if stopped / superseded / resumed

## Notes

- **GPU0 contention:** If `llava_pope_round5_matrix_200` (or any other job) is also bound to GPU0, either stop it first or reschedule — two heavy runners on the same device will corrupt timing and may OOM.
- Some older experiments do not have a dedicated remote log file because they were launched through an interactive wrapper that only streamed stdout back to the local terminal transcript.
- From this point onward, new experiments should always create a dedicated remote log file so this registry can stay complete.
