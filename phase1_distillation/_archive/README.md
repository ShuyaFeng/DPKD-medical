# Archived phase-1 experiments

Superseded code / results moved out of `phase1_distillation/` on 2026-06-16
to keep only files tied to the **current** experiments (PATE × channel-pruning,
sample-once + private calibration, DP-SGD baseline). Nothing here is read by a
kept script; everything is recoverable from git history. Grouped by why:

## Old channel×spatial water-filling line (pre-PATE headline)
The original "joint-WF + threshold" story, before the headline moved to PATE
multi-teacher + pruning.
- `drive_wf_threshold.*` — WF + release-threshold sweep
- `drive_canal_ablation.py` — early CANAL ablation (superseded by
  `drive_compare_pate_canal_uniform` + `drive_pate_canal_combined`)
- `drive_student_adapter.*`, `drive_tradeoff_updated.*`, `drive_k1_and_uniform_thr.*`

## Spatial-WF line (superseded by channel pruning)
Spatial allocation was the WACV "rescue" experiment; channel pruning replaced it.
- `drive_spatial_wf.*`, `drive_pate_spatial_joint.*`, `drive_spatial_saliency.*`

## Superseded early sweeps
Replaced by `drive_pate_K_saturation` (K sweep, 5 seeds) and the joint sweep.
- `drive_student_distill_5seed.*`, `drive_student_distill_multiseed.*`
- `drive_sensitivity_sweep.*`
- `drive_per_channel_delta_analysis.py` + `drive_per_channel_delta_results.json`

## Old cluster-path plotting
- `plot_privacy_utility.py` — reads `results/GKD_V2_*` from the mmseg cluster
  pipeline, not the local TinyUNet experiments.

## Stale outputs (producing script kept as a core import)
- `drive_pate_poc_results.json`, `drive_pate_poc.png` — early PoC run, superseded
  by `drive_pate_K_saturation_results.json`. (`drive_pate_poc.py` stays: it is a
  shared library imported by the current experiment scripts.)

## NOT archived (still wired into current pipeline)
- `drive_pate_5seed.*` — its result JSON is read by
  `drive_compare_pate_canal_uniform.py`, so it is an upstream dependency.
- `drive_per_teacher_importance.py`, `drive_shared_vs_scratch_init.py`,
  `drive_active_set_R.py` — un-run exploratory scripts kept by request.
