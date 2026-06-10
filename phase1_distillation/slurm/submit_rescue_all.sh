#!/bin/bash
# ============================================================
# MASTER submitter for the WACV "save CANAL or fall back to PATE"
# rescue experiment suite.
#
# Submits 5 jobs with SLURM dependencies so each waits on the right
# upstream. Total wall-clock ~10-12 hours given typical Cheaha queue.
#
# After submission, monitor with: squeue -u $USER
#
# What each job produces:
#   EXP-1 -> drive_active_set_R_results.json          (R at top-K)
#   EXP-2 -> drive_canal_ablation_results.json        (4x3x5 grid)
#   EXP-3 -> spatial_saliency.pt + drive_spatial_saliency_results.json
#   EXP-4 -> drive_spatial_wf_results.json            (3x3x5 grid)
#   EXP-5 -> drive_pate_spatial_joint_results.json    (2x3x5 grid)
# ============================================================

set -e
cd "$(dirname "$0")"

echo "Submitting EXP-1 (active-set R analysis, ~30 min)..."
J1=$(sbatch --parsable run_rescue_exp1_activeR.sh)
echo "  job $J1"

echo "Submitting EXP-2 (CANAL ablation, ~6-8h) — runs in parallel with EXP-1..."
J2=$(sbatch --parsable run_rescue_exp2_canal_ablation.sh)
echo "  job $J2"

echo "Submitting EXP-3 (spatial saliency, ~10 min)..."
J3=$(sbatch --parsable run_rescue_exp3_spatial_saliency.sh)
echo "  job $J3"

echo "Submitting EXP-4 (spatial-WF, ~6-8h) — depends on EXP-3..."
J4=$(sbatch --parsable --dependency=afterok:$J3 run_rescue_exp4_spatial_wf.sh)
echo "  job $J4  (waits on $J3)"

echo "Submitting EXP-5 (PATE × spatial-WF, ~4-6h) — depends on EXP-3..."
J5=$(sbatch --parsable --dependency=afterok:$J3 run_rescue_exp5_pate_spatial.sh)
echo "  job $J5  (waits on $J3)"

echo ""
echo "All submitted. Watch with:  squeue -u \$USER"
echo ""
echo "When all 5 finish, results JSON live in:"
echo "  /data/user/home/ialam/DPKD-medical/phase1_distillation/"
echo "  drive_active_set_R_results.json"
echo "  drive_canal_ablation_results.json"
echo "  drive_spatial_saliency_results.json"
echo "  drive_spatial_wf_results.json"
echo "  drive_pate_spatial_joint_results.json"
