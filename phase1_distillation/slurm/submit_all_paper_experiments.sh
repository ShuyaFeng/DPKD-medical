#!/bin/bash
# ============================================================
# Submit ALL remaining paper experiments (local TinyUNet pipeline).
#
# 5 jobs, all independent, submit together. ~9-11 GPU-h total;
# at 5 parallel slots they finish well within a day.
#
# Run from phase1_distillation/slurm/ on Cheaha after:
#   cd /data/user/home/ialam/DPKD-medical && git pull
# ============================================================
set -e
cd "$(dirname "$0")"

# SLURM --output dirs must exist before jobs launch
mkdir -p /home/fengs/DPKD-medical/slurm_out

# conda env that has torch + numpy + PIL + skimage.
# Override if yours is named differently:  CONDA_ENV=myenv bash submit_all_paper_experiments.sh
export CONDA_ENV="${CONDA_ENV:-pytorch}"
echo "Using conda env: $CONDA_ENV  (override with CONDA_ENV=...)"
echo ""

echo "=== P1: core contributions (channel-pruning + per-teacher) ==="
J1=$(sbatch --parsable run_pate_pruning_joint.sh)
echo "  pate_pruning_joint   -> $J1   (K×keep heatmap)"
J2=$(sbatch --parsable run_per_teacher_importance.sh)
echo "  per_teacher_importance -> $J2 (Jaccard + shared/union/vote)"

echo "=== P2: verification + alpha ==="
J3=$(sbatch --parsable run_shared_vs_scratch.sh)
echo "  shared_vs_scratch    -> $J3   (channel-alignment causal test)"
J4=$(sbatch --parsable run_alpha_sweep.sh)
echo "  alpha_sweep          -> $J4   (label-leak / §3.5)"

echo "=== P3: single-teacher pruning ceiling ==="
J5=$(sbatch --parsable run_single_teacher_pruning.sh)
echo "  single_teacher_pruning -> $J5 (keep down to 2%)"

echo ""
echo "Submitted 5 jobs: $J1 $J2 $J3 $J4 $J5"
echo "Monitor:  squeue -u \$USER"
echo ""
echo "When done, the 5 result JSONs live in the repo:"
echo "  drive_pate_pruning_joint_results.json"
echo "  drive_per_teacher_importance_results.json"
echo "  drive_shared_vs_scratch_init_results.json"
echo "  drive_alpha_sweep_results.json"
echo "  drive_single_teacher_pruning_results.json"
