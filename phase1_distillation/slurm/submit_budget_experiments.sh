#!/bin/bash
# Submit all budget split experiments.
#
# Jobs submitted:
#   1  caps diagnostic         (amperenodes,        ~3h)
#   8  ISIC sweep  array 0-7  (amperenodes-medium,  up to 2 days)
#   8  Kvasir sweep array 0-7 (amperenodes-medium,  up to 2 days)
#   8  BUSI sweep  array 0-7  (pascalnodes-medium,  up to 2 days)
#
# Usage (on Cheaha login node):
#   cd /home/ab36/DPKD-medical/phase1_distillation/slurm
#   bash submit_budget_experiments.sh

set -e

SLURM_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm"
LOG_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting caps diagnostic (amperenodes, 3h)..."
sbatch "$SLURM_DIR/run_caps_diagnostic.sh"

echo "Submitting ISIC budget sweep — 8 jobs (amperenodes-medium, 2 days)..."
sbatch "$SLURM_DIR/run_sweep_isic.sh"

echo "Submitting Kvasir budget sweep — 8 jobs (amperenodes-medium, 2 days)..."
sbatch "$SLURM_DIR/run_sweep_kvasir.sh"

echo "Submitting BUSI budget sweep — 8 jobs (pascalnodes-medium, 2 days)..."
sbatch "$SLURM_DIR/run_sweep_busi.sh"

echo ""
echo "All jobs submitted."
echo "Check status:  squeue -u ab36"
echo "Results saved: /home/ab36/DPKD-medical/phase1_distillation/results/budget_sweep_*.json"
