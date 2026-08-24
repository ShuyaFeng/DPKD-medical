#!/bin/bash
# Submit final CANAL vs Uniform experiments (all 3 datasets).
#
# Setup: fc=0.10, fi=0.05, fr=0.85, clip_imp=p90, eps={1,2,4,8}, 5 seeds, K=3
#
# Usage (on Cheaha login node):
#   cd /home/ialam/DPKD-medical/phase1_distillation/slurm
#   bash submit_final_experiments.sh

set -e

SLURM_DIR="/home/ialam/DPKD-medical/phase1_distillation/slurm"
LOG_DIR="/home/ialam/DPKD-medical/phase1_distillation/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting final CANAL experiment — ISIC (amperenodes-medium, 10h)..."
sbatch "$SLURM_DIR/run_final_isic.sh"

echo "Submitting final CANAL experiment — Kvasir (amperenodes-medium, 10h)..."
sbatch "$SLURM_DIR/run_final_kvasir.sh"

echo ""
echo "All 2 jobs submitted."
echo "Check status:  squeue -u ialam"
echo "Results:       /home/ialam/DPKD-medical/phase1_distillation/results/{isic,kvasir}_final_canal_results.json"
