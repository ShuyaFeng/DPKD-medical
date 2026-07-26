#!/bin/bash
# Submit channel selection ablation on all 3 datasets.
# Parameters match the paper: seeds=3, te=60, se=40.
#
# Usage (on Cheaha login node):
#   cd /home/ab36/DPKD-medical/phase1_distillation/slurm
#   bash submit_sel_abl.sh

set -e

SLURM_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm"
LOG_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting ISIC channel selection ablation..."
sbatch "$SLURM_DIR/run_sel_abl_isic.sh"

echo "Submitting Kvasir channel selection ablation..."
sbatch "$SLURM_DIR/run_sel_abl_kvasir.sh"

echo "Submitting BUSI channel selection ablation..."
sbatch "$SLURM_DIR/run_sel_abl_busi.sh"

echo ""
echo "All 3 jobs submitted. Check: squeue -u ab36"
echo "Results: phase1_distillation/results/{isic,kvasir,busi}_channel_selection_ablation_results.json"
