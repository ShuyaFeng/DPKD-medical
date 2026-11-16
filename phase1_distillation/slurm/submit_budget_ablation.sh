#!/bin/bash
# Submit budget-split ablation experiments (3 splits x 3 datasets = 9 jobs).
#
# Splits tested:
#   s1: fc=0.05  fi=0.05  fr=0.90  (more budget to release)
#   s2: fc=0.20  fi=0.10  fr=0.70  (more budget to privatization)
#   s3: fc=0.33  fi=0.33  fr=0.34  (equal split)
#
# Current split fc=0.10 fi=0.05 fr=0.85 already has results in:
#   results/{isic,kvasir,busi}_final_canal_results.json
#
# Usage (on Cheaha login node):
#   cd /home/ab36/DPKD-medical/phase1_distillation/slurm
#   bash submit_budget_ablation.sh

set -e

SLURM_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm"
LOG_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting split1 (fc=0.05 fi=0.05 fr=0.90)..."
sbatch "$SLURM_DIR/run_budget_abl_s1_isic.sh"
sbatch "$SLURM_DIR/run_budget_abl_s1_kvasir.sh"
sbatch "$SLURM_DIR/run_budget_abl_s1_busi.sh"

echo "Submitting split2 (fc=0.20 fi=0.10 fr=0.70)..."
sbatch "$SLURM_DIR/run_budget_abl_s2_isic.sh"
sbatch "$SLURM_DIR/run_budget_abl_s2_kvasir.sh"
sbatch "$SLURM_DIR/run_budget_abl_s2_busi.sh"

echo "Submitting split3 (fc=0.33 fi=0.33 fr=0.34)..."
sbatch "$SLURM_DIR/run_budget_abl_s3_isic.sh"
sbatch "$SLURM_DIR/run_budget_abl_s3_kvasir.sh"
sbatch "$SLURM_DIR/run_budget_abl_s3_busi.sh"

echo ""
echo "All 9 jobs submitted."
echo "Check status:  squeue -u ab36"
echo "Results:       /home/ab36/DPKD-medical/phase1_distillation/results/{isic,kvasir,busi}_final_canal_results_fc*.json"
