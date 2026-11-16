#!/bin/bash
# Submit budget-split ablation experiments (3 new splits, all 3 datasets each).
#
# Splits tested:
#   split1: fc=0.05  fi=0.05  fr=0.90  (more budget to release)
#   split2: fc=0.20  fi=0.10  fr=0.70  (more budget to privatization)
#   split3: fc=0.33  fi=0.33  fr=0.34  (equal split)
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

echo "Submitting budget ablation split1 (fc=0.05 fi=0.05 fr=0.90)..."
sbatch "$SLURM_DIR/run_budget_abl_split1.sh"

echo "Submitting budget ablation split2 (fc=0.20 fi=0.10 fr=0.70)..."
sbatch "$SLURM_DIR/run_budget_abl_split2.sh"

echo "Submitting budget ablation split3 (fc=0.33 fi=0.33 fr=0.34)..."
sbatch "$SLURM_DIR/run_budget_abl_split3.sh"

echo ""
echo "All 3 jobs submitted (each runs isic -> kvasir -> busi sequentially)."
echo "Check status:  squeue -u ab36"
echo "Results:       /home/ab36/DPKD-medical/phase1_distillation/results/{isic,kvasir,busi}_final_canal_results_fc*.json"
