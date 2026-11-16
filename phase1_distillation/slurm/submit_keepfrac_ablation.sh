#!/bin/bash
# Submit channel-selection threshold ablation (3 keep_frac values x 3 datasets = 9 jobs).
#
# Values tested:
#   kf05: keep_frac=0.05 (top 5%)
#   kf20: keep_frac=0.20 (top 20%)
#   kf30: keep_frac=0.30 (top 30%)
#
# Current keep_frac=0.10 already has results in:
#   results/{isic,kvasir,busi}_final_canal_results.json
#
# All runs use default budget split: fc=0.10 fi=0.05 fr=0.85
#
# Usage (on Cheaha login node):
#   cd /home/ab36/DPKD-medical/phase1_distillation/slurm
#   bash submit_keepfrac_ablation.sh

set -e

SLURM_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm"
LOG_DIR="/home/ab36/DPKD-medical/phase1_distillation/slurm_logs"
mkdir -p "$LOG_DIR"

echo "Submitting keep_frac=0.05 (top 5%)..."
sbatch "$SLURM_DIR/run_keepfrac_kf05_isic.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf05_kvasir.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf05_busi.sh"

echo "Submitting keep_frac=0.20 (top 20%)..."
sbatch "$SLURM_DIR/run_keepfrac_kf20_isic.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf20_kvasir.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf20_busi.sh"

echo "Submitting keep_frac=0.30 (top 30%)..."
sbatch "$SLURM_DIR/run_keepfrac_kf30_isic.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf30_kvasir.sh"
sbatch "$SLURM_DIR/run_keepfrac_kf30_busi.sh"

echo ""
echo "All 9 jobs submitted."
echo "Check status:  squeue -u ab36"
echo "Results:       /home/ab36/DPKD-medical/phase1_distillation/results/{isic,kvasir,busi}_final_canal_results_kf*.json"
