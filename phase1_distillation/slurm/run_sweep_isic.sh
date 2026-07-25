#!/bin/bash
#SBATCH --job-name=sweep_isic
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sweep_isic_%A_%a.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sweep_isic_%A_%a.err

set -e

# Budget split configurations
# Format: f_caps  f_imp  f_rel  clip_imp_type  clip_caps_type
case $SLURM_ARRAY_TASK_ID in
  0) F_CAPS=0.10; F_IMP=0.45; F_REL=0.45; CIMP=avg; CCAPS=avg ;;  # PRIMARY: paper split + avg
  1) F_CAPS=0.10; F_IMP=0.20; F_REL=0.70; CIMP=avg; CCAPS=avg ;;  # lighter importance
  2) F_CAPS=0.10; F_IMP=0.30; F_REL=0.60; CIMP=avg; CCAPS=avg ;;  # moderate importance
  3) F_CAPS=0.10; F_IMP=0.10; F_REL=0.80; CIMP=avg; CCAPS=avg ;;  # minimal importance
  4) F_CAPS=0.05; F_IMP=0.45; F_REL=0.50; CIMP=avg; CCAPS=avg ;;  # lighter caps
  5) F_CAPS=0.15; F_IMP=0.35; F_REL=0.50; CIMP=avg; CCAPS=avg ;;  # heavier caps
  6) F_CAPS=0.10; F_IMP=0.45; F_REL=0.45; CIMP=p99; CCAPS=avg ;;  # worst-case imp
  7) F_CAPS=0.10; F_IMP=0.45; F_REL=0.45; CIMP=avg; CCAPS=p99 ;;  # worst-case caps
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "=========================================="
echo "ISIC budget sweep — array task $SLURM_ARRAY_TASK_ID"
echo "Config: f_caps=$F_CAPS f_imp=$F_IMP f_rel=$F_REL clip_imp=$CIMP clip_caps=$CCAPS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."
    exit 1
fi

mkdir -p /home/ab36/DPKD-medical/phase1_distillation/slurm_logs
cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_budget_sweep.py \
    --dataset isic \
    --f_caps $F_CAPS --f_imp $F_IMP --f_rel $F_REL \
    --clip_imp_type $CIMP --clip_caps_type $CCAPS \
    --epsilons "1,2,4,8,16,32" \
    --seeds 5 --te 60 --se 40

echo "Finished: $(date)"
