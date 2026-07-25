#!/bin/bash
#SBATCH --job-name=sweep_busi
#SBATCH --partition=pascalnodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sweep_busi_%A_%a.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sweep_busi_%A_%a.err

set -e

case $SLURM_ARRAY_TASK_ID in
  0) F_CAPS=0.05; CCAPS=avg ;;
  1) F_CAPS=0.10; CCAPS=avg ;;
  2) F_CAPS=0.15; CCAPS=avg ;;
  3) F_CAPS=0.20; CCAPS=avg ;;
  4) F_CAPS=0.05; CCAPS=p99 ;;
  5) F_CAPS=0.10; CCAPS=p99 ;;
  6) F_CAPS=0.15; CCAPS=p99 ;;
  7) F_CAPS=0.20; CCAPS=p99 ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "=========================================="
echo "BUSI budget sweep — array task $SLURM_ARRAY_TASK_ID"
echo "Config: f_caps=$F_CAPS clip_caps=$CCAPS"
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
    --dataset busi \
    --f_caps $F_CAPS \
    --clip_caps_type $CCAPS \
    --epsilons "1,2,4,8,16,32" \
    --seeds 7 --te 60 --se 60

echo "Finished: $(date)"
