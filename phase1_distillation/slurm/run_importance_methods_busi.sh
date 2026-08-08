#!/bin/bash
#SBATCH --job-name=imp_methods_busi
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-18:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/imp_methods_busi_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/imp_methods_busi_%j.err

echo "=========================================="
echo "Importance method ablation (grad_energy / running_var / weight_mag) — BUSI"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."
    exit 1
fi

cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_importance_methods.py \
    --dataset busi \
    --seeds 3 --te 60 --se 40 \
    --epsilons "0.5,1,2,4,6"

echo "Finished: $(date)"
