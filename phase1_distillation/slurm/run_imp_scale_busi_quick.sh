#!/bin/bash
#SBATCH --job-name=imp_scale_busi_quick
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/imp_scale_busi_quick_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/imp_scale_busi_quick_%j.err

echo "=========================================="
echo "Quick BUSI test — importance scaling (bug fix verification)"
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

python -u canal_importance_scaling.py \
    --dataset busi \
    --seeds 1 --te 30 --se 20 \
    --epsilons "2" \
    --alphas "1.0"

echo "Finished: $(date)"
