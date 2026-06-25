#!/bin/bash
#SBATCH --job-name=busi_canal_k3_eps2
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ialam/DPKD-medical/phase1_distillation/slurm_logs/busi_canal_k3_eps2_%j.out
#SBATCH --error=/home/ialam/DPKD-medical/phase1_distillation/slurm_logs/busi_canal_k3_eps2_%j.err

set -e

echo "=========================================="
echo "BUSI K=3/eps=2 CANAL vs uniform run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate correctly."
    exit 1
fi

cd /home/ialam/DPKD-medical/phase1_distillation

python -u busi_canal_k3_eps2.py --seeds 5 --te 60 --se 40 --epsilons "2"

echo "Finished: $(date)"
