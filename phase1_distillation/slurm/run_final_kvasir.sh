#!/bin/bash
#SBATCH --job-name=final_canal_kvasir
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/final_canal_kvasir_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/final_canal_kvasir_%j.err

echo "=========================================="
echo "Final CANAL experiment — Kvasir"
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

python -u canal_final_experiment.py \
    --dataset kvasir \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8"

echo "Finished: $(date)"
