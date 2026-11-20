#!/bin/bash
#SBATCH --job-name=student_only_kvasir
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/student_only_kvasir_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/student_only_kvasir_%j.err

# Student-only baseline (no distillation, no DP noise) — Kvasir

echo "=========================================="
echo "Student-only baseline — Kvasir"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."; exit 1
fi

cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_student_only.py \
    --dataset kvasir \
    --seeds 5 --se 40

echo "Finished: $(date)"
