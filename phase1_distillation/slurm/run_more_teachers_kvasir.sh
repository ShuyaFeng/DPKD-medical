#!/bin/bash
#SBATCH --job-name=more_teach_kvasir
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-14:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/more_teachers_kvasir_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/more_teachers_kvasir_%j.err

echo "=========================================="
echo "More teachers K=3,5,7 — Kvasir"
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

python -u canal_more_teachers.py \
    --dataset kvasir \
    --seeds 3 --te 60 --se 40 \
    --epsilons "0.5,1,2,4,6" \
    --k_values "3,5,7"

echo "Finished: $(date)"
