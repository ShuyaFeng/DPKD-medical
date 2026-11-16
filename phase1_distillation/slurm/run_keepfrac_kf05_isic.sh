#!/bin/bash
#SBATCH --job-name=kf05_isic
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/keepfrac_kf05_isic_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/keepfrac_kf05_isic_%j.err

# Channel selection threshold ablation: keep_frac=0.05 (top 5%) — ISIC

echo "=========================================="
echo "Keep-frac ablation kf=0.05 — ISIC"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."; exit 1
fi

cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_final_experiment.py \
    --dataset isic \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --keep_frac 0.05 \
    --suffix kf05

echo "Finished: $(date)"
