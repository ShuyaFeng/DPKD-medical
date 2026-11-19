#!/bin/bash
#SBATCH --job-name=attack_isic
#SBATCH --partition=pascalnodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/attack_isic_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/attack_isic_%j.err

# Table 2 attack evaluation — ISIC

echo "=========================================="
echo "Table 2 attack — ISIC"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "=========================================="

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."; exit 1
fi

cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_table2_attack.py \
    --dataset isic \
    --epsilons "2,8" \
    --seeds 100 200 300 400 500 \
    --te 60

echo "Finished: $(date)"
