#!/bin/bash
#SBATCH --job-name=attack_kvasir
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/attack_kvasir_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/attack_kvasir_%j.err

# Table 2 attack evaluation — Kvasir

echo "=========================================="
echo "Table 2 attack — Kvasir"
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
    --dataset kvasir \
    --epsilons "2,8" \
    --seeds 100 200 300 400 500 \
    --te 60

echo "Finished: $(date)"
