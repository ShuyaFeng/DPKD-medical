#!/bin/bash
#SBATCH --job-name=table2_attack
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/table2_attack_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/table2_attack_%j.err

echo "=========================================="
echo "Table 2: MIA + Reconstruction attack"
echo "Config: K=3, keep=10%, fc=0.10/fi=0.05/fr=0.85, eps={2,8}"
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

echo ""
echo "--- ISIC (default dataset for Table 2) ---"
python -u canal_table2_attack.py \
    --dataset isic \
    --epsilons 2,8 \
    --teacher-epochs 60 \
    --n-eval 200

echo ""
echo "Finished: $(date)"
echo "Results saved to: /home/ab36/DPKD-medical/phase1_distillation/results/isic_table2_attack_results.json"
