#!/bin/bash
#SBATCH --job-name=budget_abl_s2
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/budget_abl_s2_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/budget_abl_s2_%j.err

# Budget split 2: fc=0.20  fi=0.10  fr=0.70  (more budget to privatization)

echo "=========================================="
echo "Budget ablation split2 — fc=0.20 fi=0.10 fr=0.70"
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
    --dataset isic \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --fc 0.20 --fi 0.10 --fr 0.70 \
    --suffix fc20_fi10_fr70

python -u canal_final_experiment.py \
    --dataset kvasir \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --fc 0.20 --fi 0.10 --fr 0.70 \
    --suffix fc20_fi10_fr70

python -u canal_final_experiment.py \
    --dataset busi \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --fc 0.20 --fi 0.10 --fr 0.70 \
    --suffix fc20_fi10_fr70

echo "Finished: $(date)"
