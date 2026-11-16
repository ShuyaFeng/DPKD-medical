#!/bin/bash
#SBATCH --job-name=budget_abl_s3
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/budget_abl_s3_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/budget_abl_s3_%j.err

# Budget split 3: fc=0.33  fi=0.33  fr=0.34  (equal split)

echo "=========================================="
echo "Budget ablation split3 — fc=0.33 fi=0.33 fr=0.34"
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
    --fc 0.33 --fi 0.33 --fr 0.34 \
    --suffix fc33_fi33_fr34

python -u canal_final_experiment.py \
    --dataset kvasir \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --fc 0.33 --fi 0.33 --fr 0.34 \
    --suffix fc33_fi33_fr34

python -u canal_final_experiment.py \
    --dataset busi \
    --seeds 5 --te 60 --se 40 \
    --epsilons "1,2,4,8" \
    --fc 0.33 --fi 0.33 --fr 0.34 \
    --suffix fc33_fi33_fr34

echo "Finished: $(date)"
