#!/bin/bash
#SBATCH --job-name=isic_sel_abl
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/isic_sel_abl_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/isic_sel_abl_%j.err

set -e
echo "=========================================="
echo "ISIC channel selection ablation (Tables 4 & 5)"
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

mkdir -p /home/ab36/DPKD-medical/phase1_distillation/slurm_logs

cd /home/ab36/DPKD-medical/phase1_distillation

# 3 seeds (matches paper Tables 4 & 5), 6 drop fracs x 5 eps x (uniform + CANAL@90%)
# Plus K=3 teacher training (60 epochs each).
# Estimated runtime: ~16-18 h on a single A100.
python -u isic_channel_selection_ablation.py \
    --dataset isic \
    --seeds 3 \
    --te 60 \
    --se 40 \
    --epsilons "0.5,1,2,4,6" \
    --drops "0.0,0.1,0.3,0.5,0.7,0.9"

echo "Finished: $(date)"
