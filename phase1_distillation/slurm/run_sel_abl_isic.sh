#!/bin/bash
#SBATCH --job-name=sel_abl_isic
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sel_abl_isic_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/sel_abl_isic_%j.err

echo "=========================================="
echo "ISIC channel selection ablation"
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

python -u isic_channel_selection_ablation.py \
    --dataset isic \
    --seeds 3 --te 60 --se 40 \
    --epsilons "0.5,1,2,4,6" \
    --drops "0.0,0.1,0.3,0.5,0.7,0.9"

echo "Finished: $(date)"
