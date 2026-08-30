#!/bin/bash
#SBATCH --job-name=qual_fig
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/qual_fig_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/qual_fig_%j.err

echo "=========================================="
echo "Qualitative Figure 2 generation"
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
echo "--- Kvasir (canal seed=200, uniform seed=200, eps=2.0) ---"
python -u generate_qualitative_fig.py \
    --dataset kvasir \
    --epsilon 2.0 \
    --canal-seed 200 \
    --uniform-seed 200 \
    --top 3

echo ""
echo "--- BUSI (canal seed=400, uniform seed=400, eps=2.0) ---"
python -u generate_qualitative_fig.py \
    --dataset busi \
    --epsilon 2.0 \
    --canal-seed 400 \
    --uniform-seed 400 \
    --top 3

echo ""
echo "--- ISIC (canal seed=200, uniform seed=200, eps=2.0, suffix=rerun3) ---"
python -u generate_qualitative_fig.py \
    --dataset isic \
    --epsilon 2.0 \
    --canal-seed 200 \
    --uniform-seed 200 \
    --suffix rerun3 \
    --top 3

echo ""
echo "Finished: $(date)"
echo "Images saved to: /home/ab36/DPKD-medical/phase1_distillation/qualitative_exports/"
