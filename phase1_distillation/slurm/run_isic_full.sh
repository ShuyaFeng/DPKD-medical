#!/bin/bash
#SBATCH --job-name=isic_full_comparison
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ialam/DPKD-medical/phase1_distillation/slurm_logs/isic_full_%j.out
#SBATCH --error=/home/ialam/DPKD-medical/phase1_distillation/slurm_logs/isic_full_%j.err

set -e

echo "=========================================="
echo "ISIC full comparison experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- environment setup ---
module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

# fail fast if env didn't activate correctly
if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate correctly."
    echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Active conda env: $CONDA_DEFAULT_ENV"
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

cd /home/ialam/DPKD-medical/phase1_distillation
mkdir -p slurm_logs

# --- sanity check: data must already exist (built on login node) ---
if [[ ! -d "../data/ISIC_HF/train/input" ]]; then
    echo "ERROR: data/ISIC_HF not found. Run on login node first:"
    echo "  python download_isic.py --n 0"
    exit 1
fi

N_TRAIN=$(ls ../data/ISIC_HF/train/input | wc -l)
N_VAL=$(ls ../data/ISIC_HF/val/input | wc -l)
echo "Found data: train=$N_TRAIN val=$N_VAL"

# --- run the full experiment ---
echo "=========================================="
echo "Starting full run: seeds=5 te=60 se=40"
echo "=========================================="

python -u run_comparison.py \
    --dataset isic \
    --seeds 5 \
    --te 60 \
    --se 40

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
echo "Results saved to:"
echo "  results/isic_comparison.json"
echo "  fig_isic_comparison.png"
Done