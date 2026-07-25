#!/bin/bash
#SBATCH --job-name=caps_diagnostic
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/caps_diagnostic_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/caps_diagnostic_%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."
    exit 1
fi

mkdir -p /home/ab36/DPKD-medical/phase1_distillation/slurm_logs
cd /home/ab36/DPKD-medical/phase1_distillation

python -u diagnose_clip_caps.py | tee clip_caps_diagnostic.txt

echo "Finished: $(date)"
