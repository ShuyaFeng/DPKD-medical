#!/bin/bash
#SBATCH --job-name=arxiv_table6
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/arxiv_table6_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/arxiv_table6_%j.err

echo "=========================================="
echo "ArXiv rerun — Table 6 (channel selection ablation)"
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

python -u isic_honest_subsample.py \
    --K 3 \
    --seeds 3 \
    --epsilons "0.5,1,2,4,6" \
    --keep_fracs "0.9,0.7,0.5,0.3,0.1" \
    --skip_random \
    --te 50 --se 40 \
    --out_tag "arxiv"

echo "Finished: $(date)"
