#!/bin/bash
# Submit non-private (eps=inf) reference jobs for all 3 datasets.
# Usage: bash slurm/run_noprivacy.sh [dataset]
#   No arg  → submits all 3 datasets
#   dataset → submits only that dataset (busi | isic | kvasir)

DATASETS=("busi" "isic" "kvasir")
TARGET="${1:-all}"

for DS in "${DATASETS[@]}"; do
    if [[ "$TARGET" != "all" && "$TARGET" != "$DS" ]]; then
        continue
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=noprivacy_${DS}
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/noprivacy_${DS}_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/noprivacy_${DS}_%j.err

echo "=========================================="
echo "Non-private reference (eps=inf) — ${DS}"
echo "Job ID: \$SLURM_JOB_ID  Node: \$(hostname)  Start: \$(date)"
echo "=========================================="

module load Anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

if [[ "\$CONDA_DEFAULT_ENV" != "mmseg-cu124-240" ]]; then
    echo "ERROR: conda environment did not activate."
    exit 1
fi

cd /home/ab36/DPKD-medical/phase1_distillation

python -u canal_final_experiment.py \\
    --dataset ${DS} \\
    --seeds 5 --te 60 --se 40 \\
    --no_noise

echo "Finished: \$(date)"
EOF

    echo "Submitted noprivacy_${DS}"
done
