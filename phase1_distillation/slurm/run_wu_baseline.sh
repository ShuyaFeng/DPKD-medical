#!/bin/bash
# Submit Wu et al. full experiment jobs — one job per dataset.
# Run AFTER run_wu_hparam.sh has completed and
#   results/wu_busi_hparams.json exists.
#
# Usage:
#   bash slurm/run_wu_baseline.sh            # submits all three datasets
#   bash slurm/run_wu_baseline.sh isic       # submits only isic
#   bash slurm/run_wu_baseline.sh busi       # submits only busi

DATASETS=${1:-"isic kvasir busi"}

for DS in $DATASETS; do
    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=wu_${DS}
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/wu_${DS}_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/wu_${DS}_%j.err

echo "=========================================="
echo "Wu et al. 2025 (adapted) — ${DS}"
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

python -u wu_baseline.py --dataset ${DS}

echo "Finished: \$(date)"
SLURM
    echo "Submitted wu_${DS}"
done
