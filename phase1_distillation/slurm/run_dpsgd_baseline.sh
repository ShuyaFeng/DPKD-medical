#!/bin/bash
# Submit DP-SGD baseline jobs — one job per dataset.
# Usage from Cheaha:
#   bash slurm/run_dpsgd_baseline.sh            # submits all three datasets
#   bash slurm/run_dpsgd_baseline.sh isic       # submits only isic

DATASETS=${1:-"isic kvasir busi"}

for DS in $DATASETS; do
    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=dpsgd_${DS}
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-14:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_${DS}_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_${DS}_%j.err

echo "=========================================="
echo "DP-SGD baseline (Abadi 2016) — ${DS}"
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

python -u dpsgd_baseline.py --dataset ${DS}

echo "Finished: \$(date)"
SLURM
    echo "Submitted dpsgd_${DS}"
done
