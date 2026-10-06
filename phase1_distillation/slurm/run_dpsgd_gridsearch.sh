#!/bin/bash
# Submit DP-SGD grid search job for a single dataset.
# Usage from Cheaha:
#   bash slurm/run_dpsgd_gridsearch.sh busi
#   bash slurm/run_dpsgd_gridsearch.sh isic
#   bash slurm/run_dpsgd_gridsearch.sh kvasir

DS=${1:?"Usage: bash slurm/run_dpsgd_gridsearch.sh <busi|isic|kvasir>"}

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=dpsgd_grid_${DS}
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_grid_${DS}_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_grid_${DS}_%j.err

echo "=========================================="
echo "DP-SGD Grid Search — ${DS}  ε=8.0"
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

python -u dpsgd_gridsearch.py --dataset ${DS}

echo "Finished: \$(date)"
SLURM

echo "Submitted dpsgd_grid_${DS}"
