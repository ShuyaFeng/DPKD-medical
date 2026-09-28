#!/bin/bash
# Submit Wu et al. HP search job — BUSI dataset, eps=8.0.
# Run this FIRST.  When complete, check:
#   cat /home/ab36/DPKD-medical/phase1_distillation/results/wu_busi_hparams.json
# Then submit the full experiment with run_wu_baseline.sh.
#
# Usage:
#   bash slurm/run_wu_hparam.sh

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=wu_hparam
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/wu_hparam_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/wu_hparam_%j.err

echo "=========================================="
echo "Wu et al. HP search — BUSI  eps=8.0"
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

python -u wu_baseline.py --dataset busi --hparam

echo "Finished: \$(date)"
SLURM

echo "Submitted wu_hparam"
