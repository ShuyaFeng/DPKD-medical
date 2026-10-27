#!/bin/bash
# Wu et al. full pipeline — one command.
#
# Submits HP search first, then submits 3 full-experiment jobs that
# start automatically when HP search finishes (Slurm dependency).
#
# Usage (from phase1_distillation/):
#   bash slurm/run_wu_all.sh

set -e

# ── Step 1: submit HP search, capture its job ID ─────────────────────────────
HP_JOB=$(sbatch --parsable <<SLURM
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
)

echo "Submitted HP search: job ${HP_JOB}"

# ── Step 2: submit 3 full-experiment jobs, each waiting on HP search ──────────
for DS in isic kvasir busi; do
    JOB=$(sbatch --parsable --dependency=afterok:${HP_JOB} <<SLURM
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
    )
    echo "Submitted wu_${DS}: job ${JOB} (starts after job ${HP_JOB})"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
