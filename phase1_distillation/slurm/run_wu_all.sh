#!/bin/bash
# Wu et al. full pipeline — one command.
#
# If results/wu_busi_hparams.json already exists, skips HP search and submits
# the 3 dataset jobs immediately. Otherwise runs HP search first and chains
# the dataset jobs as Slurm dependencies.
#
# Usage (from phase1_distillation/):
#   bash slurm/run_wu_all.sh

set -e

HPARAMS_FILE="/home/ab36/DPKD-medical/phase1_distillation/results/wu_busi_hparams.json"

submit_dataset_jobs() {
    local dep_flag="$1"   # e.g. "--dependency=afterok:12345" or ""
    for DS in isic kvasir busi; do
        JOB=$(sbatch --parsable ${dep_flag} <<SLURM
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
        echo "Submitted wu_${DS}: job ${JOB}${dep_flag:+ (starts after HP search)}"
    done
}

# ── Check if HP search results already exist ─────────────────────────────────
if [[ -f "${HPARAMS_FILE}" ]]; then
    echo "HP results found: ${HPARAMS_FILE}"
    echo "Skipping HP search — submitting 3 dataset jobs directly."
    submit_dataset_jobs ""
else
    echo "No HP results found. Running HP search first."

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
    submit_dataset_jobs "--dependency=afterok:${HP_JOB}"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
