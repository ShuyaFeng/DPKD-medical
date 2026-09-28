#!/bin/bash
# DP-SGD full pipeline — one command.
#
# 1. Submits grid search for ISIC and Kvasir (BUSI already done).
# 2. Submits full baseline runs for all 3 datasets, each waiting on
#    its own grid search job (Slurm dependency).
#
# Usage (from phase1_distillation/):
#   bash slurm/run_dpsgd_all.sh

set -e

# ── Helper: submit one grid search job, return job ID ────────────────────────
submit_grid() {
    local DS=$1
    sbatch --parsable <<SLURM
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
echo "DP-SGD grid search — ${DS}  ε=8.0"
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
}

# ── Helper: submit full baseline job with dependency ──────────────────────────
submit_baseline() {
    local DS=$1
    local DEP=$2   # job ID to wait on (empty = no dependency)
    local DEP_FLAG=""
    [[ -n "$DEP" ]] && DEP_FLAG="--dependency=afterok:${DEP}"

    sbatch --parsable $DEP_FLAG <<SLURM
#!/bin/bash
#SBATCH --job-name=dpsgd_${DS}
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_${DS}_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/dpsgd_${DS}_%j.err

echo "=========================================="
echo "DP-SGD baseline — ${DS}"
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
}

# ── BUSI: grid search already done, submit baseline directly ──────────────────
BUSI_BASE=$(submit_baseline busi "")
echo "Submitted dpsgd_busi baseline: job ${BUSI_BASE} (uses existing grid search result)"

# ── ISIC: grid search first, then baseline ────────────────────────────────────
ISIC_GRID=$(submit_grid isic)
echo "Submitted dpsgd_grid_isic: job ${ISIC_GRID}"
ISIC_BASE=$(submit_baseline isic "$ISIC_GRID")
echo "Submitted dpsgd_isic baseline: job ${ISIC_BASE} (starts after job ${ISIC_GRID})"

# ── Kvasir: grid search first, then baseline ─────────────────────────────────
KVASIR_GRID=$(submit_grid kvasir)
echo "Submitted dpsgd_grid_kvasir: job ${KVASIR_GRID}"
KVASIR_BASE=$(submit_baseline kvasir "$KVASIR_GRID")
echo "Submitted dpsgd_kvasir baseline: job ${KVASIR_BASE} (starts after job ${KVASIR_GRID})"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  results/busi_dpsgd_results.json"
echo "  results/isic_dpsgd_results.json"
echo "  results/kvasir_dpsgd_results.json"
