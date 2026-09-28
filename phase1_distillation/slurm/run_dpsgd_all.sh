#!/bin/bash
# DP-SGD full pipeline — one command.
#
# 1. Submits grid search for ISIC and Kvasir (BUSI already done).
# 2. Submits full baseline runs for all 3 datasets, each waiting on
#    its own grid search job (Slurm dependency).
#
# Usage (from phase1_distillation/):
#   bash slurm/run_dpsgd_all.sh              # all three datasets
#   bash slurm/run_dpsgd_all.sh isic kvasir  # skip busi (already running)
#   bash slurm/run_dpsgd_all.sh isic         # single dataset

DATASETS=${@:-"busi isic kvasir"}

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

for DS in $DATASETS; do
    if [[ "$DS" == "busi" ]]; then
        # BUSI grid search already done — submit baseline directly
        JOB=$(submit_baseline busi "")
        echo "Submitted dpsgd_busi baseline: job ${JOB} (uses existing grid search result)"
    else
        # ISIC / Kvasir — grid search first, baseline chained after
        GRID_JOB=$(submit_grid "$DS")
        echo "Submitted dpsgd_grid_${DS}: job ${GRID_JOB}"
        BASE_JOB=$(submit_baseline "$DS" "$GRID_JOB")
        echo "Submitted dpsgd_${DS} baseline: job ${BASE_JOB} (starts after job ${GRID_JOB})"
    fi
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  results/busi_dpsgd_results.json"
echo "  results/isic_dpsgd_results.json"
echo "  results/kvasir_dpsgd_results.json"
