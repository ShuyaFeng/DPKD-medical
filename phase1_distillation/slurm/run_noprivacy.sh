#!/bin/bash
# Submit non-private (eps=inf) reference job — 20-seed run for BUSI only.
# Saves to busi_noprivacy_results_20seeds.json (does NOT overwrite 5-seed or 10-seed results).
# Usage: bash slurm/run_noprivacy.sh

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=noprivacy_busi_20s
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/noprivacy_busi_20seeds_%j.out
#SBATCH --error=/home/ab36/DPKD-medical/phase1_distillation/slurm_logs/noprivacy_busi_20seeds_%j.err

echo "=========================================="
echo "Non-private reference (eps=inf) — BUSI 20 seeds"
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
    --dataset busi \\
    --seeds 20 --te 60 --se 40 \\
    --suffix 20seeds \\
    --no_noise

echo "Finished: \$(date)"
EOF

echo "Submitted noprivacy_busi_20seeds"
