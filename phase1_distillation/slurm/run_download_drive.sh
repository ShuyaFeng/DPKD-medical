#!/bin/bash
# ============================================================
# Download + reorganize DRIVE into data/DRIVE_HF/.
# Download needs network — run on a Cheaha LOGIN node:
#     CONDA_ENV=dpkd-cv bash run_download_drive.sh
# (No GPU needed. Not an sbatch job — login nodes have internet.)
# ============================================================
set -e

CONDA_ENV="${CONDA_ENV:-dpkd-cv}"

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1

REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
python3 download_drive.py

echo ""
echo "Verify:"
echo "  ls /home/fengs/DPKD-medical/data/DRIVE_HF/train/input/*.tif | wc -l   # expect ~20"
echo "  ls /home/fengs/DPKD-medical/data/DRIVE_HF/val/input/*.tif   | wc -l   # expect ~20"
