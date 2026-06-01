#!/bin/bash
# ============================================================
# Compute per-channel caps + importance on a PUBLIC retinal dataset.
#
# These are needed by the public-proxy threat model: they let us spend
# the entire user-level privacy budget on the per-image feature release
# (no budget split, no sequential composition with caps/imp estimation).
#
# Public datasets that work as proxies (all CC / academic-use):
#   HRF        — 45 high-res fundus images       https://www5.cs.fau.de/research/data/fundus-images/
#   STARE      — 20 manually segmented images    https://cecas.clemson.edu/~ahoover/stare/
#   CHASE-DB1  — 28 child fundus images          https://blogs.kingston.ac.uk/retinal/chasedb1/
#
# Download one of the above to $DATA_DIR/PUBLIC_RETINAL/<name>/ then run:
#   sbatch run_compute_public_proxy.sh HRF
# ============================================================
#SBATCH --job-name=PUBLIC_PROXY
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/public_proxy_%1_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/public_proxy_%1_%j.err

DATASET=${1:-HRF}
if [ -z "$DATASET" ]; then
    echo "Usage: sbatch run_compute_public_proxy.sh <HRF|STARE|CHASE-DB1>"
    exit 1
fi

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

export PYTHONNOUSERSITE=1

export MMSEG=/data/user/home/ialam/mmsegmentation
export TEACHER_CONFIG=/data/user/home/ialam/mmseg_models/unet_teacher/unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export TEACHER_CKPT=/data/user/home/ialam/mmseg_models/unet_teacher/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth

export PUBLIC_DATA_ROOT=/data/user/home/ialam/public_retinal/$DATASET
export OUT_DIR=$MMSEG/work_dirs/PUBLIC_PROXY_${DATASET}

export SCRIPT=$MMSEG/tools/analysis/compute_public_proxy.py

cd "$MMSEG"
mkdir -p slurm_out "$OUT_DIR"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

if [ ! -d "$PUBLIC_DATA_ROOT" ]; then
    echo "Public dataset not found at $PUBLIC_DATA_ROOT"
    echo "Download $DATASET first (see header comment for URLs)."
    exit 1
fi

echo "=========================================="
echo "Computing public proxy (caps + importance)"
echo "  dataset : $DATASET"
echo "  source  : $PUBLIC_DATA_ROOT"
echo "  output  : $OUT_DIR"
echo "=========================================="

python "$SCRIPT" \
    --teacher-config     "$TEACHER_CONFIG" \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --public-data-root   "$PUBLIC_DATA_ROOT" \
    --out-dir            "$OUT_DIR" \
    --image-size         584 565 \
    --cap-quantile       0.95 \
    --device             cuda:0

echo ""
echo "Done. To use these in distillation runs, either:"
echo "  (a) symlink:    ln -sfn $OUT_DIR $MMSEG/work_dirs/PUBLIC_PROXY_HRF"
echo "  (b) env var:    export PUBLIC_PROXY_DIR=$OUT_DIR"
echo "Then submit run_lambda_sweep.sh as usual."
