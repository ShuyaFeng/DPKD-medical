#!/bin/bash
# ============================================================
# GKD Distillation — all 8 runs (2 noise types × 4 epsilons)
#
# Submit each run separately:
#   sbatch run_gkd_distill.sh uniform    2
#   sbatch run_gkd_distill.sh uniform    4
#   sbatch run_gkd_distill.sh uniform    8
#   sbatch run_gkd_distill.sh uniform    16
#   sbatch run_gkd_distill.sh channel_WF 2
#   sbatch run_gkd_distill.sh channel_WF 4
#   sbatch run_gkd_distill.sh channel_WF 8
#   sbatch run_gkd_distill.sh channel_WF 16
# ============================================================
#SBATCH --job-name=GKD_DISTILL
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_%1_%2_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_%1_%2_%j.err

# ----------------------------
# Arguments
#   $1 = noise type: uniform or channel_WF
#   $2 = epsilon:    2, 4, 8, or 16
# ----------------------------
NOISE_TYPE=$1
EPSILON=$2

if [ -z "$NOISE_TYPE" ] || [ -z "$EPSILON" ]; then
    echo "Usage: sbatch run_gkd_distill.sh <noise_type> <epsilon>"
    echo "  noise_type: uniform or channel_WF"
    echo "  epsilon:    2, 4, 8, or 16"
    exit 1
fi

# ----------------------------
# Environment
# ----------------------------
module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ----------------------------
# Paths
# ----------------------------
export MMSEG=/data/user/home/ialam/mmsegmentation
export DATA=/data/user/home/ialam/mmsegmentation/data/DRIVE

export TEACHER_CONFIG=/data/user/home/ialam/mmseg_models/unet_teacher/unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export TEACHER_CKPT=/data/user/home/ialam/mmseg_models/unet_teacher/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth

export STUDENT_CONFIG=/data/user/home/ialam/mmseg_models/unet_student/unet-s5-d16-small_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export STUDENT_CKPT=/data/user/home/ialam/mmseg_models/unet_student/unet-s5-d16-small_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.pth

export IMPORTANCE_CSV=$MMSEG/work_dirs/UNET64_DRIVE_BOTTLENECK_CHANNEL_ANALYSIS_GRADIENT_s3405/channel_importance_scores_bottleneck_gradient.csv

export SCRIPT=$MMSEG/tools/analysis/phase1_gkd_distill.py
export OUT=$MMSEG/work_dirs/GKD_DISTILL_${NOISE_TYPE}_eps${EPSILON}

# ----------------------------
# Setup
# ----------------------------
cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "GKD Distillation"
echo "Job ID:     $SLURM_JOB_ID"
echo "Noise type: $NOISE_TYPE"
echo "Epsilon:    $EPSILON"
echo "Output:     $OUT"
echo "=========================================="

python "$SCRIPT" \
    --teacher-config      "$TEACHER_CONFIG" \
    --teacher-checkpoint  "$TEACHER_CKPT" \
    --student-config      "$STUDENT_CONFIG" \
    --student-checkpoint  "$STUDENT_CKPT" \
    --data-root           "$DATA" \
    --importance-csv      "$IMPORTANCE_CSV" \
    --out-dir             "$OUT" \
    --noise-type          "$NOISE_TYPE" \
    --epsilon             "$EPSILON" \
    --K                   1 \
    --delta-dp            1e-5 \
    --lambda-gkd          0.4 \
    --max-iters           40000 \
    --val-interval        4000 \
    --device              cuda:0 \
    --num-workers         4 \
    --pad-divisor         16 \
    --ignore-index        255 \
    --cap-quantile        0.9 \
    --seed                42

echo "=========================================="
echo "Done: noise=$NOISE_TYPE  epsilon=$EPSILON"
echo "Results in: $OUT"
echo "=========================================="