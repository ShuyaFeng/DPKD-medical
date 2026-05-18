#!/bin/bash
# ============================================================
# GKD Distillation v2 — direct feature matching + multi-seed
#
# Takes 3 arguments: noise_type, epsilon, seed
#
# Submit all 24 runs (2 types x 4 epsilons x 3 seeds):
#
#   for NOISE in uniform channel_WF; do
#     for EPS in 2 4 8 16; do
#       for SEED in 0 1 2; do
#         sbatch run_gkd_distill_v2.sh $NOISE $EPS $SEED
#       done
#     done
#   done
#
# Or start with just seed=0 for all 8 conditions first:
#   for NOISE in uniform channel_WF; do
#     for EPS in 2 4 8 16; do
#       sbatch run_gkd_distill_v2.sh $NOISE $EPS 0
#     done
#   done
# ============================================================
#SBATCH --job-name=GKD_V2
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_v2_%1_%2_s%3_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_v2_%1_%2_s%3_%j.err

# ----------------------------
# Arguments
# ----------------------------
NOISE_TYPE=$1   # uniform or channel_WF
EPSILON=$2      # 2, 4, 8, or 16
SEED=$3         # 0, 1, or 2

if [ -z "$NOISE_TYPE" ] || [ -z "$EPSILON" ] || [ -z "$SEED" ]; then
    echo "Usage: sbatch run_gkd_distill_v2.sh <noise_type> <epsilon> <seed>"
    echo "  noise_type : uniform or channel_WF"
    echo "  epsilon    : 2, 4, 8, or 16"
    echo "  seed       : 0, 1, or 2"
    exit 1
fi

# ----------------------------
# Environment
# ----------------------------
module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=$SEED
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

export SCRIPT=$MMSEG/tools/analysis/phase1_gkd_distill_v2.py

# Output folder includes seed so each run saves separately
export OUT=$MMSEG/work_dirs/GKD_V2_${NOISE_TYPE}_eps${EPSILON}_seed${SEED}

# ----------------------------
# Setup
# ----------------------------
cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "GKD v2: direct feature matching"
echo "Job ID    : $SLURM_JOB_ID"
echo "Noise type: $NOISE_TYPE"
echo "Epsilon   : $EPSILON"
echo "Seed      : $SEED"
echo "Output    : $OUT"
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
    --lambda-feat         0.4 \
    --max-iters           40000 \
    --val-interval        4000 \
    --device              cuda:0 \
    --num-workers         4 \
    --pad-divisor         16 \
    --ignore-index        255 \
    --cap-quantile        0.9 \
    --seed                "$SEED"

echo "=========================================="
echo "Done: noise=$NOISE_TYPE  eps=$EPSILON  seed=$SEED"
echo "Results in: $OUT"
echo "=========================================="