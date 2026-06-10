#!/bin/bash
# ============================================================
# GKD Distillation v4 — DP-Honest Pipeline
#
# Based on Dr. Feng's v2 structure + normalised alpha loss.
#
# Key flags:
#   --precompute-noise  → sample-once (DP-honest, reported eps = true eps)
#   --public-caps-csv   → caps from HRF/STARE/CHASE (zero DRIVE privacy cost)
#   --importance-csv    → importance from public data
#
# Takes 4 positional args: noise_type, epsilon, alpha, seed
#
# Submit full alpha sweep (120 jobs):
#   for NOISE in uniform channel_WF; do
#     for EPS in 2 4 8 16; do
#       for ALPHA in 0.5 0.6 0.7 0.8 0.9; do
#         for SEED in 0 1 2; do
#           sbatch run_gkd_v4.sh $NOISE $EPS $ALPHA $SEED
#         done
#       done
#     done
#   done
#
# Submit alpha=1.0 ablation (24 jobs):
#   for NOISE in uniform channel_WF; do
#     for EPS in 2 4 8 16; do
#       for SEED in 0 1 2; do
#         sbatch run_gkd_v4.sh $NOISE $EPS 1.0 $SEED
#       done
#     done
#   done
# ============================================================
#SBATCH --job-name=GKD_V4
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_v4_%1_%2_a%3_s%4_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/gkd_v4_%1_%2_a%3_s%4_%j.err

# ----------------------------
# Arguments
# ----------------------------
NOISE_TYPE=$1   # uniform or channel_WF
EPSILON=$2      # 2, 4, 8, or 16
ALPHA=$3        # 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0
SEED=$4         # 0, 1, or 2

if [ -z "$NOISE_TYPE" ] || [ -z "$EPSILON" ] || [ -z "$ALPHA" ] || [ -z "$SEED" ]; then
    echo "Usage: sbatch run_gkd_v4.sh <noise_type> <epsilon> <alpha> <seed>"
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

export PUBLIC_CAPS=$MMSEG/work_dirs/PUBLIC_PROXY_P05/public_caps.csv
export PUBLIC_IMP=$MMSEG/work_dirs/PUBLIC_PROXY_P05/public_importance.csv

export SCRIPT=$MMSEG/tools/analysis/phase1_gkd_distill_v4.py

# Alpha for folder name: 0.7 → a70, 1.0 → a100
ALPHA_INT=$(echo "$ALPHA * 100" | bc | cut -d'.' -f1)
export OUT=$MMSEG/work_dirs/GKD_V4_P05_${NOISE_TYPE}_eps${EPSILON}_a${ALPHA_INT}_seed${SEED}

# ----------------------------
# Setup
# ----------------------------
cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "GKD v4: DP-honest pipeline"
echo "Job ID    : $SLURM_JOB_ID"
echo "Noise type: $NOISE_TYPE"
echo "Epsilon   : $EPSILON"
echo "Alpha     : $ALPHA  (${ALPHA_INT}% teacher / $((100 - ALPHA_INT))% GT)"
echo "Seed      : $SEED"
echo "Output    : $OUT"
echo "=========================================="

python "$SCRIPT" \
    --teacher-config      "$TEACHER_CONFIG" \
    --teacher-checkpoint  "$TEACHER_CKPT" \
    --student-config      "$STUDENT_CONFIG" \
    --student-checkpoint  "$STUDENT_CKPT" \
    --data-root           "$DATA" \
    --public-caps-csv     "$PUBLIC_CAPS" \
    --importance-csv      "$PUBLIC_IMP" \
    --out-dir             "$OUT" \
    --noise-type          "$NOISE_TYPE" \
    --epsilon             "$EPSILON" \
    --alpha               "$ALPHA" \
    --precompute-noise \
    --K                   1 \
    --delta-dp            1e-5 \
    --max-iters           40000 \
    --val-interval        4000 \
    --device              cuda:0 \
    --num-workers         4 \
    --pad-divisor         16 \
    --ignore-index        255 \
    --seed                "$SEED"

echo "=========================================="
echo "Done: noise=$NOISE_TYPE  eps=$EPSILON  alpha=$ALPHA  seed=$SEED"
echo "Results in: $OUT"
echo "=========================================="