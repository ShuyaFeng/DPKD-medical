#!/bin/bash
# ============================================================
# LAMBDA SWEEP — varies feature-distillation weight
#
# Critical for Framing 1 (Diagnostic + Remedy paper):
#   show that channel-WF only beats uniform when lambda_feat
#   is large enough that feature loss dominates task loss.
#
# 4 args: lambda, noise_type, epsilon, seed
#
# Recommended priority order (most informative first):
#   # 1. Confirm task loss alone explains existing results
#   for SEED in 0 1 2; do
#     sbatch run_lambda_sweep.sh 0 uniform 8 $SEED
#   done
#
#   # 2. eps=2 — where WF should shine — sweep lambda
#   for LAM in 5 20 50; do
#     for NT in uniform channel_WF; do
#       for SEED in 0 1 2; do
#         sbatch run_lambda_sweep.sh $LAM $NT 2 $SEED
#       done
#     done
#   done
#
#   # 3. eps=16 — sanity check that gap narrows at high eps
#   for LAM in 20; do
#     for NT in uniform channel_WF; do
#       for SEED in 0 1 2; do
#         sbatch run_lambda_sweep.sh $LAM $NT 16 $SEED
#       done
#     done
#   done
# ============================================================
#SBATCH --job-name=GKD_LAMSWP
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/lamswp_l%1_%2_e%3_s%4_p%5_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/lamswp_l%1_%2_e%3_s%4_p%5_%j.err

LAMBDA=$1
NOISE_TYPE=$2
EPSILON=$3
SEED=$4
PRECOMPUTE=${5:-no}      # optional 5th arg: yes | no  (default no)

if [ -z "$LAMBDA" ] || [ -z "$NOISE_TYPE" ] || [ -z "$EPSILON" ] || [ -z "$SEED" ]; then
    echo "Usage: sbatch run_lambda_sweep.sh <lambda> <noise_type> <epsilon> <seed> [precompute]"
    echo "  lambda     : 0, 5, 20, 50, etc."
    echo "  noise_type : uniform | channel_WF | none"
    echo "  epsilon    : 2, 4, 8, 16"
    echo "  seed       : 0, 1, 2"
    echo "  precompute : yes | no  (default no)"
    echo "               yes -> sample-once-per-image threat model"
    echo "                      (sets --precompute-noise on the python script)"
    exit 1
fi

if [ "$PRECOMPUTE" != "yes" ] && [ "$PRECOMPUTE" != "no" ]; then
    echo "Invalid PRECOMPUTE='$PRECOMPUTE' — must be 'yes' or 'no'."
    exit 1
fi

# Build the optional flag to pass to the python script
PRECOMPUTE_FLAG=""
THREAT_TAG="pIter"     # per-iteration release (default)
if [ "$PRECOMPUTE" = "yes" ]; then
    PRECOMPUTE_FLAG="--precompute-noise"
    THREAT_TAG="sOnce"  # sample-once
fi

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export MMSEG=/data/user/home/ialam/mmsegmentation
export DATA=/data/user/home/ialam/mmsegmentation/data/DRIVE

export TEACHER_CONFIG=/data/user/home/ialam/mmseg_models/unet_teacher/unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export TEACHER_CKPT=/data/user/home/ialam/mmseg_models/unet_teacher/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth

export STUDENT_CONFIG=/data/user/home/ialam/mmseg_models/unet_student/unet-s5-d16-small_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export STUDENT_CKPT=/data/user/home/ialam/mmseg_models/unet_student/unet-s5-d16-small_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.pth

# Public-proxy paths produced by compute_public_proxy.py on HRF/STARE/CHASE.
# Override with PUBLIC_PROXY_DIR env var if needed.
export PUBLIC_PROXY_DIR=${PUBLIC_PROXY_DIR:-$MMSEG/work_dirs/PUBLIC_PROXY_HRF}
export IMPORTANCE_CSV=${IMPORTANCE_CSV:-$PUBLIC_PROXY_DIR/public_importance.csv}
export PUBLIC_CAPS_CSV=${PUBLIC_CAPS_CSV:-$PUBLIC_PROXY_DIR/public_caps.csv}

# Optional: fall back to the legacy private-data importance CSV for ablation runs
# (set THREAT_MODEL=fully-private to enable).
export THREAT_MODEL=${THREAT_MODEL:-public-proxy}

export SCRIPT=$MMSEG/tools/analysis/phase1_gkd_distill_v2.py
export OUT=$MMSEG/work_dirs/GKD_LAMSWP_${THREAT_MODEL}_${THREAT_TAG}_l${LAMBDA}_${NOISE_TYPE}_eps${EPSILON}_seed${SEED}

cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "LAMBDA SWEEP"
echo "  lambda       : $LAMBDA"
echo "  noise_type   : $NOISE_TYPE"
echo "  epsilon      : $EPSILON"
echo "  seed         : $SEED"
echo "  precompute   : $PRECOMPUTE  (threat=$THREAT_TAG)"
echo "  output       : $OUT"
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
    --lambda-feat         "$LAMBDA" \
    --max-iters           40000 \
    --val-interval        4000 \
    --device              cuda:0 \
    --num-workers         4 \
    --pad-divisor         16 \
    --ignore-index        255 \
    --cap-quantile        0.9 \
    --seed                "$SEED" \
    --threat-model        "$THREAT_MODEL" \
    --public-caps-csv     "$PUBLIC_CAPS_CSV" \
    $PRECOMPUTE_FLAG

echo "Done: lambda=$LAMBDA noise=$NOISE_TYPE eps=$EPSILON seed=$SEED "\
"precompute=$PRECOMPUTE threat=$THREAT_MODEL"
