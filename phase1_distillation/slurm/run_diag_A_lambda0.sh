#!/bin/bash
# ============================================================
# DIAGNOSTIC A — lambda_feat = 0
#
# Hypothesis: student is NOT learning from teacher feature.
# If lambda_feat=0 gives roughly the same mDice as the noisy
# distillation runs (~88.9%), the feature-matching path
# contributes nothing — the student is being trained by GT
# alone.
#
# Setup matches existing GKD_V2_uniform_eps8 runs as closely
# as possible, EXCEPT lambda_feat=0. Compare against:
#   GKD_V2_uniform_eps8 mean=88.77%
#
# Submit:
#   for SEED in 0 1 2; do sbatch run_diag_A_lambda0.sh $SEED; done
# ============================================================
#SBATCH --job-name=GKD_DIAG_A
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/diag_A_s%1_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/diag_A_s%1_%j.err

SEED=$1
if [ -z "$SEED" ]; then
    echo "Usage: sbatch run_diag_A_lambda0.sh <seed>"
    exit 1
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

export IMPORTANCE_CSV=$MMSEG/work_dirs/UNET64_DRIVE_BOTTLENECK_CHANNEL_ANALYSIS_GRADIENT_s3405/channel_importance_scores_bottleneck_gradient.csv

export SCRIPT=$MMSEG/tools/analysis/phase1_gkd_distill_v2.py
export OUT=$MMSEG/work_dirs/GKD_DIAG_A_lambda0_seed${SEED}

cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "DIAGNOSTIC A — lambda_feat = 0"
echo "Seed   : $SEED"
echo "Output : $OUT"
echo "=========================================="

python "$SCRIPT" \
    --teacher-config      "$TEACHER_CONFIG" \
    --teacher-checkpoint  "$TEACHER_CKPT" \
    --student-config      "$STUDENT_CONFIG" \
    --student-checkpoint  "$STUDENT_CKPT" \
    --data-root           "$DATA" \
    --importance-csv      "$IMPORTANCE_CSV" \
    --out-dir             "$OUT" \
    --noise-type          uniform \
    --epsilon             8 \
    --K                   1 \
    --delta-dp            1e-5 \
    --lambda-feat         0.0 \
    --max-iters           40000 \
    --val-interval        4000 \
    --device              cuda:0 \
    --num-workers         4 \
    --pad-divisor         16 \
    --ignore-index        255 \
    --cap-quantile        0.9 \
    --seed                "$SEED"

echo "Done: DIAG_A seed=$SEED"
