#!/bin/bash
#SBATCH --job-name=PHASE1_NOISE_V2
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/phase1_noise_v2_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/phase1_noise_v2_%j.err

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export MMSEG=/data/user/home/ialam/mmsegmentation
export DATA=/data/user/home/ialam/mmsegmentation/data/DRIVE

export CONFIG=/data/user/home/ialam/mmseg_models/unet_teacher/unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py
export CKPT=/data/user/home/ialam/mmseg_models/unet_teacher/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth
export IMPORTANCE_CSV=$MMSEG/work_dirs/UNET64_DRIVE_BOTTLENECK_CHANNEL_ANALYSIS_GRADIENT_s3405/channel_importance_scores_bottleneck_gradient.csv

export SCRIPT=$MMSEG/tools/analysis/phase1_drive_channel_noise_v2.py
export OUT=$MMSEG/work_dirs/PHASE1_CHANNEL_NOISE_V2

cd "$MMSEG"
mkdir -p slurm_out "$OUT"
export PYTHONPATH="$MMSEG:$PYTHONPATH"

echo "=========================================="
echo "Phase 1 v2: normalised channel noise"
echo "Job ID: $SLURM_JOB_ID"
echo "K=10 teachers, eps={1,2,4,8,16,32}"
echo "=========================================="

python "$SCRIPT" \
  --config          "$CONFIG" \
  --checkpoint      "$CKPT" \
  --data-root       "$DATA" \
  --importance-csv  "$IMPORTANCE_CSV" \
  --out-dir         "$OUT" \
  --device          cuda:0 \
  --num-workers     4 \
  --pad-divisor     16 \
  --foreground-class 1 \
  --ignore-index    255 \
  --delta-dp        1e-5 \
  --K               10 \
  --epsilons        1.0 2.0 4.0 8.0 16.0 32.0 \
  --cap-quantile    0.9 \
  --seed            42

echo "=========================================="
echo "Phase 1 v2 finished. Results in: $OUT"
echo "=========================================="