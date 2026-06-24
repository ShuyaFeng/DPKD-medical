#!/bin/bash
#SBATCH --job-name=SBT
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/sbt_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/sbt_%j.err

# SBT (sparsity-induced bottleneck training) pipeline - SLURM wrapper.
#
# Stage 1 (lambda sweep, ~1 hour):
#   sbatch --export=ALL,DATASET=drive,STAGE=1 slurm/run_sbt_pipeline.sh
#   sbatch --export=ALL,DATASET=isic, STAGE=1 slurm/run_sbt_pipeline.sh
#
# Stage 2 (3-method sweep on SBT teachers, ~7 hours):
#   sbatch --export=ALL,DATASET=drive,STAGE=2,LAMBDA_SBT=1e-4 slurm/run_sbt_pipeline.sh
#   sbatch --export=ALL,DATASET=isic, STAGE=2,LAMBDA_SBT=1e-4 slurm/run_sbt_pipeline.sh
#
# Optional overrides:
#   LAMBDAS=0,1e-5,5e-5,1e-4,5e-4,1e-3    (stage 1 sweep set)
#   EPS=0.1,0.5,1,2,4,6,8                  (stage 2 epsilon grid)
#   SEEDS=5                                (student seed count)
#   TE=50                                  (teacher epochs)
#   SE=40                                  (student epochs)

module purge
module load Anaconda3/2023.07-2 2>/dev/null || module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null \
  || source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"

# ---- required args ----
: "${DATASET:?set DATASET=drive|isic}"
: "${STAGE:?set STAGE=1|2}"

# ---- env sanity ----
python3 -c "import torch, numpy" 2>/dev/null \
  || { echo "ERROR: env missing (torch/numpy not importable)"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "ERROR: no CUDA visible"; exit 1; }

echo "=== SBT pipeline ==="
echo "DATASET=$DATASET  STAGE=$STAGE  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ---- defaults ----
LAMBDAS=${LAMBDAS:-0,1e-5,5e-5,1e-4,5e-4,1e-3}
EPS=${EPS:-0.1,0.5,1,2,4,6,8}
SEEDS=${SEEDS:-5}
TE=${TE:-50}
SE=${SE:-40}

if [ "$STAGE" = "1" ]; then
    echo "[stage 1] lambdas=$LAMBDAS  te=$TE"
    python3 drive_sbt_pipeline.py \
        --dataset "$DATASET" --stage 1 \
        --lambdas "$LAMBDAS" \
        --te "$TE"
elif [ "$STAGE" = "2" ]; then
    : "${LAMBDA_SBT:?set LAMBDA_SBT=<chosen lambda from stage 1>}"
    echo "[stage 2] lambda_sbt=$LAMBDA_SBT  eps=$EPS  seeds=$SEEDS  te=$TE  se=$SE"
    python3 drive_sbt_pipeline.py \
        --dataset "$DATASET" --stage 2 \
        --lambda_sbt "$LAMBDA_SBT" \
        --epsilons "$EPS" \
        --seeds "$SEEDS" \
        --te "$TE" --se "$SE"
else
    echo "ERROR: STAGE must be 1 or 2 (got: $STAGE)"
    exit 1
fi

echo
echo "=== done  $(date) ==="
