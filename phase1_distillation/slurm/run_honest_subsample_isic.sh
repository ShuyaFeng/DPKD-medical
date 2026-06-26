#!/bin/bash
#SBATCH --job-name=HonestSub
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/honestsub_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/honestsub_%j.err

# Honest CANAL + channel subsampling: pay rho_imp for noisy importance, then
# keep top-keep_frac channels by noisy importance and drop the rest.
#
# Usage:
#   K=3 single job:
#     sbatch --export=ALL,K_LIST=3,OUT_TAG=K3 slurm/run_honest_subsample_isic.sh
#
#   K=1,5,10 combined job (loops over the K values internally):
#     sbatch --export=ALL,K_LIST=1,5,10,OUT_TAG=K1_5_10 slurm/run_honest_subsample_isic.sh
#
# Optional overrides:
#   EPS=0.5,1,2
#   KEEP_FRACS=0.5,0.25,0.1,0.05
#   SEEDS=5
#   ALPHA_IMP=0.10
#   CLIP_IMP=100.0
#   TE=50 SE=40
#   TEACHER_BASE=32 STUDENT_BASE=0

module purge
module load Anaconda3/2023.07-2 2>/dev/null || module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null \
  || source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"

python3 -c "import torch, numpy" 2>/dev/null || { echo "ERROR env missing"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

echo "=== ISIC honest-subsample ==="
echo "$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

K_LIST=${K_LIST:-3}
EPS=${EPS:-0.5,1,2}
KEEP_FRACS=${KEEP_FRACS:-0.5,0.25,0.1,0.05}
SEEDS=${SEEDS:-5}
ALPHA_IMP=${ALPHA_IMP:-0.10}
CLIP_IMP=${CLIP_IMP:-100.0}
TE=${TE:-50}
SE=${SE:-40}
TEACHER_BASE=${TEACHER_BASE:-32}
STUDENT_BASE=${STUDENT_BASE:-0}
OUT_TAG=${OUT_TAG:-}
SKIP_RANDOM=${SKIP_RANDOM:-0}     # set to 1 to skip random_subsample control

echo "[honest_subsample] K_LIST=$K_LIST  eps=$EPS  keep_fracs=$KEEP_FRACS  seeds=$SEEDS  alpha=$ALPHA_IMP  clip=$CLIP_IMP  tb=$TEACHER_BASE  sb=$STUDENT_BASE  skip_random=$SKIP_RANDOM  tag=$OUT_TAG"

# Build extra flag
SKIP_FLAG=""
if [ "$SKIP_RANDOM" = "1" ]; then SKIP_FLAG="--skip_random"; fi

# Loop over each K value in K_LIST
for K in $(echo "$K_LIST" | tr ',' ' '); do
    SUB_TAG="${OUT_TAG}_K${K}"
    SUB_TAG="${SUB_TAG#_}"   # strip leading _ if OUT_TAG was empty
    echo
    echo "------ K=$K  out_tag=$SUB_TAG  $(date) ------"
    python3 isic_honest_subsample.py \
        --K "$K" \
        --epsilons "$EPS" \
        --keep_fracs "$KEEP_FRACS" \
        --seeds "$SEEDS" \
        --alpha_imp "$ALPHA_IMP" \
        --clip_imp "$CLIP_IMP" \
        --te "$TE" --se "$SE" \
        --teacher_base "$TEACHER_BASE" \
        --student_base "$STUDENT_BASE" \
        --out_tag "$SUB_TAG" \
        $SKIP_FLAG
done

echo
echo "=== done $(date) ==="
