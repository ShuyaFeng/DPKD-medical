#!/bin/bash
#SBATCH --job-name=HonestCANAL
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/honestcanal_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/honestcanal_%j.err

# Honest CANAL accounting comparison on ISIC at fixed K.
# Compares uniform / CANAL (raw, free importance) / CANAL (honest, paid importance).
#
#   sbatch slurm/run_honest_canal_isic.sh
#
# Optional overrides:
#   K=10                          fixed teacher count
#   SEEDS=5                       student seeds
#   EPS=0.1,0.5,1,2,4,6,8
#   ALPHA_IMP=0.1                 fraction of rho spent on importance release
#   CLIP_IMP=100.0                per-sample L2 clip
#   OUT_TAG=v1                    suffix on JSON/PNG
#   TE=50  SE=40

module purge
module load Anaconda3/2023.07-2 2>/dev/null || module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null \
  || source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"

python3 -c "import torch, numpy" 2>/dev/null \
  || { echo "ERROR: env missing"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available()" \
  || { echo "no CUDA"; exit 1; }

echo "=== ISIC honest-CANAL ==="
echo "$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

K=${K:-10}
SEEDS=${SEEDS:-5}
EPS=${EPS:-0.1,0.5,1,2,4,6,8}
ALPHA_IMP=${ALPHA_IMP:-0.1}
CLIP_IMP=${CLIP_IMP:-100.0}
TEACHER_BASE=${TEACHER_BASE:-32}
STUDENT_BASE=${STUDENT_BASE:-0}
OUT_TAG=${OUT_TAG:-}
TE=${TE:-50}
SE=${SE:-40}

echo "[honest CANAL] K=$K seeds=$SEEDS eps=$EPS alpha=$ALPHA_IMP clip=$CLIP_IMP teacher_base=$TEACHER_BASE student_base=$STUDENT_BASE tag=$OUT_TAG"

python3 isic_honest_canal_compare.py \
    --K "$K" \
    --seeds "$SEEDS" \
    --epsilons "$EPS" \
    --alpha_imp "$ALPHA_IMP" \
    --clip_imp "$CLIP_IMP" \
    --teacher_base "$TEACHER_BASE" \
    --student_base "$STUDENT_BASE" \
    --te "$TE" --se "$SE" \
    --out_tag "$OUT_TAG"

echo
echo "=== done $(date) ==="
