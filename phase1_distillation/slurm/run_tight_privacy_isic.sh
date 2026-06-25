#!/bin/bash
#SBATCH --job-name=TightPriv
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/tightpriv_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/tightpriv_%j.err

# Tight-privacy panel - ISIC, 3 methods, tight eps, many seeds.
#
# Default: PATE K=1 / PATE+uniform K=10 / PATE+CANAL K=10 at eps in {0.1, 0.5, 1}
# with 20 seeds, so we can resolve CANAL's +1.7-sigma trend (seen in 5-seed run
# at eps=0.1) to >= 2 sigma if real.
#
#   sbatch slurm/run_tight_privacy_isic.sh
#
# Optional overrides:
#   SEEDS=20                  number of student seeds
#   EPS=0.1,0.5,1             epsilons to sweep
#   OUT_TAG=tight20s          suffix for JSON / PNG (avoid overwriting prior runs)
#   TE=50                     teacher epochs
#   SE=40                     student epochs

module purge
module load Anaconda3/2023.07-2 2>/dev/null || module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null \
  || source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"

# ---- env sanity ----
python3 -c "import torch, numpy" 2>/dev/null \
  || { echo "ERROR: env missing (torch/numpy not importable)"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "ERROR: no CUDA visible"; exit 1; }

echo "=== tight-privacy ISIC ==="
echo "$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ---- defaults ----
SEEDS=${SEEDS:-20}
EPS=${EPS:-0.1,0.5,1}
OUT_TAG=${OUT_TAG:-tight20s}
TE=${TE:-50}
SE=${SE:-40}

echo "[tight-privacy] seeds=$SEEDS  eps=$EPS  out_tag=$OUT_TAG  te=$TE  se=$SE"

python3 isic_3method_actnorm_sweep.py \
    --seeds "$SEEDS" \
    --epsilons "$EPS" \
    --te "$TE" --se "$SE" \
    --out_tag "$OUT_TAG"

echo
echo "=== done  $(date) ==="
