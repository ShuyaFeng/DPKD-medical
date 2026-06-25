#!/bin/bash
#SBATCH --job-name=KSweep
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/ksweep_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/ksweep_%j.err

# ISIC K-sweep x eps-sweep x methods.
#
# Methods: PATE+uniform, PATE+CANAL  (K=1 PATE+uniform == K=1 PATE baseline)
# Ks:      1, 3, 5, 10
# Eps:     0.1, 0.5, 1, 2, 4, 6, 8
# Seeds:   5
# Total:   4 * 7 * 2 = 56 cells * 5 seeds = 280 student trainings (~15h on GPU)
#
#   sbatch slurm/run_K_eps_sweep_isic.sh
#
# Optional overrides:
#   KS=1,3,5,10
#   EPS=0.1,0.5,1,2,4,6,8
#   SEEDS=5
#   OUT_TAG=v1            (suffix on JSON/PNG filename)
#   TE=50  SE=40

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
  || { echo "ERROR: env missing"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "ERROR: no CUDA visible"; exit 1; }

echo "=== ISIC K x eps sweep ==="
echo "$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

KS=${KS:-1,3,5,10}
METHODS=${METHODS:-uniform,CANAL}
EPS=${EPS:-0.1,0.5,1,2,4,6,8}
SEEDS=${SEEDS:-5}
OUT_TAG=${OUT_TAG:-}
TE=${TE:-50}
SE=${SE:-40}

echo "[K-sweep] Ks=$KS methods=$METHODS eps=$EPS seeds=$SEEDS out_tag=$OUT_TAG te=$TE se=$SE"

python3 isic_K_eps_sweep.py \
    --Ks "$KS" \
    --methods "$METHODS" \
    --epsilons "$EPS" \
    --seeds "$SEEDS" \
    --te "$TE" --se "$SE" \
    --out_tag "$OUT_TAG"

echo
echo "=== done $(date) ==="
