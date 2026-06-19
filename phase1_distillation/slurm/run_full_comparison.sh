#!/bin/bash
#SBATCH --job-name=DPcompare
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/dpcompare_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/dpcompare_%j.err

# Full DP experiment suite for ONE dataset, in one job:
#   utility comparison (run_comparison) + DP-SGD baseline + MIA/reconstruction audit.
#
#   sbatch --export=ALL,DATASET=isic  slurm/run_full_comparison.sh
#   sbatch --export=ALL,DATASET=brats slurm/run_full_comparison.sh   # after run_prep_brats.sh
#   optional: SEEDS=5 TE=60 SE=40

module purge
# make conda available (Lmod module first, then fall back to sourcing the install)
module load Anaconda3/2023.07-2 2>/dev/null || module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null \
  || source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
: "${DATASET:?set DATASET=drive|isic|brats}"

# Deps + data are prepared ONCE on the LOGIN node (compute nodes may lack internet):
#   pip install remotezip opacus pyarrow nibabel scikit-image huggingface_hub
#   python3 download_isic.py --n 0 --size 96            # ISIC
#   python3 prep_brats.py --src <BraTS root> --size 96  # BraTS
python3 -c "import torch, numpy, matplotlib, opacus" 2>/dev/null || {
  echo "ERROR: env/deps missing. On the LOGIN node run:"
  echo "  conda activate ${CONDA_ENV:-dpkd-cv} && pip install remotezip opacus pyarrow nibabel scikit-image huggingface_hub"; exit 1; }
if [ "$DATASET" = "isic" ] && [ ! -e ../data/ISIC_HF/train/input ]; then
  echo "ERROR: data/ISIC_HF empty. On the LOGIN node run: python3 download_isic.py --n 0 --size 96"; exit 1; fi
if [ "$DATASET" = "brats" ] && [ ! -e ../data/BraTS_HF/train/input ]; then
  echo "ERROR: data/BraTS_HF missing. Run prep_brats.py / run_prep_brats.sh first"; exit 1; fi

SEEDS=${SEEDS:-5}; TE=${TE:-60}; SE=${SE:-40}
echo "=== [1/3] utility comparison ($DATASET) ==="
python3 run_comparison.py --dataset "$DATASET" --seeds "$SEEDS" --te "$TE" --se "$SE"
echo "=== [2/3] DP-SGD baseline ($DATASET) ==="
python3 drive_dpsgd_baseline.py --dataset "$DATASET"
echo "=== [3/3] privacy audit: MIA + reconstruction ($DATASET) ==="
python3 drive_joint_attack.py --dataset "$DATASET"
echo "=== DONE: results/${DATASET}_comparison.json, fig_${DATASET}_comparison.png, *_dpsgd_*, *_joint_attack_* ==="
