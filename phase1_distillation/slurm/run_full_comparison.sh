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
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
pip install --quiet huggingface_hub remotezip pyarrow nibabel scikit-image opacus 2>/dev/null

export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
: "${DATASET:?set DATASET=drive|isic|brats}"

# ISIC: build the full dataset if not present yet
if [ "$DATASET" = "isic" ] && [ ! -d ../data/ISIC_HF/train ]; then
  echo "[data] building full ISIC (2594) ..."
  python3 download_isic.py --n 0 --size 96
fi
# BraTS expects data/BraTS_HF from run_prep_brats.sh (sbatch run_prep_brats.sh first)

SEEDS=${SEEDS:-5}; TE=${TE:-60}; SE=${SE:-40}
echo "=== [1/3] utility comparison ($DATASET) ==="
python3 run_comparison.py --dataset "$DATASET" --seeds "$SEEDS" --te "$TE" --se "$SE"
echo "=== [2/3] DP-SGD baseline ($DATASET) ==="
python3 drive_dpsgd_baseline.py --dataset "$DATASET"
echo "=== [3/3] privacy audit: MIA + reconstruction ($DATASET) ==="
python3 drive_joint_attack.py --dataset "$DATASET"
echo "=== DONE: results/${DATASET}_comparison.json, fig_${DATASET}_comparison.png, *_dpsgd_*, *_joint_attack_* ==="
