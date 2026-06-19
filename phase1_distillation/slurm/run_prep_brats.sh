#!/bin/bash
#SBATCH --job-name=PrepBraTS
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/prep_brats_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/prep_brats_%j.err

# Build data/BraTS_HF/ from a BraTS root of per-case NIfTI folders.
# Set BRATS_SRC to the BraTS training dir (each case has *_t1/_t1ce/_t2/_flair/_seg.nii.gz).
#   sbatch --export=ALL,BRATS_SRC=/path/to/BraTS2021_Training,N=0,SIZE=96 run_prep_brats.sh

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1
pip install --quiet nibabel scikit-image 2>/dev/null

export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
: "${BRATS_SRC:?set BRATS_SRC=/path/to/BraTS root}"
python3 prep_brats.py --src "$BRATS_SRC" --n "${N:-0}" --size "${SIZE:-96}" --slices-per-case "${SLICES:-1}"
