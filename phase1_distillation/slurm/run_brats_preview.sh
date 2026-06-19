#!/bin/bash
#SBATCH --job-name=BraTSprev
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/brats_prev_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/brats_prev_%j.err

# Run AFTER run_prep_brats.sh has built data/BraTS_HF/.
# Headline series + the multi-modal CANAL test, ε∈{1,2,3,4,5}, 3 seeds.

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dpkd-cv}
export PYTHONNOUSERSITE=1

export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
python3 brats_preview_sweep.py
