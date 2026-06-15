#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/home/fengs/DPKD-medical/slurm_out/alpha_sweep_%j.out
#SBATCH --error=/home/fengs/DPKD-medical/slurm_out/alpha_sweep_%j.err

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-pytorch}
export PYTHONNOUSERSITE=1

export REPO=/home/fengs/DPKD-medical/phase1_distillation
cd "$REPO"
python3 drive_alpha_sweep.py
