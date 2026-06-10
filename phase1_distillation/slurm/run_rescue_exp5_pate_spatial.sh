#!/bin/bash
#SBATCH --job-name=R_R5_pate_sp
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/rescue_exp5_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/rescue_exp5_%j.err

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240
export PYTHONNOUSERSITE=1

export REPO=/data/user/home/ialam/DPKD-medical/phase1_distillation
cd "$REPO"
python3 drive_pate_spatial_joint.py
