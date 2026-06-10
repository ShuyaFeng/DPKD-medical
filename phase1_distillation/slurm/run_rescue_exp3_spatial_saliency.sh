#!/bin/bash
#SBATCH --job-name=R_R3_sal
#SBATCH --partition=pascalnodes-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/rescue_exp3_%j.out
#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/rescue_exp3_%j.err

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate mmseg-cu124-240
export PYTHONNOUSERSITE=1

# Optional: download HRF here if not present (script falls back to synthetic
# radial saliency if HRF missing, but real HRF gives the strongest signal)
HRF=$HOME/public_retinal/HRF
if [ ! -d "$HRF" ]; then
    echo "Downloading HRF to $HRF ..."
    mkdir -p "$HRF" && cd "$HRF"
    wget -q https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
    unzip -q all.zip && rm all.zip
fi

export REPO=/data/user/home/ialam/DPKD-medical/phase1_distillation
cd "$REPO"
python3 drive_spatial_saliency.py
