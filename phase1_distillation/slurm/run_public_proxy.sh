#!/bin/bash

#SBATCH --job-name=PUBLIC_PROXY

#SBATCH --partition=pascalnodes-medium

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=4G

#SBATCH --time=02:00:00

#SBATCH --output=/data/user/home/ialam/mmsegmentation/slurm_out/public_proxy_%j.out

#SBATCH --error=/data/user/home/ialam/mmsegmentation/slurm_out/public_proxy_%j.err



module purge

source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate mmseg-cu124-240



export MMSEG=/data/user/home/ialam/mmsegmentation

export PYTHONPATH="$MMSEG:$PYTHONPATH"



mkdir -p /data/user/ialam/Datasets/ALL_PUBLIC

cp /data/user/ialam/Datasets/HRF/*.jpg   /data/user/ialam/Datasets/ALL_PUBLIC/ 2>/dev/null

cp /data/user/ialam/Datasets/HRF/*.JPG   /data/user/ialam/Datasets/ALL_PUBLIC/ 2>/dev/null

cp /data/user/ialam/Datasets/STARE/*.ppm /data/user/ialam/Datasets/ALL_PUBLIC/ 2>/dev/null

cp /data/user/ialam/Datasets/CHASE_DB1/images/*.jpg /data/user/ialam/Datasets/ALL_PUBLIC/ 2>/dev/null



echo "Total public images:"

ls /data/user/ialam/Datasets/ALL_PUBLIC/ | wc -l



python $MMSEG/tools/analysis/compute_public_proxy.py --teacher-config /data/user/home/ialam/mmseg_models/unet_teacher/unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py --teacher-checkpoint /data/user/home/ialam/mmseg_models/unet_teacher/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth --public-data-root /data/user/ialam/Datasets/ALL_PUBLIC --out-dir /data/user/home/ialam/mmsegmentation/work_dirs/PUBLIC_PROXY_P05 --image-size 584 565 --cap-quantile 0.05 --device cuda:0



echo "Done. Results in work_dirs/PUBLIC_PROXY_P05/"
