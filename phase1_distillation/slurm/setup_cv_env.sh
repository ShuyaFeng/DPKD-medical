#!/bin/bash
# ============================================================
# Build a dedicated conda env for the CV/distillation experiments.
# Run ONCE on a Cheaha LOGIN node (no GPU needed to install):
#     bash setup_cv_env.sh
#
# Creates env "dpkd-cv" with torch + the imaging stack the
# drive_*.py scripts need. After this, submit jobs with:
#     CONDA_ENV=dpkd-cv bash submit_all_paper_experiments.sh
# ============================================================
set -e

ENV_NAME="${ENV_NAME:-dpkd-cv}"

module purge
source /share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Env '${ENV_NAME}' already exists — activating to update."
    conda activate "${ENV_NAME}"
else
    echo "Creating conda env '${ENV_NAME}' (python 3.10)..."
    conda create -n "${ENV_NAME}" python=3.10 -y
    conda activate "${ENV_NAME}"
fi

echo "Installing packages (torch + imaging stack)..."
pip install --upgrade pip
pip install torch numpy pillow scikit-image scipy matplotlib huggingface_hub datasets

echo ""
echo "=== verify ==="
python -c "import torch, numpy, PIL, skimage, scipy, matplotlib; \
print('torch', torch.__version__, '| cuda build:', torch.version.cuda, \
'| cuda avail (False on login is OK):', torch.cuda.is_available())"

echo ""
echo "Done. Env name: ${ENV_NAME}"
echo "Use it in jobs with:  CONDA_ENV=${ENV_NAME} bash submit_all_paper_experiments.sh"
