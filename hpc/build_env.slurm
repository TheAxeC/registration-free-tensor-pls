#!/bin/bash
#SBATCH --job-name=build-env
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/buildenv_%j.out
# Build the regfree-tensor-pls venv overlay ON A COMPUTE NODE (login-node NFS is
# throttled). Reuses hccnet's torch via --system-site-packages; adds POT/nibabel/kaggle.
# Also times hccnet's torch import to confirm compute-node NFS is usable.
set -e
source /software/miniconda3/25.7/etc/profile.d/conda.sh
echo "[$(date +%T)] node $(hostname)"
echo "[$(date +%T)] timing hccnet torch import (compute node):"
time ~/envs/hccnet/bin/python -c "import torch; print('hccnet torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"
echo "[$(date +%T)] (re)building venv overlay..."
rm -rf ~/envs/regfree-tensor-pls
~/envs/hccnet/bin/python -m venv --system-site-packages ~/envs/regfree-tensor-pls
source ~/envs/regfree-tensor-pls/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q POT nibabel kaggle
echo "[$(date +%T)] verifying full stack..."
python -c "import torch,ot,nibabel,kaggle,sklearn,pandas,scipy,numpy; print('OVERLAY OK | torch',torch.__version__,'| cuda_avail',torch.cuda.is_available(),'| POT',ot.__version__,'| nibabel',nibabel.__version__)"
echo BUILDENV_DONE
