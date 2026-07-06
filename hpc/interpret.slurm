#!/bin/bash
#SBATCH --job-name=rfpls-interp
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/interp_%j.out
# Train RAW-PLS on grade and compute real per-voxel transport-saliency for the
# interpretability figure -> interp_data.npz (small; scp down + plot with interp_figure.py).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
"$PY" campaigns/brats_realdata/interpret.py
echo "INTERP_JOB_DONE"
