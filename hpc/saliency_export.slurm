#!/bin/bash
#SBATCH --job-name=regfree-saliency
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Retrain the full-method struct model on GPU + export per-voxel gradient saliency mapped to
# image space + tumor-cropped anatomy, for the example BraTS2021 patients. Reads the 4 example
# case folders from ~/data/brats21_examples (copied by the login download driver) + the struct
# cache from ~/data. Writes results/brats_realdata/reg/saliency_export.npz (pulled + rendered locally).
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/brats_realdata/reg
source hpc/env_stage.sh
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
"$PY" campaigns/brats_realdata/saliency_export.py \
    --raw_root "$HOME/data/brats21_examples" \
    --ids BraTS2021_01180 BraTS2021_00525 BraTS2021_01402 BraTS2021_00383 \
    --out results/brats_realdata/reg/saliency_export.npz
echo "SALIENCY_EXPORT_DONE"
