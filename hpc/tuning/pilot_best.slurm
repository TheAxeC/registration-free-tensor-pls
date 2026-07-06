#!/bin/bash
#SBATCH --job-name=rfpls-pilot
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/pilot_%A_%a.out
# Multi-seed pilot at the sweep-winning config -> confidence intervals on RAW-PLS vs grid.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/pilot
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
    --folds 5 --seed_id "$SLURM_ARRAY_TASK_ID" \
    --K 16 --Mmax 128 --epochs 200 --lr 0.003 --wd 1e-3 --alpha 0.6 --eps 0.03 \
    --device cuda --out "results/brats_realdata/pilot/seed_${SLURM_ARRAY_TASK_ID}.json"
echo "PILOT_${SLURM_ARRAY_TASK_ID}_DONE"
