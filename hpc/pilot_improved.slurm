#!/bin/bash
#SBATCH --job-name=rfpls-improved
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/improved_%j.out
# ONE job: all 10 seeds run sequentially on a single GPU (env staged once; no array =
# no queue contention). Best config + the #1/#2 improvements. Per-fold lines are printed
# to the log as a backup in case the final json isn't reached.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/improved
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
    --folds 5 --seeds 10 \
    --K 16 --Mmax 128 --epochs 200 --lr 0.003 --wd 1e-3 --alpha 0.6 --eps 0.03 \
    --geom_rank 6 --lambda_cov 0.1 --device cuda \
    --out "results/brats_realdata/improved/improved.json"
echo "IMPROVED_PILOT_DONE"
