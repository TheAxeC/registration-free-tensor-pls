#!/bin/bash
#SBATCH --job-name=rfpls-grade
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/grade_%j.out
# AAAI-push test: BraTS2020 GRADE (HGG vs LGG, classification/AUC) — a high-signal clinical
# target. One job, 10 seeds sequential, best config + #1/#2 improvements. Per-fold AUC printed.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/grade
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2020/brats2020_grade.npz" --task clf \
    --folds 5 --seeds 10 \
    --K 16 --Mmax 128 --epochs 200 --lr 0.003 --wd 1e-3 --alpha 0.6 --eps 0.03 \
    --geom_rank 6 --lambda_cov 0.1 --device cuda \
    --out "results/brats_realdata/grade/grade.json"
echo "GRADE_PILOT_DONE"
