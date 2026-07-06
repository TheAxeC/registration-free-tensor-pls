#!/bin/bash
#SBATCH --job-name=rfpls-gsweep
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gsweep_%j.out
# Grade-specific sweep (one job, configs run sequentially). seed 0, 5-fold AUC per config.
# Includes a #1/#2 ablation (cfg 7). rank with: rank_sweep.py results/brats_realdata/grade_sweep/
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/grade_sweep
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
CACHE="$HOME/data/BraTS2020/brats2020_grade.npz"
# K Mmax ep lr wd alpha eps geom lcov
CFG=(
  "16 128 150 0.003 1e-3 0.6 0.03 6 0.1"   # 0 current best (AUC~0.935)
  "24 128 150 0.003 1e-3 0.6 0.03 6 0.1"   # 1 more atoms
  "12 128 150 0.003 1e-3 0.6 0.03 6 0.1"   # 2 fewer atoms
  "16 128 150 0.003 1e-3 0.6 0.02 6 0.1"   # 3 sharper transport
  "16 128 150 0.003 1e-3 0.4 0.03 6 0.1"   # 4 feature-weighted (grade is intensity-ish)
  "16 128 150 0.003 1e-3 0.7 0.03 6 0.1"   # 5 structure-weighted
  "16 128 150 0.003 1e-2 0.6 0.03 6 0.1"   # 6 stronger reg
  "16 128 150 0.003 1e-3 0.6 0.03 0 0.0"   # 7 ABLATION: no #1/#2
  "16 128 150 0.003 1e-3 0.5 0.03 10 0.3"  # 8 more geom + cov
)
for i in "${!CFG[@]}"; do
  read K Mmax EP LR WD AL EPS GR LC <<< "${CFG[$i]}"
  echo "=== cfg $i: K=$K eps=$EPS a=$AL wd=$WD geom=$GR lcov=$LC ==="
  "$PY" campaigns/brats_realdata/run_brats.py --cache "$CACHE" --task clf --folds 5 --seed_id 0 \
    --K "$K" --Mmax "$Mmax" --epochs "$EP" --lr "$LR" --wd "$WD" --alpha "$AL" --eps "$EPS" \
    --geom_rank "$GR" --lambda_cov "$LC" --device cuda \
    --out "results/brats_realdata/grade_sweep/cfg_${i}.json"
done
echo "GSWEEP_DONE"
