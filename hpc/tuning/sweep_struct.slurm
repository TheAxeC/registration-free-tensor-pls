#!/bin/bash
#SBATCH --job-name=rfpls-sweep
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/sweep_%A_%a.out
# Hyperparameter campaign on the structural target (seed 0, full 5-fold CV per config).
# Each array task = one config; writes results/brats_realdata/sweep/cfg_<id>.json with cfg + ours/base R^2.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/sweep
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
# config table (index = SLURM_ARRAY_TASK_ID): K Mmax epochs lr wd alpha eps
CFG=(
  "12 128 120 0.005 1e-4 0.5 0.05"   # 0 base (replicates the 0.067 point at full CV)
  "12 128 120 0.002 1e-4 0.5 0.05"   # 1 lower lr
  "12 128 120 0.010 1e-4 0.5 0.05"   # 2 higher lr
  "8  128 120 0.005 1e-4 0.5 0.05"   # 3 fewer atoms
  "20 128 120 0.005 1e-4 0.5 0.05"   # 4 more atoms
  "12 128 120 0.005 1e-4 0.5 0.02"   # 5 sharper transport
  "12 128 120 0.005 1e-4 0.5 0.10"   # 6 softer transport
  "12 128 120 0.005 1e-4 0.3 0.05"   # 7 feature-weighted
  "12 128 120 0.005 1e-4 0.7 0.05"   # 8 structure-weighted
  "12 128 120 0.005 1e-3 0.5 0.05"   # 9 more weight decay
  "12 128 120 0.005 1e-2 0.5 0.05"   # 10 strong weight decay
  "20 128 120 0.005 1e-3 0.5 0.05"   # 11 capacity + reg
  "8  128 120 0.005 1e-3 0.7 0.02"   # 12 sharp/structure/few atoms
  "12 256 120 0.005 1e-4 0.5 0.05"   # 13 more points/lesion
  "16 128 200 0.003 1e-3 0.6 0.03"   # 14 balanced + longer
  "12 128 200 0.002 1e-3 0.5 0.05"   # 15 slow + reg + long
  "24 128 120 0.005 1e-2 0.5 0.05"   # 16 big K + strong reg
  "12 128 120 0.005 5e-4 0.6 0.04"   # 17 mild combo
)
read K Mmax EP LR WD AL EPS <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "config $SLURM_ARRAY_TASK_ID: K=$K Mmax=$Mmax ep=$EP lr=$LR wd=$WD alpha=$AL eps=$EPS"
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
    --folds 5 --seed_id 0 --K "$K" --Mmax "$Mmax" --epochs "$EP" \
    --lr "$LR" --wd "$WD" --alpha "$AL" --eps "$EPS" --device cuda \
    --out "results/brats_realdata/sweep/cfg_${SLURM_ARRAY_TASK_ID}.json"
echo "SWEEP_${SLURM_ARRAY_TASK_ID}_DONE"
