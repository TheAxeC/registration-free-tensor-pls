#!/bin/bash
#SBATCH --job-name=rfpls-struct
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-9                 # 10 seeds -> 10 parallel GPU tasks
#SBATCH --output=logs/struct_%A_%a.out
# Full pilot on the structural target (ET-NCR separation), regression.
# One array task = one seed; reads the shared npz read-only, writes results/brats_realdata/main/seed_<id>.json.
# Memory-safe trainer (per-minibatch device moves, float32, Mmax 128).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/main
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
    --folds 5 --seed_id "$SLURM_ARRAY_TASK_ID" --K 12 --Mmax 128 --epochs 200 \
    --device cuda --out "results/brats_realdata/main/seed_${SLURM_ARRAY_TASK_ID}.json"
echo "STRUCT_SEED_${SLURM_ARRAY_TASK_ID}_DONE"
