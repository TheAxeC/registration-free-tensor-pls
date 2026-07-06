#!/bin/bash
#SBATCH --job-name=brats-rawpls
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-9                 # 10 seeds -> 10 parallel GPU tasks
#SBATCH --output=logs/seed_%A_%a.out
# One array task = one seed: reads the shared cache read-only, writes an isolated
# results/seed_<id>.json. No cross-task races; SLURM spreads tasks across GPUs.
# Env is staged from the single tarball to node-local /local (see env_stage.sh).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/main
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
CACHE=${CACHE:-$HOME/data/BraTS2021/brats_clf.npz}; TASK=${TASK:-clf}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" campaigns/brats_realdata/run_brats.py --cache "$CACHE" --task "$TASK" --folds 5 \
    --seed_id "$SLURM_ARRAY_TASK_ID" --K 12 --Mmax 256 --epochs 400 --device cuda \
    --out "results/brats_realdata/main/seed_${SLURM_ARRAY_TASK_ID}.json"
