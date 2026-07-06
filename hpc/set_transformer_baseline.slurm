#!/bin/bash
#SBATCH --job-name=regfree-settrans
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Upgrade track D — Set Transformer baseline (Lee et al. 2019) on the cached point clouds.
# 10 seeds x 5 folds on struct (reg) + grade (clf), same harness as DeepSets. Uses the torch env.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/settransformer
source hpc/env_stage.sh
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "=== struct (BraTS2021, R2) ==="
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" \
    --task reg --model settransformer --seeds 10 --folds 5 --epochs 200 --Mmax 128 \
    --out results/brats_realdata/settransformer/struct.json
echo "=== grade (BraTS2020, AUC) ==="
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2020/brats2020_grade.npz" \
    --task clf --model settransformer --seeds 10 --folds 5 --epochs 200 --Mmax 128 \
    --out results/brats_realdata/settransformer/grade.json
echo "SETTRANS_DONE"
