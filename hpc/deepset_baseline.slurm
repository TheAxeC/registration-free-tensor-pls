#!/bin/bash
#SBATCH --job-name=rfpls-deepset
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/deepset_%j.out
# Strong/modern baseline: permutation-invariant DeepSets on the same point clouds, SAME CV.
# Both real targets, 10 seeds each. DeepSets is light (no GW) so both fit one job.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/deepset
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
echo "=== DeepSets on GRADE (HGG/LGG, AUC) ==="
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2020/brats2020_grade.npz" --task clf \
    --folds 5 --seeds 10 --model deepset --Mmax 128 --epochs 200 --lr 1e-3 --wd 1e-4 \
    --device cuda --out "results/brats_realdata/deepset/grade_deepset.json"
echo "=== DeepSets on STRUCTURAL target (R2) ==="
"$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
    --folds 5 --seeds 10 --model deepset --Mmax 128 --epochs 200 --lr 1e-3 --wd 1e-4 \
    --device cuda --out "results/brats_realdata/deepset/struct_deepset.json"
echo "DEEPSET_DONE"
