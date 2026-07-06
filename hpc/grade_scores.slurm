#!/bin/bash
#SBATCH --job-name=rfpls-gradeauprc
#SBATCH --partition=main-gpu
#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gradescores_%j.out
# R-3b: grade AUPRC + bootstrap AUC/AUPRC CIs (per-sample scores), imbalanced cohort n=368.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/stats
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
"$PY" campaigns/brats_realdata/grade_scores.py --cache "$HOME/data/BraTS2020/brats2020_grade.npz" \
    --out "results/brats_realdata/stats/grade_scores.json"
echo "GRADE_SCORES_DONE"
