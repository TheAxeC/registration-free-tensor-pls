#!/bin/bash
#SBATCH --job-name=rfpls-radiomics
#SBATCH --partition=main-gpu
#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/radiomics_%j.out
# C-1: lesion-cropped registered radiomics baseline (breaks the registration-vs-rasterization
# confound). CPU-only work (sklearn PLS + numpy, no GW/torch); runs on main-gpu because main-cpu
# nodes are down/drained, and the short 30-min walltime lets SLURM backfill slot it into a gap
# ahead of the large pending array. Four configs: {struct, grade} x {affine (abs), deformable
# (syn)}, each paired vs the stored full-method RAW-PLS folds (improved.json / grade.json).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/reg
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

STRUCT_STORED="results/brats_realdata/improved/improved.json"
GRADE_STORED="results/brats_realdata/grade/grade.json"

echo "=== C-1 struct / AFFINE (abs SRI24) ==="
"$PY" campaigns/brats_realdata/reg_radiomics.py --cache "$HOME/data/BraTS2021/brats_struct_abs.npz" \
    --task reg --stored "$STRUCT_STORED" --out "results/brats_realdata/reg/rad_struct_abs.json"
echo "=== C-1 struct / DEFORMABLE (SyN) ==="
"$PY" campaigns/brats_realdata/reg_radiomics.py --cache "$HOME/data/BraTS2021/brats_struct_syn.npz" \
    --task reg --stored "$STRUCT_STORED" --out "results/brats_realdata/reg/rad_struct_syn.json"
echo "=== C-1 grade / AFFINE (abs SRI24) ==="
"$PY" campaigns/brats_realdata/reg_radiomics.py --cache "$HOME/data/BraTS2020/brats2020_grade_abs.npz" \
    --task clf --stored "$GRADE_STORED" --out "results/brats_realdata/reg/rad_grade_abs.json"
echo "=== C-1 grade / DEFORMABLE (SyN) ==="
"$PY" campaigns/brats_realdata/reg_radiomics.py --cache "$HOME/data/BraTS2020/brats2020_grade_syn.npz" \
    --task clf --stored "$GRADE_STORED" --out "results/brats_realdata/reg/rad_grade_syn.json"
echo "RADIOMICS_DONE"
