#!/bin/bash
#SBATCH --job-name=rfpls-build-cache
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_%j.out
# Build the TWO point-cloud caches the paper needs, ONCE, from the downloaded BraTS data (reads
# NIfTI from NFS -- slow on this cluster, hence the long walltime). The pilot/array jobs then read
# the small npz caches read-only. Both build from a per-case NIfTI ROOT: point DATA21 at the
# BraTS2021 root and DATA20 at the extracted BraTS2020 MICCAI dir. Modern kagglehub auto-extracts a
# dataset on download (no zip is kept), so the grade side reads the extracted tree, not a zip (see
# README "One-time setup" for the kagglehub download that produces these two roots).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$HOME/data/BraTS2021" "$HOME/data/BraTS2020"
source hpc/env_stage.sh
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/campaigns/brats_realdata:$PYTHONPATH"
DATA21=${DATA21:-$HOME/data/BraTS2021_raw}     # BraTS2021 per-case NIfTI root (structural target)
DATA20=${DATA20:-$HOME/data/BraTS2020_train}   # BraTS2020 extracted MICCAI dir (grade target; has name_mapping.csv + case folders)
# structural cache (ET-NCR centroid distance) -> ~/data/BraTS2021/brats_struct.npz
"$PY" campaigns/brats_realdata/build_seg_target.py --root "$DATA21" --out "$HOME/data/BraTS2021/brats_struct.npz"
# grade cache (HGG/LGG) from the extracted BraTS2020 dir -> ~/data/BraTS2020/brats2020_grade.npz
"$PY" campaigns/brats_realdata/build_grade_from_dir.py --root "$DATA20" --out "$HOME/data/BraTS2020/brats2020_grade.npz"
