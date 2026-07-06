#!/bin/bash
#SBATCH --job-name=rfpls-etacurve
#SBATCH --partition=main-gpu
#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/etacurve_%j.out
# C-3: representation-change vs non-isometric distortion eta on real lesions (stability, Prop. non-rigid).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/stats
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
"$PY" campaigns/brats_realdata/eta_curve.py --cache "$HOME/data/BraTS2021/brats_struct.npz" \
    --ncases 120 --out "results/brats_realdata/stats/eta_curve.json"
echo "ETA_CURVE_DONE"
