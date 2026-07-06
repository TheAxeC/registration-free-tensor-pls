#!/bin/bash
#SBATCH --job-name=rfpls-r10sweep
#SBATCH --partition=main-gpu
#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r10_%j.out
# R-10: alpha (structure-vs-feature) and epsilon (entropy) sensitivity around the headline
# operating point (alpha=0.6, eps=0.03). The eps axis also gives the empirical O(eps)
# identification-bias evidence the theory predicts. Lean CROSS design (eps axis at alpha=0.6, alpha
# axis at eps=0.03), 2 seeds x 5 folds per config; eps starts at 0.02 (eps<=0.01 makes the Sinkhorn
# transport numerically stiff and pathologically slow). Single job (QOS submit quota saturated).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results/brats_realdata/sweep_ae
source hpc/env_stage.sh; export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

run() {  # alpha eps
  echo "=== R-10 config alpha=$1 eps=$2 (struct, 2 seeds) ==="
  "$PY" campaigns/brats_realdata/run_brats.py --cache "$HOME/data/BraTS2021/brats_struct.npz" --task reg \
      --model rawpls --folds 5 --seeds 2 --K 16 --Mmax 128 --epochs 200 --lr 0.003 --wd 1e-3 \
      --alpha "$1" --eps "$2" --geom_rank 6 --lambda_cov 0.1 --device cuda \
      --out "results/brats_realdata/sweep_ae/a${1}_e${2}.json"
  echo "R10_CONFIG_DONE alpha=$1 eps=$2"
}
# eps axis at headline alpha=0.6 (O(eps) bias trend)
for E in 0.02 0.03 0.05 0.10; do run 0.6 "$E"; done
# alpha axis at headline eps=0.03 (structure-weight sensitivity)
for A in 0.0 0.3 0.9; do run "$A" 0.03; done
echo "R10_ALL_DONE"
