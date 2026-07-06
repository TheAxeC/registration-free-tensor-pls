#!/bin/bash
# FULL from-scratch reproduction of the RAW-PLS real-data results. Nothing computed is assumed to
# pre-exist: with FRESH=1 it wipes every computed cache, then rebuilds BOTH point-cloud caches from
# the raw BraTS, trains every model, and writes every table. The only inputs a fresh downloader must
# supply are the packed env(s) and the raw BraTS data (credentialed on Kaggle; see README setup).
#
# Each stage is a SLURM job chained with afterok (one stage active at a time). The script EXITS in
# seconds; SLURM walks the chain over the next hours/days (log out freely).
#
# The heavy registration + radiomics arm (C-1 / Table 2) is OPT-IN via REG_ARM=1. It does NOT queue
# ~60 jobs at once: a post-core helper fires hpc/run_reg.sh, which SELF-CHAINS the arm and submits
# the two 25-shard deformable-SyN arrays in waves of WAVE (default 8), so the queue never floods.
#
#   FRESH=1 bash hpc/run_all.sh                    # from-scratch CORE chain (headline + baselines + cache-only hardening)
#   FRESH=1 REG_ARM=1 bash hpc/run_all.sh          # ...ALSO the registration arm (wave-chained, self-driven, multi-day)
#   DRY_RUN=1 REG_ARM=1 bash hpc/run_all.sh        # print the WHOLE plan (every stage) without submitting anything
#   (WAVE=5 lowers the SyN queue footprint further; FRESH=0 keeps existing caches - NOT a from-scratch run)
#
# PREREQUISITES a fresh downloader supplies (checked loudly below; see README "One-time setup"):
#   - packed env  ~/envs/regfree_env.tar.gz   (+ ~/envs/regfree_ants_env.tar.gz for REG_ARM)
#   - raw BraTS   ~/data/BraTS2021_raw/  (struct)  +  ~/data/BraTS2020_train/  (grade, holds name_mapping.csv)
# After the chain finishes, regenerate the figures locally:  python campaigns/brats_realdata/make_figures.py
set -e
DRY="${DRY_RUN:-0}"
fail(){ echo "[run_all] MISSING PREREQUISITE: $1" >&2; echo "  fix: $2" >&2; exit 1; }
[ -f "$HOME/envs/regfree_env.tar.gz" ] || fail "packed env ~/envs/regfree_env.tar.gz" "build it once via hpc/build_env.slurm (see README)"
[ -d "$HOME/data/BraTS2021_raw" ]      || fail "BraTS2021 raw NIfTI ~/data/BraTS2021_raw/" "kagglehub-download dschettler8845/brats-2021-task1 and point it there (README setup)"
[ -f "$HOME/data/BraTS2020_train/name_mapping.csv" ] || fail "BraTS2020 extracted dir ~/data/BraTS2020_train/ (with name_mapping.csv)" "kagglehub-download awsaf49/brats20-dataset-training-validation and symlink its MICCAI_BraTS2020_TrainingData there (README setup)"
if [ "${REG_ARM:-0}" = "1" ]; then
  [ -f "$HOME/envs/regfree_ants_env.tar.gz" ] || fail "packed antspyx env ~/envs/regfree_ants_env.tar.gz" "build it once via hpc/build_env.slurm (the SyN/registration arm needs it)"
fi

cd "$(dirname "$0")/.."                 # -> repo root ($HOME/projects/regfree-tensor-pls on the cluster)
ROOT="$PWD"
mkdir -p logs results/brats_realdata/reg results/brats_realdata/stats results/brats_realdata/sweep_ae

# --- FRESH: wipe every COMPUTED intermediate so build_cache + the chain genuinely rebuild all -------
if [ "${FRESH:-0}" = "1" ]; then
  echo "[run_all] FRESH=1 -> wiping computed caches + results (build_cache + the chain rebuild them all)"
  if [ "$DRY" = 1 ]; then echo "[dry] rm point-cloud + abs + syn caches + results/brats_realdata/{pilot,improved,grade,deepset,settransformer,reg,stats,sweep_ae,syn_*}"; else
    rm -f "$HOME/data/BraTS2021/brats_struct.npz" "$HOME/data/BraTS2021/brats_struct_abs.npz" "$HOME/data/BraTS2021/brats_struct_syn.npz"
    rm -f "$HOME/data/BraTS2020/brats2020_grade.npz" "$HOME/data/BraTS2020/brats2020_grade_abs.npz" "$HOME/data/BraTS2020/brats2020_grade_syn.npz"
    rm -rf results/brats_realdata/pilot results/brats_realdata/improved results/brats_realdata/grade \
           results/brats_realdata/deepset results/brats_realdata/settransformer results/brats_realdata/reg \
           results/brats_realdata/stats results/brats_realdata/sweep_ae results/brats_realdata/syn_struct results/brats_realdata/syn_grade
    mkdir -p results/brats_realdata/reg results/brats_realdata/stats results/brats_realdata/sweep_ae
  fi
fi

# submit helper (echoes + returns a fake id under DRY_RUN so the whole plan can be walked).
sub(){ if [ "$DRY" = 1 ]; then echo "[dry] sbatch $*" >&2; echo "DRY"; else sbatch --parsable "$@"; fi; }

# --- stage 0: rebuild BOTH point-cloud caches from the raw BraTS -------------------------------------
BC=$(sub --job-name=rfpls-build-cache hpc/build_cache.slurm);                       echo "0  build caches (struct+grade) = $BC  -> ~/data/BraTS20{21,20}/*.npz"

# --- core result chain (each waits for the previous; all read the two caches) -----------------------
PB=$(sub  --dependency=afterok:$BC  hpc/tuning/pilot_best.slurm);                    echo "1  base struct pilot      = $PB   -> results/brats_realdata/pilot/"
STR=$(sub --dependency=afterok:$PB  hpc/pilot_improved.slurm);                       echo "2  struct pilot (headline)= $STR  -> results/brats_realdata/improved/improved.json"
GRD=$(sub --dependency=afterok:$STR hpc/grade_pilot.slurm);                          echo "3  grade pilot (headline) = $GRD  -> results/brats_realdata/grade/grade.json"
DST=$(sub --dependency=afterok:$GRD hpc/deepset_baseline.slurm);                     echo "4  DeepSets baseline      = $DST  -> results/brats_realdata/deepset/"
SET=$(sub --dependency=afterok:$DST hpc/set_transformer_baseline.slurm);             echo "5  Set Transformer        = $SET  -> results/brats_realdata/settransformer/"
SAL=$(sub --dependency=afterok:$SET hpc/saliency_export.slurm);                      echo "6  saliency export        = $SAL  -> results/brats_realdata/reg/saliency_export.npz"
GS=$(sub  --dependency=afterok:$SAL hpc/grade_scores.slurm);                         echo "7  grade AUPRC (R-3b)     = $GS   -> results/brats_realdata/stats/grade_scores.json"
ETA=$(sub --dependency=afterok:$GS  hpc/eta_curve.slurm);                            echo "8  eta stability (C-3)    = $ETA  -> results/brats_realdata/stats/eta_curve.json"
R10=$(sub --dependency=afterok:$ETA hpc/r10_sweep.slurm);                            echo "9  alpha/eps sweep (R-10) = $R10  -> results/brats_realdata/sweep_ae/"

# --- registration + radiomics arm (opt-in): fire the self-chaining walker AFTER the core drains -----
if [ "${REG_ARM:-0}" = "1" ]; then
  if [ "$DRY" = 1 ]; then
    echo "10 registration walker    -> after core: run_reg.sh absaff (self-chains; SyN in waves of ${WAVE:-8})"; echo "   --- registration plan ---"; DRY_RUN=1 bash hpc/run_reg.sh absaff
  else
    REG=$(sbatch --parsable --dependency=afterok:$R10 --job-name=rfpls-reg-kick --partition=main-cpu \
        --cpus-per-task=1 --mem=1G --time=00:05:00 --output=logs/reg_kick_%j.out \
        --wrap="cd $ROOT && bash hpc/run_reg.sh absaff")
    echo "10 registration walker kick = $REG  -> self-chains absaff -> SyN(waves of ${WAVE:-8}) -> asmrad (C-1 / Table 2)"
  fi
else
  echo "-- registration arm NOT submitted (opt-in). Add REG_ARM=1 for the C-1 / Table-2 arm (wave-chained, self-driven)."
fi

echo
[ "$DRY" = 1 ] && echo "[dry] plan only - nothing submitted." || echo "submitted. watch: squeue -u $(id -un) ; logs: logs/ ; you can log out."
echo "when the whole chain finishes, locally: python campaigns/brats_realdata/make_figures.py"
