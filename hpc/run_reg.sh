#!/bin/bash
# Self-chaining registration + radiomics arm for the RAW-PLS full reproduction (the C-1 / Table-2
# confound-breaker). It walks the arm one SMALL piece at a time: each piece, on success, fires a tiny
# afterok helper that re-invokes THIS script for the next piece. So the queue never holds more than
# the current step (<= WAVE deformable-SyN shards) plus one pending helper -- the ~50-shard arm never
# lands in the queue at once. Fully cluster-driven: submit the first phase once and log out.
#
#   bash hpc/run_reg.sh absaff             # entry point (run_all.sh fires this after the core drains)
#   DRY_RUN=1 bash hpc/run_reg.sh absaff   # print the WHOLE plan (every phase) without submitting
#
# Phase order:  absaff -> synstruct 0,8,16,24 -> syngrade 0,8,16,24 -> asmrad
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> repo root ($HOME/projects/regfree-tensor-pls on cluster)
ROOT="$PWD"
mkdir -p logs results/brats_realdata/reg results/brats_realdata/syn_struct results/brats_realdata/syn_grade
B21="$HOME/data/BraTS2021_raw"                # BraTS2021 per-case NIfTI (struct)
B20="$HOME/data/BraTS2020_train"             # BraTS2020 extracted MICCAI dir (grade) - same source as the grade cache
WAVE="${WAVE:-8}"                             # deformable-SyN shards submitted per wave (queue footprint cap)
NSHARDS=25
ENVP='source hpc/env_stage.sh && export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/campaigns/brats_realdata'

PHASE="${1:?usage: run_reg.sh <absaff|synstruct|syngrade|asmrad> [wave_offset]}"
OFF="${2:-0}"

# submit a bash job (sbatch --wrap defaults to /bin/sh, which lacks `source`, so force bash -c).
# in DRY_RUN, echo the command and return a fake job id so the chain can be walked end to end.
job() {  # job <name> <sbatch-opts...> -- <bash-command>
  local name="$1"; shift; local opts=(); while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[dry] sbatch --job-name=$name ${opts[*]} --wrap bash -c: $1" >&2; echo "DRY"; return; fi
  sbatch --parsable --job-name="$name" --output="logs/%x_%j.out" "${opts[@]}" --wrap="bash -c '$1'"
}
# chain: fire a tiny afterok helper that re-invokes this script for the next phase.
chain() {  # chain <dep_jobid> <phase> [offset]
  local dep="$1" nxt="$2" noff="${3:-0}"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[dry] -> after $dep: run_reg.sh $nxt $noff" >&2; DRY_RUN=1 bash hpc/run_reg.sh "$nxt" "$noff"; return; fi
  sbatch --parsable --job-name=rfpls-chain --partition=main-cpu --cpus-per-task=1 --mem=1G \
    --time=00:20:00 --dependency=afterok:"$dep" --output=logs/chain_%j.out \
    --wrap="cd $ROOT && bash hpc/run_reg.sh $nxt $noff" >/dev/null
}
# submit one SyN wave (array OFF..min(OFF+WAVE-1,24)) for struct|grade.
syn_wave() {  # syn_wave <struct|grade> <offset>
  local kind="$1" off="$2" end raw ref outd tmpl
  end=$(( off + WAVE - 1 )); [ "$end" -gt 24 ] && end=24
  if [ "$kind" = struct ]; then raw="$B21"; ref="$HOME/data/BraTS2021/brats_struct.npz"; outd="$ROOT/results/brats_realdata/syn_struct"
  else raw="$B20"; ref="$HOME/data/BraTS2020/brats2020_grade.npz"; outd="$ROOT/results/brats_realdata/syn_grade"; fi
  tmpl=$(find -L "$raw" -maxdepth 2 -name "*_t1.nii*" 2>/dev/null | head -1)   # -L follows the $raw symlink (BraTS2020_train); find|head stops at the first match (the glob stats all ~1250 dirs, which timed the helper out on NFS)
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[dry] sbatch --array=$off-$end (NSHARDS=$NSHARDS) syn_register $kind  ref=$(basename "$ref") tmpl=$(basename "${tmpl:-MISSING}")" >&2; echo "DRY"; return; fi
  sbatch --parsable --job-name="rfpls-syn$kind-$off" --array="$off-$end" --exclude=spark-head2 \
    --output="logs/%x_%A_%a.out" \
    --export=ALL,RAW_ROOT="$raw",REF_CACHE="$ref",TEMPLATE="$tmpl",OUT_DIR="$outd",NSHARDS="$NSHARDS" \
    hpc/syn_register.slurm
}

case "$PHASE" in
  absaff)
    # abs (affine SRI24-coord) caches from raw NIfTI, then the affine grid-PLS baseline.
    ABS=$(job rfpls-build-abs --partition=main-cpu --cpus-per-task=4 --mem=16G --time=02:00:00 -- \
      "cd $ROOT && $ENVP && \$PY campaigns/brats_realdata/build_abs_cache.py --raw_root $B21 --ref_cache $HOME/data/BraTS2021/brats_struct.npz --out $HOME/data/BraTS2021/brats_struct_abs.npz && \$PY campaigns/brats_realdata/build_abs_cache.py --raw_root $B20 --ref_cache $HOME/data/BraTS2020/brats2020_grade.npz --out $HOME/data/BraTS2020/brats2020_grade_abs.npz")
    AFF=$(job rfpls-reg-affine --partition=main-cpu --cpus-per-task=4 --mem=16G --time=01:00:00 --dependency=afterok:"$ABS" -- \
      "cd $ROOT && $ENVP && \$PY campaigns/brats_realdata/reg_baseline.py --abs_cache $HOME/data/BraTS2021/brats_struct_abs.npz --task reg --stored results/brats_realdata/pilot --out results/brats_realdata/reg/reg_struct.json && \$PY campaigns/brats_realdata/reg_baseline.py --abs_cache $HOME/data/BraTS2020/brats2020_grade_abs.npz --task clf --stored results/brats_realdata/grade/grade.json --out results/brats_realdata/reg/reg_grade.json")
    chain "$AFF" synstruct 0 ;;
  synstruct)
    SJ=$(syn_wave struct "$OFF")
    if [ $(( OFF + WAVE )) -le 24 ]; then chain "$SJ" synstruct $(( OFF + WAVE )); else chain "$SJ" syngrade 0; fi ;;
  syngrade)
    SJ=$(syn_wave grade "$OFF")
    if [ $(( OFF + WAVE )) -le 24 ]; then chain "$SJ" syngrade $(( OFF + WAVE )); else chain "$SJ" asmrad 0; fi ;;
  asmrad)
    # assemble the SyN shards into deformable caches, evaluate the deformable grid baseline, then C-1 radiomics.
    ASM=$(job rfpls-reg-eval --partition=main-cpu --cpus-per-task=4 --mem=16G --time=01:00:00 -- \
      "cd $ROOT && $ENVP && \$PY campaigns/brats_realdata/syn_assemble.py --shard_dir results/brats_realdata/syn_struct --ref_cache $HOME/data/BraTS2021/brats_struct.npz --out $HOME/data/BraTS2021/brats_struct_syn.npz && \$PY campaigns/brats_realdata/syn_assemble.py --shard_dir results/brats_realdata/syn_grade --ref_cache $HOME/data/BraTS2020/brats2020_grade.npz --out $HOME/data/BraTS2020/brats2020_grade_syn.npz && \$PY campaigns/brats_realdata/reg_baseline.py --abs_cache $HOME/data/BraTS2021/brats_struct_syn.npz --task reg --stored results/brats_realdata/pilot --out results/brats_realdata/reg/reg_struct_syn.json && \$PY campaigns/brats_realdata/reg_baseline.py --abs_cache $HOME/data/BraTS2020/brats2020_grade_syn.npz --task clf --stored results/brats_realdata/grade/grade.json --out results/brats_realdata/reg/reg_grade_syn.json")
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "[dry] sbatch --dependency=afterok:$ASM hpc/reg_radiomics.slurm  (C-1 Table 2, TERMINAL)" >&2
    else RAD=$(sbatch --parsable --dependency=afterok:"$ASM" hpc/reg_radiomics.slurm); echo "registration arm complete through radiomics ($RAD); it self-drives to the end."; fi ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac
