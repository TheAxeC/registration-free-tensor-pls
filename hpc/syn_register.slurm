#!/bin/bash
#SBATCH --job-name=regfree-syn
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/%x_%A_%a.out
# Arm 2 — ANTs SyN deformable registration baseline (array over shards).
# Stages the antspyx env tarball to node-local /local, then registers this shard's cases to the
# template and writes per-case .npz to $OUT_DIR (on NFS, small files). Env vars set at submit:
#   RAW_ROOT  = dir with the case folders (on NFS, readable by compute nodes)
#   REF_CACHE = the (ids,Y)-order cache (~/data/BraTS2021/brats_struct.npz or BraTS2020 grade)
#   TEMPLATE  = fixed reference T1 NIfTI (skull-stripped, SRI24 space)
#   OUT_DIR   = per-case output dir
#   NSHARDS   = number of array tasks (match --array)
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
U=$(id -un)
STAGE=/local/$U/ants_env
mkdir -p "$STAGE"
exec 9>"$STAGE/.lock"; flock 9
if [ ! -f "$STAGE/.ready" ] || [ "$HOME/envs/regfree_ants_env.tar.gz" -nt "$STAGE/.ready" ]; then
  rm -rf "$STAGE/ants_env"; tar xzf "$HOME/envs/regfree_ants_env.tar.gz" -C "$STAGE"; touch "$STAGE/.ready"
fi
flock -u 9
export PYTHONPATH="$STAGE/ants_env/lib/python3.10/site-packages:$SLURM_SUBMIT_DIR"
PY=/usr/bin/python3
# isolate ANTs temp files per array task (concurrent tasks on one node share /tmp -> antsApplyTransforms
# exit-1 collisions; this caused ~145 struct case failures on the first run).
export TMPDIR=/local/$U/antstmp/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
"$PY" campaigns/brats_realdata/syn_register.py \
    --raw_root "$RAW_ROOT" --ref_cache "$REF_CACHE" --template "$TEMPLATE" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "${NSHARDS:-50}" --out_dir "$OUT_DIR"
echo "SYN_SHARD_${SLURM_ARRAY_TASK_ID}_DONE"
