# Source this in every SLURM job to get a fast, working Python env.
# Why: this cluster's home dir is on NFS which is slow (DOCUMENTED on the HPC wiki); the
# sanctioned fix is the node-local /local scratch. Conda/venv on NFS => 10-min torch imports.
# Pattern: ship the env as ONE tarball, extract to /local, run the SYSTEM python3 with
# PYTHONPATH at the extracted site-packages (relocation-proof; no venv hardcoded-path dep).
#
# Optimization (per wiki: /local is wiped only if untouched for 90 days): extract to a fixed
# per-user path and reuse it across jobs on the same node, guarded by flock so concurrent
# array tasks on one node share a single extract instead of racing.
set -euo pipefail
ENV_TARBALL="${ENV_TARBALL:-$HOME/envs/regfree_env.tar.gz}"
STAGE="${STAGE:-/local/$USER/regfree-tensor-pls}"
mkdir -p "$STAGE"
exec 9>"$STAGE/.lock"
flock 9
if [ ! -f "$STAGE/.ready" ] || [ "$ENV_TARBALL" -nt "$STAGE/.ready" ]; then
  echo "[env_stage $(date +%T)] extracting env to $STAGE (one-time per node; NFS read ~8 min)..."
  rm -rf "$STAGE/regfree_env"
  tar xzf "$ENV_TARBALL" -C "$STAGE"
  touch "$STAGE/.ready"
  echo "[env_stage $(date +%T)] env staged."
else
  touch "$STAGE/.ready"   # keep fresh so the 90-day cleaner doesn't remove it
  echo "[env_stage $(date +%T)] reusing cached env at $STAGE."
fi
flock -u 9
export PYTHONPATH="$STAGE/regfree_env/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PY="/usr/bin/python3"
# kaggle CLI: invoke as "$PY -m kaggle ..." (the bin/kaggle shebang is non-relocatable)
