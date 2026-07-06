"""Gather the per-case SyN-registered .npz files (from syn_register.py array) into ONE ragged
cache aligned to the reference cache's (ids, Y) order, so reg_baseline.py can grid-PLS it against
the stored RAW-PLS folds. Missing/failed cases are reported; full alignment is required for a valid
paired test.

Usage:
  python syn_assemble.py --shard_dir <dir> --ref_cache <abs/centered cache> --out <brats_*_syn.npz>
"""
from __future__ import annotations
import os, argparse
import numpy as np
from data_brats import load_npz, save_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--ref_cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _, _, Y, ids = load_npz(a.ref_cache)
    P_list, F_list, Yk, idk, missing = [], [], [], [], []
    for i, cid in enumerate(ids):
        f = os.path.join(a.shard_dir, f"{cid}.npz")
        if not os.path.exists(f):
            missing.append(cid); continue
        d = np.load(f)
        P_list.append(d["P"]); F_list.append(d["F"]); Yk.append(float(Y[i])); idk.append(cid)

    if missing:
        print(f"WARNING: {len(missing)}/{len(ids)} cases missing (e.g. {missing[:5]}). "
              f"reg_baseline pairs only the present cases - investigate before trusting CIs.", flush=True)
    save_npz(a.out, P_list, F_list, np.asarray(Yk, np.float64), idk)
    print(f"WROTE {a.out} | {len(idk)} cases | missing {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
