"""
Build a cache with a STRUCTURAL regression target derived from the BraTS segmentation,
designed to showcase the registration-free advantage (mirrors the synthetic
structure-predictive experiment): the normalized spatial separation between the
enhancing-tumor (label 4) and necrotic-core (label 1) centroids.

Features given to the model are ONLY the 4 multiparametric intensities (NOT the seg) per
tumor voxel, so the model must identify subregions by intensity and recover their
arrangement -- a geometric relationship that grid-alignment scrambles on unaligned lesions
but intra-sample GW structure can capture. Non-circular (the target is a geometry, not an
intensity statistic).

Usage:
  python build_seg_target.py --root /tmp/brats_ex --out ~/data/BraTS2021/brats_struct.npz \
         --target et_ncr_dist [--limit N]
"""
import os
import argparse
import numpy as np
from data_brats import MODS, SUFFIX, _load, _find, _znorm, save_npz

# BraTS seg labels: 1 = necrotic/non-enhancing core (NCR), 2 = peritumoral edema (ED),
# 4 = GD-enhancing tumor (ET).
LBL = {"NCR": 1, "ED": 2, "ET": 4}


def _centroid(seg, lbl):
    v = np.argwhere(seg == lbl)
    return v.mean(0) if len(v) > 0 else None


def compute_target(seg, ijk, mode):
    """Return a scalar structural target, or None to skip the case."""
    if mode == "et_ncr_dist":
        cET, cNCR = _centroid(seg, LBL["ET"]), _centroid(seg, LBL["NCR"])
        if cET is None or cNCR is None:
            return None
        rad = np.linalg.norm(ijk - ijk.mean(0), axis=1).mean() + 1e-6   # tumor extent
        return float(np.linalg.norm(cET - cNCR) / rad)                  # normalized
    if mode == "et_frac":                                               # sanity / composition
        et = (seg == LBL["ET"]).sum(); tot = (seg >= 1).sum()
        return float(et / (tot + 1e-9))
    raise ValueError(mode)


def build(root, out, mode="et_ncr_dist", max_points=4000, limit=None, seed=0):
    g = np.random.default_rng(seed)
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    P_list, F_list, Y, ids = [], [], [], []
    skipped = 0
    for cid in cases:
        cdir = os.path.join(root, cid)
        try:
            vols = {m: _load(_find(cdir, cid, SUFFIX[m]))[0] for m in MODS}
            seg, aff = _load(_find(cdir, cid, SUFFIX["seg"]))
        except (FileNotFoundError, ValueError):
            skipped += 1; continue
        ijk = np.argwhere(seg >= 1)
        if len(ijk) < 50:
            skipped += 1; continue
        y = compute_target(seg, ijk, mode)
        if y is None:
            skipped += 1; continue
        brain = vols["flair"] > 0
        for m in MODS:
            vols[m] = _znorm(vols[m], brain)
        sub = ijk if len(ijk) <= max_points else ijk[g.choice(len(ijk), max_points, replace=False)]
        xyz = (aff @ np.c_[sub, np.ones(len(sub))].T).T[:, :3]
        P = (xyz - xyz.mean(0)).astype(np.float64)
        F = np.stack([vols[m][tuple(sub.T)] for m in MODS], axis=1).astype(np.float64)
        P_list.append(P); F_list.append(F); Y.append(float(y)); ids.append(cid)
        if limit and len(ids) >= limit:
            break
    Y = np.asarray(Y, np.float64)
    save_npz(out, P_list, F_list, Y, ids)
    print(f"built {len(ids)} cases (skipped {skipped}) | target={mode} | "
          f"Y mean {Y.mean():.3f} std {Y.std():.3f} range [{Y.min():.2f},{Y.max():.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="et_ncr_dist")
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.root, a.out, a.target, a.max_points, a.limit)
