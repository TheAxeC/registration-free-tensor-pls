"""Re-extract per-lesion point clouds keeping ABSOLUTE (SRI24-atlas) mm coordinates,
aligned 1:1 to an existing centered cache.

WHY: the locked caches centered each case's coords (`xyz - xyz.mean(0)`), which destroys
absolute anatomical position. The current `grid-PLS` baseline therefore only does
centroid/translation alignment. To give the registration-based baseline a FAIR shot we
rebuild the clouds in the BraTS-native SRI24 atlas frame (BraTS is distributed already
affine-co-registered to SRI24 @1mm, skull-stripped) -> absolute coords ARE cross-subject
comparable = anatomical registration. The new cache feeds reg_baseline.py.

Alignment: we read the canonical (ids, Y) order from the existing centered cache and
re-extract abs coords for exactly those ids in that order, so cross-validation folds
(indexing arange(N)) map to the same cases as the stored RAW-PLS runs.

Usage (on the cluster, raw NIfTI on /local):
  python build_abs_cache.py --raw_root /local/$USER/khub/.../BraTS2021_root \
      --ref_cache ~/data/BraTS2021/brats_struct.npz \
      --out ~/data/BraTS2021/brats_struct_abs.npz
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
from data_brats import MODS, SUFFIX, _load, _find, _znorm, load_npz, save_npz


def _case_dir(raw_root, cid):
    """Locate the folder for case id `cid` under raw_root (handles nested layouts)."""
    direct = os.path.join(raw_root, cid)
    if os.path.isdir(direct):
        return direct
    hits = glob.glob(os.path.join(raw_root, "**", cid), recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    return hits[0] if hits else None


def extract_abs(case_dir, cid, max_points=4000, seed=0):
    """(P[m,3] ABSOLUTE atlas mm, F[m,4] z-normed mods) over tumor voxels (seg>=1)."""
    vols = {m: _load(_find(case_dir, cid, SUFFIX[m]))[0] for m in MODS}
    seg, aff = _load(_find(case_dir, cid, SUFFIX["seg"]))
    brain = vols["flair"] > 0
    for m in MODS:
        vols[m] = _znorm(vols[m], brain)
    ijk = np.argwhere(seg >= 1)
    if len(ijk) == 0:
        raise ValueError(f"{cid}: empty tumor mask")
    g = np.random.default_rng(seed)
    if len(ijk) > max_points:
        ijk = ijk[g.choice(len(ijk), max_points, replace=False)]
    xyz = (aff @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    P = xyz.astype(np.float64)                                   # ABSOLUTE - no centering
    F = np.stack([vols[m][tuple(ijk.T)] for m in MODS], axis=1).astype(np.float64)
    return P, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="dir containing the case folders")
    ap.add_argument("--ref_cache", required=True, help="existing centered cache (for ids+Y order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_points", type=int, default=4000)
    a = ap.parse_args()

    _, _, Y, ids = load_npz(a.ref_cache)
    print(f"ref cache: {len(ids)} cases | will re-extract abs coords in this exact order", flush=True)

    P_list, F_list, Yk, idk, missing = [], [], [], [], []
    aff_check = []
    for i, cid in enumerate(ids):
        cdir = _case_dir(a.raw_root, cid)
        if cdir is None:
            missing.append(cid); continue
        try:
            P, F = extract_abs(cdir, cid, a.max_points)
        except (FileNotFoundError, ValueError) as e:
            missing.append(cid); print("skip", cid, e, flush=True); continue
        P_list.append(P); F_list.append(F); Yk.append(float(Y[i])); idk.append(cid)
        if i < 5:
            seg, aff = _load(_find(cdir, cid, SUFFIX["seg"])); aff_check.append(aff[:3, :3].diagonal())
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(ids)}", flush=True)

    if missing:
        print(f"WARNING: {len(missing)} ids not re-extracted (e.g. {missing[:5]}). "
              f"Alignment to stored folds REQUIRES the full set — investigate before trusting CIs.", flush=True)
    print("affine voxel-size check (first cases, expect ~1mm isotropic & identical):", flush=True)
    for d in aff_check:
        print("   diag", np.round(d, 3), flush=True)

    save_npz(a.out, P_list, F_list, np.asarray(Yk, np.float64), idk)
    print(f"WROTE {a.out} | {len(idk)} cases | missing {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
