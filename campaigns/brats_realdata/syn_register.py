"""Arm 2 of the registration baseline: DEFORMABLE (ANTs SyN) registration to a common template.

For a shard of cases: SyN-register each subject's T1 to a fixed template, warp the 4 modalities
(linear) + the segmentation (nearest-neighbor) into template space, then extract the warped tumor
voxels' template-space coordinates + warped modality features. One per-case .npz is written; the
shard is array-parallel. `syn_assemble.py` gathers them into a cache with the same ragged format as
the abs cache, and `reg_baseline.py` then runs the grid-PLS comparison vs the stored RAW-PLS folds.

This is the "we tried state-of-the-art nonlinear registration" arm. Thesis prediction: even a SyN
warp of the surrounding parenchyma cannot register one patient's lesion sub-structure to another's,
so RAW-PLS still wins (and, per Arm 1, anatomical alignment already underperforms centering).

Usage (one array task):
  python syn_register.py --raw_root <root> --ref_cache <abs_or_centered_cache.npz> \
      --template <ref_T1.nii.gz> --shard $SLURM_ARRAY_TASK_ID --nshards 50 \
      --out_dir <dir> --max_points 4000
"""
from __future__ import annotations
import sys, types
# antspyx's `import ants` pulls in ants.plotting -> matplotlib, and on this cluster the system
# (old) mpl_toolkits clashes with the env's matplotlib 3.10 ("cannot import name 'docstring'").
# We only need registration, not plotting: pre-insert an empty ants.plotting so the broken import
# is skipped entirely. (Verified: registration + apply_transforms work fine without it.)
sys.modules["ants.plotting"] = types.ModuleType("ants.plotting")
import os, glob, argparse
import numpy as np
from data_brats import MODS, SUFFIX, _find, load_npz


def _case_dir(root, cid):
    d = os.path.join(root, cid)
    if os.path.isdir(d):
        return d
    h = [x for x in glob.glob(os.path.join(root, "**", cid), recursive=True) if os.path.isdir(x)]
    return h[0] if h else None


def _znorm(vol, brain):
    m, s = vol[brain].mean(), vol[brain].std() + 1e-6
    return (vol - m) / s


def register_case(ants, case_dir, cid, fixed, max_points=4000, seed=0):
    """SyN-register T1->template, warp mods+seg, return (P template-mm, F warped z-normed)."""
    mov_t1 = ants.image_read(_find(case_dir, cid, SUFFIX["t1"]))
    reg = ants.registration(fixed=fixed, moving=mov_t1, type_of_transform="SyN")
    tx = reg["fwdtransforms"]
    vols = {}
    for m in MODS:
        w = ants.apply_transforms(fixed, ants.image_read(_find(case_dir, cid, SUFFIX[m])), tx,
                                  interpolator="linear")
        vols[m] = w.numpy()
    wseg = ants.apply_transforms(fixed, ants.image_read(_find(case_dir, cid, SUFFIX["seg"])), tx,
                                 interpolator="nearestNeighbor")
    seg = wseg.numpy()
    brain = vols["flair"] > 0
    for m in MODS:
        vols[m] = _znorm(vols[m], brain)
    ijk = np.argwhere(seg >= 1)
    if len(ijk) == 0:
        raise ValueError(f"{cid}: empty warped tumor mask")
    g = np.random.default_rng(seed)
    if len(ijk) > max_points:
        ijk = ijk[g.choice(len(ijk), max_points, replace=False)]
    spacing = np.array(fixed.spacing, float)                  # common template frame for ALL subjects
    P = (ijk * spacing).astype(np.float64)                    # absolute template mm (cross-subject comparable)
    F = np.stack([vols[m][tuple(ijk.T)] for m in MODS], axis=1).astype(np.float64)
    return P, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--ref_cache", required=True, help="cache giving the canonical ids+Y order")
    ap.add_argument("--template", required=True, help="fixed reference T1 NIfTI (skull-stripped, SRI24 space)")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_points", type=int, default=4000)
    a = ap.parse_args()

    import ants
    os.makedirs(a.out_dir, exist_ok=True)
    _, _, Y, ids = load_npz(a.ref_cache)
    fixed = ants.image_read(a.template)
    mine = list(range(a.shard, len(ids), a.nshards))
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} cases | template {a.template} spacing {fixed.spacing}", flush=True)

    for i in mine:
        cid = ids[i]
        out = os.path.join(a.out_dir, f"{cid}.npz")
        if os.path.exists(out):
            continue
        cd = _case_dir(a.raw_root, cid)
        if cd is None:
            print("MISSING", cid, flush=True); continue
        try:
            P, F = register_case(ants, cd, cid, fixed, a.max_points)
        except Exception as e:                                # registration can fail on odd cases
            print("FAIL", cid, repr(e)[:120], flush=True); continue
        np.savez_compressed(out, P=P, F=F, y=float(Y[i]), idx=i, cid=cid)
        print(f"  done {cid} ({i}) n={len(P)}", flush=True)
    print(f"shard {a.shard} DONE", flush=True)


if __name__ == "__main__":
    main()
