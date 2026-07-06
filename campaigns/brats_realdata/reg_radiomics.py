"""C-1: lesion-cropped registered radiomics baseline.

The devil's-advocate objection to the registration ladder: the grid baselines rasterize the
registered lesion onto a whole-brain H<=24 grid, so the loss to RAW-PLS could be a rasterization /
field-of-view dilution artifact rather than anything about registration. This baseline removes the
dilution while keeping registration: it extracts a standard radiomics feature vector (first-order
intensity per channel + whole-lesion shape) from the lesion in the REGISTERED atlas frame
(abs = affine SRI24, or syn = deformable), then fits the SAME PLS head the grid baseline uses. Only
the representation changes (lesion-cropped radiomics vs whole-brain grid); registration is held on.
If RAW-PLS still wins, the registration-free advantage is not a rasterization artifact.

Non-circular for the structural target: the cache holds only whole-tumor voxels (seg>=1) with
4-channel intensity + coordinates, no sub-region labels, so the ET-NCR centroid distance cannot be
read off directly. The registered atlas-frame centroid and shape give registration its strongest
fair shot (anatomical location + tumor geometry in the shared frame).

Usage (cluster, CPU):
  # struct, affine (abs) and deformable (syn), paired vs the full-method folds in improved.json
  python reg_radiomics.py --cache ~/data/BraTS2021/brats_struct_abs.npz --task reg \
      --stored ../../../results/brats_realdata/improved/improved.json --out rad_struct_abs.json
  python reg_radiomics.py --cache ~/data/BraTS2021/brats_struct_syn.npz --task reg \
      --stored ../../../results/brats_realdata/improved/improved.json --out rad_struct_syn.json
  # grade, paired vs grade.json
  python reg_radiomics.py --cache ~/data/BraTS2020/brats2020_grade_abs.npz --task clf \
      --stored ../../../results/brats_realdata/grade/grade.json --out rad_grade_abs.json
  python reg_radiomics.py --cache ~/data/BraTS2020/brats2020_grade_syn.npz --task clf \
      --stored ../../../results/brats_realdata/grade/grade.json --out rad_grade_syn.json
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, KFold

from data_brats import load_npz
from run_brats import baseline_eval


def load_stored(path):
    """Return (ours, base_old) per-fold lists in seed-major fold order from the locked run(s)."""
    if os.path.isdir(path):
        ours, base = [], []
        for f in sorted(glob.glob(os.path.join(path, "seed_*.json")),
                        key=lambda p: int(p.split("seed_")[1].split(".")[0])):
            d = json.load(open(f)); ours += list(d["ours"]); base += list(d["base"])
        return ours, base
    d = json.load(open(path))
    return list(d["ours"]), list(d["base"])


def _firstorder(x):
    """First-order stats for one intensity channel over the lesion voxels."""
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    h, _ = np.histogram(x, bins=16, density=True); h = h[h > 0]
    entropy = float(-(h * np.log(h)).sum()) if h.size else 0.0
    mu, sd = x.mean(), x.std()
    skew = float(((x - mu) ** 3).mean() / (sd ** 3 + 1e-9))
    kurt = float(((x - mu) ** 4).mean() / (sd ** 4 + 1e-9))
    return [mu, sd, x.min(), x.max(), p50, p10, p90, p90 - p10,
            float((x ** 2).mean()), entropy, skew, kurt]


def radiomics_vector(P, F):
    """Standard-style radiomics vector from one registered lesion cloud.
    First-order intensity per channel + whole-lesion shape/geometry in the registered atlas frame."""
    feats = []
    C = F.shape[1]
    for c in range(C):
        feats += _firstorder(F[:, c])
    # shape / geometry from registered coordinates (atlas mm)
    n = len(P)
    ext = P.max(0) - P.min(0)                              # bbox extents (3)
    bbox_vol = float(np.prod(ext) + 1e-9)
    centroid = P.mean(0)                                   # atlas-frame location (registration signal)
    Pc = P - centroid
    cov = np.cov(Pc.T) if n > 3 else np.eye(3)
    evals = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]  # l1>=l2>=l3
    axlen = np.sqrt(evals + 1e-9)
    elong = float(np.sqrt((evals[1] + 1e-12) / (evals[0] + 1e-12)))
    flat = float(np.sqrt((evals[2] + 1e-12) / (evals[0] + 1e-12)))
    rad = np.linalg.norm(Pc, axis=1)
    feats += [np.log1p(n), *ext.tolist(), np.log1p(bbox_vol), float(n) / bbox_vol,
              *centroid.tolist(), *axlen.tolist(), elong, flat, rad.mean(), rad.std()]
    return np.asarray(feats, np.float64)


def radiomics_matrix(P, F):
    return np.vstack([radiomics_vector(P[i], F[i]) for i in range(len(P))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="registered cache (*_abs.npz affine, *_syn.npz deformable)")
    ap.add_argument("--task", choices=["reg", "clf"], required=True)
    ap.add_argument("--stored", required=True, help="improved.json / grade.json (or seed_*.json dir) to pair against")
    ap.add_argument("--nseeds", type=int, default=10); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P, F, Y, ids = load_npz(a.cache)
    N = len(Y)
    ours, base_old = load_stored(a.stored)
    metric = "R2" if a.task == "reg" else "AUC"
    X = radiomics_matrix(P, F)
    print(f"{N} cases | task={a.task} ({metric}) | radiomics dim={X.shape[1]} | stored ours={len(ours)}", flush=True)

    rad = []
    for s in range(a.nseeds):
        if a.task == "clf":
            splitter = StratifiedKFold(a.folds, shuffle=True, random_state=s); ystrat = Y.astype(int)
        else:
            splitter = KFold(a.folds, shuffle=True, random_state=s); ystrat = np.zeros(N)
        for tr, va in splitter.split(np.arange(N), ystrat):
            rad.append(baseline_eval(X[tr], Y[tr], X[va], Y[va], a.task))

    n = min(len(ours), len(rad))
    if len(ours) != len(rad):
        print(f"WARNING: stored ours={len(ours)} vs radiomics={len(rad)} fold misalignment "
              f"(cache may be missing cases). Truncating to {n}; FIX before trusting.", flush=True)
    o, r = np.array(ours[:n]), np.array(rad[:n])
    try:
        W, p = wilcoxon(o, r)
    except ValueError:
        W, p = float("nan"), float("nan")

    print("\n================ C-1  LESION-CROPPED REGISTERED RADIOMICS ================", flush=True)
    print(f"  RAW-PLS (stored full method) : {o.mean():.4f} +/- {o.std():.4f}", flush=True)
    print(f"  registered radiomics         : {r.mean():.4f} +/- {r.std():.4f}", flush=True)
    print(f"  margin RAW-PLS - radiomics    : {(o - r).mean():+.4f}   (paired Wilcoxon p={p:.2e})", flush=True)
    verdict = ("RAW-PLS STILL WINS vs lesion-cropped registered radiomics -> confound broken, claim airtight"
               if o.mean() > r.mean() and p < 0.05 else
               "MARGIN NARROWED / NOT SIGNIFICANT -> iterate the method (no-null rule); do NOT report a weakened result")
    print("  VERDICT:", verdict, flush=True)

    if a.out:
        json.dump({"task": a.task, "N": N, "cache": os.path.basename(a.cache), "radiomics_dim": int(X.shape[1]),
                   "rawpls_mean": float(o.mean()), "radiomics_mean": float(r.mean()),
                   "margin_mean": float((o - r).mean()), "wilcoxon_p": float(p),
                   "ours": o.tolist(), "radiomics": r.tolist()}, open(a.out, "w"), indent=2)
        print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
