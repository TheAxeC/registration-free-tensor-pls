"""Arm 1 of the registration baseline: ANATOMICAL (absolute-SRI24-coordinate) grid-PLS,
paired against the LOCKED RAW-PLS results on identical folds.

The locked `grid-PLS` baseline rasterizes per-case-CENTERED clouds (centroid alignment only).
This evaluates the FAIR registration baseline: the same rasterize representation but on the
absolute atlas coordinates (built by build_abs_cache.py) -> lesions keep their true anatomical
position, i.e. the alignment BraTS's SRI24 registration actually provides. We sweep the grid
resolution H and report the baseline's BEST config (no strawman). Folds replay the exact
StratifiedKFold/KFold(random_state=s) splits of the stored runs, so the paired Wilcoxon vs the
stored RAW-PLS per-fold metrics is valid.

Usage (cluster):
  # struct (10 seeds x 5 folds; stored ours in pilot/seed_*.json)
  python reg_baseline.py --abs_cache ~/data/BraTS2021/brats_struct_abs.npz \
      --task reg --stored ../../../results/brats_realdata/pilot --out reg_struct.json
  # grade (stored ours in grade/grade.json)
  python reg_baseline.py --abs_cache ~/data/BraTS2020/brats2020_grade_abs.npz \
      --task clf --stored ../../../results/brats_realdata/grade/grade.json --out reg_grade.json
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, KFold

from data_brats import load_npz
from run_brats import rasterize, baseline_eval


def load_stored(path):
    """Return (ours, base_old) flat lists in seed-major,fold order from the locked run(s)."""
    if os.path.isdir(path):                                  # struct: one file per seed
        ours, base = [], []
        for f in sorted(glob.glob(os.path.join(path, "seed_*.json")),
                        key=lambda p: int(p.split("seed_")[1].split(".")[0])):
            d = json.load(open(f)); ours += list(d["ours"]); base += list(d["base"])
        return ours, base
    d = json.load(open(path))                                # grade: single file, 50 evals
    return list(d["ours"]), list(d["base"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abs_cache", required=True)
    ap.add_argument("--task", choices=["reg", "clf"], required=True)
    ap.add_argument("--stored", required=True, help="dir of seed_*.json (struct) or grade.json (grade)")
    ap.add_argument("--nseeds", type=int, default=10); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grids", type=int, nargs="+", default=[12, 16, 20, 24])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P, F, Y, ids = load_npz(a.abs_cache)
    N = len(Y)
    ours, base_old = load_stored(a.stored)
    metric = "R2" if a.task == "reg" else "AUC"
    print(f"{N} cases | task={a.task} ({metric}) | stored evals: ours={len(ours)} base_old={len(base_old)}", flush=True)

    # replay the exact fold loop the stored runs used, evaluate the anatomical baseline
    results = {}
    for H in a.grids:
        X = rasterize(P, F, H)                               # global-box anatomical grid, computed once
        base_anat = []
        for s in range(a.nseeds):
            if a.task == "clf":
                splitter = StratifiedKFold(a.folds, shuffle=True, random_state=s); ystrat = Y.astype(int)
            else:
                splitter = KFold(a.folds, shuffle=True, random_state=s); ystrat = np.zeros(N)
            for tr, va in splitter.split(np.arange(N), ystrat):
                base_anat.append(baseline_eval(X[tr], Y[tr], X[va], Y[va], a.task))
        results[H] = base_anat
        m = float(np.mean(base_anat))
        print(f"  H={H:>2}: anat-grid-PLS {metric} {m:.4f} ± {np.std(base_anat):.4f}  (dim={X.shape[1]})", flush=True)

    # baseline's BEST grid (give registration its strongest config)
    bestH = max(results, key=lambda h: np.mean(results[h]))
    best = results[bestH]
    n = min(len(ours), len(best))
    if len(ours) != len(best):
        print(f"WARNING: stored ours={len(ours)} vs anat={len(best)} — fold misalignment "
              f"(abs cache may be missing cases). Truncating to {n} for the test; FIX before trusting.", flush=True)
    o, b_new, b_old = np.array(ours[:n]), np.array(best[:n]), np.array(base_old[:n])
    try:
        W, p = wilcoxon(o, b_new)
    except ValueError:
        W, p = float("nan"), float("nan")

    print("\n================ ARM-1 ANATOMICAL REGISTRATION BASELINE ================", flush=True)
    print(f"  RAW-PLS (stored)        : {o.mean():.4f} ± {o.std():.4f}", flush=True)
    print(f"  anat-grid-PLS (best H={bestH}) : {b_new.mean():.4f} ± {b_new.std():.4f}", flush=True)
    print(f"  centered grid-PLS (old) : {b_old.mean():.4f} ± {b_old.std():.4f}", flush=True)
    print(f"  margin RAW-PLS - anat   : {(o - b_new).mean():+.4f}   (paired Wilcoxon p={p:.2e})", flush=True)
    print(f"  did anatomical coords help the baseline vs centered? {b_new.mean()-b_old.mean():+.4f}", flush=True)
    verdict = ("RAW-PLS STILL WINS vs the stronger anatomical baseline"
               if o.mean() > b_new.mean() and p < 0.05 else
               "⚠ MARGIN NARROWED / NOT SIGNIFICANT — iterate the method (no-null rule), do NOT report weakened")
    print("  VERDICT:", verdict, flush=True)

    if a.out:
        json.dump({"task": a.task, "N": N, "bestH": bestH, "grids": list(results.keys()),
                   "rawpls_mean": o.mean(), "anat_mean": b_new.mean(), "centered_old_mean": b_old.mean(),
                   "margin_mean": float((o - b_new).mean()), "wilcoxon_p": float(p),
                   "anat_by_H": {h: float(np.mean(v)) for h, v in results.items()},
                   "ours": o.tolist(), "anat": b_new.tolist()}, open(a.out, "w"), indent=2)
        print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
