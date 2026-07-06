"""R-3b: grade AUPRC + bootstrap AUC CI (per-sample scores).

The stored grade run keeps only per-fold AUC. For the imbalanced grade cohort (n=368, 292 HGG /
76 LGG, prevalence 0.207) a methodology reviewer asked for AUPRC and a bootstrap/DeLong CI. This
replays the exact grade CV (StratifiedKFold random_state=s, s=0..9), collects per-sample RAW-PLS
scores from train_rawpls, and reports per-fold AUPRC mean + bootstrap CI, plus a single-partition
(seed 0, each sample held out once) pooled AUC and AUPRC with bootstrap CIs.

Config matches the headline grade run (grade.json cfg): K16 Mmax128 ep200 lr0.003 wd1e-3
alpha0.6 eps0.03 geom6 cov0.1.

Usage (cluster, GPU):
  python grade_scores.py --cache ~/data/BraTS2020/brats2020_grade.npz --out grade_scores.json
"""
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from data_brats import load_npz
from core.raw_pls import train_rawpls


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_ci(fn, *arrs, n=10000, seed=0):
    rng = np.random.default_rng(seed); m = len(arrs[0]); vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals.append(fn(*[a[idx] for a in arrs]))
        except ValueError:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nseeds", type=int, default=10); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P, F, Y, ids = load_npz(a.cache)
    Y = np.asarray(Y, np.float64); N = len(Y)
    prev = float(Y.mean())
    print(f"{N} cases | prevalence (AUPRC null) = {prev:.3f}", flush=True)

    fold_auc, fold_auprc = [], []
    seed0_scores = np.full(N, np.nan)                       # single partition: each sample once
    for s in range(a.nseeds):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=s)
        for tr, va in skf.split(np.arange(N), Y.astype(int)):
            _, _, pred = train_rawpls(P, F, Y, tr, va, K=16, Mmax=128, epochs=200, lr=0.003, wd=1e-3,
                                      alpha=0.6, eps=0.03, geom_rank=6, lambda_cov=0.1,
                                      task="clf", device="cuda", seed=s, verbose=False)
            sc = sigmoid(np.asarray(pred, np.float64)); yv = Y[va]
            fold_auc.append(roc_auc_score(yv, sc)); fold_auprc.append(average_precision_score(yv, sc))
            if s == 0:
                seed0_scores[va] = sc
        print(f"  seed {s}: mean fold AUC {np.mean(fold_auc[-a.folds:]):.4f} "
              f"AUPRC {np.mean(fold_auprc[-a.folds:]):.4f}", flush=True)

    fa, fp = np.array(fold_auc), np.array(fold_auprc)
    # seed-0 pooled (each sample predicted exactly once) -> bootstrap CI over samples
    m0 = ~np.isnan(seed0_scores)
    y0, s0 = Y[m0], seed0_scores[m0]
    pooled_auc = roc_auc_score(y0, s0); pooled_auprc = average_precision_score(y0, s0)
    auc_lo, auc_hi = boot_ci(roc_auc_score, y0, s0)
    ap_lo, ap_hi = boot_ci(average_precision_score, y0, s0)

    print("\n================ R-3b GRADE AUPRC / AUC ================", flush=True)
    print(f"  per-fold AUPRC : {fp.mean():.4f} +/- {fp.std():.4f}  (50 folds)", flush=True)
    print(f"  per-fold AUC   : {fa.mean():.4f} +/- {fa.std():.4f}", flush=True)
    print(f"  AUPRC null (prevalence) : {prev:.4f}", flush=True)
    print(f"  seed-0 pooled AUC   {pooled_auc:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]", flush=True)
    print(f"  seed-0 pooled AUPRC {pooled_auprc:.4f}  95% CI [{ap_lo:.4f}, {ap_hi:.4f}]", flush=True)

    if a.out:
        json.dump({"N": N, "prevalence": prev,
                   "fold_auc_mean": float(fa.mean()), "fold_auc_sd": float(fa.std()),
                   "fold_auprc_mean": float(fp.mean()), "fold_auprc_sd": float(fp.std()),
                   "pooled_auc": float(pooled_auc), "pooled_auc_ci": [auc_lo, auc_hi],
                   "pooled_auprc": float(pooled_auprc), "pooled_auprc_ci": [ap_lo, ap_hi],
                   "fold_auc": fa.tolist(), "fold_auprc": fp.tolist()}, open(a.out, "w"), indent=2)
        print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
