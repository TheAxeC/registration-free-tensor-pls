"""
BraTS pilot runner for RAW-PLS. Cross-validated, with the same alignment-assuming
baselines as the synthetic study so the real-data comparison is apples-to-apples.

Usage (after caching the dataset once with --build):
  python run_brats.py --build --root /data/BraTS2021 --labels mgmt.csv \
        --id_col BraTS21ID --target MGMT_value --task clf --cache brats_clf.npz
  python run_brats.py --cache brats_clf.npz --task clf --folds 5 --seeds 3
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.cross_decomposition import PLSRegression

from core.raw_pls import train_rawpls
from deepset import train_deepset
from set_transformer import train_settransformer
from data_brats import build_dataset, save_npz, load_npz


def rasterize(P, F, H=12):
    """Alignment-assuming baseline rep: fixed mm grid, per-cell mean feature + mass."""
    C = F[0].shape[1]
    allp = np.vstack(P); lo, hi = np.percentile(allp, [2, 98], axis=0)
    out = []
    for p, f in zip(P, F):
        idx = np.clip(((p - lo) / (hi - lo + 1e-9) * H).astype(int), 0, H - 1)
        mass = np.zeros((H, H, H)); feat = np.zeros((H, H, H, C))
        for j in range(len(p)):
            t = tuple(idx[j]); mass[t] += 1; feat[t] += f[j]
        nz = mass > 0; feat[nz] /= mass[nz][:, None]
        out.append(np.concatenate([mass.ravel() / max(1, len(p)), feat.ravel()]))
    return np.array(out)


def baseline_eval(Xtr, ytr, Xva, yva, task):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    pls = PLSRegression(n_components=min(10, Xtr.shape[1], Xtr.shape[0] - 1))
    pls.fit(Xtr, ytr if task == "reg" else ytr.astype(float))
    pred = pls.predict(Xva).ravel()
    return r2_score(yva, pred) if task == "reg" else roc_auc_score(yva, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root"); ap.add_argument("--labels")
    ap.add_argument("--id_col", default="BraTS21ID"); ap.add_argument("--target")
    ap.add_argument("--cache", default="brats.npz")
    ap.add_argument("--task", choices=["clf", "reg"], default="clf")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_id", type=int, default=None,
                    help="run ONE seed only (for SLURM job arrays); writes --out json")
    ap.add_argument("--out", default=None, help="json result path for single-seed mode")
    ap.add_argument("--K", type=int, default=12); ap.add_argument("--Mmax", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.005); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--geom_rank", type=int, default=0); ap.add_argument("--lambda_cov", type=float, default=0.0)
    ap.add_argument("--dist_readout", action="store_true", help="add per-atom feature variance to the barycentric readout")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", choices=["rawpls", "deepset", "settransformer"], default="rawpls")
    a = ap.parse_args()

    if a.build:
        P, F, Y, ids = build_dataset(a.root, a.labels, a.id_col, a.target,
                                     task=a.task, limit=a.limit)
        save_npz(a.cache, P, F, Y, ids); print("cached ->", a.cache); return

    P, F, Y, ids = load_npz(a.cache)
    print(f"{len(ids)} cases | task={a.task} | metric={'AUC' if a.task=='clf' else 'R2'}", flush=True)
    # single-seed mode: one SLURM array task = one seed; cache is read-only, output isolated
    seeds = [a.seed_id] if a.seed_id is not None else list(range(a.seeds))
    X = rasterize(P, F)                                        # baseline rep, computed once
    ours, base = [], []
    for s in seeds:
        splitter = (StratifiedKFold(a.folds, shuffle=True, random_state=s) if a.task == "clf"
                    else KFold(a.folds, shuffle=True, random_state=s))
        ystrat = Y.astype(int) if a.task == "clf" else np.zeros(len(Y))
        for fold, (tr, va) in enumerate(splitter.split(np.arange(len(Y)), ystrat)):
            if a.model == "deepset":
                _, m, _ = train_deepset(P, F, Y, tr, va, Mmax=a.Mmax, epochs=a.epochs,
                                        lr=a.lr, wd=a.wd, task=a.task, device=a.device, seed=s)
            elif a.model == "settransformer":
                _, m, _ = train_settransformer(P, F, Y, tr, va, Mmax=a.Mmax, epochs=a.epochs,
                                               lr=a.lr, wd=a.wd, task=a.task, device=a.device, seed=s)
            else:
                _, m, _ = train_rawpls(P, F, Y, tr, va, K=a.K, Mmax=a.Mmax, epochs=a.epochs,
                                       lr=a.lr, wd=a.wd, alpha=a.alpha, eps=a.eps,
                                       geom_rank=a.geom_rank, lambda_cov=a.lambda_cov,
                                       dist_readout=a.dist_readout,
                                       task=a.task, device=a.device, seed=s, verbose=False)
            ours.append(m)
            base.append(baseline_eval(X[tr], Y[tr], X[va], Y[va], a.task))
            print(f"  seed{s} fold: ours={m:.3f}  grid-baseline={base[-1]:.3f}")
    print(f"\n=== RAW-PLS:  {np.mean(ours):.3f} +/- {np.std(ours):.3f}")
    print(f"=== grid-PLS: {np.mean(base):.3f} +/- {np.std(base):.3f}")
    if a.out:                                                  # isolated per-seed result file
        import json
        with open(a.out, "w") as fh:
            json.dump({"seed": a.seed_id, "task": a.task, "metric": a.task,
                       "ours": ours, "base": base,
                       "cfg": {"K": a.K, "Mmax": a.Mmax, "epochs": a.epochs, "lr": a.lr,
                               "wd": a.wd, "alpha": a.alpha, "eps": a.eps,
                               "geom_rank": a.geom_rank, "lambda_cov": a.lambda_cov}}, fh, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
