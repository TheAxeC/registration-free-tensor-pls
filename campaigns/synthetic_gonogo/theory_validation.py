"""Upgrade track C (Rigor + Evidence): turn the theory's guarantees into demonstrated facts.

(1) INVARIANCE (Lemma 1): the unrolled fused-GW representation depends only on intra-sample
    distances + features, so it is exactly invariant to isometric reparametrization (rotation +
    translation + relabeling) of a sample's coordinates, whereas the aligned grid representation
    is not. We measure the relative change of each representation under random isometries.

(2) sqrt(N)-CONSISTENCY (Thm 2): the ERM's excess test error should shrink with the training
    size N. We train on a geometric ladder of N on a FIXED held-out test set and fit the log-log
    slope of test MSE vs N (predicted ~ -1 for the squared excess risk, i.e. RMSE ~ 1/sqrt(N)).

Reuses joint_supervised.py (CH=6). Run (local, CPU), from code/:
  PYTHONPATH=. python campaigns/synthetic_gonogo/theory_validation.py
"""
import os, json
import numpy as np
import torch
import joint_supervised as js
from joint_supervised import gen_dataset, pad_batch, fgw_layer, rasterize, train_joint

RNG = np.random.default_rng
CH = 6


# ----------------------------- (1) invariance -----------------------------
def _kmeans_template(F_list, K, seed):
    g = RNG(seed); pool = np.vstack(F_list)
    cent = pool[g.choice(len(pool), K, replace=False)].copy()
    for _ in range(15):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    return cent


def _rep_ours(P, F, G, Mmax):
    Fb, Cb, ab = pad_batch([P], [F], Mmax)
    K = G.shape[0]
    coords = torch.tensor(G[:, :2]); Cref = torch.cdist(coords, coords)
    b = torch.full((K,), 1.0 / K)
    with torch.no_grad():
        z = fgw_layer(Fb, Cb, ab, torch.tensor(G), Cref, b, torch.zeros(CH))
    return z[0].numpy()


def invariance_test(n_samples=12, n_iso=8, seed=0):
    P, F, _ = gen_dataset(seed, warp=0.4, n=n_samples)
    Mmax = max(len(p) for p in P) + 1                      # no subsample => exact relabel-invariance
    G = _kmeans_template(F, K=8, seed=0)
    ours, grid = [], []
    g = RNG(seed + 7)
    for p, f in zip(P, F):
        z0 = _rep_ours(p, f, G, Mmax); x0 = rasterize([p], [f])[0]
        for _ in range(n_iso):
            th = g.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            t = g.uniform(-1.5, 1.5, size=2)
            perm = g.permutation(len(p))
            p1 = (p @ R.T + t)[perm]; f1 = f[perm]           # isometry + relabel
            z1 = _rep_ours(p1, f1, G, Mmax); x1 = rasterize([p1], [f1])[0]
            ours.append(np.linalg.norm(z1 - z0) / (np.linalg.norm(z0) + 1e-12))
            grid.append(np.linalg.norm(x1 - x0) / (np.linalg.norm(x0) + 1e-12))
    return dict(ours_rel_change_mean=float(np.mean(ours)), ours_rel_change_max=float(np.max(ours)),
                grid_rel_change_mean=float(np.mean(grid)), grid_rel_change_max=float(np.max(grid)))


# ----------------------------- (2) consistency -----------------------------
def consistency_test(Ns=(40, 80, 160, 320, 640), n_test=400, seeds=(0, 1, 2), warp=0.5):
    P_te, F_te, Y_te = gen_dataset(777, warp, n_test)       # FIXED held-out test set
    mse = {}
    for N in Ns:
        errs = []
        for s in seeds:
            P_tr, F_tr, Y_tr = gen_dataset(s, warp, N)
            P = P_tr + P_te; F = F_tr + F_te; Y = np.concatenate([Y_tr, Y_te])
            tr = np.arange(N); te = np.arange(N, N + n_test)
            js.Y_GLOBAL = Y
            # train_joint returns test R2; convert to test MSE on the fixed test set
            r2 = train_joint(P, F, Y, tr, te, seed=s)
            errs.append((1.0 - r2) * float(np.var(Y_te)))    # MSE = (1-R2)*Var(Y_te)
        mse[N] = (float(np.mean(errs)), float(np.std(errs)))
    Ns_a = np.array(list(mse.keys()), float)
    m = np.array([mse[n][0] for n in mse])
    floor = m.min() * 0.5                                   # rough approximation-error floor
    excess = np.clip(m - floor, 1e-6, None)
    slope = float(np.polyfit(np.log(Ns_a), np.log(excess), 1)[0])  # ~ -1 for MSE ~ 1/N
    return dict(mse={int(n): mse[n] for n in mse}, loglog_slope_excess_mse=slope,
                note="MSE~1/N => slope~-1 (RMSE~1/sqrt(N)); approximation floor subtracted")


def main():
    print("[1] invariance ...", flush=True)
    inv = invariance_test()
    print(f"  ours rel-change mean {inv['ours_rel_change_mean']:.2e} (max {inv['ours_rel_change_max']:.2e}) "
          f"vs grid mean {inv['grid_rel_change_mean']:.3f} (max {inv['grid_rel_change_max']:.3f})", flush=True)
    print("[2] consistency (this takes a few minutes) ...", flush=True)
    con = consistency_test()
    for n, (mm, sd) in con["mse"].items():
        print(f"  N={n:>4}  test MSE {mm:.4f} +/- {sd:.4f}", flush=True)
    print(f"  log-log slope of excess MSE vs N: {con['loglog_slope_excess_mse']:+.2f} (predicted ~ -1)", flush=True)

    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "synthetic_gonogo")
    os.makedirs(od, exist_ok=True)
    json.dump(dict(invariance=inv, consistency=con),
              open(os.path.join(od, "theory_validation.json"), "w"), indent=2)
    print("wrote theory_validation.json", flush=True)


if __name__ == "__main__":
    main()
