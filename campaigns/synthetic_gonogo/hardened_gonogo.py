"""
Hardened GO/NO-GO. Adds to the first experiment:
  (1) a real multilinear-PLS baseline (N-PLS / HOPLS family: rank-1 multilinear
      loadings on the aligned grid tensor) -- not just flatten+PLS;
  (2) the SUPERVISED template loop (the actual novelty) vs the unsupervised template;
  (3) a harder noise/atom regime where supervision should matter more;
  (4) a classification (AUC) read-out alongside regression R^2;
  (5) strict no-leakage: template built on TRAIN samples only.

Decision rules:
  - HOPLS-family baseline must show the same alignment-collapse as flatten naive.
  - ours_sup must be >= ours_unsup (especially under warp / hard regime) to justify
    the supervised contribution. Otherwise the novelty is not yet earned.
"""
import numpy as np
import ot
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

from synthetic_gonogo import (gen_dataset, rasterize, rigid_register, eval_rep, C)

RNG = np.random.default_rng


# ----------------------- N-PLS / HOPLS-family baseline -----------------------
def _rank1_als(T, iters=30):
    """Rank-1 CP/HOSVD approx of tensor T -> unit loading vector per mode."""
    shape = T.shape
    facs = [np.ones(s) / np.sqrt(s) for s in shape]
    for _ in range(iters):
        for m in range(T.ndim):
            # contract all modes except m with current factors
            v = T
            for k in range(T.ndim):
                if k == m:
                    continue
                v = np.tensordot(v, facs[k], axes=([1 if k > m else 0], [0]))
            n = np.linalg.norm(v) + 1e-12
            facs[m] = v / n
    w = facs[0]
    for m in range(1, T.ndim):
        w = np.multiply.outer(w, facs[m])
    return w.ravel()


def nway_pls_eval(Xtr, ytr, Xte, yte, ncomp=5):
    """Multilinear PLS on aligned grid tensors X:(N,d1,d2,d3). Rank-1 multilinear
    weights per component (the alignment-assuming tensor-PLS baseline)."""
    N = Xtr.shape[0]; modes = Xtr.shape[1:]
    Xtr = Xtr.reshape(N, -1).astype(float); Xte = Xte.reshape(Xte.shape[0], -1).astype(float)
    mu = Xtr.mean(0); Xtr = Xtr - mu; Xte = Xte - mu
    yc = ytr - ytr.mean()
    Xd = Xtr.copy(); W = []
    for _ in range(ncomp):
        cross = (Xd * yc[:, None]).sum(0).reshape(modes)        # cross-cov tensor
        w = _rank1_als(cross); w /= np.linalg.norm(w) + 1e-12
        t = Xd @ w
        W.append(w)
        Xd = Xd - np.outer(t, (t @ Xd) / (t @ t + 1e-12))       # deflate X block
    W = np.array(W).T
    Ttr = Xtr @ W; Tte = Xte @ W
    beta = np.linalg.lstsq(np.c_[np.ones(N), Ttr], ytr, rcond=None)[0]
    pred = np.c_[np.ones(len(yte)), Tte] @ beta
    return r2_score(yte, pred)


# --------------------- registration-free features (un/supervised) -------------
def fgw_features(P, F, train_idx, n_atoms=8, alpha=0.5, seed=0, Y=None):
    """Build shared template from TRAIN pool only (no leakage); transport ALL
    samples via fused-GW; barycentric-project. If Y is given -> SUPERVISED template:
    cluster in a Y-discriminant-weighted feature space so atoms separate predictive
    feature regions."""
    g = RNG(seed + 7)
    pool = np.vstack([F[i] for i in train_idx])

    if Y is not None:                                            # supervised template
        sizes = [len(F[i]) for i in train_idx]
        ylab = np.concatenate([[Y[i]] * sizes[j] for j, i in enumerate(train_idx)])
        d = Ridge(alpha=1.0).fit(pool, ylab).coef_
        d = d / (np.linalg.norm(d) + 1e-9)
        space = np.concatenate([pool, (3.0 * (pool @ d))[:, None]], axis=1)
    else:                                                        # unsupervised template
        space = pool

    idx = g.choice(len(pool), size=n_atoms, replace=False)
    cent = space[idx].copy()
    lab = np.zeros(len(space), dtype=int)
    for _ in range(15):                                          # Lloyd in (sup) space
        dd = ((space[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = dd.argmin(1)
        for k in range(n_atoms):
            if (lab == k).any():
                cent[k] = space[lab == k].mean(0)
    proto = np.zeros((n_atoms, pool.shape[1]))                   # prototypes in feat space
    for k in range(n_atoms):
        m = lab == k
        proto[k] = pool[m].mean(0) if m.any() else pool[g.integers(len(pool))]

    C_ref = ot.dist(proto, proto); C_ref /= C_ref.max() + 1e-9
    q = np.full(n_atoms, 1.0 / n_atoms)
    out = []
    for p, f in zip(P, F):
        C_s = ot.dist(p, p); C_s /= C_s.max() + 1e-9
        M = ot.dist(f, proto); M /= M.max() + 1e-9
        ps = np.full(len(p), 1.0 / len(p))
        T = ot.gromov.fused_gromov_wasserstein(
            M, C_s, C_ref, ps, q, loss_fun="square_loss", alpha=alpha, verbose=False)
        mass = T.sum(0); denom = mass[:, None] + 1e-12
        out.append(np.concatenate([mass, ((T.T @ f) / denom).ravel()]))
    return np.array(out)


def grid_tensor(P, F, H=12):
    """Aligned grid tensor (N,H,H,C+1) for the multilinear-PLS baseline."""
    flat = rasterize(P, F, H)                                    # [mass(H*H), feat(H*H*C)]
    N = len(flat)
    mass = flat[:, :H * H].reshape(N, H, H, 1)
    feat = flat[:, H * H:].reshape(N, H, H, C)
    return np.concatenate([mass, feat], axis=-1)


# ----------------------------- evaluation -----------------------------
def run(regime="standard"):
    if regime == "hard":
        kw = dict(); noise_note = "more atoms / more noise / fewer points"
    warps = [0.0, 0.5, 1.0]
    seeds = [0, 1, 2, 3, 4]
    n = 160
    methods = ["flatten_naive", "hopls_grid", "rigid_reg", "ours_unsup", "ours_sup"]
    r2 = {m: {w: [] for w in warps} for m in methods}
    auc = {m: {w: [] for w in warps} for m in methods}

    for w in warps:
        for s in seeds:
            P, F, Y = gen_dataset(s, w, n)
            if regime == "hard":                                # inject extra feature noise
                F = [f + RNG(s * 97 + 5).normal(scale=0.25, size=f.shape) for f in F]
            ntr = int(0.7 * n); tr = slice(0, ntr); te = slice(ntr, n)
            tr_idx = np.arange(ntr)
            ythr = np.median(Y[tr])
            ycls_te = (Y[te] > ythr).astype(int)

            def record(name, Xtr, Xte):
                r2[name][w].append(eval_rep(Xtr, Y[tr], Xte, Y[te]))
                # classification AUC: reuse PLS scores via 1-comp proj on standardized X
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                pls = PLSRegression(n_components=min(5, Xtr.shape[1], Xtr.shape[0] - 1))
                pls.fit((Xtr - mu) / sd, (Y[tr] > ythr).astype(float))
                sc = pls.predict((Xte - mu) / sd).ravel()
                auc[name][w].append(roc_auc_score(ycls_te, sc) if ycls_te.min() != ycls_te.max() else np.nan)

            Xn = rasterize(P, F); record("flatten_naive", Xn[tr], Xn[te])
            Xr = rasterize(rigid_register(P), F); record("rigid_reg", Xr[tr], Xr[te])
            Xu = fgw_features(P, F, tr_idx, seed=s); record("ours_unsup", Xu[tr], Xu[te])
            Xs = fgw_features(P, F, tr_idx, seed=s, Y=Y); record("ours_sup", Xs[tr], Xs[te])

            GT = grid_tensor(P, F)
            r2["hopls_grid"][w].append(nway_pls_eval(GT[tr], Y[tr], GT[te], Y[te]))
            mu, sd = GT[tr].reshape(ntr, -1).mean(0), GT[tr].reshape(ntr, -1).std(0) + 1e-9
            auc["hopls_grid"][w].append(np.nan)                  # regression-focused baseline
        print(f"[{regime}] warp={w:.2f} done")

    print(f"\n=== [{regime}] R^2 (mean over seeds) ===")
    print(f"{'method':<15}" + "".join(f"  w={w:<4}" for w in warps))
    for m in methods:
        print(f"{m:<15}" + "".join(f"  {np.mean(r2[m][w]):+.2f} " for w in warps))
    print(f"\n=== [{regime}] classification AUC (mean over seeds) ===")
    print(f"{'method':<15}" + "".join(f"  w={w:<4}" for w in warps))
    for m in methods:
        vals = [np.nanmean(auc[m][w]) for w in warps]
        print(f"{m:<15}" + "".join(f"  {v:.2f} " if not np.isnan(v) else "   -   " for v in vals))

    print(f"\n=== [{regime}] novelty check: ours_sup - ours_unsup (R^2) ===")
    diff = [np.mean(r2["ours_sup"][w]) - np.mean(r2["ours_unsup"][w]) for w in warps]
    print("per warp:", np.round(diff, 3),
          "  -> supervision helps" if np.mean(diff) > 0.01 else "  -> supervision NOT yet earned")
    return r2, auc


if __name__ == "__main__":
    run("standard")
    run("hard")
