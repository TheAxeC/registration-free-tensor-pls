"""
Does the SUPERVISED template actually add lift? The standard synthetic world gives
unsupervised k-means zero headroom (atoms separable + Y depends on mass). Here we
build a regime WITH headroom:
  - n_nuis nuisance atoms: many points, LARGE feature variance, Y-IRRELEVANT;
  - n_pred predictive atoms: fewer points, signal in a SUBTLE low-variance subspace,
    activations drive Y.
With a limited template budget K, unsupervised k-means should spend atoms on dominant
nuisance variance; a supervised template (cluster + transport in a Y-predictive metric)
should focus atoms on the predictive subspace.

Compares: ours_unsup vs ours_sup (+ flatten_naive & hopls for context).
"""
import numpy as np
import ot
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from synthetic_gonogo import eval_rep
from hardened_gonogo import nway_pls_eval


def rasterize(P, F, H=12):                               # CH-channel local version
    out = []
    for p, f in zip(P, F):
        lo, hi = -2.0, 2.0
        ix = np.clip(((p[:, 0] - lo) / (hi - lo) * H).astype(int), 0, H - 1)
        iy = np.clip(((p[:, 1] - lo) / (hi - lo) * H).astype(int), 0, H - 1)
        mass = np.zeros((H, H)); feat = np.zeros((H, H, CH))
        for j in range(len(p)):
            mass[ix[j], iy[j]] += 1; feat[ix[j], iy[j]] += f[j]
        nz = mass > 0; feat[nz] /= mass[nz][:, None]
        out.append(np.concatenate([mass.ravel() / max(1, len(p)), feat.ravel()]))
    return np.array(out)


def grid_tensor(P, F, H=12):
    flat = rasterize(P, F, H); N = len(flat)
    mass = flat[:, :H * H].reshape(N, H, H, 1)
    feat = flat[:, H * H:].reshape(N, H, H, CH)
    return np.concatenate([mass, feat], axis=-1)

RNG = np.random.default_rng
CH = 6                      # feature channels
N_PRED, N_NUIS = 3, 4
PRED_W = np.array([1.5, -1.0, 0.9])


def make_world(seed):
    g = RNG(seed)
    K = N_PRED + N_NUIS
    centers = g.uniform(0.1, 0.9, size=(K, 2))
    sig = np.zeros((K, CH))
    # predictive atoms: signal only in channels 0,1, SMALL norm (subtle)
    sig[:N_PRED, :2] = g.normal(scale=0.5, size=(N_PRED, 2))
    # nuisance atoms: LARGE norm in channels 2..5 (dominant variance, irrelevant)
    sig[N_PRED:, 2:] = g.normal(scale=2.5, size=(N_NUIS, CH - 2))
    return centers, sig


def gen_sample(g, centers, sig, warp):
    K = N_PRED + N_NUIS
    act = g.uniform(0, 1, size=K)
    pts, feats = [], []
    for k in range(K):
        base = 18 if k < N_PRED else 55              # nuisance atoms = more mass
        nk = int(round(act[k] * base))
        if nk == 0:
            continue
        pts.append(centers[k] + g.normal(scale=0.04, size=(nk, 2)))
        feats.append(sig[k] + g.normal(scale=0.2, size=(nk, CH)))
    if not pts:
        pts.append(centers[0] + g.normal(scale=0.04, size=(6, 2)))
        feats.append(sig[0] + g.normal(scale=0.2, size=(6, CH)))
    P = np.vstack(pts); F = np.vstack(feats)
    th = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = P @ R.T + g.uniform(-warp, warp, size=2)
    keep = g.uniform(size=len(P)) < 0.75
    if keep.sum() >= 5:
        P, F = P[keep], F[keep]
    Y = float(PRED_W @ act[:N_PRED] + g.normal(scale=0.05))
    return P, F, Y


def gen_dataset(seed, warp, n):
    g = RNG(seed + 1000)
    centers, sig = make_world(seed)
    d = [gen_sample(g, centers, sig, warp) for _ in range(n)]
    return [x[0] for x in d], [x[1] for x in d], np.array([x[2] for x in d])


def fgw_features(P, F, train_idx, Y=None, n_atoms=6, alpha=0.5, seed=0):
    g = RNG(seed + 7)
    pool = np.vstack([F[i] for i in train_idx])
    transform = None
    if Y is not None:                                   # supervised predictive metric
        sizes = [len(F[i]) for i in train_idx]
        ylab = np.concatenate([[Y[i]] * sizes[j] for j, i in enumerate(train_idx)])
        pls = PLSRegression(n_components=2).fit(pool, ylab)
        # scale-restore so the metric is a linear map W: feat -> predictive coords
        W = pls.x_rotations_                            # (CH, 2)
        transform = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
        space = pool @ transform
    else:
        space = pool

    idx = g.choice(len(pool), size=n_atoms, replace=False)
    cent = space[idx].copy(); lab = np.zeros(len(space), dtype=int)
    for _ in range(20):
        lab = ((space[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(n_atoms):
            if (lab == k).any():
                cent[k] = space[lab == k].mean(0)
    proto = np.array([pool[lab == k].mean(0) if (lab == k).any()
                      else pool[g.integers(len(pool))] for k in range(n_atoms)])

    # cost metric: supervised uses predictive-subspace distance; else raw
    if transform is not None:
        proto_m = proto @ transform
        feat_m = lambda f: f @ transform
    else:
        proto_m = proto; feat_m = lambda f: f
    C_ref = ot.dist(proto_m, proto_m); C_ref /= C_ref.max() + 1e-9
    q = np.full(n_atoms, 1.0 / n_atoms)
    out = []
    for p, f in zip(P, F):
        C_s = ot.dist(p, p); C_s /= C_s.max() + 1e-9
        M = ot.dist(feat_m(f), proto_m); M /= M.max() + 1e-9
        ps = np.full(len(p), 1.0 / len(p))
        T = ot.gromov.fused_gromov_wasserstein(M, C_s, C_ref, ps, q,
                                               loss_fun="square_loss", alpha=alpha)
        mass = T.sum(0); denom = mass[:, None] + 1e-12
        out.append(np.concatenate([mass, ((T.T @ f) / denom).ravel()]))
    return np.array(out)


def run():
    warps = [0.0, 0.5, 1.0]; seeds = [0, 1, 2, 3, 4]; n = 180
    methods = ["flatten_naive", "hopls_grid", "ours_unsup", "ours_sup"]
    r2 = {m: {w: [] for w in warps} for m in methods}
    for w in warps:
        for s in seeds:
            P, F, Y = gen_dataset(s, w, n)
            ntr = int(0.7 * n); tr = slice(0, ntr); te = slice(ntr, n)
            ti = np.arange(ntr)
            Xn = rasterize(P, F); r2["flatten_naive"][w].append(
                eval_rep(Xn[tr], Y[tr], Xn[te], Y[te]))
            GT = grid_tensor(P, F); r2["hopls_grid"][w].append(
                nway_pls_eval(GT[tr], Y[tr], GT[te], Y[te]))
            Xu = fgw_features(P, F, ti, seed=s); r2["ours_unsup"][w].append(
                eval_rep(Xu[tr], Y[tr], Xu[te], Y[te]))
            Xs = fgw_features(P, F, ti, Y=Y, seed=s); r2["ours_sup"][w].append(
                eval_rep(Xs[tr], Y[tr], Xs[te], Y[te]))
        print(f"warp={w:.2f} done")

    print("\n=== R^2 (mean over seeds), nuisance regime ===")
    print(f"{'method':<15}" + "".join(f"  w={w:<4}" for w in warps))
    for m in methods:
        print(f"{m:<15}" + "".join(f"  {np.mean(r2[m][w]):+.2f} " for w in warps))
    diff = [np.mean(r2["ours_sup"][w]) - np.mean(r2["ours_unsup"][w]) for w in warps]
    print("\nours_sup - ours_unsup:", np.round(diff, 3),
          "  -> SUPERVISION EARNED" if np.mean(diff) > 0.03 else "  -> still not earned")


if __name__ == "__main__":
    run()
