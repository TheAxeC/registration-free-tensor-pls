"""
Joint-supervised registration-free tensor-PLS via a differentiable entropic
fused-Gromov-Wasserstein layer.

Trainable: template atom features G, latent atom structure C_ref, a diagonal feature
metric (can suppress nuisance channels), learnable atom marginals b, and a linear
head z -> Y. All trained end-to-end through a differentiable fused-GW transport so the
template is shaped to MAXIMIZE prediction of Y (the genuine supervised objective, not a
two-stage proxy).

Benchmark v2 makes SPATIAL STRUCTURE predictive: Y depends on the intra-sample distance
between two predictive atoms -> recoverable only by (a) identifying atoms via features
AND (b) measuring their separation via structure. This justifies the GW machinery and is
something bag-of-features cannot capture. Nuisance atoms (high mass, high feature
variance, Y-irrelevant) create headroom where an UNsupervised template fails.

Reports R^2 vs warp for: flatten+PLS, HOPLS/N-PLS, bag-of-features, ours(unsup template),
ours(JOINT supervised). The supervised contribution is earned only if joint-sup is
clearly positive AND beats the unsupervised template + bag-of-features.
"""
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from synthetic_gonogo import eval_rep
from hardened_gonogo import nway_pls_eval

torch.set_default_dtype(torch.float64)
RNG = np.random.default_rng
CH = 6
N_PRED, N_NUIS = 3, 3
A_IDX, B_IDX = 0, 1          # the two atoms whose separation drives Y


# ----------------------------- benchmark v2 -----------------------------
def make_world(seed):
    g = RNG(seed)
    sig = np.zeros((N_PRED + N_NUIS, CH))
    sig[:N_PRED, :3] = g.normal(scale=1.2, size=(N_PRED, 3))      # predictive: ch 0-2
    sig[N_PRED:, 3:] = g.normal(scale=2.5, size=(N_NUIS, 3))      # nuisance: ch 3-5, big
    base = g.uniform(0.2, 0.8, size=(N_PRED + N_NUIS, 2))
    return sig, base


def gen_sample(g, sig, base, warp):
    centers = base.copy()
    # jitter predictive atom centers so the A-B separation varies across samples
    centers[:N_PRED] += g.uniform(-0.25, 0.25, size=(N_PRED, 2))
    pts, feats, owner = [], [], []
    for k in range(N_PRED + N_NUIS):
        nk = 16 if k < N_PRED else 45                            # nuisance = more mass
        pts.append(centers[k] + g.normal(scale=0.03, size=(nk, 2)))
        feats.append(sig[k] + g.normal(scale=0.25, size=(nk, CH)))
        owner += [k] * nk
    P = np.vstack(pts); F = np.vstack(feats); owner = np.array(owner)
    # structure-predictive target (computed on clean centroids, pre-warp)
    cA = P[owner == A_IDX].mean(0); cB = P[owner == B_IDX].mean(0)
    dist_AB = np.linalg.norm(cA - cB)
    actA = (owner == A_IDX).sum() / 16.0
    Y = float(2.0 * dist_AB + 0.5 * actA + g.normal(scale=0.03))
    # warp (rigid preserves dist_AB; nonrigid adds noise)
    th = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = P @ R.T + g.uniform(-warp, warp, size=2)
    if warp > 0:
        a = g.uniform(P.min(), P.max(), size=(2, 2))
        for anc in a:
            d2 = ((P - anc) ** 2).sum(1)
            P = P + np.exp(-d2 / 0.08)[:, None] * g.normal(scale=0.08 * warp, size=2)
    keep = g.uniform(size=len(P)) < 0.8
    P, F = P[keep], F[keep]
    return P, F, Y


def gen_dataset(seed, warp, n):
    g = RNG(seed + 1000); sig, base = make_world(seed)
    d = [gen_sample(g, sig, base, warp) for _ in range(n)]
    return [x[0] for x in d], [x[1] for x in d], np.array([x[2] for x in d])


# ------------------- classical reps for baselines -------------------
def rasterize(P, F, H=12):
    out = []
    for p, f in zip(P, F):
        lo, hi = -2.5, 2.5
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
    return np.concatenate([flat[:, :H * H].reshape(N, H, H, 1),
                           flat[:, H * H:].reshape(N, H, H, CH)], axis=-1)


def bag_of_features(P, F, train_idx, n_atoms=8, seed=0):
    """Feature-only clustering + per-cluster mass & mean feature. NO spatial structure
    -> should miss the distance-based predictive signal."""
    g = RNG(seed + 3)
    pool = np.vstack([F[i] for i in train_idx])
    cent = pool[g.choice(len(pool), n_atoms, replace=False)].copy()
    for _ in range(20):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(n_atoms):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    out = []
    for f in F:
        d = ((f[:, None] - cent[None]) ** 2).sum(-1); lab = d.argmin(1)
        mass = np.array([(lab == k).mean() for k in range(n_atoms)])
        mean = np.array([f[lab == k].mean(0) if (lab == k).any() else np.zeros(CH)
                         for k in range(n_atoms)])
        out.append(np.concatenate([mass, mean.ravel()]))
    return np.array(out)


# ------------------- differentiable entropic fused-GW -------------------
def pad_batch(P, F, Mmax):
    """Subsample/pad each sample to Mmax points; return features, structure, marginals."""
    B = len(P)
    Fb = torch.zeros(B, Mmax, CH); Cb = torch.zeros(B, Mmax, Mmax); ab = torch.zeros(B, Mmax)
    g = RNG(0)
    for i, (p, f) in enumerate(zip(P, F)):
        m = len(p)
        idx = np.arange(m) if m <= Mmax else g.choice(m, Mmax, replace=False)
        pp, ff = p[idx], f[idx]
        mm = len(idx)
        Fb[i, :mm] = torch.tensor(ff)
        d = np.linalg.norm(pp[:, None] - pp[None], axis=-1)
        d = d / (d.max() + 1e-9)
        Cb[i, :mm, :mm] = torch.tensor(d)
        ab[i, :mm] = 1.0 / mm
    return Fb, Cb, ab


def fgw_layer(Fb, Cb, ab, G, Cref, b, log_metric, alpha=0.5, eps=0.05,
              outer=5, sink=25):
    """Batched differentiable entropic fused-GW -> barycentric z. log_metric: per-channel
    log-scale (diagonal feature metric; suppresses nuisance channels when learned)."""
    B, Mmax, _ = Fb.shape; K = G.shape[0]
    w = torch.exp(log_metric)                                   # (CH,) positive weights
    Fm = Fb * torch.sqrt(w)[None, None]                         # metric-scaled features
    Gm = G * torch.sqrt(w)[None]
    M = ((Fm[:, :, None, :] - Gm[None, None]) ** 2).sum(-1)     # (B,Mmax,K) feature cost
    M = M / (M.amax(dim=(1, 2), keepdim=True) + 1e-9)
    Crefn = Cref / (Cref.max() + 1e-9)
    bb = b[None].expand(B, K)
    Ca = (Cb ** 2) @ ab[:, :, None]                            # (B,Mmax,1)
    Cbt = (Crefn ** 2) @ b                                     # (K,)
    T = ab[:, :, None] * bb[:, None, :]                         # init plan (B,Mmax,K)
    for _ in range(outer):
        cross = Cb @ T @ Crefn                                  # (B,Mmax,K)
        cost = (1 - alpha) * M + alpha * (Ca + Cbt[None, None] - 2 * cross)
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        Km = torch.exp(-cost / eps)                             # (B,Mmax,K)
        u = torch.ones(B, Mmax); v = torch.ones(B, K)
        for _ in range(sink):
            u = ab / (torch.bmm(Km, v[:, :, None]).squeeze(-1) + 1e-30)
            v = bb / (torch.bmm(Km.transpose(1, 2), u[:, :, None]).squeeze(-1) + 1e-30)
        T = u[:, :, None] * Km * v[:, None, :]
    mass = T.sum(1)                                            # (B,K)
    zfeat = torch.bmm(T.transpose(1, 2), Fb) / (mass[:, :, None] + 1e-9)  # (B,K,CH)
    return torch.cat([mass, zfeat.reshape(B, -1)], dim=1)      # (B, K + K*CH)


class JointModel(torch.nn.Module):
    def __init__(self, K, seed, init_G):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.G = torch.nn.Parameter(torch.tensor(init_G))
        L = torch.randn(K, 2, generator=g) * 0.3               # latent atom coords
        self.coords = torch.nn.Parameter(L)
        self.b_logit = torch.nn.Parameter(torch.zeros(K))
        self.log_metric = torch.nn.Parameter(torch.zeros(CH))
        self.head = torch.nn.Linear(K + K * CH, 1)

    def forward(self, Fb, Cb, ab, alpha=0.5):
        Cref = torch.cdist(self.coords, self.coords)
        b = torch.softmax(self.b_logit, 0)
        z = fgw_layer(Fb, Cb, ab, self.G, Cref, b, self.log_metric, alpha=alpha)
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        return self.head(z).squeeze(-1), z


def train_joint(P, F, Y, tr_idx, te_idx, K=8, seed=0, epochs=250, Mmax=50):
    Fb, Cb, ab = pad_batch(P, F, Mmax)
    y = torch.tensor(Y)
    # init template from unsupervised kmeans (warm start)
    g = RNG(seed); pool = np.vstack([F[i] for i in tr_idx])
    cent = pool[g.choice(len(pool), K, replace=False)].copy()
    for _ in range(15):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    model = JointModel(K, seed, cent)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    tr = torch.tensor(tr_idx)
    ymu, ysd = y[tr].mean(), y[tr].std()
    for ep in range(epochs):
        opt.zero_grad()
        pred, _ = model(Fb, Cb, ab)
        loss = ((pred[tr] - (y[tr] - ymu) / ysd) ** 2).mean() \
            + 1e-3 * (model.G ** 2).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred, _ = model(Fb, Cb, ab)
        pred = pred * ysd + ymu
    return r2_score(Y[te_idx], pred[te_idx].numpy())


def fgw_unsup(P, F, tr_idx, te_idx, K=8, seed=0, Mmax=50):
    """Unsupervised template (kmeans) + ridge head, same fgw layer, no training."""
    Fb, Cb, ab = pad_batch(P, F, Mmax)
    g = RNG(seed); pool = np.vstack([F[i] for i in tr_idx])
    cent = pool[g.choice(len(pool), K, replace=False)].copy()
    for _ in range(15):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    G = torch.tensor(cent)
    coords = torch.tensor(cent[:, :2])
    Cref = torch.cdist(coords, coords)
    b = torch.full((K,), 1.0 / K)
    with torch.no_grad():
        z = fgw_layer(Fb, Cb, ab, G, Cref, b, torch.zeros(CH)).numpy()
    return eval_rep(z[tr_idx], Y_GLOBAL[tr_idx], z[te_idx], Y_GLOBAL[te_idx])


def run():
    global Y_GLOBAL
    warps = [0.0, 0.5, 1.0]; seeds = [0, 1, 2]; n = 180
    methods = ["flatten", "hopls", "bag_feat", "ours_unsup", "ours_joint_sup"]
    r2 = {m: {w: [] for w in warps} for m in methods}
    for w in warps:
        for s in seeds:
            P, F, Y = gen_dataset(s, w, n); Y_GLOBAL = Y
            ntr = int(0.7 * n); tr = np.arange(ntr); te = np.arange(ntr, n)
            Xn = rasterize(P, F); r2["flatten"][w].append(
                eval_rep(Xn[tr], Y[tr], Xn[te], Y[te]))
            GT = grid_tensor(P, F); r2["hopls"][w].append(
                nway_pls_eval(GT[tr], Y[tr], GT[te], Y[te]))
            Xb = bag_of_features(P, F, tr, seed=s); r2["bag_feat"][w].append(
                eval_rep(Xb[tr], Y[tr], Xb[te], Y[te]))
            r2["ours_unsup"][w].append(fgw_unsup(P, F, tr, te, seed=s))
            r2["ours_joint_sup"][w].append(train_joint(P, F, Y, tr, te, seed=s))
        print(f"warp={w:.2f} done  "
              f"(joint_sup R2={np.mean(r2['ours_joint_sup'][w]):+.2f}, "
              f"unsup={np.mean(r2['ours_unsup'][w]):+.2f}, bag={np.mean(r2['bag_feat'][w]):+.2f})")

    print("\n=== R^2 (mean over seeds), structure-predictive + nuisance regime ===")
    print(f"{'method':<16}" + "".join(f"  w={w:<4}" for w in warps))
    for m in methods:
        print(f"{m:<16}" + "".join(f"  {np.mean(r2[m][w]):+.2f} " for w in warps))
    js = np.array([np.mean(r2["ours_joint_sup"][w]) for w in warps])
    un = np.array([np.mean(r2["ours_unsup"][w]) for w in warps])
    bf = np.array([np.mean(r2["bag_feat"][w]) for w in warps])
    print("\nEARNED check:")
    print(f"  joint_sup positive at all warps: {np.all(js > 0.15)}")
    print(f"  joint_sup beats unsup template : {np.all(js > un + 0.05)}  (margins {np.round(js-un,2)})")
    print(f"  joint_sup beats bag-of-features: {np.all(js > bf + 0.05)}  (margins {np.round(js-bf,2)})")
    verdict = np.all(js >= 0.15) and np.all(js > un + 0.05) and np.all(js > bf + 0.05)
    print("  VERDICT:", "SUPERVISED NOVELTY EARNED" if verdict else "not yet -> iterate")
    import pickle
    with open("joint_results.pkl", "wb") as fh:
        pickle.dump({"r2": r2, "warps": warps, "methods": methods}, fh)
    print("saved joint_results.pkl")
    return r2


if __name__ == "__main__":
    run()
