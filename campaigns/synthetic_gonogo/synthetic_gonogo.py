"""
Synthetic GO/NO-GO for registration-free supervised tensor-PLS (fused-GW route).

Core claim under test:
  When samples are point clouds drawn from a SHARED set of latent atoms but each
  sample is independently warped (rotation+translation+nonrigid+resampling), so that
  cross-sample voxel correspondence is destroyed, a registration-free representation
  (fused Gromov-Wasserstein transport to a shared template) recovers the predictive
  latent activations, whereas alignment-naive grid/flatten PLS and cheap rigid
  registration degrade as warp grows.

GO criterion:
  Ours >= baselines at every warp level AND the advantage WIDENS with warp magnitude
  (baselines decay, ours stays roughly flat).

This is a private de-risking experiment. A null outcome is guidance, not a manuscript.
"""
import numpy as np
import ot
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

RNG = np.random.default_rng  # seedable factory

# ----------------------------- ground-truth world -----------------------------
C = 4          # modalities (feature channels)
K_TRUE = 6     # true latent atoms
PRED_ATOMS = [0, 2, 4]          # only these atoms drive Y
PRED_W = np.array([1.5, -1.0, 0.8])  # their regression weights


def make_world(seed):
    g = RNG(seed)
    centers = g.uniform(0.1, 0.9, size=(K_TRUE, 2))      # atom spatial prototypes
    sig = g.normal(size=(K_TRUE, C))                      # atom feature signatures
    sig = sig / np.linalg.norm(sig, axis=1, keepdims=True)
    return centers, sig


def gen_sample(g, centers, sig, warp):
    """One warped point cloud + its target Y (Y depends on atom activations only)."""
    act = g.uniform(0.0, 1.0, size=K_TRUE)               # activation per atom
    act[g.uniform(size=K_TRUE) < 0.25] = 0.0             # some atoms absent
    pts, feats = [], []
    for k in range(K_TRUE):
        n_k = int(round(act[k] * 40))                    # mass encodes activation
        if n_k == 0:
            continue
        p = centers[k] + g.normal(scale=0.04, size=(n_k, 2))
        f = sig[k] + g.normal(scale=0.15, size=(n_k, C))
        pts.append(p); feats.append(f)
    if not pts:                                           # guarantee non-degenerate
        p = centers[0] + g.normal(scale=0.04, size=(6, 2))
        pts.append(p); feats.append(sig[0] + g.normal(scale=0.15, size=(6, C)))
    P = np.vstack(pts); F = np.vstack(feats)

    # ---- per-sample warp: rotation + translation + nonrigid + resampling ----
    theta = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    P = P @ R.T
    P = P + g.uniform(-warp, warp, size=2)
    if warp > 0:                                         # nonrigid RBF displacement
        anchors = g.uniform(P.min(), P.max(), size=(3, 2))
        for a in anchors:
            d2 = ((P - a) ** 2).sum(1)
            disp = g.normal(scale=0.15 * warp, size=2)
            P = P + np.exp(-d2 / 0.05)[:, None] * disp
    keep = g.uniform(size=len(P)) < 0.75                 # random resampling
    if keep.sum() >= 5:
        P, F = P[keep], F[keep]

    Y = float(PRED_W @ act[PRED_ATOMS] + g.normal(scale=0.05))
    return P, F, Y


def gen_dataset(seed, warp, n):
    g = RNG(seed + 1000)
    centers, sig = make_world(seed)
    data = [gen_sample(g, centers, sig, warp) for _ in range(n)]
    P = [d[0] for d in data]; F = [d[1] for d in data]
    Y = np.array([d[2] for d in data])
    return P, F, Y


# ----------------------------- representations -----------------------------
def rasterize(P, F, H=12):
    """Alignment-naive: bin points to a fixed HxH grid; cell = [mass, mean-feature]."""
    out = []
    for p, f in zip(P, F):
        lo, hi = -2.0, 2.0                               # fixed global frame
        ix = np.clip(((p[:, 0] - lo) / (hi - lo) * H).astype(int), 0, H - 1)
        iy = np.clip(((p[:, 1] - lo) / (hi - lo) * H).astype(int), 0, H - 1)
        mass = np.zeros((H, H)); feat = np.zeros((H, H, C))
        for j in range(len(p)):
            mass[ix[j], iy[j]] += 1
            feat[ix[j], iy[j]] += f[j]
        nz = mass > 0
        feat[nz] /= mass[nz][:, None]
        out.append(np.concatenate([mass.ravel() / max(1, len(p)), feat.ravel()]))
    return np.array(out)


def rigid_register(P):
    """Cheap registration: center centroid, rotate principal axis to x. Removes
    rotation+translation but NOT nonrigid warp."""
    out = []
    for p in P:
        q = p - p.mean(0)
        if len(q) >= 2:
            cov = np.cov(q.T) + 1e-9 * np.eye(2)          # 2x2 -> always 2D basis
            _, vecs = np.linalg.eigh(cov)
            q = q @ vecs                                  # rotate to principal axes
        out.append(q)
    return out


def fgw_features(P, F, n_atoms=8, alpha=0.5, seed=0):
    """Registration-free: build a shared template from pooled features, transport each
    sample to it via fused-GW (intra-sample structure only -> warp-invariant), then
    barycentric-project. z = [mass_k, mean-feature_k] per template node."""
    g = RNG(seed + 7)
    # template: k-means-lite on pooled features (feature-space prototypes)
    pool = np.vstack(F)
    idx = g.choice(len(pool), size=n_atoms, replace=False)
    proto = pool[idx].copy()
    for _ in range(15):                                  # Lloyd iterations
        d = ((pool[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(n_atoms):
            if (lab == k).any():
                proto[k] = pool[lab == k].mean(0)
    C_ref = ot.dist(proto, proto); C_ref /= C_ref.max() + 1e-9
    q = np.full(n_atoms, 1.0 / n_atoms)

    out = []
    for p, f in zip(P, F):
        C_s = ot.dist(p, p); C_s /= C_s.max() + 1e-9     # intra-sample structure
        M = ot.dist(f, proto); M /= M.max() + 1e-9       # feature cost to template
        ps = np.full(len(p), 1.0 / len(p))
        T = ot.gromov.fused_gromov_wasserstein(
            M, C_s, C_ref, ps, q, loss_fun="square_loss", alpha=alpha, verbose=False)
        mass = T.sum(0)                                  # mass on each template node
        denom = mass[:, None] + 1e-12
        zfeat = (T.T @ f) / denom                        # barycentric mean feature
        out.append(np.concatenate([mass, zfeat.ravel()]))
    return np.array(out)


# ----------------------------- evaluation -----------------------------
def eval_rep(Xtr, ytr, Xte, yte, ncomp=5):
    ncomp = min(ncomp, Xtr.shape[1], Xtr.shape[0] - 1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    pls = PLSRegression(n_components=ncomp)
    pls.fit(Xtr, ytr)
    return r2_score(yte, pls.predict(Xte).ravel())


def run():
    warps = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [0, 1, 2, 3, 4]
    n = 160
    methods = ["flatten_naive", "rigid_reg", "meanpool", "ours_fgw"]
    results = {m: {w: [] for w in warps} for m in methods}

    for w in warps:
        for s in seeds:
            P, F, Y = gen_dataset(s, w, n)
            ntr = int(0.7 * n)
            tr, te = slice(0, ntr), slice(ntr, n)

            Xn = rasterize(P, F)
            results["flatten_naive"][w].append(
                eval_rep(Xn[tr], Y[tr], Xn[te], Y[te]))

            Pr = rigid_register(P)
            Xr = rasterize(Pr, F)
            results["rigid_reg"][w].append(
                eval_rep(Xr[tr], Y[tr], Xr[te], Y[te]))

            Xm = np.array([f.mean(0) for f in F])         # spatial-blind sanity
            results["meanpool"][w].append(
                eval_rep(Xm[tr], Y[tr], Xm[te], Y[te]))

            Xo = fgw_features(P, F, seed=s)
            results["ours_fgw"][w].append(
                eval_rep(Xo[tr], Y[tr], Xo[te], Y[te]))
        print(f"warp={w:.2f} done")

    print("\n=== R^2 (mean +/- std over seeds), higher is better ===")
    print(f"{'method':<16}" + "".join(f"  w={w:<5}" for w in warps))
    for m in methods:
        row = "".join(f"  {np.mean(results[m][w]):+.2f}" + " " * 4 for w in warps)
        print(f"{m:<16}{row}")

    print("\n=== GO/NO-GO check ===")
    ours = np.array([np.mean(results["ours_fgw"][w]) for w in warps])
    best_base = np.array([max(np.mean(results[m][w]) for m in methods if m != "ours_fgw")
                          for w in warps])
    gap = ours - best_base
    print("ours - best_baseline per warp:", np.round(gap, 3))
    widens = gap[-1] > gap[0] + 0.03
    ahead = np.all(gap > -0.02)
    print(f"ahead at all warps: {ahead}   |   advantage widens with warp: {widens}")
    print("VERDICT:", "GO" if (ahead and widens) else "NO-GO / rethink")


if __name__ == "__main__":
    run()
