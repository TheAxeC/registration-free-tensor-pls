"""
RAW-PLS: registration-free supervised tensor-PLS via a differentiable entropic
fused-Gromov-Wasserstein layer. Clean, device-aware, configurable-channel,
minibatch-capable reuse module (CPU / MPS / CUDA).

Estimand matches theory.tex: a FIXED `outer`-step unrolled entropic FGW transport
(the algorithmic estimand) -> barycentric projection -> linear/low-rank head.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_dtype(dev: torch.device) -> torch.dtype:
    # float32 on accelerators (memory; MPS has no float64). float64 only on CPU
    # (synthetic-scale precision). Entropic transport is stable in f32 at eps>=~0.02.
    return torch.float64 if dev.type == "cpu" else torch.float32


def pad_clouds(P, F, Mmax, seed=0):
    """List of (m_i, d) coords + (m_i, C) features -> padded tensors with marginals.
    Subsamples each cloud to <= Mmax points; padded rows get zero marginal mass."""
    B = len(P); C = F[0].shape[1]
    Fb = np.zeros((B, Mmax, C), np.float64)
    Cb = np.zeros((B, Mmax, Mmax), np.float64)
    ab = np.zeros((B, Mmax), np.float64)
    g = np.random.default_rng(seed)
    for i, (p, f) in enumerate(zip(P, F)):
        m = len(p)
        idx = np.arange(m) if m <= Mmax else g.choice(m, Mmax, replace=False)
        pp, ff = p[idx], f[idx]
        mm = len(idx)
        Fb[i, :mm] = ff
        d = np.linalg.norm(pp[:, None] - pp[None], axis=-1)
        d /= (d.max() + 1e-9)
        Cb[i, :mm, :mm] = d
        ab[i, :mm] = 1.0 / mm
    return (torch.tensor(Fb), torch.tensor(Cb), torch.tensor(ab))


def fgw_unrolled(Fb, Cb, ab, G, Cref, b, log_metric, alpha=0.5, eps=0.05,
                 outer=5, sink=25, dist=False):
    """Batched differentiable T-step unrolled entropic fused-GW -> barycentric z.
    All ops are smooth compositions (Sinkhorn layers) -> autograd-friendly and,
    per theory.tex Lemma 2', Lipschitz in (G, Cref, b, metric)."""
    dev = Fb.device
    B, Mmax, C = Fb.shape
    K = G.shape[0]
    w = torch.exp(log_metric)                                   # (C,) diagonal metric
    Fm = Fb * torch.sqrt(w)[None, None]
    Gm = G * torch.sqrt(w)[None]
    M = ((Fm[:, :, None, :] - Gm[None, None]) ** 2).sum(-1)
    M = M / (M.amax(dim=(1, 2), keepdim=True) + 1e-9)
    Crefn = Cref / (Cref.max() + 1e-9)
    bb = b[None].expand(B, K)
    Ca = (Cb ** 2) @ ab[:, :, None]
    Cbt = (Crefn ** 2) @ b
    T = ab[:, :, None] * bb[:, None, :]
    for _ in range(outer):
        cross = Cb @ T @ Crefn
        cost = (1 - alpha) * M + alpha * (Ca + Cbt[None, None] - 2 * cross)
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        Km = torch.exp(-cost / eps)
        u = torch.ones(B, Mmax, device=dev, dtype=Fb.dtype)
        v = torch.ones(B, K, device=dev, dtype=Fb.dtype)
        for _ in range(sink):
            u = ab / (torch.bmm(Km, v[:, :, None]).squeeze(-1) + 1e-30)
            v = bb / (torch.bmm(Km.transpose(1, 2), u[:, :, None]).squeeze(-1) + 1e-30)
        T = u[:, :, None] * Km * v[:, None, :]
    mass = T.sum(1)                                            # (B,K)
    zfeat = torch.bmm(T.transpose(1, 2), Fb) / (mass[:, :, None] + 1e-9)
    if dist:
        # distributional barycentric readout: transport-weighted per-atom soft-MIN and soft-MAX of
        # each feature channel, alongside the mean. Captures the within-atom intensity extremes/tails
        # (e.g. the darkest ADC focus that signals csPCa) that the mean washes out. Both are smooth
        # (stable weighted log-sum-exp), so the representation stays Lipschitz.
        Tn = T / (mass[:, None, :] + 1e-9)                 # (B,Mmax,K) per-atom weights over points
        tau = 0.5
        a = -Fb / tau; am = a.amax(dim=1, keepdim=True)    # soft-min
        wmin = torch.einsum("bik,bic->bkc", Tn, torch.exp(a - am))
        smin = -tau * (am.squeeze(1)[:, None, :] + torch.log(wmin + 1e-9))
        a2 = Fb / tau; am2 = a2.amax(dim=1, keepdim=True)  # soft-max
        wmax = torch.einsum("bik,bic->bkc", Tn, torch.exp(a2 - am2))
        smax = tau * (am2.squeeze(1)[:, None, :] + torch.log(wmax + 1e-9))
        return torch.cat([mass, zfeat.reshape(B, -1), smin.reshape(B, -1), smax.reshape(B, -1)], dim=1), T
    return torch.cat([mass, zfeat.reshape(B, -1)], dim=1), T


class RawPLS(nn.Module):
    def __init__(self, K, C, init_G=None, head_rank=0, n_out=1, geom_rank=0,
                 alpha=0.5, eps=0.05, outer=5, sink=25, seed=0, dtype=torch.float64,
                 dist_readout=False):
        super().__init__()
        self.K, self.C, self.geom_rank = K, C, geom_rank
        self.alpha, self.eps, self.outer, self.sink = alpha, eps, outer, sink
        self.dist_readout = dist_readout
        g = torch.Generator().manual_seed(seed)
        G0 = torch.tensor(init_G) if init_G is not None else torch.randn(K, C, generator=g)
        self.G = nn.Parameter(G0.to(dtype))
        self.coords = nn.Parameter((torch.randn(K, 3, generator=g) * 0.3).to(dtype))
        self.b_logit = nn.Parameter(torch.zeros(K, dtype=dtype))
        self.log_metric = nn.Parameter(torch.zeros(C, dtype=dtype))
        # #2: low-rank readout of the transport-weighted inter-atom geometry (T^T C T).
        # r rank-1 bilinear probes a_j^T G b_j -> r scalars: geometry enters through a
        # learned, regularized pathway (r*2K params) instead of K^2 raw features (which overfit).
        din = K * (1 + (3 if dist_readout else 1) * C) + geom_rank
        if geom_rank > 0:
            self.geomA = nn.Parameter((torch.randn(geom_rank, K, generator=g) * 0.1).to(dtype))
            self.geomB = nn.Parameter((torch.randn(geom_rank, K, generator=g) * 0.1).to(dtype))
        if head_rank > 0:                                      # low-rank multilinear head
            self.head = nn.Sequential(nn.Linear(din, head_rank), nn.Linear(head_rank, n_out))
        else:
            self.head = nn.Linear(din, n_out)
        self.head.to(dtype)
        self.register_buffer("zmu", torch.zeros(din, dtype=dtype))
        self.register_buffer("zsd", torch.ones(din, dtype=dtype))

    def represent(self, Fb, Cb, ab):
        diff = self.coords[:, None, :] - self.coords[None, :, :]   # portable pdist
        Cref = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
        b = torch.softmax(self.b_logit, 0)
        z, T = fgw_unrolled(Fb, Cb, ab, self.G, Cref, b, self.log_metric,
                            self.alpha, self.eps, self.outer, self.sink, dist=self.dist_readout)
        if self.geom_rank > 0:
            mass = T.sum(1)                                   # (B,K)
            Gw = torch.bmm(T.transpose(1, 2), torch.bmm(Cb, T))   # (B,K,K) inter-atom geom
            Gw = Gw / (mass[:, :, None] * mass[:, None, :] + 1e-9)
            GA = torch.einsum("rk,bkl->brl", self.geomA, Gw)
            gfeat = torch.einsum("brl,rl->br", GA, self.geomB)    # (B, geom_rank)
            z = torch.cat([z, gfeat], dim=1)
        return z, T

    def forward(self, Fb, Cb, ab):
        z, T = self.represent(Fb, Cb, ab)
        zs = (z - self.zmu) / (self.zsd + 1e-6)
        return self.head(zs).squeeze(-1), z, T


def kmeans_init(F_list, K, seed=0, iters=15):
    pool = np.vstack(F_list)
    g = np.random.default_rng(seed)
    cent = pool[g.choice(len(pool), K, replace=False)].copy()
    for _ in range(iters):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    return cent


def _batched(model, Fb, Cb, ab, idx, dev, dt, want, batch=64):
    """Run represent ('z') or full forward ('pred') over idx in minibatches, moving
    each batch to device on demand. Returns a CPU tensor. Bounds memory to one batch."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            fb, cb, a = Fb[bi].to(dev, dt), Cb[bi].to(dev, dt), ab[bi].to(dev, dt)
            if want == "z":
                z, _ = model.represent(fb, cb, a); outs.append(z.cpu())
            else:
                p, _, _ = model(fb, cb, a); outs.append(p.cpu())
    return torch.cat(outs)


def train_rawpls(P, F, Y, tr_idx, va_idx, K=12, Mmax=128, epochs=300, lr=0.005,
                 batch=32, task="reg", device=None, seed=0, head_rank=0, wd=1e-4,
                 geom_rank=0, lambda_cov=0.0,
                 alpha=0.5, eps=0.05, outer=5, sink=25, verbose=True, dist_readout=False):
    """Train RAW-PLS. task in {'reg','clf'}. Memory-safe: data stays on CPU, each
    minibatch is moved to the device on demand (scales to thousands of cases)."""
    dev = pick_device(device)
    dt = device_dtype(dev)
    C = F[0].shape[1]
    Fb, Cb, ab = pad_clouds(P, F, Mmax, seed=seed)            # CPU tensors, stay on CPU
    y = torch.tensor(np.asarray(Y, np.float64))
    init = kmeans_init([F[i] for i in tr_idx], K, seed=seed)
    model = RawPLS(K, C, init_G=init, head_rank=head_rank, geom_rank=geom_rank, alpha=alpha,
                   eps=eps, outer=outer, sink=sink, seed=seed, dtype=dt,
                   dist_readout=dist_readout).to(dev)
    tr, va = np.asarray(tr_idx), np.asarray(va_idx)
    # z-normalization from a batched pass over train
    z0 = _batched(model, Fb, Cb, ab, tr, dev, dt, want="z")
    model.zmu.copy_(z0.mean(0).to(dev, dt)); model.zsd.copy_(z0.std(0).to(dev, dt))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ymu, ysd = y[tr].mean().item(), max(y[tr].std().item(), 1e-6)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); rng.shuffle(tr)
        for s in range(0, len(tr), batch):
            bi = tr[s:s + batch]
            fb, cb, a = Fb[bi].to(dev, dt), Cb[bi].to(dev, dt), ab[bi].to(dev, dt)
            yb = y[bi].to(dev, dt)
            opt.zero_grad()
            pred, z, _ = model(fb, cb, a)
            loss = (((pred - (yb - ymu) / ysd) ** 2).mean() if task == "reg"
                    else bce(pred, yb)) + 1e-3 * (model.G ** 2).mean()
            if lambda_cov > 0:                                 # #1: PLS covariance objective
                zc = (z - z.mean(0)) / (z.std(0) + 1e-6)        # shape template/transport to
                yc = (yb - yb.mean()) / (yb.std() + 1e-6)       # be predictive, not just the head
                loss = loss - lambda_cov * ((zc * yc[:, None]).mean(0) ** 2).sum()
            loss.backward(); opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}")
    model.eval()
    pred = _batched(model, Fb, Cb, ab, va, dev, dt, want="pred").numpy()
    yv = np.asarray(Y, np.float64)[va]
    if task == "reg":
        pred = pred * ysd + ymu
        metric = float(1 - ((yv - pred) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-12))
    else:
        from sklearn.metrics import roc_auc_score
        metric = float(roc_auc_score(yv, 1 / (1 + np.exp(-pred))))
    return model, metric, pred
