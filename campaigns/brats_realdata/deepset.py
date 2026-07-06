"""DeepSets baseline - a permutation-invariant DEEP encoder on the same unaligned point
cloud RAW-PLS uses (per-point MLP -> masked mean+max pool -> MLP head). The fair "why not a
flexible black-box set encoder?" comparison: registration-free, more params than RAW-PLS,
run through the SAME CV harness (run_brats --model deepset). Memory-safe (per-minibatch).
"""
import numpy as np
import torch
import torch.nn as nn
from core.raw_pls import pick_device, device_dtype


def pad_pf(P, F, Mmax, seed=0):
    """List of clouds -> padded coords (B,Mmax,3), feats (B,Mmax,C), mask (B,Mmax)."""
    B = len(P); C = F[0].shape[1]; d = P[0].shape[1]
    Pb = np.zeros((B, Mmax, d)); Fb = np.zeros((B, Mmax, C)); m = np.zeros((B, Mmax))
    g = np.random.default_rng(seed)
    for i, (p, f) in enumerate(zip(P, F)):
        n = len(p); idx = np.arange(n) if n <= Mmax else g.choice(n, Mmax, replace=False)
        mm = len(idx); Pb[i, :mm] = p[idx]; Fb[i, :mm] = f[idx]; m[i, :mm] = 1.0
    return torch.tensor(Pb), torch.tensor(Fb), torch.tensor(m)


class DeepSet(nn.Module):
    def __init__(self, din, h=64, n_out=1, dtype=torch.float32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, n_out))
        self.to(dtype)

    def forward(self, X, mask):                              # X (B,M,din), mask (B,M)
        e = self.phi(X) * mask[:, :, None]                  # (B,M,h)
        s = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean = e.sum(1) / s                                 # masked mean
        mx = e.masked_fill(mask[:, :, None] == 0, -1e9).max(1).values  # masked max
        return self.rho(torch.cat([mean, mx], -1)).squeeze(-1)


def _batched(model, Xb, mb, idx, dev, dt, batch=64):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            out.append(model(Xb[bi].to(dev, dt), mb[bi].to(dev, dt)).cpu())
    return torch.cat(out).numpy()


def train_deepset(P, F, Y, tr_idx, va_idx, Mmax=128, epochs=200, lr=1e-3, batch=32,
                  task="reg", device=None, seed=0, h=64, wd=1e-4, **_ignore):
    """Same signature/return as train_rawpls: (model, metric, preds)."""
    dev = pick_device(device); dt = device_dtype(dev)
    torch.manual_seed(seed)
    Pb, Fb, mb = pad_pf(P, F, Mmax, seed=seed)
    # registration-free input: center coords per cloud, then concat features
    Pb = Pb - (Pb * mb[:, :, None]).sum(1, keepdim=True) / mb.sum(1)[:, None, None].clamp_min(1)
    Xb = torch.cat([Pb, Fb], -1)                            # (B,M,3+C)
    y = torch.tensor(np.asarray(Y, np.float64))
    tr, va = np.asarray(tr_idx), np.asarray(va_idx)
    model = DeepSet(Xb.shape[-1], h=h, dtype=dt).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ymu, ysd = y[tr].mean().item(), max(y[tr].std().item(), 1e-6)
    bce = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); rng.shuffle(tr)
        for s in range(0, len(tr), batch):
            bi = tr[s:s + batch]
            xb, msk, yb = Xb[bi].to(dev, dt), mb[bi].to(dev, dt), y[bi].to(dev, dt)
            opt.zero_grad(); pred = model(xb, msk)
            loss = (((pred - (yb - ymu) / ysd) ** 2).mean() if task == "reg" else bce(pred, yb))
            loss.backward(); opt.step()
    model.eval()
    pred = _batched(model, Xb, mb, va, dev, dt)
    yv = np.asarray(Y, np.float64)[va]
    if task == "reg":
        pred = pred * ysd + ymu
        metric = float(1 - ((yv - pred) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-12))
    else:
        from sklearn.metrics import roc_auc_score
        metric = float(roc_auc_score(yv, 1 / (1 + np.exp(-pred))))
    return model, metric, pred
