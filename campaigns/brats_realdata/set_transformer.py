"""Set Transformer baseline (Lee et al., 2019) - a STRONGER permutation-invariant set encoder
than DeepSets: induced self-attention blocks (ISAB) + pooling-by-multihead-attention (PMA), on
the same registration-free per-case point cloud, through the SAME CV harness. Same signature and
return as train_rawpls / train_deepset so run_brats.py can call it via `--model settransformer`.
Memory-safe (per-minibatch device moves). Reuses deepset.pad_pf + _batched."""
import numpy as np
import torch
import torch.nn as nn
from core.raw_pls import pick_device, device_dtype
from deepset import pad_pf, _batched


class MAB(nn.Module):
    """Multihead Attention Block with key masking + row-wise feedforward (LayerNorm form)."""
    def __init__(self, dq, dk, d, nh=4):
        super().__init__()
        self.nh, self.d = nh, d
        self.fq, self.fk, self.fv, self.fo = (nn.Linear(dq, d), nn.Linear(dk, d),
                                              nn.Linear(dk, d), nn.Linear(d, d))
        self.ln0, self.ln1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, Q, K, kmask=None):
        q, k, v = self.fq(Q), self.fk(K), self.fv(K)
        B, nq, _ = q.shape; nk = k.shape[1]; h = self.nh; dh = self.d // h
        qh = q.view(B, nq, h, dh).transpose(1, 2)
        kh = k.view(B, nk, h, dh).transpose(1, 2)
        vh = v.view(B, nk, h, dh).transpose(1, 2)
        a = (qh @ kh.transpose(-1, -2)) / dh ** 0.5
        if kmask is not None:
            a = a.masked_fill(kmask[:, None, None, :] == 0, -1e9)
        a = a.softmax(-1)
        o = (a @ vh).transpose(1, 2).reshape(B, nq, self.d)
        H = self.ln0(q + self.fo(o))
        return self.ln1(H + self.ff(H))


class ISAB(nn.Module):
    def __init__(self, d, m=16, nh=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.1)
        self.mab0, self.mab1 = MAB(d, d, d, nh), MAB(d, d, d, nh)

    def forward(self, X, xmask=None):
        Hh = self.mab0(self.I.expand(X.shape[0], -1, -1), X, kmask=xmask)   # (B,m,d)
        return self.mab1(X, Hh)                                             # (B,n,d), keys unmasked


class PMA(nn.Module):
    def __init__(self, d, k=1, nh=4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d) * 0.1)
        self.mab = MAB(d, d, d, nh)

    def forward(self, X, xmask=None):
        return self.mab(self.S.expand(X.shape[0], -1, -1), X, kmask=xmask)  # (B,k,d)


class SetTransformer(nn.Module):
    def __init__(self, din, d=64, nh=4, dtype=torch.float32):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.enc0, self.enc1 = ISAB(d, nh=nh), ISAB(d, nh=nh)
        self.pma = PMA(d, 1, nh)
        self.out = nn.Linear(d, 1)
        self.to(dtype)

    def forward(self, X, mask):
        h = self.inp(X)
        h = self.enc0(h, xmask=mask)
        h = self.enc1(h, xmask=mask)
        z = self.pma(h, xmask=mask).squeeze(1)     # (B,d)
        return self.out(z).squeeze(-1)


def train_settransformer(P, F, Y, tr_idx, va_idx, Mmax=128, epochs=200, lr=1e-3, batch=32,
                         task="reg", device=None, seed=0, d=64, wd=1e-4, **_ignore):
    """Same signature/return as train_deepset: (model, metric, preds)."""
    dev = pick_device(device); dt = device_dtype(dev)
    torch.manual_seed(seed)
    Pb, Fb, mb = pad_pf(P, F, Mmax, seed=seed)
    Pb = Pb - (Pb * mb[:, :, None]).sum(1, keepdim=True) / mb.sum(1)[:, None, None].clamp_min(1)
    Xb = torch.cat([Pb, Fb], -1)                   # registration-free input (centered coords + feats)
    y = torch.tensor(np.asarray(Y, np.float64))
    tr, va = np.asarray(tr_idx), np.asarray(va_idx)
    model = SetTransformer(Xb.shape[-1], d=d, dtype=dt).to(dev)
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
