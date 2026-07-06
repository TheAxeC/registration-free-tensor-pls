"""Generate REAL per-subject saliency for the interpretability figure on the STRUCTURAL target
(ET-NCR distance) -- the geometric target where saliency should LOCALIZE to lesion sub-structure.
Uses a proper gradient saliency |dY_hat/dF_voxel| (full nonlinear path through the FGW representation
+ geometry readout + head), which is far sharper than the flat back-projection. Also a per-modality
importance bar (mean |gradient| per channel). Saves interp_data.npz; plot with interp_figure.py.
"""
import numpy as np, torch
from data_brats import load_npz
from core.raw_pls import train_rawpls, device_dtype, pick_device
from core import paths

MODS = ["T1", "T1ce", "T2", "FLAIR"]
P, F, Y, ids = load_npz(paths.data_path("BraTS2021/brats_struct.npz"))
n = len(P); tr = np.arange(n)
# full method on the structural target (regression)
model, _, _ = train_rawpls(P, F, Y, tr, tr, K=16, Mmax=128, epochs=200, lr=0.003, wd=1e-3,
                           alpha=0.6, eps=0.03, geom_rank=6, lambda_cov=0.1, task="reg",
                           seed=0, verbose=False)
model.eval(); dev = pick_device(None); dt = device_dtype(dev)
C = model.C

def grad_saliency(p, f, Mmax=128, seed=0):
    """Per-voxel gradient saliency |dY_hat/dF_i| and per-channel mean |grad|."""
    m = len(p); g = np.random.default_rng(seed)
    idx = np.arange(m) if m <= Mmax else g.choice(m, Mmax, replace=False)
    pp, ff = p[idx], f[idx]
    d = np.linalg.norm(pp[:, None] - pp[None], axis=-1); d /= d.max() + 1e-9
    Fb = torch.tensor(ff, dtype=dt, device=dev)[None].requires_grad_(True)
    Cb = torch.tensor(d, dtype=dt, device=dev)[None]
    ab = torch.full((1, len(pp)), 1.0 / len(pp), dtype=dt, device=dev)
    pred, _, _ = model(Fb, Cb, ab)
    if Fb.grad is not None: Fb.grad = None
    pred.sum().backward()
    gr = Fb.grad[0]                                    # (m, C)
    vimp = gr.norm(dim=1).detach().cpu().numpy()       # (m,)  per-voxel sensitivity
    mod = gr.abs().mean(0).detach().cpu().numpy()      # (C,)  per-modality sensitivity
    return pp, vimp, mod, float(pred.detach().cpu())

# pick examples spanning the target range: 2 large ET-NCR distance, 2 small
Yv = np.asarray(Y, float); order = np.argsort(Yv)
chosen = list(order[-2:][::-1]) + list(order[:2])      # high, high, low, low

coords_l, vimp_l, yv_l, pred_l, mods_acc = [], [], [], [], []
for i in chosen:
    pp, vimp, mod, pred = grad_saliency(P[i], F[i])
    coords_l.append(pp); vimp_l.append(vimp); yv_l.append(float(Yv[i])); pred_l.append(pred)
    mods_acc.append(mod)
    print(f"{ids[i]} y={Yv[i]:.3f} pred={pred:.3f} n={len(pp)} "
          f"vimp[min,max,cv]=[{vimp.min():.3g},{vimp.max():.3g},{vimp.std()/(vimp.mean()+1e-9):.2f}]", flush=True)

mod_imp = np.mean(mods_acc, 0)
np.savez_compressed("interp_data.npz",
                    coords=np.array(coords_l, dtype=object),
                    vimp=np.array(vimp_l, dtype=object),
                    yval=np.array(yv_l), pred=np.array(pred_l),
                    ids=np.array([ids[i] for i in chosen]),
                    mod_imp=mod_imp, mods=np.array(MODS))
print("INTERP_DONE -> interp_data.npz", flush=True)
