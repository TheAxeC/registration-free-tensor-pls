"""Local (no-HPC) ablation of the two method improvements on the synthetic
structure-predictive benchmark (Y = intra-sample distance between two predictive atoms
+ nuisance atoms) -- the analog of the BraTS ET-NCR target. Tests whether:
  #2 geom_rank (low-rank inter-atom geometry readout) and
  #1 lambda_cov (PLS covariance objective shaping template/transport)
lift val R^2 over the base method. Runs on CPU."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brats"))
from joint_supervised import gen_dataset           # structure-predictive + nuisance
from core.raw_pls import train_rawpls

P, F, Y = gen_dataset(seed=0, warp=0.5, n=320)
n = len(P); tr = np.arange(int(0.7 * n)); va = np.arange(int(0.7 * n), n)
print(f"{n} samples, feat-dim {F[0].shape[1]}, Y std {Y.std():.3f}\n")

configs = {
    "base":          dict(geom_rank=0, lambda_cov=0.0),
    "+geom (#2)":    dict(geom_rank=6, lambda_cov=0.0),
    "+cov  (#1)":    dict(geom_rank=0, lambda_cov=0.1),
    "+both":         dict(geom_rank=6, lambda_cov=0.1),
}
for name, cfg in configs.items():
    r2s = []
    for seed in range(3):                            # 3 seeds for a stable read
        _, m, _ = train_rawpls(P, F, Y, tr, va, K=12, Mmax=64, epochs=150, lr=0.005,
                               wd=1e-3, device="cpu", seed=seed, verbose=False, **cfg)
        r2s.append(m)
    print(f"{name:12s} val R2 = {np.mean(r2s):+.3f} +/- {np.std(r2s):.3f}")
