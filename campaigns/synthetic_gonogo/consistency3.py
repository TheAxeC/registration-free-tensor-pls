"""Track C, consistency (v3 = the DIRECT Thm-2 quantity). Proxies (test risk, generalization
gap) are contaminated by the approximation floor / interpolation. The theorem is about PARAMETER
recovery: the learned atom features should converge to the ground-truth predictive-atom signatures
at the sqrt(N) rate. We measure min-permutation feature-recovery error vs N and fit the log-log
slope (predicted ~ -0.5 for an L2 error ~ N^{-1/2}). This is exactly what Lemma 5 / Prop 5 (track E)
guarantee for the prediction-relevant atoms. Honest: report whatever slope it shows.
Run (local): PYTHONPATH=. python campaigns/synthetic_gonogo/consistency3.py
"""
import os, json
import numpy as np
import torch
from joint_supervised import gen_dataset, make_world, pad_batch, JointModel, N_PRED

RNG = np.random.default_rng


def train_atoms(P, F, Y, tr, K=8, seed=0, epochs=250, Mmax=50):
    Fb, Cb, ab = pad_batch(P, F, Mmax); y = torch.tensor(np.asarray(Y, float))
    g = RNG(seed); pool = np.vstack([F[i] for i in tr])
    cent = pool[g.choice(len(pool), K, replace=False)].copy()
    for _ in range(15):
        lab = ((pool[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = pool[lab == k].mean(0)
    model = JointModel(K, seed, cent); opt = torch.optim.Adam(model.parameters(), lr=0.02)
    trt = torch.tensor(tr); ymu, ysd = y[trt].mean(), y[trt].std()
    for _ in range(epochs):
        opt.zero_grad(); pred, _ = model(Fb, Cb, ab)
        loss = ((pred[trt] - (y[trt] - ymu) / ysd) ** 2).mean() + 1e-3 * (model.G ** 2).mean()
        loss.backward(); opt.step()
    return model.G.detach().numpy()                          # (K, C) learned atom features


def recovery_err(Ghat, sig_true):
    """Mean over true predictive atoms of the distance to the nearest learned atom (feature space)."""
    return float(np.mean([np.linalg.norm(Ghat - gt[None], axis=1).min() for gt in sig_true]))


def main():
    Ns = [60, 120, 240, 480, 960, 1920]; seeds = range(6); warp = 0.5
    rows = {}
    for N in Ns:
        es = []
        for s in seeds:
            P, F, Y = gen_dataset(s, warp, N)
            sig, _ = make_world(s); sig_true = sig[:N_PRED]   # predictive-atom signatures (recoverable)
            Gh = train_atoms(P, F, Y, np.arange(N), seed=s)
            es.append(recovery_err(Gh, sig_true))
        rows[N] = (float(np.mean(es)), float(np.std(es)))
        print(f"N={N:>5}  atom-recovery err {rows[N][0]:.4f} +/- {rows[N][1]:.4f}", flush=True)
    Na = np.array(Ns, float); err = np.clip(np.array([rows[n][0] for n in Ns]), 1e-4, None)
    slope = float(np.polyfit(np.log(Na), np.log(err), 1)[0])
    print(f"\nlog-log slope of atom-recovery error vs N: {slope:+.2f} (predicted ~ -0.5 for sqrt(N) rate)", flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "synthetic_gonogo")
    json.dump(dict(rows={int(n): rows[n] for n in rows}, recovery_loglog_slope=slope),
              open(os.path.join(od, "consistency_recovery.json"), "w"), indent=2)
    print("wrote consistency_recovery.json", flush=True)


if __name__ == "__main__":
    main()
