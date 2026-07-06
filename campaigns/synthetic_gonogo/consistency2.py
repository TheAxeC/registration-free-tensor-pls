"""Track C, consistency (clean v2): the GENERALIZATION GAP (test MSE - train MSE) is pure
estimation error with no approximation-error floor, so it should decay ~ O(1/N) (Thm 2:
parameter error ~ N^{-1/2} => squared-error gap ~ N^{-1}), even when the task saturates. We
measure the gap vs N (more seeds, wider N range) and fit the log-log slope.
Run (local): PYTHONPATH=. python campaigns/synthetic_gonogo/consistency2.py
"""
import os, json
import numpy as np
import torch
from joint_supervised import gen_dataset, pad_batch, JointModel

RNG = np.random.default_rng


def train_gap(P, F, Y, tr, te, K=8, seed=0, epochs=250, Mmax=50):
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
    model.eval()
    with torch.no_grad():
        pred, _ = model(Fb, Cb, ab); pr = (pred * ysd + ymu).numpy()
    Y = np.asarray(Y, float)
    return float(((Y[tr] - pr[tr]) ** 2).mean()), float(((Y[te] - pr[te]) ** 2).mean())


def main():
    Ns = [60, 120, 240, 480, 960]; seeds = range(6); warp = 0.5; n_test = 400
    P_te, F_te, Y_te = gen_dataset(777, warp, n_test)
    rows = {}
    for N in Ns:
        gaps, tes, trs = [], [], []
        for s in seeds:
            P_tr, F_tr, Y_tr = gen_dataset(s, warp, N)
            P = P_tr + P_te; F = F_tr + F_te; Y = np.concatenate([Y_tr, Y_te])
            tr_mse, te_mse = train_gap(P, F, Y, np.arange(N), np.arange(N, N + n_test), seed=s)
            gaps.append(te_mse - tr_mse); tes.append(te_mse); trs.append(tr_mse)
        rows[N] = dict(gap=float(np.mean(gaps)), gap_sd=float(np.std(gaps)),
                       test=float(np.mean(tes)), train=float(np.mean(trs)))
        print(f"N={N:>4}  gap(test-train) {rows[N]['gap']:+.4f}+/-{rows[N]['gap_sd']:.4f}  "
              f"(test {rows[N]['test']:.3f}, train {rows[N]['train']:.3f})", flush=True)
    Na = np.array(Ns, float); gap = np.clip(np.array([rows[n]["gap"] for n in Ns]), 1e-4, None)
    slope = float(np.polyfit(np.log(Na), np.log(gap), 1)[0])
    print(f"\nlog-log slope of generalization gap vs N: {slope:+.2f} (predicted ~ -1 for squared-error gap)", flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "synthetic_gonogo")
    json.dump(dict(rows={int(n): rows[n] for n in rows}, gap_loglog_slope=slope),
              open(os.path.join(od, "consistency_gap.json"), "w"), indent=2)
    print("wrote consistency_gap.json", flush=True)


if __name__ == "__main__":
    main()
