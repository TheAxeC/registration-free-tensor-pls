"""Upgrade track B (Evidence): show the necessity ablations are NOT an artifact of a single
synthetic generator. We re-run structure-necessity (joint-supervised vs bag-of-features) and
supervision-necessity (joint-supervised vs unsupervised template) across a GRID of regimes,
varying one axis at a time around a base regime: warp level, nuisance-atom count, nuisance mass.
Reports both margins (mean +/- std over seeds) per regime + a robustness verdict.

Reuses the model + baselines from joint_supervised.py (CH=6); only the WORLD generator is
parametrized here. Run (local, CPU), from code/:
  PYTHONPATH=. python campaigns/synthetic_gonogo/ablation_robustness.py
"""
import os, json
import numpy as np
import joint_supervised as js
from joint_supervised import train_joint, fgw_unsup, bag_of_features, eval_rep

RNG = np.random.default_rng
CH = 6


def make_world_p(seed, n_nuis):
    g = RNG(seed); n_pred = 2                              # A, B drive Y via their separation
    sig = np.zeros((n_pred + n_nuis, CH))
    sig[:n_pred, :3] = g.normal(scale=1.2, size=(n_pred, 3))    # predictive on ch 0-2
    sig[n_pred:, 3:] = g.normal(scale=2.5, size=(n_nuis, 3))    # nuisance on ch 3-5
    base = g.uniform(0.2, 0.8, size=(n_pred + n_nuis, 2))
    return sig, base, n_pred


def gen_sample_p(g, sig, base, n_pred, n_nuis, warp, nuis_mass):
    centers = base.copy()
    centers[:n_pred] += g.uniform(-0.25, 0.25, size=(n_pred, 2))
    pts, feats, owner = [], [], []
    for k in range(n_pred + n_nuis):
        nk = 16 if k < n_pred else nuis_mass
        pts.append(centers[k] + g.normal(scale=0.03, size=(nk, 2)))
        feats.append(sig[k] + g.normal(scale=0.25, size=(nk, CH)))
        owner += [k] * nk
    P = np.vstack(pts); F = np.vstack(feats); owner = np.array(owner)
    cA = P[owner == 0].mean(0); cB = P[owner == 1].mean(0)
    Y = float(2.0 * np.linalg.norm(cA - cB) + 0.5 * ((owner == 0).sum() / 16.0) + g.normal(scale=0.03))
    th = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = P @ R.T + g.uniform(-warp, warp, size=2)
    if warp > 0:
        for anc in g.uniform(P.min(), P.max(), size=(2, 2)):
            d2 = ((P - anc) ** 2).sum(1)
            P = P + np.exp(-d2 / 0.08)[:, None] * g.normal(scale=0.08 * warp, size=2)
    keep = g.uniform(size=len(P)) < 0.8
    return P[keep], F[keep], Y


def gen_dataset_p(seed, warp, n_nuis, nuis_mass, n):
    g = RNG(seed + 1000); sig, base, n_pred = make_world_p(seed, n_nuis)
    d = [gen_sample_p(g, sig, base, n_pred, n_nuis, warp, nuis_mass) for _ in range(n)]
    return [x[0] for x in d], [x[1] for x in d], np.array([x[2] for x in d])


def run_regime(warp, n_nuis, nuis_mass, seeds=(0, 1, 2), n=180):
    struct_margin, sup_margin, joint_abs = [], [], []
    for s in seeds:
        P, F, Y = gen_dataset_p(s, warp, n_nuis, nuis_mass, n)
        js.Y_GLOBAL = Y                                   # fgw_unsup reads this module global
        ntr = int(0.7 * n); tr = np.arange(ntr); te = np.arange(ntr, n)
        joint = train_joint(P, F, Y, tr, te, seed=s)
        unsup = fgw_unsup(P, F, tr, te, seed=s)
        Xb = bag_of_features(P, F, tr, seed=s)
        bag = eval_rep(Xb[tr], Y[tr], Xb[te], Y[te])
        joint_abs.append(joint); struct_margin.append(joint - bag); sup_margin.append(joint - unsup)
    return (np.mean(joint_abs), np.std(joint_abs),
            np.mean(struct_margin), np.std(struct_margin),
            np.mean(sup_margin), np.std(sup_margin))


def main():
    base = dict(warp=0.6, n_nuis=3, nuis_mass=45)
    regimes = [("base", base)]
    for w in (0.3, 1.0):
        regimes.append((f"warp={w}", {**base, "warp": w}))
    for k in (2, 5):
        regimes.append((f"n_nuis={k}", {**base, "n_nuis": k}))
    for m in (30, 60):
        regimes.append((f"nuis_mass={m}", {**base, "nuis_mass": m}))

    print(f"{'regime':<14}{'joint R2':>16}{'struct-nec margin':>20}{'sup-nec margin':>18}", flush=True)
    out = {}
    for name, cfg in regimes:
        ja, js_, sm, ss, pm, ps = run_regime(**cfg)
        out[name] = dict(cfg=cfg, joint=ja, joint_sd=js_, struct_margin=sm, struct_sd=ss,
                         sup_margin=pm, sup_sd=ps)
        print(f"{name:<14}{ja:+.3f}+/-{js_:.3f}    {sm:+.3f}+/-{ss:.3f}      {pm:+.3f}+/-{ps:.3f}", flush=True)

    sm_all = [v["struct_margin"] for v in out.values()]
    pm_all = [v["sup_margin"] for v in out.values()]
    verdict = all(m > 0 for m in sm_all) and all(m > 0 for m in pm_all)
    print(f"\nROBUSTNESS VERDICT: structure-necessity positive in ALL {len(sm_all)} regimes: "
          f"{all(m>0 for m in sm_all)} (min {min(sm_all):+.3f}); "
          f"supervision-necessity positive in ALL: {all(m>0 for m in pm_all)} (min {min(pm_all):+.3f})", flush=True)
    print("=> NECESSITY ABLATIONS ROBUST" if verdict else "=> NOT robust in some regime (investigate)", flush=True)

    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "synthetic_gonogo")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "ablation_robustness.json"), "w"), indent=2)
    print("wrote ablation_robustness.json", flush=True)


if __name__ == "__main__":
    main()
