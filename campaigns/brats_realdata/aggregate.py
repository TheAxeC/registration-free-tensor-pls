"""Aggregate per-seed JSON results from the SLURM array into mean +/- 95% CI and a
paired RAW-PLS vs grid-baseline test. Usage: python aggregate.py results/"""
import sys
import glob
import json
import numpy as np


def ci95(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main(folder):
    files = sorted(glob.glob(f"{folder.rstrip('/')}/seed_*.json"))
    if not files:
        print("no result files in", folder); return
    ours, base, seeds = [], [], []
    for f in files:
        d = json.load(open(f))
        ours += d["ours"]; base += d["base"]; seeds.append(d.get("seed"))
    ours, base = np.array(ours), np.array(base)
    metric = "AUC" if json.load(open(files[0]))["metric"] == "clf" else "R2"
    print(f"{len(files)} seeds {seeds} | {len(ours)} (seed x fold) evals | metric={metric}\n")
    mo, eo = ci95(ours); mb, eb = ci95(base)
    print(f"  RAW-PLS    : {mo:.3f} +/- {eo:.3f}")
    print(f"  grid-PLS   : {mb:.3f} +/- {eb:.3f}")
    diff = ours - base
    md, ed = ci95(diff)
    print(f"  difference : {md:+.3f} +/- {ed:.3f}  (paired, per fold)")
    try:
        from scipy.stats import wilcoxon
        if np.any(diff != 0):
            stat, p = wilcoxon(ours, base)
            print(f"  Wilcoxon signed-rank p = {p:.4g}")
    except Exception as e:
        print("  (wilcoxon skipped:", e, ")")
    verdict = (md > 0) and (md - ed > 0)
    print(f"\n  PILOT: {'RAW-PLS beats baseline (CI excludes 0)' if verdict else 'inconclusive — iterate method, do NOT write up a null'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
