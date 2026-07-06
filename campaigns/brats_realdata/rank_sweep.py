"""Rank sweep configs by mean CV val-R^2 (ours) and margin over the grid baseline.
Usage: python rank_sweep.py results_sweep/"""
import sys, glob, json
import numpy as np

rows = []
for f in sorted(glob.glob(f"{sys.argv[1].rstrip('/')}/cfg_*.json")):
    d = json.load(open(f))
    o, b = np.array(d["ours"]), np.array(d["base"])
    rows.append((d.get("cfg", {}), o.mean(), o.std(), b.mean(), o.mean() - b.mean(), f))
rows.sort(key=lambda r: r[1], reverse=True)
print(f"{'rank':<5}{'ours R2':<18}{'grid R2':<10}{'margin':<9}config")
for i, (cfg, om, osd, bm, marg, f) in enumerate(rows):
    c = f"K{cfg.get('K')} eps{cfg.get('eps')} a{cfg.get('alpha')} wd{cfg.get('wd')} lr{cfg.get('lr')} ep{cfg.get('epochs')} M{cfg.get('Mmax')}"
    print(f"{i:<5}{om:+.3f}+/-{osd:.3f}     {bm:+.3f}    {marg:+.3f}   {c}")
if rows:
    best = rows[0]
    print(f"\nBEST: ours R2={best[1]:+.3f} (grid {best[3]:+.3f}, margin {best[4]:+.3f})")
    print("cfg:", best[0])
