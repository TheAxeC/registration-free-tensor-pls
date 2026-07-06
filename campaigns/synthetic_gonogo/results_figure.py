"""Money plot for the confirmed GO: R^2 and classification AUC vs warp level
(standard regime), registration-free method vs alignment-assuming baselines."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hardened_gonogo import run

r2, auc = run("standard")
warps = [0.0, 0.5, 1.0]
order = ["flatten_naive", "hopls_grid", "rigid_reg", "ours_unsup", "ours_sup"]
nice = {"flatten_naive": "flatten + PLS", "hopls_grid": "HOPLS / N-PLS (grid)",
        "rigid_reg": "rigid-register + PLS", "ours_unsup": "ours (unsup. template)",
        "ours_sup": "ours (sup. template)"}
col = {"flatten_naive": "#888", "hopls_grid": "#d62728", "rigid_reg": "#ff7f0e",
       "ours_unsup": "#1f77b4", "ours_sup": "#2ca02c"}
mk = {"flatten_naive": "o", "hopls_grid": "s", "rigid_reg": "^",
      "ours_unsup": "D", "ours_sup": "*"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for m in order:
    y = [np.mean(r2[m][w]) for w in warps]
    e = [np.std(r2[m][w]) for w in warps]
    lw = 2.6 if m.startswith("ours") else 1.6
    a1.errorbar(warps, y, yerr=e, marker=mk[m], color=col[m], lw=lw, ms=9 if m == "ours_sup" else 7,
                capsize=3, label=nice[m])
a1.axhline(0, color="k", lw=0.7, ls=":")
a1.set_xlabel("warp / misalignment level"); a1.set_ylabel(r"test $R^2$")
a1.set_title("Regression: registration-free stays flat,\nalignment-assuming methods collapse")
a1.set_ylim(-1.0, 0.85); a1.legend(fontsize=8.5, loc="lower left"); a1.grid(alpha=0.25)

for m in order:
    vals = [np.nanmean(auc[m][w]) for w in warps]
    if all(np.isnan(vals)):
        continue
    lw = 2.6 if m.startswith("ours") else 1.6
    a2.plot(warps, vals, marker=mk[m], color=col[m], lw=lw, ms=9 if m == "ours_sup" else 7,
            label=nice[m])
a2.axhline(0.5, color="k", lw=0.7, ls=":")
a2.set_xlabel("warp / misalignment level"); a2.set_ylabel("classification AUC")
a2.set_title("Classification: same story\n(AUC, median split)")
a2.set_ylim(0.45, 0.97); a2.legend(fontsize=8.5, loc="lower left"); a2.grid(alpha=0.25)

fig.suptitle("Synthetic GO/NO-GO (standard regime): the alignment assumption is the failure mode",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results_figure.png", dpi=160, facecolor="white")
print("wrote results_figure.png")
