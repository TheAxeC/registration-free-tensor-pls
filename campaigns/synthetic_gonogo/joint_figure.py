"""Figure for the joint-supervised result (structure-predictive + nuisance regime).
Reads joint_results.pkl."""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("joint_results.pkl", "rb") as fh:
    d = pickle.load(fh)
r2, warps = d["r2"], d["warps"]

order = ["flatten", "hopls", "bag_feat", "ours_unsup", "ours_joint_sup"]
nice = {"flatten": "flatten + PLS", "hopls": "HOPLS / N-PLS (grid)",
        "bag_feat": "bag-of-features (no structure)",
        "ours_unsup": "ours (unsup. template)",
        "ours_joint_sup": "ours (JOINT supervised)"}
col = {"flatten": "#999", "hopls": "#d62728", "bag_feat": "#9467bd",
       "ours_unsup": "#1f77b4", "ours_joint_sup": "#2ca02c"}
mk = {"flatten": "o", "hopls": "s", "bag_feat": "v", "ours_unsup": "D",
      "ours_joint_sup": "*"}

fig, ax = plt.subplots(figsize=(7.6, 5.4))
for m in order:
    y = [np.mean(r2[m][w]) for w in warps]
    e = [np.std(r2[m][w]) for w in warps]
    big = m == "ours_joint_sup"
    ax.errorbar(warps, y, yerr=e, marker=mk[m], color=col[m],
                lw=3.0 if big else 1.7, ms=15 if big else 7, capsize=3,
                zorder=5 if big else 2, label=nice[m])
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("warp / misalignment level", fontsize=11)
ax.set_ylabel(r"test $R^2$", fontsize=11)
ax.set_ylim(-1.6, 0.75)
ax.set_title("Joint-supervised registration-free tensor-PLS\n"
             "structure-predictive + nuisance regime: only the supervised method predicts",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(alpha=0.25)
ax.annotate("structure needed\n(beats bag-of-features)", xy=(0.5, -1.30), xytext=(0.25, -0.72),
            fontsize=8.5, color="#9467bd", ha="center",
            arrowprops=dict(arrowstyle="->", color="#9467bd"))
ax.annotate("supervision needed\n(beats unsup. template)", xy=(0.5, -0.23), xytext=(0.18, -0.30),
            fontsize=8.5, color="#1f77b4", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
fig.tight_layout()
fig.savefig("joint_figure.png", dpi=160, facecolor="white")
print("wrote joint_figure.png")
