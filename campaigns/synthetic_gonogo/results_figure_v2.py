"""Results figure (2x2, clean): (A) robustness to misalignment (synthetic), (B) structural
target R^2, (C) grade HGG/LGG AUC, (D) method ablation. Panels B/C now show 3 methods
(RAW-PLS vs DeepSets vs aligned grid-PLS). Measured values from this project's runs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTE = "#222222", "#666666"
OURS = "#2ca02c"; DSET = "#5b6cc4"; GRID = "#c0392b"; BASE = "#999999"; RIG = "#e0922a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "axes.linewidth": 1.0})

fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13.5, 9.2))
fig.patch.set_facecolor("white")
fig.suptitle("Registration-Free Supervised Tensor-PLS — results", fontsize=18,
             fontweight="bold", color=INK, y=0.99)

# ---------- (A) robustness to misalignment (synthetic) ----------
warps = [0.0, 0.5, 1.0]
for name, c, mk, y in [("ours (registration-free)", OURS, "*", [0.70, 0.68, 0.67]),
                       ("HOPLS / N-PLS (aligned)", GRID, "s", [0.64, 0.06, -0.25]),
                       ("rigid-register + PLS", RIG, "^", [0.50, 0.51, 0.31]),
                       ("flatten + PLS", BASE, "o", [0.70, -0.51, -0.73])]:
    big = name.startswith("ours")
    axA.plot(warps, y, marker=mk, color=c, lw=3.0 if big else 1.8, ms=15 if big else 7,
             zorder=5 if big else 2, label=name)
axA.axhline(0, color="#444", lw=0.8, ls=":")
axA.set_xlabel("misalignment (warp level)"); axA.set_ylabel(r"test $R^2$")
axA.set_ylim(-0.95, 0.85); axA.set_xticks(warps)
axA.set_title("(A)  Robustness to misalignment  (synthetic)", fontsize=11.5, fontweight="bold")
axA.legend(fontsize=8.3, loc="lower left", framealpha=.95); axA.grid(alpha=.25)
axA.annotate("aligned methods\ncollapse", xy=(1.0, -0.40), xytext=(0.55, -0.75),
             fontsize=8.5, color=GRID, ha="center", arrowprops=dict(arrowstyle="->", color=GRID))

def bars3(ax, means, errs, ylim, ylab, title, fmt="{:+.3f}", baseline=None):
    labels = ["RAW-PLS\n(ours)", "DeepSets\n(deep)", "grid-PLS\n(aligned)"]
    cols = [OURS, DSET, GRID]; x = np.arange(3)
    ax.bar(x, means, yerr=errs, color=cols, width=0.62, capsize=4, edgecolor="#333",
           linewidth=1.2, zorder=3, error_kw=dict(lw=1.4))
    if baseline is not None:
        ax.axhline(baseline, color="#444", lw=0.9, ls="--"); ax.text(2.35, baseline+0.006, "chance", fontsize=7.5, color=MUTE)
    else:
        ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel(ylab)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=11.5, fontweight="bold")
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + (e+0.012 if m >= (baseline or 0) else -e-0.03), fmt.format(m),
                ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(0, ylim[1]-0.04*(ylim[1]-ylim[0]), "RAW-PLS best", ha="center", fontsize=8.2,
            color=OURS, fontweight="bold")

# ---------- (B) structural target (R^2) ----------
bars3(axB, [0.086, -0.337, -0.267], [0.023, 0.049, 0.038], (-0.45, 0.22), r"test $R^2$",
      "(B)  Real: BraTS structural target (R²)")
axB.text(1.0, -0.41, "geometric target — both baselines fail\n(DeepSets pooling discards geometry)",
         ha="center", fontsize=7.8, color=MUTE, style="italic")

# ---------- (C) grade HGG/LGG (AUC) ----------
bars3(axC, [0.935, 0.890, 0.834], [0.010, 0.012, 0.017], (0.5, 1.04), "AUC",
      "(C)  Real: BraTS2020 grade HGG/LGG (AUC)", fmt="{:.3f}", baseline=0.5)
axC.text(1.0, 1.012, "clinical target — RAW-PLS > DeepSets > aligned", ha="center",
         fontsize=7.8, color=MUTE, style="italic")

# ---------- (D) method ablation (synthetic) ----------
abl = [("base", 0.476, 0.060, BASE), ("+geom\n(#2)", 0.547, 0.071, "#7ec6c2"),
       ("+cov\n(#1)", 0.485, 0.196, "#c3a8e6"), ("+both", 0.582, 0.025, OURS)]
x = np.arange(4)
axD.bar(x, [a[1] for a in abl], yerr=[a[2] for a in abl], color=[a[3] for a in abl],
        width=0.62, capsize=4, edgecolor="#333", linewidth=1.2, zorder=3, error_kw=dict(lw=1.4))
axD.set_xticks(x); axD.set_xticklabels([a[0] for a in abl]); axD.set_ylabel(r"val $R^2$")
axD.set_ylim(0.0, 0.70); axD.set_title("(D)  Method ablation  (synthetic)", fontsize=11.5, fontweight="bold")
for xi, a in zip(x, abl):
    axD.text(xi, a[1]+a[2]+0.015, f"{a[1]:.3f}", ha="center", fontsize=8.8, color=INK)
axD.axhline(0.476, color=BASE, lw=1.0, ls="--", zorder=1)
axD.annotate("+0.106\n(geometry readout\n+ PLS objective)", xy=(3, 0.582), xytext=(1.6, 0.64),
             fontsize=8.3, color=OURS, ha="center", arrowprops=dict(arrowstyle="->", color=OURS))
axD.grid(alpha=.2, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("results_figure_v2.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote results_figure_v2.png")
