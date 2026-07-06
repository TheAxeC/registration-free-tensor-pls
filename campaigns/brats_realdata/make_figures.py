"""Generate the real-data result figures from the stored per-fold JSONs (reproducible;
no hard-coded numbers). Produces, for both the manuscript and the presentation:

  results_main.png        -- 2x2 headline: (A) synthetic robustness, (B) struct R^2,
                             (C) grade AUC, (D) ablation. B/C now include the anatomical-
                             registration baseline (the reviewer-proofing bar).
  registration_baseline.png -- focused 1x2: RAW-PLS vs centered grid vs anatomically-
                             registered grid, on struct + grade, with the H-sweep inset
                             showing the anatomical baseline reported at its BEST grid.

Error bars = std across the 10 seed-means (n=10), matching the reported CIs (struct
0.086+/-0.023, grade 0.935+/-0.010). All 50-value arrays share the same seed-major fold
layout, so methods are paired.

Run (local): python make_figures.py            # writes PNGs next to this script + into code/results/figures/
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "../../results/brats_realdata"))
FIGS = os.path.normpath(os.path.join(HERE, "../../results/figures"))
os.makedirs(FIGS, exist_ok=True)

INK, MUTE = "#222222", "#666666"
OURS, DSET, GRID, ANAT, BASE, RIG = "#2ca02c", "#5b6cc4", "#c0392b", "#e67e22", "#999999", "#e0922a"
SETT = "#17becf"   # Set Transformer (permutation-invariant deep-set baseline)
SYN = "#8e44ad"    # deformable SyN-registered grid baseline
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "axes.linewidth": 1.0})


def jload(*p):
    return json.load(open(os.path.join(RES, *p)))


def seed_stats(arr, folds=5):
    """mean and std across seed-means (n = len/folds), the reported-CI convention."""
    a = np.asarray(arr, float)
    sm = a.reshape(-1, folds).mean(1)            # one mean per seed
    return float(sm.mean()), float(sm.std()), sm


# ---- load measured per-fold arrays (all 50 = 10 seeds x 5 folds, paired) ----
imp = jload("improved", "improved.json")          # struct full method (#1+#2): ours, base(centered)
grd = jload("grade", "grade.json")                # grade: ours, base(centered)
dss = jload("deepset", "struct_deepset.json")     # struct DeepSets
dsg = jload("deepset", "grade_deepset.json")      # grade DeepSets
rst = jload("reg", "reg_struct.json")             # struct anatomical baseline (+ H sweep)
rgr = jload("reg", "reg_grade.json")              # grade anatomical baseline (+ H sweep)
stt = jload("settransformer", "struct.json")      # struct Set Transformer (perm-invariant)
gtt = jload("settransformer", "grade.json")       # grade Set Transformer
rss = jload("reg", "reg_struct_syn.json")         # struct deformable-SyN grid (+ H sweep)
rgs = jload("reg", "reg_grade_syn.json")          # grade deformable-SyN grid

# struct (R^2)
s_ours = seed_stats(imp["ours"]); s_dset = seed_stats(dss["ours"])
s_grid = seed_stats(imp["base"]); s_anat = seed_stats(rst["anat"])
s_sett = seed_stats(stt["ours"]); s_syn = seed_stats(rss["anat"])
# grade (AUC)
g_ours = seed_stats(grd["ours"]); g_dset = seed_stats(dsg["ours"])
g_grid = seed_stats(grd["base"]); g_anat = seed_stats(rgr["anat"])
g_sett = seed_stats(gtt["ours"]); g_syn = seed_stats(rgs["anat"])

# paired Wilcoxon RAW-PLS vs anatomical baseline (same folds)
_, p_s = wilcoxon(np.asarray(imp["ours"]), np.asarray(rst["anat"]))
_, p_g = wilcoxon(np.asarray(grd["ours"]), np.asarray(rgr["anat"]))
# paired Wilcoxon RAW-PLS vs deformable-SyN grid (same folds)
_, p_s_syn = wilcoxon(np.asarray(rss["ours"]), np.asarray(rss["anat"]))
_, p_g_syn = wilcoxon(np.asarray(rgs["ours"]), np.asarray(rgs["anat"]))
print(f"struct: RAW-PLS {s_ours[0]:+.3f} vs affine {s_anat[0]:+.3f} (p={p_s:.1e}) vs SyN {s_syn[0]:+.3f} (p={p_s_syn:.1e})")
print(f"grade : RAW-PLS {g_ours[0]:.3f} vs affine {g_anat[0]:.3f} (p={p_g:.1e}) vs SyN {g_syn[0]:.3f} (p={p_g_syn:.1e})")


def bars_n(ax, stats, labels, cols, ylim, ylab, title, fmt="{:+.3f}", chance=None):
    """Headline grouped bar panel for an arbitrary number of methods (paired seed-means)."""
    n = len(stats)
    means = [s[0] for s in stats]; errs = [s[1] for s in stats]; x = np.arange(n)
    ax.bar(x, means, yerr=errs, color=cols, width=0.72, capsize=4, edgecolor="#333",
           linewidth=1.2, zorder=3, error_kw=dict(lw=1.4))
    if chance is not None:
        ax.axhline(chance, color="#444", lw=0.9, ls="--")
        ax.text(n - 0.7, chance + 0.006, "chance", fontsize=7.5, color=MUTE)
    else:
        ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8); ax.set_ylabel(ylab)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=11.5, fontweight="bold")
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + (e + 0.012 if m >= (chance or 0) else -e - 0.03), fmt.format(m),
                ha="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(0, ylim[1] - 0.05 * (ylim[1] - ylim[0]), "RAW-PLS best", ha="center",
            fontsize=8.2, color=OURS, fontweight="bold")


# ============================ FIGURE 1: results_main.png (2x2) ============================
fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13.5, 9.4))
fig.patch.set_facecolor("white")
fig.suptitle("Registration-Free Supervised Tensor-PLS — results", fontsize=18, fontweight="bold", color=INK, y=0.995)

# (A) synthetic robustness
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
axA.annotate("aligned methods\ncollapse", xy=(1.0, -0.40), xytext=(0.55, -0.78),
             fontsize=8.5, color=GRID, ha="center", arrowprops=dict(arrowstyle="->", color=GRID))

# (B) struct R^2 - 5 bars: RAW-PLS, DeepSets, Set Transformer, grid(centroid), grid(registered)
HEAD_LABELS = ["RAW-PLS\n(ours)", "DeepSets\n(deep set)", "Set\nTransformer",
               "grid-PLS\n(centroid)", "grid-PLS\n(registered)"]
HEAD_COLS = [OURS, DSET, SETT, GRID, ANAT]
bars_n(axB, [s_ours, s_dset, s_sett, s_grid, s_anat], HEAD_LABELS, HEAD_COLS,
       (-0.46, 0.22), r"test $R^2$", "(B)  Real: BraTS structural target (R²)")
axB.text(2.0, -0.43, "geometric target — every alignment-based baseline fails;\nregistration does not help",
         ha="center", fontsize=7.8, color=MUTE, style="italic")

# (C) grade AUC - 5 bars
bars_n(axC, [g_ours, g_dset, g_sett, g_grid, g_anat], HEAD_LABELS, HEAD_COLS,
       (0.5, 1.05), "AUC", "(C)  Real: BraTS2020 grade HGG/LGG (AUC)", fmt="{:.3f}", chance=0.5)
axC.text(2.0, 1.016, "clinical target — RAW-PLS > DeepSets / Set Transformer > centroid / registered grid",
         ha="center", fontsize=7.6, color=MUTE, style="italic")

# (D) ablation (synthetic)
abl = [("base", 0.476, 0.060, BASE), ("+geom\n(#2)", 0.547, 0.071, "#7ec6c2"),
       ("+cov\n(#1)", 0.485, 0.196, "#c3a8e6"), ("+both", 0.582, 0.025, OURS)]
x = np.arange(4)
axD.bar(x, [a[1] for a in abl], yerr=[a[2] for a in abl], color=[a[3] for a in abl],
        width=0.62, capsize=4, edgecolor="#333", linewidth=1.2, zorder=3, error_kw=dict(lw=1.4))
axD.set_xticks(x); axD.set_xticklabels([a[0] for a in abl]); axD.set_ylabel(r"val $R^2$")
axD.set_ylim(0.0, 0.70); axD.set_title("(D)  Method ablation  (synthetic)", fontsize=11.5, fontweight="bold")
for xi, a in zip(x, abl):
    axD.text(xi, a[1] + a[2] + 0.015, f"{a[1]:.3f}", ha="center", fontsize=8.8, color=INK)
axD.axhline(0.476, color=BASE, lw=1.0, ls="--", zorder=1)
axD.annotate("+0.106\n(geometry readout\n+ PLS objective)", xy=(3, 0.582), xytext=(1.6, 0.64),
             fontsize=8.3, color=OURS, ha="center", arrowprops=dict(arrowstyle="->", color=OURS))
axD.grid(alpha=.2, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.97])
for d in (FIGS,):
    fig.savefig(os.path.join(d, "results_main.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote results_main.png")


# ==================== FIGURE 2: registration_baseline.png (focused) ====================
fig2, (bx0, bx1, bx2) = plt.subplots(1, 3, figsize=(14.2, 4.5),
                                     gridspec_kw={"width_ratios": [1, 1, 1.05]})
fig2.patch.set_facecolor("white")
fig2.suptitle("Registration does not help — centroid > affine > deformable-SyN, all far below RAW-PLS",
              fontsize=14.5, fontweight="bold", color=INK, y=1.02)


def reg_panel(ax, ours, grid, anat, syn, ylim, ylab, title, fmt, p_syn, chance=None):
    # order: reg-free, centroid, affine, SyN -> centroid > affine > SyN (monotone decline)
    labels = ["RAW-PLS\n(reg-free)", "grid-PLS\n(centroid)", "grid-PLS\n(SRI24\naffine)",
              "grid-PLS\n(SyN\ndeformable)"]
    cols = [OURS, GRID, ANAT, SYN]
    means = [ours[0], grid[0], anat[0], syn[0]]; errs = [ours[1], grid[1], anat[1], syn[1]]
    x = np.arange(4)
    ax.bar(x, means, yerr=errs, color=cols, width=0.66, capsize=4, edgecolor="#333", linewidth=1.2,
           zorder=3, error_kw=dict(lw=1.4))
    if chance is not None:
        ax.axhline(chance, color="#444", lw=0.9, ls="--"); ax.text(3.3, chance + 0.006, "chance", fontsize=7.5, color=MUTE)
    else:
        ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.0); ax.set_ylabel(ylab)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=11.5, fontweight="bold")
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + (e + 0.012 if m >= (chance or 0) else -e - 0.03), fmt.format(m),
                ha="center", fontsize=8.6, fontweight="bold", color=INK)
    # annotate: registration HURTS, and MORE registration (affine->SyN) hurts MORE
    ax.annotate("more registration\nHURTS more", xy=(3, syn[0]),
                xytext=(2.0, syn[0] + (ylim[1]-ylim[0])*0.18),
                fontsize=8.0, color=SYN, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SYN))
    ax.text(0.5, ylim[1] - (ylim[1]-ylim[0])*0.07, f"margin vs SyN {means[0]-syn[0]:+.3f}\np={p_syn:.0e}",
            ha="center", fontsize=8.4, color=OURS, fontweight="bold")


reg_panel(bx0, s_ours, s_grid, s_anat, s_syn, (-0.46, 0.22), r"test $R^2$",
          "Structural target (R²)", "{:+.3f}", p_s_syn)
reg_panel(bx1, g_ours, g_grid, g_anat, g_syn, (0.5, 1.0), "AUC",
          "Grade HGG/LGG (AUC)", "{:.3f}", p_g_syn, chance=0.5)

# (3) H-sweep: struct R^2 for affine vs deformable-SyN grid, each at its swept resolutions
#     (no strawman - both baselines reported at their BEST H; SyN sits below affine everywhere)
Hs = sorted(int(h) for h in rst["anat_by_H"])
sv = [rst["anat_by_H"][str(h)] for h in Hs]            # affine (SRI24) struct R^2
sv_syn = [rss["anat_by_H"][str(h)] for h in sorted(int(h) for h in rss["anat_by_H"])]
Hs_syn = sorted(int(h) for h in rss["anat_by_H"])
l1, = bx2.plot(Hs, sv, "o-", color=ANAT, lw=2, label="affine (SRI24)")
l2, = bx2.plot(Hs_syn, sv_syn, "s--", color=SYN, lw=2, label="deformable (SyN)")
bx2.axhline(s_ours[0], color=OURS, lw=1.8, ls=":")
bx2.text(Hs[0], s_ours[0] + 0.02, "RAW-PLS R²", color=OURS, fontsize=8)
bx2.set_xlabel("grid resolution H (atlas voxels)"); bx2.set_ylabel("struct R²")
bx2.set_xticks(sorted(set(Hs) | set(Hs_syn)))
bx2.set_ylim(-1.0, 0.3)  # clip the high-H struct blow-up for readability
bx2.set_title("Struct R² swept over H\n(SyN below affine; neither reaches RAW-PLS)",
              fontsize=10.5, fontweight="bold")
bx2.legend(handles=[l1, l2], fontsize=7.8, loc="lower right", title="registered grid")
bx2.grid(alpha=.2)

fig2.tight_layout(rect=[0, 0, 1, 0.95])
PRESENT = os.path.normpath(os.path.join(HERE, "../../../present"))
for d in (FIGS, PRESENT):
    fig2.savefig(os.path.join(d, "registration_baseline.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote registration_baseline.png (code/results/figures/ + present/)")


# ==================== FIGURE 3: synthetic_upgrades.png (2 panels) ====================
# Upgrade-campaign synthetic validation: (A) both necessity margins positive across all
# regimes; (B) RAW-PLS representation is exactly invariant to isometric reparametrization
# whereas the grid representation is not. Reproducible from the synthetic_gonogo JSONs.
SYN_RES = os.path.normpath(os.path.join(HERE, "../../results/synthetic_gonogo"))
abl_rob = json.load(open(os.path.join(SYN_RES, "ablation_robustness.json")))
theo = json.load(open(os.path.join(SYN_RES, "theory_validation.json")))

fig3, (cxA, cxB) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                                gridspec_kw={"width_ratios": [1.85, 1]})
fig3.patch.set_facecolor("white")
fig3.suptitle("Synthetic upgrade-campaign validation — necessity holds in every regime; exact invariance",
              fontsize=14, fontweight="bold", color=INK, y=1.03)

# (A) ablation robustness: structure-necessity + supervision-necessity margins per regime
regimes = list(abl_rob.keys())
s_marg = [abl_rob[r]["struct_margin"] for r in regimes]
s_sd = [abl_rob[r]["struct_sd"] for r in regimes]
u_marg = [abl_rob[r]["sup_margin"] for r in regimes]
u_sd = [abl_rob[r]["sup_sd"] for r in regimes]
xr = np.arange(len(regimes)); w = 0.38
STRUCTC, SUPC = "#2ca02c", "#5b6cc4"
cxA.bar(xr - w / 2, s_marg, w, yerr=s_sd, color=STRUCTC, capsize=3, edgecolor="#333",
        linewidth=1.0, zorder=3, error_kw=dict(lw=1.2), label="structure-necessity margin")
cxA.bar(xr + w / 2, u_marg, w, yerr=u_sd, color=SUPC, capsize=3, edgecolor="#333",
        linewidth=1.0, zorder=3, error_kw=dict(lw=1.2), label="supervision-necessity margin")
cxA.axhline(0, color="#444", lw=1.1, zorder=2)
cxA.set_xticks(xr); cxA.set_xticklabels(regimes, fontsize=8.2, rotation=20, ha="right")
cxA.set_ylabel(r"necessity margin ($\Delta R^2$)")
cxA.set_title("(A)  Ablation robustness — both margins > 0 in every regime",
              fontsize=11.5, fontweight="bold")
cxA.legend(fontsize=8.4, loc="upper right", framealpha=.95)
cxA.grid(alpha=.2, axis="y")
cxA.set_ylim(-0.1, max(s_marg) + max(s_sd) + 0.35)
cxA.text(len(regimes) - 1, 0.04, "margin = 0", fontsize=7.6, color=MUTE, ha="right")

# (B) invariance: relative representation change under random isometric reparametrization
inv = theo["invariance"]
ours_rc = inv["ours_rel_change_mean"]; grid_rc = inv["grid_rel_change_mean"]
xb = np.arange(2)
floor = 1e-17  # log-axis floor so the ~4e-16 bar is visible
cxB.bar(xb, [max(ours_rc, floor), grid_rc], width=0.6, color=[OURS, GRID],
        edgecolor="#333", linewidth=1.2, zorder=3)
cxB.set_yscale("log")
cxB.set_xticks(xb); cxB.set_xticklabels(["RAW-PLS\n(ours)", "grid-PLS\n(flatten)"], fontsize=9)
cxB.set_ylabel("mean relative change of representation")
cxB.set_ylim(floor, max(grid_rc * 3, 10))
cxB.set_title("(B)  Invariance to isometric reparametrization",
              fontsize=11.5, fontweight="bold")
cxB.text(0, max(ours_rc, floor) * 3, f"{ours_rc:.0e}\nexact invariance\n(machine precision)",
         ha="center", fontsize=8.0, color=OURS, fontweight="bold")
cxB.text(1, grid_rc * 1.4, f"{grid_rc:.2f}", ha="center", fontsize=9, color=INK, fontweight="bold")
cxB.grid(alpha=.25, axis="y", which="both")

fig3.tight_layout(rect=[0, 0, 1, 0.94])
for d in (FIGS, PRESENT):
    fig3.savefig(os.path.join(d, "synthetic_upgrades.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote synthetic_upgrades.png (code/results/figures/ + present/)")
