"""Interpretability figure from real BraTS transport-saliency (interp_data.npz produced by
brats/interpret.py). Per example subject: lesion point cloud colored by per-voxel importance
(back-projected through T); plus a per-modality importance bar. Run: python interp_figure.py interp_data.npz"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "interp_data.npz", allow_pickle=True)
coords, vimp = list(d["coords"]), list(d["vimp"])
yval, pred = list(d["yval"]), list(d["pred"]); ids = list(d["ids"])
mod_imp, mods = d["mod_imp"], list(d["mods"])
ncase = len(coords)

fig = plt.figure(figsize=(3.2 * ncase + 3.6, 4.1)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, ncase + 2, width_ratios=[1] * ncase + [0.07, 0.85], wspace=0.28)
fig.suptitle(r"Per-subject gradient saliency $|\partial \hat Y/\partial F_{\rm voxel}|$ on the "
             "structural target  (registration-free)", fontsize=13, fontweight="bold", y=1.02)

for j in range(ncase):
    P = np.asarray(coords[j], float); v = np.asarray(vimp[j], float)
    # project to the lesion's 2 principal axes (registration-free view)
    Pc = P - P.mean(0); u, s, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
    # percentile-clip normalization: heavy-tailed saliency -> reveal mid-range structure,
    # not just a few outliers. Important voxels also drawn larger.
    lo, hi_p = np.percentile(v, 20), np.percentile(v, 98)
    vn = np.clip((v - lo) / (hi_p - lo + 1e-9), 0, 1)
    order = np.argsort(vn)                                  # draw important voxels on top
    hi = yval[j] >= np.median(yval)
    ax = fig.add_subplot(gs[0, j]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f4f4f6")
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=vn[order], cmap="viridis",
                    s=12 + 75 * vn[order]**1.5, edgecolors="none", vmin=0, vmax=1)
    ax.set_title(f"{ids[j]}\nET-NCR dist {yval[j]:.2f}  (pred {pred[j]:.2f})", fontsize=9,
                 color=("#c0392b" if hi else "#2e86ab"), fontweight="bold")
    ax.set_aspect("equal")
cax = fig.add_subplot(gs[0, ncase])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("voxel saliency (percentile-scaled)", fontsize=8); cb.ax.tick_params(labelsize=7)

axm = fig.add_subplot(gs[0, ncase + 1])
order = np.argsort(mod_imp)
axm.barh(range(len(mods)), mod_imp[order], color="#5b6cc4", edgecolor="#333")
axm.set_yticks(range(len(mods))); axm.set_yticklabels([mods[i] for i in order], fontsize=9)
axm.set_title("which modality\n(head importance)", fontsize=9.5, fontweight="bold")
axm.set_xlabel("importance", fontsize=8); axm.tick_params(labelsize=7)

fig.subplots_adjust(top=0.86, bottom=0.06, left=0.02, right=0.98)
fig.savefig("interp_figure.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote interp_figure.png")
