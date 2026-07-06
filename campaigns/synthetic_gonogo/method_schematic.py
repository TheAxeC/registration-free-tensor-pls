"""Publication-style schematic of the registration-free supervised tensor-PLS
(fused-GW route). Uses real warped point clouds from the synthetic generator so
the illustration is faithful to the actual method."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RNG = np.random.default_rng

# ---- small faithful world: 3 atoms in 2D, 2 "feature" channels for color ----
def make_cloud(seed, warp):
    g = RNG(seed)
    centers = np.array([[0.25, 0.7], [0.7, 0.65], [0.5, 0.25]])
    cols = np.array([0, 1, 2])              # atom id -> color
    act = g.uniform(0.4, 1.0, size=3)
    P, lab = [], []
    for k in range(3):
        nk = int(round(act[k] * 30))
        P.append(centers[k] + g.normal(scale=0.05, size=(nk, 2)))
        lab += [cols[k]] * nk
    P = np.vstack(P); lab = np.array(lab)
    th = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = P @ R.T + g.uniform(-warp * 0.4, warp * 0.4, size=2)
    return P, lab

ATOM_C = ["#e4572e", "#17bebb", "#4361ee"]   # 3 atom colors

fig = plt.figure(figsize=(15, 7.2))
fig.suptitle("Registration-free supervised tensor-PLS  (fused Gromov-Wasserstein route)",
             fontsize=15, fontweight="bold", y=0.985)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 15); ax.set_ylim(0, 7.2)

def box(x, y, w, h, fc, ec, lw=1.5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1))

def arrow(x1, y1, x2, y2, color="#333", lw=2.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=18, color=color, lw=lw, linestyle=ls, zorder=5))

def inset(rect):
    a = fig.add_axes(rect); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    return a

# ============================ (A) unaligned samples ============================
box(0.25, 0.5, 3.0, 6.0, "#f7f7fb", "#888")
ax.text(1.75, 6.15, "(A) Unaligned samples", ha="center", fontsize=11, fontweight="bold")
ax.text(1.75, 5.8, r"each sample $n=\{(p_i,\,f_i)\}$", ha="center", fontsize=9, style="italic")
ax.text(1.75, 5.5, "spatial mode = exchangeable\nmeasure (no correspondence)",
        ha="center", fontsize=8, color="#555")
for j, (seed, yb) in enumerate([(1, 3.55), (4, 1.9), (9, 0.55)]):
    P, lab = make_cloud(seed, warp=0.6 + 0.15 * j)
    a = inset([0.045, 0.075 + j * 0.205, 0.13, 0.16])
    for k in range(3):
        m = lab == k
        a.scatter(P[m, 0], P[m, 1], s=10, c=ATOM_C[k], alpha=0.85, edgecolors="none")
    a.set_title(f"subject {j+1}", fontsize=7.5, pad=1)
    a.set_xlim(-1, 1.6); a.set_ylim(-1, 1.6)
ax.text(1.75, 0.72, "different warp / pose each", ha="center", fontsize=7.5, color="#a00")

# ============================ (B) per-sample structure =========================
arrow(3.35, 3.5, 4.05, 3.5)
box(4.1, 1.4, 2.2, 4.2, "#eef6ff", "#3b6")
ax.text(5.2, 5.25, "(B) Intra-sample\nstructure", ha="center", fontsize=10, fontweight="bold")
a = inset([0.292, 0.549, 0.075, 0.111]); a.imshow(np.abs(np.subtract.outer(
    np.sort(RNG(2).uniform(size=14)), np.sort(RNG(3).uniform(size=14)))), cmap="magma")
ax.text(5.2, 3.7, r"$C^{(n)}$: pairwise dist. (warp-inv.)", ha="center", fontsize=8)
a = inset([0.292, 0.326, 0.075, 0.090]); a.imshow(RNG(5).uniform(size=(14, 4)), cmap="viridis",
    aspect="auto")
ax.text(5.2, 2.1, r"$F^{(n)}$: features", ha="center", fontsize=8)

# ============================ fused-GW transport ==============================
arrow(6.35, 3.5, 7.15, 3.5, color="#c0392b", lw=2.6)
ax.text(6.75, 3.95, "fused", ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")
ax.text(6.75, 3.05, "GW  " + r"$T^{(n)}$", ha="center", fontsize=8.5, color="#c0392b",
        fontweight="bold")

# ============================ (C) shared template =============================
box(7.2, 3.0, 2.55, 3.5, "#fff7ee", "#e08a1e", lw=2.0)
ax.text(8.47, 6.15, "(C) Shared template", ha="center", fontsize=11, fontweight="bold")
ax.text(8.47, 5.82, r"$K$ latent atoms", ha="center", fontsize=9, style="italic")
a = inset([0.493, 0.46, 0.135, 0.135])
tc = np.array([[0.3, 0.7], [0.72, 0.6], [0.5, 0.28]])
for k in range(3):
    a.scatter(*tc[k], s=320, c=ATOM_C[k], edgecolors="k", lw=1.2, zorder=3)
    a.text(tc[k, 0], tc[k, 1], f"$g_{k+1}$", ha="center", va="center", fontsize=9,
           color="w", fontweight="bold", zorder=4)
for i in range(3):
    for j in range(i + 1, 3):
        a.plot(tc[[i, j], 0], tc[[i, j], 1], "k-", lw=0.8, alpha=0.4, zorder=1)
a.set_xlim(0.1, 0.95); a.set_ylim(0.1, 0.9)
ax.text(8.47, 3.25, r"atoms + structure $C^{\mathrm{ref}}$", ha="center", fontsize=8)

# ============================ (D) barycentric projection ======================
arrow(9.8, 3.5, 10.5, 3.5)
box(10.55, 1.9, 1.85, 3.2, "#f0fff4", "#27ae60")
ax.text(11.47, 4.75, "(D) Barycentric\nprojection", ha="center", fontsize=10, fontweight="bold")
a = inset([0.715, 0.34, 0.082, 0.18])
z = RNG(11).uniform(size=(3, 5)); z[:, 0] = [0.9, 0.5, 0.7]
a.imshow(z, cmap="cividis", aspect="auto")
for k in range(3):
    a.add_patch(plt.Rectangle((-0.5, k - 0.5), 0.18, 1, color=ATOM_C[k], clip_on=False))
ax.text(11.47, 2.25, r"$z^{(n)}\!\in\!\mathbb{R}^{K\times(C{+}1)}$", ha="center", fontsize=8.5)
ax.text(11.47, 1.98, "mass + mean feat / atom", ha="center", fontsize=7.5, color="#555")

# ============================ (E) PLS head + supervision ======================
arrow(12.45, 3.5, 13.15, 3.5)
box(13.2, 2.3, 1.55, 2.4, "#fdeef2", "#c0392b")
ax.text(13.97, 4.35, "(E) tensor-PLS\nhead", ha="center", fontsize=10, fontweight="bold")
ax.text(13.97, 3.55, r"scores $t^{(n)}$", ha="center", fontsize=9)
ax.text(13.97, 3.15, r"$\Rightarrow\ \hat{Y}$", ha="center", fontsize=12, fontweight="bold")
ax.text(13.97, 2.6, "grade / survival", ha="center", fontsize=7.5, color="#555")

# supervised feedback loop: maximize cov(z, Y) -> update template + loadings
ax.add_patch(FancyArrowPatch((13.6, 4.7), (8.6, 6.5),
             connectionstyle="arc3,rad=0.30", arrowstyle="-|>", mutation_scale=16,
             color="#8e44ad", lw=2.0, linestyle=(0, (5, 3)), zorder=4))
ax.text(11.3, 6.35, r"supervised update: maximize $\mathrm{cov}(z, Y)$  "
        r"$\rightarrow$  template $\{g_k, C^{\mathrm{ref}}\}$ + loadings",
        ha="center", fontsize=9.5, color="#8e44ad", fontweight="bold")

# ============================ interpretability branch =========================
ax.add_patch(FancyArrowPatch((11.47, 1.9), (5.2, 0.95),
             connectionstyle="arc3,rad=0.18", arrowstyle="-|>", mutation_scale=15,
             color="#2c7", lw=1.8, linestyle=(0, (4, 2)), zorder=4))
ax.text(8.3, 0.55, r"per-subject maps via $T^{(n)}$:  which atom/modality  $\cdot$  "
        r"where (spatial)  $\cdot$  when (time)", ha="center", fontsize=9,
        color="#1a7", fontweight="bold")

fig.savefig("method_schematic.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote method_schematic.png")
