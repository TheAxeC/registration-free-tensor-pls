"""Publication-grade schematic of Registration-Free Supervised Tensor-PLS.
All geometry in axis coords (W x H); insets converted via ins(). Clean vertical budget:
  title (top) | training-objective arc | template | pipeline band | bottom callouts."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RNG = np.random.default_rng
ATOM = ["#e4572e", "#2e86ab", "#8ac926"]
INK, MUTE = "#222222", "#666666"
W, H = 16.5, 9.4

def cloud(seed, warp, centers):
    g = RNG(seed); P, lab = [], []
    for k in range(3):
        nk = int(g.uniform(.6, 1.) * 26)
        P.append(centers[k] + g.normal(scale=.05, size=(nk, 2))); lab += [k] * nk
    P = np.vstack(P); lab = np.array(lab)
    th = g.uniform(0, warp * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return P @ R.T + g.uniform(-warp * .5, warp * .5, size=2), lab

fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)

def box(x, y, w, h, fc, ec, lw=1.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.14",
                 fc=fc, ec=ec, lw=lw, zorder=z))
def arrow(x1, y1, x2, y2, c=INK, lw=2.4, ls="-", rad=0.0, ms=20, z=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                 color=c, lw=lw, linestyle=ls, zorder=z, connectionstyle=f"arc3,rad={rad}"))
def ins(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor("#bbbbbb")
    return a
def txt(x, y, s, size=10, c=INK, w="normal", st="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=c, fontweight=w, style=st, ha=ha, va=va, zorder=8)

# ---- title ----
txt(W/2, 9.05, "Registration-Free Supervised Tensor-PLS", 20, INK, "bold")
txt(W/2, 8.66, "fused Gromov–Wasserstein transport of unaligned spatial measures onto a "
    "learnable, response-supervised template", 11.5, MUTE, st="italic")

C0 = np.array([[.30, .70], [.68, .62], [.50, .28]])
BY = 4.55                                   # pipeline band center

# ===== (A) unaligned subjects =====
box(0.30, 2.55, 3.05, 4.55, "#f6f8fb", "#9bb0c4")
txt(1.82, 6.82, "(A)  Unaligned subjects", 12.5, INK, "bold")
txt(1.82, 6.48, r"each subject $n=\{(p_i,\,f_i)\}_i$", 9.5, MUTE)
txt(1.82, 6.16, "exchangeable spatial mode —\nno cross-subject correspondence", 8.4, "#a23")
for j in range(3):
    P, lab = cloud(j + 1, .55 + .12 * j, C0)
    a = ins(0.6, 4.78 - j * 1.02, 2.35, 0.86)          # compressed + raised to stay inside box (A)
    for k in range(3):
        m = lab == k; a.scatter(P[m, 0], P[m, 1], s=11, c=ATOM[k], alpha=.85, edgecolors="none")
    a.set_xlim(-1.1, 1.7); a.set_ylim(-1.1, 1.7)
    a.set_title(f"subject {j+1}", fontsize=7.6, pad=1.5, color=MUTE)

# ===== (B) sample summaries =====
arrow(3.40, BY, 3.95, BY)
box(4.0, 2.95, 1.95, 3.9, "#eef7ef", "#5fa86a")
txt(4.97, 6.58, "(B)  Sample summaries", 11, INK, "bold")
a = ins(4.28, 5.0, 1.4, 1.05)
a.imshow(np.abs(np.subtract.outer(np.sort(RNG(2).uniform(size=14)),
         np.sort(RNG(3).uniform(size=14)))), cmap="magma")
txt(4.97, 4.72, r"$C^{(n)}$ geometry (warp-inv.)", 8.5)
a = ins(4.45, 3.35, 1.05, 0.85); a.imshow(RNG(5).uniform(size=(14, 4)), cmap="viridis", aspect="auto")
txt(4.97, 3.12, r"$F^{(n)}$ features", 8.7)

# ===== learnable template (above transport; kept left of D, x<=8.6) =====
box(6.35, 5.95, 2.3, 1.95, "#fff6e9", "#e0922a", lw=2.0)
txt(7.5, 7.62, "Learnable template $\\Theta$", 10.5, INK, "bold")
a = ins(6.5, 6.02, 1.1, 1.15)
tc = np.array([[.30, .68], [.70, .60], [.50, .28]])
for i in range(3):
    for jj in range(i + 1, 3): a.plot(tc[[i, jj], 0], tc[[i, jj], 1], "-", color="#ccc", lw=1.1, zorder=1)
for k in range(3):
    a.scatter(*tc[k], s=240, c=ATOM[k], edgecolors="k", lw=1.1, zorder=3)
    a.text(*tc[k], f"$g_{k+1}$", ha="center", va="center", fontsize=7.5, color="w", fontweight="bold", zorder=4)
a.set_xlim(.12, .92); a.set_ylim(.12, .84)
txt(7.95, 6.95, "atoms $G$\n$C^{\\mathrm{ref}},\\,W,\\,b$", 8.2, MUTE, ha="left")

# ===== (C) fused-GW transport =====
arrow(5.97, BY, 6.55, BY, c="#c0392b", lw=2.8)
arrow(7.3, 5.93, 7.3, 5.72, c="#e0922a", lw=1.8, ms=14)         # template -> transport
box(6.6, 3.35, 1.7, 2.35, "#fdeef2", "#c0392b")
txt(7.45, 5.45, "(C) Fused-GW\ntransport", 10.5, INK, "bold")
a = ins(6.92, 3.6, 1.05, 1.0); a.imshow(RNG(9).uniform(size=(16, 6))**2, cmap="inferno", aspect="auto")
txt(7.45, 3.5, r"$T^{(n)}=\mathcal{T}^{S,J}_{\varepsilon}(\mu_n,\Theta)$", 8.2)

# ===== (D) representation z =====
arrow(8.32, BY, 8.95, BY)
box(9.0, 2.75, 2.7, 4.1, "#eef0fb", "#5b6cc4")
txt(10.35, 6.55, "(D)  Representation $z^{(n)}$", 11, INK, "bold")
for name, col, yy in [("mass per atom", "#9fb0e8", 5.55),
                      ("mean feature / atom", "#c3a8e6", 4.6),
                      ("geometry readout  (#2)", "#7ec6c2", 3.65)]:
    ax.add_patch(plt.Rectangle((9.25, yy), 2.2, 0.6, fc=col, ec="#444", lw=1.0, zorder=3))
    txt(10.35, yy + 0.3, name, 8.4, INK)
txt(10.35, 3.18, r"low-rank $a_j^{\top}(T^{\top}\!C\,T)\,b_j$", 7.6, "#2a7d77")

# ===== (E) head -> prediction =====
arrow(11.75, BY, 12.4, BY)
box(12.45, 3.55, 1.8, 2.0, "#eaf6ee", "#3a9d57")
txt(13.35, 5.25, "(E)  Head", 11, INK, "bold")
txt(13.35, 4.75, r"$\hat{Y}=\beta^{\top} z$", 11)
txt(13.35, 4.25, "grade / survival /\nstructure", 7.8, MUTE)
arrow(14.3, BY, 15.5, BY, lw=2.6)
txt(15.55, 4.95, r"$\hat{Y}$", 13, INK, "bold")

# ===== supervised objective (#1) - arc from head back to template =====
# Feedback arrow (hook): leaves the top of the head (E) box going straight UP (right of panel D),
# turns LEFT across the clear band above D (D top y=6.85, label y=7.98), and arrives on the right
# edge of the template box with the arrowhead pointing in. angle3 makes the up-then-across corner.
ax.add_patch(FancyArrowPatch((13.35, 5.6), (8.66, 7.5), connectionstyle="angle3,angleA=90,angleB=180",
             arrowstyle="-|>", mutation_scale=17, color="#7b4fb5", lw=2.4, linestyle=(0, (6, 3)), zorder=5))
txt(11.0, 8.28, "supervised training:  min $\\|Y-\\hat Y\\|^2 - \\lambda\\,\\mathrm{cov}(z,Y)$  "
    "(#1 PLS objective)", 10, "#7b4fb5", "bold")
txt(11.0, 7.98, "shapes template + transport to be predictive, not only the head", 8.4,
    "#7b4fb5", st="italic")

# ===== registration-invariance callout (bottom left) =====
box(0.30, 0.35, 7.05, 1.95, "#f4fbf6", "#86c79a")
txt(0.55, 2.05, "Key property — registration invariance", 10.5, "#2c7a44", "bold", ha="left")
for j, w_ in enumerate([0.0, 0.5, 1.0]):
    P, lab = cloud(7, w_, C0)
    a = ins(0.55 + j * 1.05, 0.55, 0.92, 0.95)
    for k in range(3):
        m = lab == k; a.scatter(P[m, 0], P[m, 1], s=7, c=ATOM[k], edgecolors="none")
    a.set_xlim(-1.4, 1.8); a.set_ylim(-1.4, 1.8)
    a.set_title(["original", "rotated", "rot+shift"][j], fontsize=7, pad=1, color=MUTE)
arrow(3.78, 1.05, 4.35, 1.05, lw=2.0)
ax.add_patch(plt.Rectangle((4.5, 0.72), 0.55, 0.66, fc="#7ec6c2", ec="#444", lw=1, zorder=3))
txt(4.775, 1.05, r"$z$", 12, "w", "bold")
txt(5.2, 1.45, "identical", 9, "#2c7a44", "bold", ha="left")
txt(5.2, 1.02, "$T,z$ see $\\mu$ only via\n$(F,C)$ $\\Rightarrow$ warp-invariant\n(grid-PLS collapses)",
    8.2, "#2c7a44", ha="left")

# ===== interpretability output (bottom right) =====
box(7.6, 0.35, 8.6, 1.95, "#f7f5fb", "#a48bd0")
txt(7.85, 2.05, "Per-subject interpretability  (back-project through $T^{(n)}$)", 10.5,
    "#6a4aa0", "bold", ha="left")
gg = RNG(11)
for j, lb in enumerate(["which atom /\nmodality", "where\n(spatial map)", "when\n(visit)"]):
    a = ins(11.5 + j * 1.55, 0.55, 1.25, 1.0)
    a.imshow(gg.uniform(size=(12, 12)) * np.linspace(.2, 1, 12)[None], cmap="magma")
    a.set_title(lb, fontsize=7.4, pad=2, color=MUTE)
txt(7.95, 1.25, "the transport plan $T^{(n)}$\ngives a principled\nsaliency map per subject",
    8.4, "#6a4aa0", ha="left")

fig.savefig("method_figure_v2.png", dpi=180, bbox_inches="tight", facecolor="white")
print("wrote method_figure_v2.png")
