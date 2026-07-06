"""Render the gradient-saliency overlay on REAL BraTS patients (local; matplotlib).

Input: results/brats_realdata/reg/saliency_export.npz (built on the cluster by saliency_export.py)
  per case: sub (saliency voxels in crop coords), vimp (|dY/dF|), lo (crop offset), t1ce/flair/seg
  crops (raw), y (true ET-NCR distance), pred (model prediction).

Output: a multi-patient axial overlay + a single-patient orthogonal-view figure, saved to code/results/figures/
and present/ for PRESENT.md. The overlay = grayscale T1ce + the model's per-voxel saliency (where it
"looks" to predict tumor sub-structure geometry), localizing to coherent lesion sub-regions, differently
per subject (each in its own coordinates - the registration-free point).

Usage: python saliency_overlay_figure.py [--npz <path>] [--patients id1 id2 id3]
"""
from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEF_NPZ = os.path.join(PROJ, "results", "brats_realdata", "reg", "saliency_export.npz")
FIGS = os.path.join(PROJ, "code", "results", "figures")
os.makedirs(FIGS, exist_ok=True)
PRESENT = os.path.join(PROJ, "present")


def load(npzs):
    """Merge one or more saliency_export npz files into a single {cid: rec} dict."""
    if isinstance(npzs, str):
        npzs = [npzs]
    out = {}
    for p in npzs:
        out.update(np.load(p, allow_pickle=True)["data"].item())
    return out


def saliency_plane(rec, axis, idx, slab=2, smooth=2.0, clip=99.0):
    """2D saliency map on the slice `axis`=idx, from voxels within +/-slab; smoothed + clipped."""
    sub, vimp = rec["sub"].astype(int), rec["vimp"].astype(float)
    shp = rec["t1ce"].shape
    keep = np.abs(sub[:, axis] - idx) <= slab
    s, v = sub[keep], vimp[keep]
    other = [d for d in range(3) if d != axis]
    H = np.zeros((shp[other[0]], shp[other[1]]), float)
    for (coord, val) in zip(s[:, other], v):
        H[coord[0], coord[1]] += val
    H = gaussian_filter(H, smooth)
    if H.max() > 0:
        hi = np.percentile(H[H > 0], clip)
        H = np.clip(H / (hi + 1e-9), 0, 1)
    return H


def anat_plane(vol, axis, idx):
    return [vol[idx, :, :], vol[:, idx, :], vol[:, :, idx]][axis]


def best_index(rec, axis):
    """Slice index with the most saliency mass along `axis`."""
    sub, vimp = rec["sub"].astype(int), rec["vimp"].astype(float)
    n = rec["t1ce"].shape[axis]
    mass = np.zeros(n)
    np.add.at(mass, sub[:, axis], vimp)
    return int(np.argmax(mass))


def overlay(ax, anat, sal, title=None, thr=0.08, gamma=0.65):
    a = anat.T[::-1]                      # display orientation
    s = sal.T[::-1]
    if (a > 0).any():
        lo, hi = np.percentile(a[a > 0], [1, 99])
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) ** gamma   # gamma-brighten anatomy
    ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    rgba = plt.cm.inferno(s)
    rgba[..., 3] = np.clip((s - thr) / (1 - thr), 0, 1) ** 0.8   # alpha ramps with saliency
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def fig_patients(data, patients, out):
    n = len(patients)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.7))
    if n == 1:
        axes = [axes]
    for ax, cid in zip(axes, patients):
        rec = data[cid]
        z = best_index(rec, 2)
        sal = saliency_plane(rec, 2, z)
        overlay(ax, anat_plane(rec["t1ce"], 2, z), sal,
                title=f"{cid}\nET–NCR dist  true {rec['y']:.2f} / pred {rec['pred']:.2f}")
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("model saliency  |∂Ŷ/∂F| (per-patient norm.)", fontsize=9)
    fig.suptitle("Where RAW-PLS looks — gradient saliency on real BraTS tumours (T1ce, axial)",
                 fontsize=12, y=1.02)
    for d in (FIGS, PRESENT):
        fig.savefig(os.path.join(d, out), dpi=160, bbox_inches="tight")
    print("wrote", out, "->", FIGS, "and", PRESENT)
    plt.close(fig)


def fig_ortho(data, cid, out):
    rec = data[cid]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
    for ax, axis, name in zip(axes, (2, 1, 0), ("axial", "coronal", "sagittal")):
        idx = best_index(rec, axis)
        sal = saliency_plane(rec, axis, idx)
        overlay(ax, anat_plane(rec["t1ce"], axis, idx), sal, title=name)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("saliency |∂Ŷ/∂F|", fontsize=9)
    fig.suptitle(f"RAW-PLS saliency localizes in 3D — {cid} (T1ce; true {rec['y']:.2f}/pred {rec['pred']:.2f})",
                 fontsize=12, y=1.02)
    for d in (FIGS, PRESENT):
        fig.savefig(os.path.join(d, out), dpi=160, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", default=[DEF_NPZ])
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--ortho_case", default=None)
    a = ap.parse_args()
    data = load(a.npz)
    avail = list(data.keys())
    print("cases:", [(k, round(data[k]["y"], 2), round(data[k]["pred"], 2)) for k in avail])
    patients = a.patients or avail[:3]
    fig_patients(data, patients, "saliency_patients.png")
    fig_ortho(data, a.ortho_case or patients[0], "saliency_ortho.png")


if __name__ == "__main__":
    main()
