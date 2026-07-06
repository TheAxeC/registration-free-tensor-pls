"""
BraTS -> unaligned point-cloud dataset for RAW-PLS.

Each case becomes a point cloud over its tumor region: one point per (sub-sampled)
tumor voxel, with a 4-D multiparametric feature [T1, T1ce, T2, FLAIR] and a 3-D mm
coordinate. NO cross-subject registration is performed (that's the whole point):
coordinates are only centered per-case, so absolute position carries no shared frame.

Targets:
  - MGMT methylation (binary classification), from a labels CSV, or
  - overall survival (regression), from a survival CSV.

Expected layout (BraTS 2021-style), one folder per case:
  <root>/<CaseID>/<CaseID>_t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, _flair.nii.gz, _seg.nii.gz
Modality filename suffixes are configurable for other BraTS releases.
"""
from __future__ import annotations
import os
import glob
import numpy as np

SUFFIX = {"t1": "_t1", "t1ce": "_t1ce", "t2": "_t2", "flair": "_flair", "seg": "_seg"}
MODS = ["t1", "t1ce", "t2", "flair"]


def _load(path):
    import nibabel as nib
    img = nib.load(path)
    return np.asanyarray(img.dataobj).astype(np.float32), img.affine


def _find(case_dir, cid, suf):
    hits = glob.glob(os.path.join(case_dir, f"*{suf}.nii*"))
    if not hits:
        raise FileNotFoundError(f"{cid}: no file matching *{suf}.nii* in {case_dir}")
    return hits[0]


def _znorm(vol, brain):
    m, s = vol[brain].mean(), vol[brain].std() + 1e-6
    return (vol - m) / s


def extract_case(case_dir, cid, max_points=4000, tumor_label_min=1, seed=0):
    """Return (P[m,3] mm coords centered, F[m,4] z-normed modalities) for one case."""
    vols = {}
    aff = None
    for mod in MODS:
        vols[mod], aff = _load(_find(case_dir, cid, SUFFIX[mod]))
    seg, _ = _load(_find(case_dir, cid, SUFFIX["seg"]))
    brain = vols["flair"] > 0
    for mod in MODS:
        vols[mod] = _znorm(vols[mod], brain)
    mask = seg >= tumor_label_min
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        raise ValueError(f"{cid}: empty tumor mask")
    g = np.random.default_rng(seed)
    if len(ijk) > max_points:
        ijk = ijk[g.choice(len(ijk), max_points, replace=False)]
    # voxel index -> mm via affine (registration-free: we only center per-case)
    homog = np.c_[ijk, np.ones(len(ijk))]
    xyz = (aff @ homog.T).T[:, :3]
    P = (xyz - xyz.mean(0)).astype(np.float64)
    F = np.stack([vols[m][tuple(ijk.T)] for m in MODS], axis=1).astype(np.float64)
    return P, F


def build_dataset(root, labels_csv, id_col, target_col, task="clf",
                  max_points=4000, limit=None, seed=0):
    """Scan <root> for case folders present in labels_csv; return P_list, F_list, Y, ids."""
    import pandas as pd
    df = pd.read_csv(labels_csv, dtype={id_col: str})       # keep leading zeros (e.g. 00000)
    df = df.dropna(subset=[id_col, target_col])
    # index by raw id, zero-padded 5-digit, and zero-stripped form so it matches
    # case dirs named BraTS2021_<5-digit> regardless of how the CSV stored the id
    lut = {}
    for _, r in df.iterrows():
        k = str(r[id_col]).strip()
        for variant in {k, k.zfill(5), k.lstrip("0") or "0"}:
            lut[variant] = r[target_col]
    P_list, F_list, Y, ids = [], [], [], []
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for cid in cases:
        key = cid if cid in lut else cid.split("_")[-1]
        if key not in lut:
            continue
        try:
            P, F = extract_case(os.path.join(root, cid), cid, max_points, seed=seed)
        except (FileNotFoundError, ValueError) as e:
            print("skip", cid, e); continue
        P_list.append(P); F_list.append(F)
        Y.append(float(lut[key])); ids.append(cid)
        if limit and len(ids) >= limit:
            break
    Y = np.asarray(Y, np.float64)
    if task == "clf":
        Y = (Y > 0.5).astype(np.float64)
    print(f"built dataset: {len(ids)} cases, "
          f"feat-dim={F_list[0].shape[1] if ids else 0}, "
          f"median points={int(np.median([len(p) for p in P_list])) if ids else 0}")
    return P_list, F_list, Y, ids


def save_npz(path, P_list, F_list, Y, ids):
    """Persist ragged dataset (avoids re-reading NIfTI on every HPC run)."""
    np.savez_compressed(
        path, Y=Y, ids=np.array(ids),
        lengths=np.array([len(p) for p in P_list]),
        P=np.vstack(P_list), F=np.vstack(F_list))


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    lens = d["lengths"]; bounds = np.cumsum(np.r_[0, lens])
    # IMPORTANT: read each array from the NpzFile ONCE. Indexing d["P"] re-decompresses
    # the full array every call, and the slice views keep each copy alive -> with 1000+
    # cases that is 100s of GB. Materialize once, then slice views into the single array.
    Pall, Fall = d["P"], d["F"]
    P = [Pall[bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    F = [Fall[bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    return P, F, d["Y"], list(d["ids"])
