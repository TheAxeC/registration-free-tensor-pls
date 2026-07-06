"""Export gradient saliency mapped to IMAGE-VOXEL space + the surrounding anatomy, for a few
real BraTS2021 patients, so an overlay figure can be rendered locally on the ACTUAL MRI.

Heavy compute (retrain the full-method struct model + gradient saliency) runs on the cluster;
output is a compact npz with, per case: the saliency voxels in image coords, their |dY/dF|
values, and a TUMOR-CROPPED T1ce + segmentation block (raw intensities, for the grayscale
background). Render with `saliency_overlay_figure.py` (local, matplotlib).

Same model + saliency as interpret.py (K16/ep200/lr3e-3/wd1e-3/alpha0.6/eps0.03/geom6/cov0.1,
struct target) so the figure is consistent with the locked interpretability result.

Usage (cluster):
  python saliency_export.py --raw_root /local/$USER/brats21_sal \
      --ids BraTS2021_01180 BraTS2021_00525 BraTS2021_01402 --out saliency_export.npz
"""
import os, glob, argparse
import numpy as np, torch
from data_brats import MODS, SUFFIX, _load, _find, _znorm, load_npz
from core.raw_pls import train_rawpls, device_dtype, pick_device
from core import paths


def case_dir(root, cid):
    d = os.path.join(root, cid)
    if os.path.isdir(d):
        return d
    h = [x for x in glob.glob(os.path.join(root, "**", cid), recursive=True) if os.path.isdir(x)]
    return h[0] if h else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--msal", type=int, default=1500, help="voxels fed to the saliency gradient")
    ap.add_argument("--margin", type=int, default=18, help="crop margin (voxels) around the tumor bbox")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cache = a.cache or paths.data_path("BraTS2021/brats_struct.npz")
    P, F, Y, ids = load_npz(cache)
    ids = list(ids)
    print(f"retraining full-method struct model on {len(P)} cases ...", flush=True)
    model, _, _ = train_rawpls(P, F, Y, np.arange(len(P)), np.arange(len(P)),
                               K=16, Mmax=128, epochs=200, lr=0.003, wd=1e-3, alpha=0.6,
                               eps=0.03, geom_rank=6, lambda_cov=0.1, task="reg", seed=0, verbose=False)
    model.eval(); dev = pick_device(None); dt = device_dtype(dev)

    out = {}
    for cid in a.ids:
        cd = case_dir(a.raw_root, cid)
        if cd is None:
            print("MISSING", cid, flush=True); continue
        raw = {m: _load(_find(cd, cid, SUFFIX[m]))[0] for m in MODS}
        seg, aff = _load(_find(cd, cid, SUFFIX["seg"]))
        brain = raw["flair"] > 0
        feat = {m: _znorm(raw[m], brain) for m in MODS}

        ijk = np.argwhere(seg >= 1)
        g = np.random.default_rng(0)
        sub = ijk if len(ijk) <= a.msal else ijk[g.choice(len(ijk), a.msal, replace=False)]
        xyz = (aff @ np.c_[sub, np.ones(len(sub))].T).T[:, :3]
        pp = (xyz - xyz.mean(0))
        ff = np.stack([feat[m][tuple(sub.T)] for m in MODS], 1)

        d = np.linalg.norm(pp[:, None] - pp[None], axis=-1); d /= d.max() + 1e-9
        Fb = torch.tensor(ff, dtype=dt, device=dev)[None].requires_grad_(True)
        Cb = torch.tensor(d, dtype=dt, device=dev)[None]
        ab = torch.full((1, len(pp)), 1.0 / len(pp), dtype=dt, device=dev)
        pred, _, _ = model(Fb, Cb, ab)
        if Fb.grad is not None:
            Fb.grad = None
        pred.sum().backward()
        vimp = Fb.grad[0].norm(dim=1).detach().cpu().numpy().astype(np.float32)

        # crop a tumor-centered block of the RAW anatomy (T1ce) + seg for the background
        lo = np.maximum(ijk.min(0) - a.margin, 0)
        hi = np.minimum(ijk.max(0) + a.margin + 1, np.array(seg.shape))
        sl = tuple(slice(lo[k], hi[k]) for k in range(3))
        t1ce_crop = raw["t1ce"][sl].astype(np.float32)
        flair_crop = raw["flair"][sl].astype(np.float32)
        seg_crop = seg[sl].astype(np.uint8)
        sub_local = (sub - lo).astype(np.int16)              # saliency voxels in crop coords

        out[cid] = {"sub": sub_local, "vimp": vimp, "lo": lo.astype(np.int16),
                    "t1ce": t1ce_crop, "flair": flair_crop, "seg": seg_crop,
                    "y": float(Y[ids.index(cid)]) if cid in ids else float("nan"),
                    "pred": float(pred.detach().cpu()),
                    "shape_full": np.array(seg.shape)}
        print(f"  {cid}: y={out[cid]['y']:.3f} pred={out[cid]['pred']:.3f} "
              f"vimp[max]={vimp.max():.3g} crop={t1ce_crop.shape} nsal={len(vimp)}", flush=True)

    np.savez_compressed(a.out, data=np.array(out, dtype=object))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
