"""Build the BraTS2020 grade (HGG/LGG) cache by reading cases INCREMENTALLY from the
downloaded zip (extract one case's 5 NIfTI to a temp dir -> point cloud -> discard), so the
31 GB uncompressed data never all hits disk. Reuses data_brats.extract_case. Also builds a
survival-regression cache from survival_info.csv (secondary target).

Usage: python build_grade_from_zip.py /tmp/b20dl/brats20-dataset-training-validation.zip
"""
import os, sys, io, re, zipfile, shutil
import numpy as np, pandas as pd
from data_brats import extract_case, save_npz
from core import paths

ZIP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b20dl/brats20-dataset-training-validation.zip"
OUTDIR = paths.data_path("BraTS2020"); os.makedirs(OUTDIR, exist_ok=True)
TMP = "/tmp/onecase"

z = zipfile.ZipFile(ZIP)
names = z.namelist()

# --- label tables from inside the zip ---
nm_path = [n for n in names if n.endswith("name_mapping.csv")][0]
nm = pd.read_csv(io.BytesIO(z.read(nm_path)))
# pick the 2020 ID column (case dirs are BraTS20_Training_XXX); name_mapping has per-year cols
_id2020 = [c for c in nm.columns if "2020" in c and "subject" in c.lower()]
idc = _id2020[0] if _id2020 else [c for c in nm.columns if "subject_id" in c.lower()][-1]
grade = {str(r[idc]).strip(): str(r["Grade"]).strip() for _, r in nm.iterrows()
         if pd.notna(r.get(idc)) and pd.notna(r.get("Grade"))}
G2N = {"HGG": 1.0, "LGG": 0.0}
print(f"name_mapping: {nm_path} | id col '{idc}' | grades {pd.Series(list(grade.values())).value_counts().to_dict()}", flush=True)

surv = {}
sp = [n for n in names if n.endswith("survival_info.csv")]
if sp:
    sv = pd.read_csv(io.BytesIO(z.read(sp[0])))
    sid = [c for c in sv.columns if "id" in c.lower()][0]
    sd = [c for c in sv.columns if "survival" in c.lower()][0]
    for _, r in sv.iterrows():
        try: surv[str(r[sid]).strip()] = float(r[sd])
        except (ValueError, TypeError): pass
    print(f"survival_info: {len(surv)} cases with survival_days", flush=True)

# --- case dirs in the zip ---
cases = sorted(set(m.group(1) for n in names
               for m in [re.search(r"(BraTS20_Training_\d+)/.*_t1\.nii", n)] if m))
print(f"{len(cases)} case dirs in zip", flush=True)

Pg, Fg, Yg, ig = [], [], [], []          # grade
Ps, Fs, Ys, isv = [], [], [], []         # survival
for n_done, cid in enumerate(cases):
    if cid not in grade or grade[cid] not in G2N:
        continue
    shutil.rmtree(TMP, ignore_errors=True); cd = os.path.join(TMP, cid); os.makedirs(cd)
    for n in names:
        if f"/{cid}/" in n and n.endswith(".nii"):
            open(os.path.join(cd, os.path.basename(n)), "wb").write(z.read(n))
    try:
        P, F = extract_case(cd, cid, max_points=4000)
    except Exception as e:
        print("skip", cid, e, flush=True); continue
    Pg.append(P); Fg.append(F); Yg.append(G2N[grade[cid]]); ig.append(cid)
    if cid in surv:
        Ps.append(P); Fs.append(F); Ys.append(surv[cid]); isv.append(cid)
    if (n_done + 1) % 50 == 0:
        print(f"  ...{len(ig)} cases done", flush=True)
shutil.rmtree(TMP, ignore_errors=True)

Yg = np.array(Yg)
save_npz(os.path.join(OUTDIR, "brats2020_grade.npz"), Pg, Fg, Yg, ig)
print(f"GRADE: {len(ig)} cases | HGG {int(Yg.sum())} / LGG {int((Yg==0).sum())}", flush=True)
if isv:
    Ys = np.array(Ys)
    save_npz(os.path.join(OUTDIR, "brats2020_survival.npz"), Ps, Fs, Ys, isv)
    print(f"SURVIVAL: {len(isv)} cases | days mean {Ys.mean():.0f} std {Ys.std():.0f}", flush=True)
print("BUILD_DONE", flush=True)
