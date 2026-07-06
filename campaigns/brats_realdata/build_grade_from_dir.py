"""Build the BraTS2020 grade (HGG/LGG) cache from the EXTRACTED kagglehub directory.

Modern kagglehub (>=0.3) auto-extracts a dataset on download, so `dataset_download` returns a
directory tree, NOT the zip that build_grade_from_zip.py expects. This builder reads that extracted
tree (the MICCAI_BraTS2020_TrainingData folder holding name_mapping.csv + the per-case NIfTI),
mirroring build_grade_from_zip's label logic exactly and reusing data_brats.build_dataset for the
point-cloud extraction. It is symmetric with build_seg_target.py, which already builds from --root.

Same output as build_grade_from_zip (same 369 HGG/LGG cases in sorted order, same seed-0 extract_case,
same G2N mapping) -> ~/data/BraTS2020/brats2020_grade.npz.

Usage:
  python build_grade_from_dir.py --root <.../MICCAI_BraTS2020_TrainingData>
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from core import paths
from data_brats import build_dataset, save_npz

G2N = {"HGG": 1.0, "LGG": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="MICCAI_BraTS2020_TrainingData dir (has name_mapping.csv + BraTS20_Training_* folders)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N cases (smoke test)")
    a = ap.parse_args()

    nm = pd.read_csv(os.path.join(a.root, "name_mapping.csv"))
    idc = [c for c in nm.columns if "2020" in c and "subject" in c.lower()][0]   # BraTS_2020_subject_ID
    nm = nm[nm["Grade"].isin(G2N)].copy()
    nm["grade_num"] = nm["Grade"].map(G2N)
    print(f"name_mapping: id col '{idc}' | grades {nm['Grade'].value_counts().to_dict()}", flush=True)

    # build_dataset reads a labels CSV and float()s the target, so hand it a numeric grade column.
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        nm[[idc, "grade_num"]].to_csv(tf.name, index=False)
        labels_csv = tf.name
    try:
        P, F, Y, ids = build_dataset(a.root, labels_csv, id_col=idc, target_col="grade_num",
                                     task="clf", max_points=a.max_points, limit=a.limit)
    finally:
        os.remove(labels_csv)

    out = a.out or os.path.join(paths.data_path("BraTS2020"), "brats2020_grade.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_npz(out, P, F, Y, ids)
    print(f"GRADE: {len(ids)} cases | HGG {int(np.sum(Y == 1))} / LGG {int(np.sum(Y == 0))} -> {out}", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
