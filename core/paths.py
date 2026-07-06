"""Path anchors for RAW-PLS - replaces hard-coded `~/data/BraTS*` and CWD-relative result
dirs so runners work from any cwd and on the HPC layout.

Resolution (env override > default):
  data caches   $RAWPLS_DATA     else  ~/data/        (BraTS2021/ struct, BraTS2020/ grade; see ../README.md)
  results root  $RAWPLS_RESULTS  else  code/results/  (inside code/; results/<campaign>/<sub>/)

Local:  run from `code/` with `PYTHONPATH=.`. BraTS data is HPC-only (synthetic campaign needs no data).
HPC:    code at ~/projects/regfree-tensor-pls/code; data at ~/data/BraTS2021|BraTS2020; env ~/envs/regfree_env.tar.gz.
"""
import os
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]            # .../code

DATA_DIR = Path(os.environ.get("RAWPLS_DATA", Path.home() / "data"))
RESULTS_ROOT = Path(os.environ.get("RAWPLS_RESULTS", _CODE / "results"))   # outputs live INSIDE code/


def data_path(name):
    """Absolute path to a data cache (e.g. 'BraTS2021/brats_struct.npz')."""
    return str(DATA_DIR / name)


def results_dir(campaign, sub=""):
    """results/<campaign>[/<sub>]/, created on demand. Returns a Path.
    `sub` distinguishes brats runs (e.g. 'sweep', 'pilot', 'grade')."""
    d = RESULTS_ROOT / campaign / sub if sub else RESULTS_ROOT / campaign
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_path(campaign, name, sub=""):
    """Absolute path to results/<campaign>[/<sub>]/<name> (string)."""
    return str(results_dir(campaign, sub) / name)
