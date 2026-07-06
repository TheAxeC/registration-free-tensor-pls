# brats_realdata - real-data BraTS pipeline (HPC)

Registration-free supervised tensor-PLS on real, genuinely unaligned lesions: BraTS tumours sit in different places/shapes per subject, so cross-subject voxel correspondence does not exist - exactly the regime the method targets. No registration is performed; coordinates are only centered per case. The model is in `../../core/raw_pls.py`; the public deposit overview is `../../README.md`.

## Two targets
- **Structural**: ET-NCR centroid distance (regression), BraTS2021, n=1181. The registration-free showcase (a geometry target, chosen to avoid intensity-circular leakage). Margin +0.354 R^2 (p=1.8e-15).
- **Grade**: HGG vs LGG (classification), BraTS2020, n=368. The high-absolute clinical result: AUC 0.935 vs grid-PLS 0.834 (+0.101, p=7.6e-10).

## Files (code -> result)
- `data_brats.py` - NIfTI to per-lesion point cloud (P[m,3] mm coords, F[m,4] modalities); ragged npz cache (`load_npz` reads arrays once).
- `build_seg_target.py` - the structural (ET-NCR distance) cache -> `~/data/BraTS2021/brats_struct.npz`.
- `build_grade_from_zip.py` - the grade cache, built incrementally from the BraTS2020 zip -> `~/data/BraTS2020/brats2020_grade.npz`.
- `run_brats.py` - the cross-validated runner; RAW-PLS vs the alignment-assuming grid+PLS baseline. CLI: `--cache --task {reg,clf} --folds --seed_id --K --Mmax --epochs --lr --wd --alpha --eps --geom_rank --lambda_cov --device --out`.
- `aggregate.py` / `rank_sweep.py` - mean + 95% CI + paired Wilcoxon; rank a config sweep.
- `deepset.py` / `set_transformer.py` - the permutation-invariant deep baselines (same clouds, more params).
- `interpret.py` - gradient-saliency interpretability (localizes to lesion sub-structure) -> `results/brats_realdata/interp_data.npz`.
- `build_abs_cache.py` - re-extract clouds keeping ABSOLUTE SRI24-atlas coords (no per-case centering), aligned 1:1 to a centered cache's id/Y order -> `*_abs.npz`. Feeds the registration baseline.
- `syn_register.py` / `syn_assemble.py` - the deformable ANTs SyN registration arm: SyN-register each case to a reference, warp the lesion to template space, extract the cloud -> `*_syn.npz`.
- `reg_baseline.py` - the registration baseline: anatomical (absolute-coord) grid-PLS, grid-resolution H swept (best-H reported), paired Wilcoxon vs the stored RAW-PLS folds -> `results/brats_realdata/reg/reg_*.json`. RAW-PLS beats it on both targets; registration HURTS the baseline vs centering (lesions do not correspond).
- `saliency_export.py` (cluster GPU) - retrain the struct model + per-voxel saliency mapped to image voxels + tumour-cropped raw anatomy -> `results/brats_realdata/reg/saliency_export.npz`. Render with `saliency_overlay_figure.py` (local matplotlib) -> `results/figures/` + `present/`.
- `make_figures.py` - regenerate the real-data figures from the JSONs (runs locally, matplotlib): `results_main.png` (the 2x2 headline; panels B/C show 5 bars: RAW-PLS, DeepSets, Set Transformer, grid-centroid, grid-registered), `registration_baseline.png` (the reg head-to-head, 4 bars: reg-free, centroid, SRI24-affine, deformable-SyN; the monotone decline + the affine-vs-SyN H-sweep), and `synthetic_upgrades.png` (ablation-robustness necessity margins + isometric-invariance). Writes to `results/figures/` + `present/`.

## Run on the cluster (`ssh utwente`), from `~/projects/regfree-tensor-pls`

```bash
# structural target: sweep -> rank -> 10-seed pilot at the winner
sbatch hpc/sweep_struct.slurm
python campaigns/brats_realdata/rank_sweep.py results/brats_realdata/sweep/
sbatch hpc/pilot_best.slurm
python campaigns/brats_realdata/aggregate.py results/brats_realdata/pilot/
# grade target
sbatch hpc/grade_sweep.slurm ; sbatch hpc/grade_pilot.slurm
# baselines + interpretability
sbatch hpc/deepset_baseline.slurm ; sbatch hpc/interpret.slurm
```

SLURM jobs `cd $SLURM_SUBMIT_DIR`, source `hpc/env_stage.sh` (the venv-tarball pattern - NOT conda, which is NFS-slow + ToS-gated; staged to node-local `/local`, sets `$PY` + `PYTHONPATH`), run `campaigns/brats_realdata/run_brats.py`, and write per-task JSON to `results/brats_realdata/<sub>/`; logs go to `logs/`. Data at `~/data/BraTS2021|BraTS2020`. Monitor with `squeue -u $USER` / `tail -f logs/...`; never launch parallel runs over ssh.

## Knobs
- `--Mmax` voxel subsample per lesion (128). `--K` template atoms.
- `--geom_rank` (#2 low-rank geometry readout) + `--lambda_cov` (#1 PLS-covariance objective) - the two method improvements.
- `--eps` (entropic) trades transport sharpness vs smoothness (theory Cor. 1, the eps_N -> 0 debiasing).
- `lr=0.005` (0.02 diverges); `wd~1e-3` matters. float64 on CPU/CUDA for transport stability; float32 auto on MPS.
- A null pilot is private guidance, not a publishable negative - iterate the method before concluding (the no-null rule).
