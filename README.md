# RAW-PLS: registration-free supervised tensor-PLS

Code for *"Registration-Free Supervised Tensor-PLS for Unaligned Multiparametric MRI"* (Faes, van den Berg, Amir Haeri). A lesion sits in a different place and shape in every patient, so tensor-PLS methods (HOPLS, N-PLS) that assume voxel correspondence break. RAW-PLS treats each sample as a point cloud of multivariate-feature voxels, transports it onto a single learnable cross-subject template via entropic fused Gromov-Wasserstein, barycentrically projects it to a fixed-size tensor, and feeds a response head - template, transport, and head trained jointly to maximize X->Y covariance. No registration; interpretable; with identifiability and consistency guarantees. The algorithm is in `core/raw_pls.py`.

## What it does (the result)

On two genuinely unaligned BraTS targets RAW-PLS is the only consistently-positive estimator under warp and beats centroid-aligned grid-PLS, an anatomically registered SRI24 grid, a deformable SyN grid, HOPLS/N-PLS, DeepSets, a Set Transformer, and a lesion-cropped registered radiomics baseline:
- **Structural** (ET-NCR centroid distance, BraTS2021, n=1181): RAW-PLS R^2 0.086 (95% CI [0.069, 0.105]); registration-free margin +0.354 over the centroid grid (95% CI [0.338, 0.372], Wilcoxon p=1.8e-15).
- **Grade** (HGG/LGG, BraTS2020, n=368): AUC 0.935 (95% CI [0.932, 0.938]) vs 0.834 (+0.101); AUPRC 0.975 (95% CI [0.954, 0.990]) against a 0.793 prevalence null, so the clinical result is not a class-imbalance artifact.
- **The grid's large deficit is largely a rasterization effect.** A registered whole-brain grid (affine SRI24, then deformable SyN) underperforms simple centering. A lesion-cropped registered radiomics baseline - same PLS head, no whole-brain rasterization - is far stronger than the grid (struct affine +0.056 vs grid -0.348), and RAW-PLS still beats it on both targets and both registration strengths (struct +0.030 / +0.053, grade +0.039 / +0.012, all p<0.05). The registration-free advantage holds against the strongest fair registered baseline.
- **Stability.** Under real non-isometric warps the representation changes at most about 3% (bounded, roughly linear in the metric distortion) against about 140% for the aligned grid - the empirical realization of the stability guarantee.

The two algorithm-level improvements: **#1** a PLS-covariance objective (`lambda_cov`, shapes the template + transport to be predictive) and **#2** a low-rank readout of the transport-weighted inter-atom geometry (`geom_rank`, recovers spatial arrangement a structural target needs). Ship both: `geom_rank=6, lambda_cov=0.1`.

## Layout (a Python package; run from this directory with `PYTHONPATH=.`)

```
core/        raw_pls.py (the algorithm: the unrolled fused-GW layer + barycentric projection + the
             geometry readout + the joint trainer), paths.py (data / results anchors)
campaigns/
  synthetic_gonogo/   the controlled warp benchmark + the method-iteration ablations + the figures (runs local)
  brats_realdata/     the real-data pipeline (cohort caches, runners, baselines, interpretability, figures; HPC)
hpc/         the SLURM runners + the env-staging script; hpc/tuning/ holds the config-search jobs
results/     generated JSONs + figures (produced by the runs; not part of an external deposit)
attic/       runnable dead-ends kept for the record but excluded from the deposit (external-validation
             probes that never beat radiomics - PI-CAI/ACDC/pediatric - and dev/smoke helpers)
paper.py     regenerate the figures from the committed JSONs + print the HPC pipeline map
```

Paths resolve through `core/paths.py` (`RAWPLS_DATA` and `RAWPLS_RESULTS` override the defaults; data lives under `~/data/`).

## Reproduce the paper

Two ways to run it.

**A. Just the figures** (fastest - the result JSONs are committed, no cluster needed):

```bash
pip install -r requirements.txt    # numpy scipy scikit-learn torch matplotlib POT nibabel kagglehub pandas
python paper.py                    # regenerate the paper figures from the committed result JSONs
```

**B. Full reproduction from scratch** (regenerate every number on the UTwente HPC), three steps:

1. **One-time setup** - build the env, download BraTS, then build the two point-cloud caches:
   ```bash
   # 1a. environments (packed venvs, staged to node-local /local at run time)
   sbatch hpc/build_env.slurm      # -> ~/envs/regfree_env.tar.gz  (the SyN + affine arm also needs ~/envs/regfree_ants_env.tar.gz)

   # 1b. download the raw BraTS data (free/public via kagglehub; needs kaggle creds in ~/.kaggle).
   #     The two targets use two collections: structural -> BraTS2021, grade -> BraTS2020.
   #     NB modern kagglehub (>=0.3) AUTO-EXTRACTS a dataset, so both return a DIRECTORY tree (no zip).
   python -c "import kagglehub; print(kagglehub.dataset_download('dschettler8845/brats-2021-task1'))"              # -> BraTS2021 per-case NIfTI root (use as DATA21)
   python -c "import kagglehub; print(kagglehub.dataset_download('awsaf49/brats20-dataset-training-validation'))"  # -> .../MICCAI_BraTS2020_TrainingData dir (has name_mapping.csv; use as DATA20)

   # 1c. build the two caches the pipeline actually reads (struct from the BraTS2021 NIfTI root, grade from the BraTS2020 extracted dir)
   sbatch --export=ALL,DATA21=<BraTS2021_nifti_root>,DATA20=<BraTS2020_MICCAI_dir> hpc/build_cache.slurm
   #   -> ~/data/BraTS2021/brats_struct.npz  +  ~/data/BraTS2020/brats2020_grade.npz

   # registration arms only: also stage the raw NIfTI to ~/data/BraTS{2021,2020}_raw (compute nodes cannot see the login /local)
   ```
2. **Run the whole chain:** `bash hpc/run_all.sh` (submits all stages as one afterok chain, respects the account's 8-job cap, exits in seconds; `bash hpc/run_all.sh 6` caps the SyN arrays; watch with `squeue -u $USER`). It writes every result JSON under `results/brats_realdata/`. The registration + radiomics arms are skipped if the raw NIfTI are not staged.
3. **Make the figures:** `python paper.py` (regenerates the figures from the JSONs; the synthetic ablation runs locally via `python campaigns/synthetic_gonogo/local_method_test.py`).

Each manuscript table/figure maps to a run as follows (each real-data run writes a JSON under `results/brats_realdata/`; `paper.py` / `make_figures.py` turn the JSONs into figures):

| Manuscript element | Produced by | Result JSON |
|---|---|---|
| Table 1 (headline + CIs + AUPRC) | `pilot_improved.slurm` + `grade_pilot.slurm`; CIs recomputed into `results/brats_realdata/stats/ci.json`; AUPRC from `grade_scores.py` | `pilot/`, `improved/`, `grade/`, `stats/ci.json`, `stats/grade_scores.json` |
| Table 2 (registered radiomics, C-1) | `reg_radiomics.py` (via `reg_radiomics.slurm`) | `reg/rad_*.json`, `stats/c1_radiomics_summary.json` |
| Table 3 (non-isometric stability, C-3) | `eta_curve.py` (via `eta_curve.slurm`) | `stats/eta_curve.json` |
| Table 4 (per-conjunct novelty) | prose only, no run | - |
| Fig 1 (method schematic) | `campaigns/synthetic_gonogo/method_figure_v2.py` | - |
| Fig 2 (headline bars) | `make_figures.py` | from the Table-1 JSONs |
| Fig 3 (registration ladder + H-sweep) | `make_figures.py` | `reg/reg_struct*.json`, `reg/reg_grade*.json` |
| Fig 4 (per-patient saliency overlay) | `saliency_export.py` -> `saliency_overlay_figure.py` | `reg/saliency_export*.npz` |
| Fig 5 (ablation robustness + invariance) | `make_figures.py` | synthetic ablation + invariance JSONs |

### The operating point (what the runs use)

The estimand is the *fixed unrolled* transport (not the nonconvex FGW optimum): S=5 outer linearized-cost steps, each with J=25 inner Sinkhorn iterations. Known-good settings, used for every reported number: K=16 atoms, Mmax=128 points per cloud, alpha=0.6, epsilon=0.03, lr=0.003 (larger rates diverge), weight decay 1e-3, geom_rank=6, lambda_cov=0.1, 200 epochs. The trainer is memory-safe (data stays on CPU, each minibatch moves to the device on demand); float32 on GPU, float64 on CPU.

## Campaigns

- **`synthetic_gonogo/`** (local, CPU/MPS). The controlled warp benchmark where the ground truth is known: two response-predictive atoms whose intra-sample distance is the target, plus high-mass high-variance nuisance atoms, under a per-sample warp. It carries the structure- and supervision-necessity ablations across the seven nuisance regimes (`ablation_robustness.py`), the invariance demonstration and theory checks (`theory_validation.py`, `consistency2.py`, `consistency3.py`), the method-iteration tests (`local_method_test.py`, `supervised_test.py`, `joint_supervised.py`, `hardened_gonogo.py`), and the figure scripts (`method_figure_v2.py` / `method_schematic.py`, `results_figure*.py`, `joint_figure.py`, `interp_figure.py`). Needs no external data.
- **`brats_realdata/`** (HPC). The real-data pipeline: cache builders, the RAW-PLS runner, every baseline, the registration arms, interpretability, and the figure assembler. See the file-by-file map in `campaigns/brats_realdata/README.md`.
- **`attic/external_validation/`** (dead-ends, excluded from the deposit). The external-validation experiments that never beat a radiomics baseline and so are not in the paper: prostate (PI-CAI csPCa), cardiac (ACDC), and pediatric glioma. Kept runnable for the record; the method's advantage is specific to the adult-glioma multi-compartment-geometry setting.

### Key `brats_realdata` scripts

- `data_brats.py` - point-cloud extraction from NIfTI and the ragged-cache IO (`load_npz` / `save_npz`).
- `build_seg_target.py` - build the structural cache (ET-NCR centroid distance) from BraTS2021.
- `build_grade_from_dir.py` - build the grade cache (HGG/LGG) from the extracted BraTS2020 dir (modern kagglehub auto-extracts, so there is no zip; reads name_mapping.csv + the per-case NIfTI). `build_grade_from_zip.py` is kept as a fallback for anyone who still has the raw training zip.
- `build_abs_cache.py` - rebuild a cache in absolute SRI24-atlas coordinates (the affine registration baseline).
- `run_brats.py` - the runner: trains RAW-PLS / DeepSets / Set Transformer over the CV folds; `rasterize` + `baseline_eval` are the aligned-grid baseline.
- `rank_sweep.py` / `aggregate.py` - pick the sweep-winning config; aggregate the per-seed pilot into means + CIs.
- `reg_baseline.py` - the anatomical registration baseline (affine SRI24 / deformable SyN whole-brain grid, best H).
- `reg_radiomics.py` - **C-1**: the lesion-cropped registered radiomics baseline (same PLS head, no whole-brain grid).
- `syn_register.py` / `syn_assemble.py` - deformable ANTs SyN registration (array over shards) and cache assembly.
- `deepset.py` / `set_transformer.py` - the two permutation-invariant set-encoder baselines.
- `grade_scores.py` - **R-3b**: grade AUPRC + bootstrap AUC/AUPRC CIs from per-sample scores.
- `eta_curve.py` - **C-3**: representation change vs measured non-isometric distortion on real lesions.
- `interpret.py` / `saliency_export.py` / `saliency_overlay_figure.py` - gradient saliency, its export with anatomy, and the overlay figure.
- `make_figures.py` - assemble the manuscript figures from the committed JSONs.

## SLURM files (`hpc/`)

Every job sources `env_stage.sh` (stages the packed venv to node-local `/local`, then runs the system `python3` with `PYTHONPATH` at the extracted site-packages - NFS is slow on this cluster). The SyN and radiomics arms use the separate antspyx env tarball. **`run_all.sh` submits the full reproduction chain** (`bash hpc/run_all.sh`); the individual jobs it chains are:

| file | what it runs |
|---|---|
| `env_stage.sh` | sourced by every job: stage the venv to `/local`, set `PYTHONPATH` (not submitted directly) |
| `build_env.slurm` | build the packed venv overlay on a compute node (built once, prerequisite) |
| `build_cache.slurm` | build the two BraTS point-cloud caches once (struct from the `DATA21` NIfTI root, grade from the `DATA20` extracted dir); prerequisite |
| `pilot_improved.slurm` | the full-method (geom_rank + lambda_cov) struct pilot, 10 seeds -> `improved/` (headline R^2) |
| `grade_pilot.slurm` | the grade pilot (AUC) -> `grade/` |
| `deepset_baseline.slurm` / `set_transformer_baseline.slurm` | the two set-encoder baselines on the same CV |
| `grade_scores.slurm` | **R-3b**: grade AUPRC + bootstrap CIs |
| `eta_curve.slurm` | **C-3**: the non-isometric stability curve on real lesions |
| `r10_sweep.slurm` | **R-10**: the alpha / epsilon sensitivity sweep around the operating point |
| `syn_register.slurm` | deformable ANTs SyN registration baseline, array over shards (antspyx env) |
| `reg_radiomics.slurm` | **C-1**: the lesion-cropped registered radiomics baseline, four target x registration configs |
| `interpret.slurm` / `saliency_export.slurm` | gradient saliency and its export for the overlay figure |

`run_all.sh` also submits `hpc/tuning/pilot_best.slurm` (the base-config struct pilot -> `pilot/`, which the affine/SyN arm pairs against) and the affine `build_abs_cache` + `reg_baseline` steps inline. **`hpc/tuning/`** holds the config-search jobs (`sweep_struct`, `pilot_best`, `grade_sweep`, `array_struct`, `array_seeds`) that found the operating point; the dead-end external-validation probes and the dev/smoke helpers live in **`attic/`** (excluded from the deposit).

## Data

All data is free / public. BraTS2021 (`kagglehub` slug `dschettler8845/brats-2021-task1`, structural target) and BraTS2020 (`awsaf49/brats20-dataset-training-validation`, grade target) come through `kagglehub` (the kaggle credentials are user-local and not in the repo). Raw volumes and the point-cloud caches are read from `~/data/` (paths in `core/paths.py`) and are not redistributed here. The synthetic campaign needs no external data. The proofs of the invariance, identifiability, and consistency guarantees are in the paper. (The dead-end external-validation probes in `attic/` use their own public datasets - PI-CAI, ACDC, BraTS-2023 - documented in their own scripts; they are not part of the paper.)

## Reproducibility

Every reported number comes from fixed seeds (the pilots sweep seeds 0-9; each seed fixes the CV split and the model init) at the single operating point above, so the runs are deterministic given the caches. `requirements.txt` pins the environment (numpy, scipy, scikit-learn, torch, matplotlib, POT, nibabel, kagglehub, pandas); the cluster uses the packed `regfree_env` tarball plus `regfree_ants_env` for the ANTs SyN arm. Public inputs are fetched, not redistributed (BraTS via `kagglehub`). `paper.py` regenerates the figures from the committed JSONs on a normal machine.

License: MIT (`LICENSE`). Cite the paper (on acceptance) and the BraTS challenge collections (Menze et al. 2015; Bakas et al. 2017, 2018).
