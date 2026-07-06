"""Rebuild the paper's figures + map the HPC pipeline.

The BraTS headline numbers are HPC-produced (credentialed data + cluster), but the figures regenerate
locally from the committed result JSONs in results/, and the synthetic ablation runs locally. Run from
this directory with PYTHONPATH=.

    python paper.py    # regenerate the real-data figures from the committed JSONs (-> results/figures/, present/)

Local synthetic ablation (the method-iteration #1/#2 check, no external data):
    python campaigns/synthetic_gonogo/local_method_test.py

HPC pipeline (governed BraTS data; results copied back to results/):
    bash hpc/run_all.sh    # submits the full 14-stage reproduction chain (see the code README);
                           # hpc/tuning/ holds the config-search jobs that found the operating point.

Figure -> manuscript:
    campaigns/synthetic_gonogo/method_figure_v2.py   -> Fig 1 (method schematic)
    campaigns/brats_realdata/make_figures.py         -> results_main.png (Fig 2) + registration_baseline.png
                                                        (Fig 3) + synthetic_upgrades.png (Fig 5)
    campaigns/brats_realdata/saliency_overlay_figure.py -> Fig 4 (real-patient saliency)
The theory proofs are in ../theory (a separate LaTeX document).
"""
import os
import subprocess
import sys


def main():
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    print("== regenerate the real-data figures from the committed result JSONs ==", flush=True)
    subprocess.run([sys.executable, "campaigns/brats_realdata/make_figures.py"], check=True, env=env)
    print("\nDONE (figures). The headline BraTS numbers need the HPC pipeline (credentialed data + cluster); "
          "the synthetic ablation runs locally (see this file's header).")


if __name__ == "__main__":
    main()
