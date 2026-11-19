#!/usr/bin/env bash
# Re-run the four March-2026 THEORY scripts, unchanged, on the Nov-2026 v5 manifold (checkpoint A).
set -e
cd "$(dirname "$0")"
T=../../OLD_THEORY_ATTEMPT
python3 convert_ckpt.py ../../Nov-2026_tests/results/checkpoints/A_full_prior.pt qg_vae_v5A.pth
python3 $T/qg_extract_fast.py --vae qg_vae_v5A.pth --samples 100 --output qg_geometry_v5A.npz > extract_v5A.txt 2>&1
python3 $T/analyze_manifold.py qg_geometry_v5A.npz > manifold_analysis_v5A.txt 2>&1
python3 $T/plot_manifold.py qg_geometry_v5A.npz plots_v5A > /dev/null 2>&1
python3 $T/qg_extract_pdes_fast.py qg_geometry_v5A.npz qg_pdes_analysis_v5A.txt > pdes_stdout_v5A.txt 2>&1
echo done
