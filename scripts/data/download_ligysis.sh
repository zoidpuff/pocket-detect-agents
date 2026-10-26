#!/usr/bin/env bash
# LIGYSIS (human subset) reference binding sites + per-method predictions from
# Utges & Barton 2024 (J Cheminform 16:126) "Comparative evaluation of methods for the
# prediction of protein-ligand binding sites".
#  - Zenodo 14645504 "LBS-Comparison results" (CC-BY-4.0): pickled pandas/dict objects
#  - Zenodo 14033286 bartongroup/LBS-comparison v2.0.0 (analysis code + notebooks; CC-BY-4.0)
#  - Zenodo 14178252 JavierSanchez-Utges/LIGYSIS v1.0.0 pipeline (CC-BY-4.0)
source "$(dirname "$0")/_common.sh"
OUT="$RAW/ligysis"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14645504 "$OUT/lbs_comparison_results"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14033286 "$OUT/lbs_comparison_code"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14178252 "$OUT/ligysis_pipeline"
for z in "$OUT"/lbs_comparison_code/*.zip "$OUT"/ligysis_pipeline/*.zip; do
  d="${z%.zip}"; [[ -d "$d" ]] || { mkdir -p "$d"; unzip -q "$z" -d "$d"; }
done
