#!/usr/bin/env bash
# LIGYSIS (human subset) reference binding sites + per-method predictions from
# Utges & Barton 2024 (J Cheminform 16:126) "Comparative evaluation of methods for the
# prediction of protein-ligand binding sites".
#  - Zenodo 14645504 "LBS-Comparison results" v2 (CC-BY-4.0): pickled pandas/dict objects
#    (reference sites, per-method predictions, residue labels); v1 = Zenodo 13121414
#  - Zenodo 14033286 bartongroup/LBS-comparison v2.0.0 (analysis code + notebooks; CC-BY-4.0)
#  - Zenodo 14178252 JavierSanchez-Utges/LIGYSIS v1.0.0 pipeline (CC-BY-4.0)
source "$(dirname "$0")/_common.sh"
OUT="$RAW/ligysis"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14645504 "$OUT/lbs_comparison_results"
# Zenodo 13121414 = v1 (2024-07-29) of the same concept record (13121413). Its 78 files are
# byte-identical (md5) to the v2 files above; v2 only adds LIGYSIS_ress_binary_labels.pkl.
# Running the fetcher on the same directory downloads nothing and verifies v1 md5s.
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 13121414 "$OUT/lbs_comparison_results"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14033286 "$OUT/lbs_comparison_code"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 14178252 "$OUT/ligysis_pipeline"
for z in "$OUT"/lbs_comparison_code/*.zip "$OUT"/ligysis_pipeline/*.zip; do
  d="${z%.zip}"; [[ -d "$d" ]] || { mkdir -p "$d"; unzip -q "$z" -d "$d"; }
done
