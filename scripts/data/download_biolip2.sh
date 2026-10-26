#!/usr/bin/env bash
# BioLiP2 annotation (Zhang et al. 2024, NAR 52:D404) — annotations + sequence/ligand lists only.
# Structures are NOT mirrored (use the local PDB mirror). BioLiP is updated weekly and files are not
# versioned, so we snapshot into a dated directory and record HTTP Last-Modified + sha256.
source "$(dirname "$0")/_common.sh"
BASE=https://zhanggroup.org/BioLiP
SNAP="${BIOLIP_SNAPSHOT:-$(date -u +%Y-%m-%d)}"
OUT="$RAW/biolip2/$SNAP"
mkdir -p "$OUT"
for p in download/BioLiP.txt.gz download/BioLiP_nr.txt.gz data/protein.fasta.gz data/protein_nr.fasta.gz \
         data/ligand.tsv.gz download/readme.txt download/readme_ligand.txt; do
  f="$OUT/$(basename "$p")"
  [[ -s "$f" ]] || curl -fsSIL "$BASE/$p" | grep -i '^last-modified' | sed "s#^#$(basename "$p")\t#" >> "$OUT/LAST_MODIFIED.txt"
  fetch "$BASE/$p" "$f"
done
ln -sfn "$SNAP" "$RAW/biolip2/current"
