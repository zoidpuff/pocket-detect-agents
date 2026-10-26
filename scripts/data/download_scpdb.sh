#!/usr/bin/env bash
# scPDB release 2017 (Desaphy et al. NAR 2015, doi:10.1093/nar/gku928): 16034 entries / 4782 proteins / 6326 ligands.
# Public, no registration: linked from http://bioinfo-pharma.u-strasbg.fr/scPDB/ ("Download sc-PDB").
# Server is HTTP-only. No explicit license on the site -> treat as academic use, do not redistribute.
source "$(dirname "$0")/_common.sh"
OUT="$RAW/scpdb"
fetch "http://bioinfo-pharma.u-strasbg.fr/scPDB/ressources/2016/scPDB.tar.gz" "$OUT/scPDB.tar.gz" a7fd8e142db6868cc65937f88124750365b976d36a8ec264e523289c9a75c079
[[ -d "$OUT/extracted" ]] || { mkdir -p "$OUT/extracted"; tar -xzf "$OUT/scPDB.tar.gz" -C "$OUT/extracted"; }
