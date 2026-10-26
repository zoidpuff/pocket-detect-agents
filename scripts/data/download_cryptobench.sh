#!/usr/bin/env bash
# CryptoBench (Skrhak et al. 2025, Bioinformatics 41:btae745) from OSF project pz4a9 (MIT).
# Downloads the dataset (json + folds + cif-files.zip + pymol scripts + non-cryptic pockets)
# and the small reference benchmark model/scripts.
source "$(dirname "$0")/_common.sh"
OUT="$RAW/cryptobench"
python3 -I "$ROOT/scripts/data/osf_fetch.py" pz4a9 "$OUT/osf"
CIF="$OUT/osf/cryptobench/cryptobench-dataset/auxiliary-data/cif-files.zip"
[[ -d "$OUT/cif-files" ]] || { mkdir -p "$OUT/cif-files"; unzip -q "$CIF" -d "$OUT/cif-files"; }
# companion code repo (dataset construction scripts), pinned commit
COMMIT="${CRYPTOBENCH_COMMIT:-9a3432f479325ca3c9fdc3b8612715ab5b6edecc}"  # master as of 2026-10-26
fetch "https://codeload.github.com/skrhakv/CryptoBench/tar.gz/$COMMIT" "$OUT/github/CryptoBench-$COMMIT.tar.gz" "$([[ -z "${CRYPTOBENCH_COMMIT:-}" ]] && echo 65845db113b53b5de3f90218c0b254280ff4de6dc60f89b5220a6ebe195eb9f0)"
[[ -d "$OUT/github/CryptoBench-$COMMIT" ]] || tar -xzf "$OUT/github/CryptoBench-$COMMIT.tar.gz" -C "$OUT/github"
