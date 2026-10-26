#!/usr/bin/env bash
# AF2BIND (Gazizov et al., Nat Methods 2026; bioRxiv 10.1101/2023.10.15.562410), MIT.
# Repo contains training labels (training/new_labels_2k*.pkl, assignments_2k.pkl = splits,
# tmscores_2k.txt.gz), model weights (attempt_*.zip) and human-proteome predictions (preds/).
source "$(dirname "$0")/_common.sh"
COMMIT="${AF2BIND_COMMIT:-6867239a99d40cb67f3d40c837550fb91b575337}"  # main @ 2026-10-14
OUT="$RAW/af2bind"
fetch "https://codeload.github.com/sokrypton/af2bind/tar.gz/$COMMIT" "$OUT/af2bind-$COMMIT.tar.gz" "$([[ -z "${AF2BIND_COMMIT:-}" ]] && echo 92d5acdc30fe76ee8661972da6f19dc24c256cd52e84b3255dce51218c4a58e3)"
[[ -d "$OUT/af2bind-$COMMIT" ]] || tar -xzf "$OUT/af2bind-$COMMIT.tar.gz" -C "$OUT"
