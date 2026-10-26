#!/usr/bin/env bash
# P2Rank datasets (COACH420, HOLO4K, CHEN11, JOINED, FPTRAIN, AH4H, + mlig variants and
# third-party predictions). Pinned to a commit of github.com/rdk/p2rank-datasets.
source "$(dirname "$0")/_common.sh"
COMMIT="${P2RANK_DS_COMMIT:-0236ecb38cbb60b89849a1ea36fbd9ee93f3e906}"   # master @ 2024-11-08
OUT="$RAW/p2rank_datasets"
mkdir -p "$OUT"
fetch "https://codeload.github.com/rdk/p2rank-datasets/tar.gz/$COMMIT" "$OUT/p2rank-datasets-$COMMIT.tar.gz" "$([[ -z "${P2RANK_DS_COMMIT:-}" ]] && echo b608bc941d12087f60c94095e822edac53a0a8ce5fd6b7cf960254a383949ad2)"
if [[ ! -d "$OUT/p2rank-datasets-$COMMIT" ]]; then
  tar -xzf "$OUT/p2rank-datasets-$COMMIT.tar.gz" -C "$OUT"
fi
ln -sfn "p2rank-datasets-$COMMIT" "$OUT/current"
echo "extracted to $OUT/current"
