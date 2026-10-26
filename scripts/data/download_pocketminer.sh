#!/usr/bin/env bash
# PocketMiner cryptic-pocket test/val set (Meller et al. 2023, Nat Commun 14:1177):
# 38 apo-holo pairs / 39 cryptic pockets, residue-level labels on apo structures.
# Only data/pm-dataset is fetched (repo also has ~300 MB of MD trajectories we don't need).
source "$(dirname "$0")/_common.sh"
COMMIT="${POCKETMINER_COMMIT:-187062df3c94127e991669768009141a08fd5d8b}"  # Mickdub/gvp branch pocket_pred @ 2023-07-21
OUT="$RAW/pocketminer"
python3 -I "$ROOT/scripts/data/github_subtree_fetch.py" Mickdub/gvp "$COMMIT" data/pm-dataset "$OUT"
for f in README.md LICENSE LICENSE.txt; do
  curl -fsSL "https://raw.githubusercontent.com/Mickdub/gvp/$COMMIT/$f" -o "$OUT/repo_$f" 2>/dev/null || rm -f "$OUT/repo_$f"
done
