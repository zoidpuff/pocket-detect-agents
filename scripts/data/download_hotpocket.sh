#!/usr/bin/env bash
# HOTPocket = "Human Omnibus of Targetable Pockets" (Carpenter & Altman, J Cheminform 2025,
# doi:10.1186/s13321-025-01125-x). NOT a classic benchmark: 2.4M candidate pockets from 7 pocket
# finders (AutoSite, CASTp, CavitySpace, fpocket, LIGSITEcs, P2Rank, PocketMiner) on the canonical
# human proteome (PDB + AF2), scored by hotpocketNN, with BioLiP2-derived known pockets.
#  - Zenodo 17653326 (data v2, 2025-11-19, CC-BY-4.0); v1 = Zenodo 16891050 (superseded, not fetched)
#  - code: github.com/Helix-Research-Lab/HOTPocket (MIT), pinned commit
source "$(dirname "$0")/_common.sh"
OUT="$RAW/hotpocket"
python3 -I "$ROOT/scripts/data/zenodo_fetch.py" 17653326 "$OUT/zenodo_v2"
for t in "$OUT"/zenodo_v2/*.tar.gz; do
  case "$t" in *model_inputs*|*model_checkpoints*) continue;; esac  # large feature arrays / NN weights: keep archive only
  b=$(basename "$t" .tar.gz); [[ -d "$OUT/extracted/$b" ]] || { mkdir -p "$OUT/extracted/$b"; tar -xzf "$t" -C "$OUT/extracted/$b"; }
done
COMMIT="${HOTPOCKET_COMMIT:-1ca16ce0c9c902568f1e9a1c83fd6978ec91acd8}"  # main @ 2026-03-17
fetch "https://codeload.github.com/Helix-Research-Lab/HOTPocket/tar.gz/$COMMIT" "$OUT/github/HOTPocket-$COMMIT.tar.gz" "$([[ -z "${HOTPOCKET_COMMIT:-}" ]] && echo 0575c1639cee0af6b487727e0ba9ccb41d4015505f429bfa34ed11226ca81c07)"
[[ -d "$OUT/github/HOTPocket-$COMMIT" ]] || tar -xzf "$OUT/github/HOTPocket-$COMMIT.tar.gz" -C "$OUT/github"
