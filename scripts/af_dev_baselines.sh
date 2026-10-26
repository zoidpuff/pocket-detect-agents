#!/bin/sh
# P2Rank 2.5.1 (default and -c alphafold) on AF-dev models, full and BURROW-pruned input.
# Raw wrapper output -> results/dev_af/baselines_raw/<name>/ ; eval-format copies (<sid>.csv) ->
# results/dev_af/preds/<name>/ ; then the pLDDT<70 filter baseline for every method.
# usage: scripts/af_dev_baselines.sh [JOBS=12]
set -e
cd "$(dirname "$0")/.."
J=${1:-12}
for inp in full pruned; do
  for cfg in default alphafold; do
    name=p2rank_${cfg}_${inp}
    raw=results/dev_af/baselines_raw/$name
    .venv/bin/python baselines/run_p2rank.py @results/dev_af/inputs/$inp.list -o $raw -j $J --config $cfg
    mkdir -p results/dev_af/preds/$name
    for f in $raw/*.pockets.csv; do
      b=$(basename "$f" .pockets.csv)
      cp "$f" results/dev_af/preds/$name/$b.csv
    done
  done
done
echo BASELINES_DONE
