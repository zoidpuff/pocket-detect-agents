#!/bin/sh
# Final dev predictions through the deployable CLI (exactly what users run), sharded over N processes.
# usage: scripts/burrow_dev_predict.sh MODEL.json NAME [RANKER=all] [N=8]
set -e
cd "$(dirname "$0")/.."
MODEL=$1; NAME=$2; RANKER=${3:-all}; N=${4:-8}
OUT=results/dev/biolip/preds_cli/$NAME
mkdir -p "$OUT" /tmp/burrow_shards_$$
split -n l/$N results/dev/biolip/inputs_audited.list /tmp/burrow_shards_$$/s_
for f in /tmp/burrow_shards_$$/s_*; do
  OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pocketdetect.burrow predict @"$f" --out "$OUT" \
      --ranker "$RANKER" --model "$MODEL" --no-points > "$f.log" 2>&1 &
done
wait
cat /tmp/burrow_shards_$$/s_*.log | grep "ok ->"
rm -rf /tmp/burrow_shards_$$
