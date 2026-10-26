#!/bin/bash
# Fast-backend equivalence on BioLiP dev structures for the 4 frozen s0 models, CPU and GPU.
cd "$(dirname "$0")/../.."
for m in all_default small_default all_accurate small_accurate; do
  for d in cpu cuda:0; do
    tag=${m}_$(echo $d | tr -d ':')
    .venv/bin/python scripts/verify_burrow_fast.py --n 64 --device $d --model results/dev/freeze/models/burrow_${m}_s0.json \
      --out results/perf/verify_${tag}.json > results/perf/verify_${tag}.log 2>&1 &
  done
done
wait
echo ALLDONE
