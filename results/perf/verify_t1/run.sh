#!/bin/bash
# T1 re-verification: fast backend (CPU + GPU, standard + AF mode) vs reference, 40 dev structures per run
cd /cluster/project/beltrao/gmagnus/pocket_detect
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
O=results/perf/verify_t1
i=0
for m in results/dev/freeze_fast/models/burrow_all_fast_s0.json results/dev/freeze_v2/models/burrow_all_default_s0.json results/dev/freeze_v2/models/burrow_all_accurate_s0.json results/dev/freeze_fast/models/burrow_small_fast_s0.json; do
  st=$(basename $m .json)
  for dev in cpu cuda:0 cuda:1 cuda:3; do
    case $dev in cuda:*) [ $((i % 3)) = 0 ] && g=cuda:0; [ $((i % 3)) = 1 ] && g=cuda:1; [ $((i % 3)) = 2 ] && g=cuda:3; d=$g;; *) d=cpu;; esac
    [ $dev = cuda:1 ] && continue; [ $dev = cuda:3 ] && continue
    .venv/bin/python scripts/verify_burrow_fast.py --n 40 --model $m --device $d --out $O/${st}_${d/:/}.json > $O/${st}_${d/:/}.log 2>&1 &
    .venv/bin/python scripts/verify_burrow_fast.py --n 40 --af-mode --path-col model_path --manifest results/dev_af/af_dev_entries.csv --model $m --device $d --out $O/${st}_${d/:/}_af.json > $O/${st}_${d/:/}_af.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo VERIFY_DONE
