#!/bin/bash
# Track A2 (apo counterparts for train/dev). Launch detached: setsid nohup scripts/data/build_apo_train_all.sh >/dev/null 2>&1 &
cd "$(dirname "$0")/../.."
set -e
W=data/processed/apo_train/work; mkdir -p $W; L=$W/run.log
echo "--- run $(date)" >> $L
[ -f data/processed/apo_train/candidates.parquet ] && [ "$REUSE_SEARCH" = 1 ] || .venv/bin/python scripts/data/build_apo_train_search.py --threads 24 > $W/search.log 2>&1; echo SEARCH_DONE >> $L
rm -rf data/processed/apo_train/train/entries data/processed/apo_train/dev/entries
.venv/bin/python scripts/data/build_apo_train_pool.py --procs 24 > $W/pool.log 2>&1; echo POOL_DONE >> $L
.venv/bin/python scripts/data/build_apo_train_leakcheck.py --threads 24 > $W/leakcheck.log 2>&1; echo LEAK_DONE >> $L
.venv/bin/python scripts/data/verify_apo_train.py --procs 24 > $W/verify.log 2>&1 || echo VERIFY_FAILED >> $L
echo VERIFY_DONE >> $L
