#!/bin/bash
# Full Track A build in order (docs/TRAINSET.md). Logs: data/processed/train/work/*.log
# Launch detached, e.g.: setsid nohup scripts/data/build_trainset_all.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."
set -e
W=data/processed/train/work
mkdir -p $W
L=$W/run_all.log
echo "--- full run $(date)" >> $L
.venv/bin/python scripts/data/build_trainset_testunion.py --procs 32 > $W/testunion.log 2>&1; echo TESTUNION_DONE >> $L
.venv/bin/python scripts/data/build_trainset_candidates.py > $W/candidates.log 2>&1; echo CANDIDATES_DONE >> $L
.venv/bin/python scripts/data/build_trainset_split.py --threads 24 > $W/split.log 2>&1; echo SPLIT_DONE >> $L
rm -rf data/processed/train/entries data/processed/dev/entries $W/clu50_*
.venv/bin/python scripts/data/build_trainset_pool.py --procs 32 --max-per-cluster 3 > $W/pool.log 2>&1; echo POOL_DONE >> $L
.venv/bin/python scripts/data/build_trainset_asmcheck.py --threads 16 > $W/asmcheck.log 2>&1; echo ASM_DONE >> $L
.venv/bin/python scripts/data/build_trainset_finalcheck.py --procs 24 > $W/finalcheck.log 2>&1; echo FINALCHECK_DONE >> $L
.venv/bin/python scripts/data/verify_trainset.py --procs 24 > $W/verify.log 2>&1 || echo VERIFY_FAILED >> $L
echo VERIFY_DONE >> $L
