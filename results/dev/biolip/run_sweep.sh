#!/bin/sh
# point-model speed/accuracy sweep (ranker `all`, seed 0); waits for the reference runs
cd /cluster/project/beltrao/gmagnus/pocket_detect
until [ -f results/dev/biolip/train_s0.done ]; do sleep 20; done
PY=.venv/bin/python
$PY -u scripts/burrow_train_biolip.py --variant all --seed-idx 0 --point-rounds 100 --point-leaves 31 --name burrow_all_s0_B100x31 > results/dev/biolip/train_all_s0_B.log 2>&1
$PY -u scripts/burrow_train_biolip.py --variant all --seed-idx 0 --point-rounds 100 --point-leaves 31 --score-floor 0.4 --name burrow_all_s0_C100x31f04 > results/dev/biolip/train_all_s0_C.log 2>&1
$PY -u scripts/burrow_train_biolip.py --variant all --seed-idx 0 --point-rounds 60 --point-leaves 15 --score-floor 0.4 --name burrow_all_s0_D60x15f04 > results/dev/biolip/train_all_s0_D.log 2>&1
echo SWEEP_DONE > results/dev/biolip/sweep.done
