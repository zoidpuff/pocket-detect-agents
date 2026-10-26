#!/bin/sh
# usage: final_run.sh PRESET VARIANT SEED "FEATURE ARGS" "TRAIN ARGS"   (audited D6 data; caches kept)
cd /cluster/project/beltrao/gmagnus/pocket_detect
export BURROW_TRAINSET=/cluster/project/beltrao/gmagnus/pocket_detect/results/dev/cache/trainset_snapshot/audited_d6
PY=.venv/bin/python
P=$1; VAR=$2; SEED=$3; FARGS=$4; TARGS=$5
OUT=burrow_${VAR}_${P}_s${SEED}
$PY -u scripts/burrow_featurize_biolip.py --split dev --jobs 14 $FARGS > results/dev/biolip/feat_final_$P.log 2>&1
$PY -u scripts/burrow_featurize_biolip.py --split train --assembly --jobs 14 $FARGS >> results/dev/biolip/feat_final_$P.log 2>&1
$PY -u scripts/burrow_train_biolip.py --variant $VAR --seed-idx $SEED --jobs 14 --name $OUT $FARGS $TARGS > results/dev/biolip/train_$OUT.log 2>&1
echo "$OUT done $(date)" >> results/dev/biolip/final_progress.txt
