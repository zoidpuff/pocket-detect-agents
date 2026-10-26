#!/bin/sh
# usage: sweep_variant.sh NAME VARIANT "FEATURE ARGS" "TRAIN ARGS"   (seed 0; output burrow_<VARIANT>_s0[_NAME])
cd /cluster/project/beltrao/gmagnus/pocket_detect
export BURROW_TRAINSET=/cluster/project/beltrao/gmagnus/pocket_detect/results/dev/cache/trainset_snapshot/audited_d6
PY=.venv/bin/python
NAME=$1; VAR=$2; FARGS=$3; TARGS=$4
OUTNAME=burrow_${VAR}_s0; [ "$NAME" != "REF" ] && OUTNAME=${OUTNAME}_$NAME
if [ -n "$FARGS" ]; then
  $PY -u scripts/burrow_featurize_biolip.py --split dev --jobs 14 $FARGS > results/dev/biolip/feat_$NAME.log 2>&1
  $PY -u scripts/burrow_featurize_biolip.py --split train --assembly --jobs 14 $FARGS >> results/dev/biolip/feat_$NAME.log 2>&1
fi
$PY -u scripts/burrow_train_biolip.py --variant $VAR --seed-idx 0 --jobs 14 --name $OUTNAME $FARGS $TARGS > results/dev/biolip/train_$OUTNAME.log 2>&1
if [ -n "$FARGS" ]; then
  D=$(grep -m1 "^config" results/dev/biolip/feat_$NAME.log | awk '{print $NF}')
  cp "$D"/timing_dev.csv results/dev/biolip/timing_feat_dev_$NAME.csv 2>/dev/null
  case "$D" in */feats_biolip/*) rm -rf -- "$D" ;; esac
fi
echo "$OUTNAME done $(date)" >> results/dev/biolip/sweep_progress.txt
