#!/bin/sh
# CLI predictions (deployable path) for all final models + paired evaluation vs P2Rank on the audited dev set.
cd /cluster/project/beltrao/gmagnus/pocket_detect
export BURROW_TRAINSET=/cluster/project/beltrao/gmagnus/pocket_detect/results/dev/cache/trainset_snapshot/audited_d6
for v in ${VARIANTS:-all small}; do
  M=""
  for p in default accurate; do for s in 0 1 2; do
    n=burrow_${v}_${p}_s${s}
    [ -f results/dev/biolip/models/$n.json ] || continue
    [ -d results/dev/biolip/preds_cli/$n ] || scripts/burrow_dev_predict.sh results/dev/biolip/models/$n.json $n $v 6 > /dev/null
    M="$M ${p}_s${s}=results/dev/biolip/preds_cli/$n"
  done; done
  [ "$v" = small ] && M="$M all_default_s0=results/dev/biolip/preds_cli/burrow_all_default_s0"
  .venv/bin/python scripts/burrow_eval_biolip.py --variant $v --baselines --n-boot 1000 --methods $M \
      burrow_geo=results/dev/biolip/preds/burrow_geo --out results/dev/biolip/eval_final_$v.csv > results/dev/biolip/eval_final_$v.log 2>&1
done
echo FINAL_EVAL_DONE
