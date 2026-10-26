#!/bin/sh
# spot-check of frozen default `all` seeds s1/s2: AF dev naive / filter / prune+soft + no-op on PDB dev
cd /cluster/project/beltrao/gmagnus/pocket_detect
until grep -q AF_FINAL_DONE results/dev_af/logs/final.log; do sleep 30; done
PY=.venv/bin/python; NO=results/dev_af/preds_noop; P=results/dev_af/preds
for s in 1 2; do
  M=results/dev/freeze/models/burrow_all_default_s$s.json; T=frz_s$s
  for c in naive prune_soft; do $PY scripts/af_dev_predict.py --config $c --model $M --name ${c}_$T --jobs 16; done
  $PY scripts/af_plddt_filter_baseline.py --pred-dir $P/naive_$T --out $P/naive_plddt70_$T --sites results/dev_af/af_dev_sites_all.tsv
  $PY scripts/af_dev_predict.py --config naive --model $M --sites results/dev/biolip/dev_sites_all.tsv --out-root $NO --name standard_$T --jobs 16
  $PY scripts/af_dev_predict.py --config prune_soft --const-plddt 95 --model $M --sites results/dev/biolip/dev_sites_all.tsv --out-root $NO --name prune_soft_c95_$T --jobs 16
  n=0; for f in $NO/standard_$T/*.csv; do b=$(basename $f); [ "$b" = _stats.csv ] || cmp -s $f $NO/prune_soft_c95_$T/$b || n=$((n+1)); done
  echo "NOOP s$s prune_soft c95: $n files differ" | tee -a results/dev_af/logs/noop_frz_s12.log
  $PY scripts/af_dev_eval.py --methods naive=$P/naive_$T naive_plddt70=$P/naive_plddt70_$T prune_soft=$P/prune_soft_$T --ref naive_plddt70 --out results/dev_af/eval_spot_$T.csv > results/dev_af/logs/eval_spot_$T.log 2>&1
done
echo SPOT_DONE
