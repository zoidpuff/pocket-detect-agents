#!/bin/sh
# Post-freeze AF-mode confirmation on the frozen model: full dev pipeline + strata + tables.
# usage: scripts/af_dev_final.sh FROZEN_MODEL.json [TAG=final]
set -e
cd "$(dirname "$0")/.."
TAG=${2:-final}
scripts/af_dev_run_all.sh "$1" $TAG
PY=.venv/bin/python
P=results/dev_af/preds; T=results/dev_af/preds_trainoof
$PY scripts/af_plddt_filter_baseline.py --pred-dir $P/prune_soft_$TAG --out $P/prune_soft_plddt70_$TAG --sites results/dev_af/af_dev_sites_all.tsv
$PY scripts/af_dev_strata.py --sites results/dev_af/af_dev_sites_all.tsv --jobs 16 --ref naive_plddt70 \
  --methods naive=$P/naive_$TAG naive_plddt70=$P/naive_plddt70_$TAG prune_soft=$P/prune_soft_$TAG afr_afpdb_pl=$P/prune_soft_afr_${TAG}_afpdb_pl \
  --out results/dev_af/strata_dev_$TAG.csv > results/dev_af/logs/strata_dev_$TAG.log 2>&1
$PY scripts/af_dev_strata.py --sites results/dev_af/af_train_sites_all.tsv --jobs 16 --ref naive_plddt70 \
  --methods naive=$T/naive_$TAG naive_plddt70=$T/naive_plddt70_$TAG prune_soft=$T/prune_soft_$TAG cv_pdb_std=$T/prune_soft_cv_pdb_std_$TAG cv_afpdb_pl=$T/prune_soft_cv_afpdb_pl_$TAG \
  --out results/dev_af/strata_trainoof_$TAG.csv > results/dev_af/logs/strata_trainoof_$TAG.log 2>&1
$PY scripts/af_dev_strata.py --sites results/dev_af/af_train_sites_all.tsv --jobs 16 --ref cv_pdb_std \
  --methods cv_pdb_std=$T/prune_soft_cv_pdb_std_$TAG cv_afpdb_pl=$T/prune_soft_cv_afpdb_pl_$TAG \
  --out results/dev_af/strata_trainoof_rankers_$TAG.csv > results/dev_af/logs/strata_trainoof_rankers_$TAG.log 2>&1
$PY scripts/af_dev_report.py --tag $TAG > results/dev_af/tables_$TAG.md
$PY scripts/af_dev_report.py --strata > results/dev_af/tables_strata.md
echo AF_FINAL_DONE $TAG
