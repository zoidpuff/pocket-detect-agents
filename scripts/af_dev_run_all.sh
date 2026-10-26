#!/bin/sh
# AF-mode development pipeline (dev only; never touches data/processed/af = AF TEST set).
# usage: scripts/af_dev_run_all.sh BASE_MODEL.json TAG [TRAINSET_SNAPSHOT]
#   BASE_MODEL  standard BURROW model JSON (copied to results/dev_af/cache/base_<TAG>.json first)
#   TAG         suffix for every output (preds: results/dev_af/preds/<cfg>_<TAG>, models: afr_<TAG>_*)
# Steps: inputs/site tables -> AF-dev geometric ablation + controls (standard ranker) -> pLDDT<70
# filter baseline -> no-op check on PDB dev -> AF ranker training (cross-fitted) -> AF ranker on
# AF dev and on PDB dev (uniform pLDDT 95) -> out-of-fold geometric ablation on AF train models
# -> evaluations (results/dev_af/eval_*_<TAG>.csv).  ~45 min with 24 workers.
set -e
cd "$(dirname "$0")/.."
BASE=$1; TAG=$2
export BURROW_TRAINSET=${3:-$PWD/results/dev/cache/trainset_snapshot/audited_d6}
PY=.venv/bin/python
J=24
mkdir -p results/dev_af/cache results/dev_af/logs
M=results/dev_af/cache/base_$TAG.json
[ -f $M ] || cp -p "$BASE" $M
sha256sum $M | tee results/dev_af/logs/base_$TAG.sha256
[ -f results/dev_af/af_dev_sites_all.tsv ] || $PY scripts/af_dev_prep.py --split dev
[ -f results/dev_af/af_train_sites_all.tsv ] || $PY scripts/af_dev_prep.py --split train
[ -f results/dev_af/inputs/pruned.list ] || $PY scripts/af_dev_inputs.py
[ -d results/dev_af/preds/p2rank_alphafold_pruned ] || scripts/af_dev_baselines.sh 12

CFGS="naive prune soft prune_soft prune_soft_lam0.5 prune_soft_lam0.25 prune_soft_lam0.5low prune_soft_lam0.25low rand_seg rand_res crop70"
# 1. AF dev, standard ranker
for c in $CFGS; do $PY scripts/af_dev_predict.py --config $c --model $M --name ${c}_$TAG --jobs $J; done
$PY scripts/af_plddt_filter_baseline.py --pred-dir results/dev_af/preds/naive_$TAG --out results/dev_af/preds/naive_plddt70_$TAG --sites results/dev_af/af_dev_sites_all.tsv
# 2. no-op check on PDB dev (uniform pLDDT 95 must give byte-identical predictions)
NO=results/dev_af/preds_noop
$PY scripts/af_dev_predict.py --config naive --model $M --sites results/dev/biolip/dev_sites_all.tsv --out-root $NO --name standard_$TAG --jobs $J
for c in prune_soft prune_soft_lam0.25low; do
  $PY scripts/af_dev_predict.py --config $c --const-plddt 95 --model $M --sites results/dev/biolip/dev_sites_all.tsv --out-root $NO --name ${c}_c95_$TAG --jobs $J
  n=0; for f in $NO/standard_$TAG/*.csv; do b=$(basename $f); [ "$b" = _stats.csv ] || cmp -s $f $NO/${c}_c95_$TAG/$b || n=$((n+1)); done
  echo "NOOP $c c95: $n files differ" | tee -a results/dev_af/logs/noop_$TAG.log
done
# 3. AF ranker (cross-fitted), AF geometry = prune_soft
$PY scripts/af_dev_train_ranker.py --model $M --config prune_soft --tag $TAG --jobs $J
for r in results/dev_af/models/afr_${TAG}_*.json; do
  n=$(basename $r .json)
  $PY scripts/af_dev_predict.py --config prune_soft --model $M --af-ranker $r --name prune_soft_$n --jobs $J
  $PY scripts/af_dev_predict.py --config prune_soft --const-plddt 95 --model $M --af-ranker $r --sites results/dev/biolip/dev_sites_all.tsv --out-root $NO --name ${n}_c95 --jobs $J
done
# 4. out-of-fold geometric ablation on AF *train* models (supplementary power; standard ranker in-sample)
for c in naive prune soft prune_soft prune_soft_lam0.25low rand_seg rand_res crop70; do
  $PY scripts/af_dev_predict.py --config $c --model $M --oof --sites results/dev_af/af_train_sites_all.tsv --out-root results/dev_af/preds_trainoof --name ${c}_$TAG --jobs $J
done
$PY scripts/af_plddt_filter_baseline.py --pred-dir results/dev_af/preds_trainoof/naive_$TAG --out results/dev_af/preds_trainoof/naive_plddt70_$TAG --sites results/dev_af/af_train_sites_all.tsv
# 4b. cluster-CV rankers -> fully out-of-fold AF-train predictions (pdb_std = CV analogue of the standard ranker)
$PY scripts/af_dev_ranker_cv.py --model $M --tag $TAG --variants pdb_std afpdb_pl --jobs $J
for v in pdb_std afpdb_pl; do
  $PY scripts/af_dev_predict.py --config prune_soft --model $M --oof --af-ranker "results/dev_af/models/cv/afrcv_${TAG}_${v}_k{k}.json" --sites results/dev_af/af_train_sites_all.tsv --out-root results/dev_af/preds_trainoof --name prune_soft_cv_${v}_$TAG --jobs $J
done
# 5. evaluations
P=results/dev_af/preds
A=""; for c in naive naive_plddt70 $CFGS; do A="$A $c=$P/${c}_$TAG"; done
A="$A p2rank_default=$P/p2rank_default_full p2rank_alphafold=$P/p2rank_alphafold_full p2rank_alphafold_plddt70=$P/p2rank_alphafold_full_plddt70 p2rank_alphafold_pruned=$P/p2rank_alphafold_pruned"
$PY scripts/af_dev_eval.py --methods $A --ref naive --out results/dev_af/eval_ablation_$TAG.csv > results/dev_af/logs/eval_ablation_$TAG.log 2>&1
R="naive_plddt70=$P/naive_plddt70_$TAG prune_soft=$P/prune_soft_$TAG"
for r in results/dev_af/models/afr_${TAG}_*.json; do n=$(basename $r .json); R="$R $n=$P/prune_soft_$n"; done
$PY scripts/af_dev_eval.py --methods $R --ref naive_plddt70 --out results/dev_af/eval_ranker_$TAG.csv > results/dev_af/logs/eval_ranker_$TAG.log 2>&1
Q="standard=$NO/standard_$TAG"
for r in results/dev_af/models/afr_${TAG}_*.json; do n=$(basename $r .json); Q="$Q $n=$NO/${n}_c95"; done
$PY scripts/af_dev_eval.py --pdb --sites results/dev/biolip/dev_sites_all.tsv --methods $Q --ref standard --out results/dev_af/eval_pdbdev_noop_$TAG.csv > results/dev_af/logs/eval_pdbdev_$TAG.log 2>&1
T=results/dev_af/preds_trainoof
O=""; for c in naive naive_plddt70 prune soft prune_soft prune_soft_lam0.25low rand_seg rand_res crop70; do O="$O $c=$T/${c}_$TAG"; done
$PY scripts/af_dev_eval.py --sites results/dev_af/af_train_sites_all.tsv --methods $O --ref naive --out results/dev_af/eval_trainoof_$TAG.csv > results/dev_af/logs/eval_trainoof_$TAG.log 2>&1
$PY scripts/af_dev_eval.py --sites results/dev_af/af_train_sites_all.tsv --methods naive_plddt70=$T/naive_plddt70_$TAG prune_soft=$T/prune_soft_$TAG prune=$T/prune_$TAG soft=$T/soft_$TAG crop70=$T/crop70_$TAG --ref naive_plddt70 --out results/dev_af/eval_trainoof_vs_plddt70_$TAG.csv > results/dev_af/logs/eval_trainoof2_$TAG.log 2>&1
$PY scripts/af_dev_eval.py --sites results/dev_af/af_train_sites_all.tsv --methods cv_pdb_std=$T/prune_soft_cv_pdb_std_$TAG naive_plddt70=$T/naive_plddt70_$TAG prune_soft_stdranker=$T/prune_soft_$TAG cv_afpdb_pl=$T/prune_soft_cv_afpdb_pl_$TAG --ref cv_pdb_std --out results/dev_af/eval_trainoof_rankers_$TAG.csv > results/dev_af/logs/eval_trainoof3_$TAG.log 2>&1
echo AF_DEV_ALL_DONE $TAG
