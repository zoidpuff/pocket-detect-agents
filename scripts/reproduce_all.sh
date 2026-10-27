#!/bin/bash
# =====================================================================================================
# BURROW — end-to-end reproduction, in order:
#   raw data -> test union & benchmark ground truth -> train/dev build -> leakage audit & exclusions
#   -> baselines -> featurise / train / freeze (v1, v2) -> test predictions -> pre-registered analysis
#
#   scripts/reproduce_all.sh smoke        # 2-minute end-to-end check on 5 structures (run this first)
#   scripts/reproduce_all.sh <step>       # run one step (names below), e.g.  scripts/reproduce_all.sh train
#   scripts/reproduce_all.sh all          # every step in order (days of CPU; NOT needed to use the models)
#
# Steps marked [LONG] take hours (64 cores); [GPU] uses CUDA if available. Every step is idempotent or
# documented as overwriting. Hardware used: 64-core EPYC 9554, 4x RTX 4090, 128 GB RAM.
# Seeds: pocketdetect.seeds.SEEDS (single source of truth). Data provenance: DATA.md.
# Test-set gating (docs/PREREG.md): steps `test_predict` / `prereg` must only run on a frozen version.
# =====================================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SNAP=results/dev/cache/trainset_snapshot/audited_d6        # D6-audited manifests (see step `snapshot`)
FF="--n-rays 30 --max-len 16 --subsample 3"                  # default-preset feature design
TF="--score-floor 0.7"                                       # default preset: score buried points only
log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

step_smoke() {                       # < 2 min: model integrity, 5 dev inputs, reference == fast, scoring
  $PY scripts/burrow_smoke.py
}

step_data() {                        # [LONG] downloads + checksums (DATA.md); PDB/AFDB mirrors are read-only
  for s in scripts/data/download_*.sh; do bash "$s"; done
  bash scripts/data/install_tools.sh                        # MMseqs2, Foldseek
  bash scripts/data/collect_manifests.sh
  $PY -I scripts/data/verify_datasets.py
}

step_benchmarks() {                  # [LONG] benchmark ground truth (test sets: build only, never tune)
  $PY -I scripts/build_gt_ligysis.py --jobs 32
  $PY scripts/build_gt_cryptobench.py
  $PY scripts/build_gt_cryptobench_trainfolds.py              # apo-dev (CryptoBench TRAIN folds; dev only)
  $PY scripts/build_gt_pocketwatch.py
  $PY scripts/data/build_af_set.py && $PY scripts/data/verify_af_set.py && $PY scripts/build_gt_af.py
  $PY scripts/data/build_dimer_scan.py && $PY scripts/data/build_dimer_select.py && $PY scripts/data/verify_dimer_set.py
  $PY scripts/build_test_clusters.py
}

step_trainset() {                    # [LONG] Track A: test union -> candidates -> leakage split -> pooling
  bash scripts/data/build_trainset_all.sh                   # docs/TRAINSET.md (incl. verify_trainset.py)
  $PY scripts/data/build_trainset_foldseek_strata.py        # fold-novelty strata (descriptive)
}

step_audit() {                       # independent leakage audit + decision D6 exclusions (filters manifests)
  for a in scripts/audit/a1_extract.py scripts/audit/a2_search.py scripts/audit/a3_analyze.py; do $PY "$a"; done
  $PY scripts/data/apply_audit_exclusions.py                # data/processed/{train,dev}/manifest.csv (post-D6)
}

step_aux_sets() {                    # [LONG] AF models / apo chains of training proteins (v2 cycle; negative)
  $PY scripts/data/build_af_train_index.py && $PY scripts/data/build_af_train_set.py && $PY scripts/data/build_af_train_verify.py
  bash scripts/data/build_apo_train_all.sh
}

step_snapshot() {                    # freeze the (post-D6) manifests for training (protects against rebuilds)
  $PY scripts/burrow_relabel_biolip.py                      # snapshot + label refresh of cached features
  S=$(ls -td results/dev/cache/trainset_snapshot/*/ | head -1)
  mkdir -p $SNAP/train $SNAP/dev
  for sp in train dev; do cp data/processed/$sp/manifest.csv $SNAP/$sp/; ln -sfn "$(realpath $S/$sp/entries)" $SNAP/$sp/entries; done
}

step_baselines() {                   # [LONG][GPU] baselines on dev and test inputs (baselines/README.md)
  export BURROW_TRAINSET=$SNAP
  $PY scripts/burrow_biolip_prep.py                          # dev inputs + site tables (all / small)
  $PY baselines/run_p2rank.py  @results/dev/biolip/inputs.list -o results/dev/biolip/baselines/p2rank  -j 8
  $PY baselines/run_fpocket.py @results/dev/biolip/inputs.list -o results/dev/biolip/baselines/fpocket -j 8 --timeout 300
  log "test-set baselines: see baselines/README.md and scripts/predict_benchmarks.py (shared inputs)"
}

step_featurize() {                   # [LONG] point features + labels for train (+assemblies) and dev, both presets
  export BURROW_TRAINSET=$SNAP
  for cfg in "" "$FF"; do
    $PY scripts/burrow_featurize_biolip.py --split dev --jobs 16 $cfg
    $PY scripts/burrow_featurize_biolip.py --split train --assembly --jobs 16 $cfg
  done
}

step_train() {                       # [LONG] 12 models: SEEDS[:3] x {all, small} x {default, accurate}
  export BURROW_TRAINSET=$SNAP
  for s in 0 1 2; do for v in all small; do
    $PY scripts/burrow_train_biolip.py --variant $v --seed-idx $s --jobs 14 --name burrow_${v}_default_s$s $FF $TF
    $PY scripts/burrow_train_biolip.py --variant $v --seed-idx $s --jobs 14 --name burrow_${v}_accurate_s$s
  done; done
}

step_freeze() {                      # FREEZE v1 (provenance) and v2 (= v1 + output floor, tau from train-CV)
  export BURROW_TRAINSET=$SNAP
  $PY scripts/burrow_freeze.py --ranker-variant V0_base      # writes results/dev/freeze/, FREEZE.md
  # tau selection (train cluster-CV; needs pocket caches from --save-pockets full runs of the s0 default models)
  $PY scripts/burrow_train_biolip.py --variant small --seed-idx 0 --jobs 14 --name p3_small_default_s0 --save-pockets full $FF $TF
  $PY scripts/burrow_train_biolip.py --variant all --seed-idx 0 --jobs 14 --name p3_all_default_s0 --save-pockets full $FF $TF
  $PY scripts/burrow_tau_cv.py --run p3_small_default_s0 --variant small
  $PY scripts/burrow_tau_cv.py --run p3_all_default_s0 --variant all
  $PY scripts/burrow_freeze_v2.py                            # writes results/dev/freeze_v2/, FREEZE_v2.md
}

step_dev_eval() {                    # dev evaluations reported in results/dev/biolip/{PHASE2,PHASE3,V2_CYCLE,V3_CYCLE}.md
  export BURROW_TRAINSET=$SNAP
  bash results/dev/biolip/run_freeze_eval.sh                 # 12 frozen v1 models via the CLI, paired vs P2Rank
  for v in small all; do $PY scripts/burrow_v2_eval.py --name H5_${v}_default_s0 \
      --model results/dev/freeze_v2/models/burrow_${v}_default_s0.json --variant $v; done
  $PY scripts/burrow_asm_gt.py && $PY scripts/burrow_asm_oof.py && \
      $PY scripts/burrow_asm_eval.py --split train --methods v2=results/dev/biolip/asm/preds/oof_v2 \
      p2rank=results/dev/biolip/asm/preds/p2rank --ref p2rank
}

step_test_predict() {                # [LONG][GPU] test predictions of a FROZEN version (gated; never before the freeze)
  $PY scripts/predict_benchmarks.py --version v2 --freeze results/dev/FREEZE_v2.md --device cuda:0
}

step_prereg() {                      # pre-registered analysis (gated runner, TEST_LEDGER) + report number check
  $PY scripts/run_prereg_analysis.py --version v2
  $PY scripts/check_report_numbers.py
}

step_speed() {                       # runtime benchmarks (fast backend; ms/structure/core and structures/s/GPU)
  $PY scripts/verify_burrow_fast.py --n 60 --device cpu
  $PY scripts/bench_burrow.py
}

STEPS="data benchmarks trainset audit aux_sets snapshot baselines featurize train freeze dev_eval test_predict prereg speed"
target="${1:-smoke}"
if [ "$target" = all ]; then
  for s in $STEPS; do log "step $s"; "step_$s"; done
elif declare -f "step_$target" > /dev/null; then
  log "step $target"; "step_$target"
else
  echo "unknown step '$target'; steps: smoke $STEPS all"; exit 2
fi
