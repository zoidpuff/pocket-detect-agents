#!/bin/sh
cd /cluster/project/beltrao/gmagnus/pocket_detect
until grep -q FEAT_DONE results/dev/biolip/featurize2.log; do sleep 20; done
V=results/dev/biolip/sweep_variant.sh
( $V REF all "" ""; $V B all "" "--point-rounds 100 --point-leaves 31"; $V C all "" "--point-rounds 100 --point-leaves 31 --score-floor 0.7"; $V F1 all "--n-rays 30" "--point-rounds 100 --point-leaves 31" ) &
( $V REF small "" ""; $V F4 all "--n-rays 30 --max-len 16 --subsample 3" "--point-rounds 100 --point-leaves 31 --score-floor 0.7"; $V F2 all "--max-len 16" "--point-rounds 100 --point-leaves 31"; $V F3 all "--subsample 3" "--point-rounds 100 --point-leaves 31" ) &
wait
echo PHASE2_DONE >> results/dev/biolip/sweep_progress.txt
