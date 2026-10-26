# Test-set prediction commands for ablations A1–A3 (prepared, NOT run)

To be run by the eval agent after the final version is fixed, as descriptive rows. Same harness as the v1 test predictions (`scripts/predict_benchmarks.py`), with the ablation model JSON instead of the frozen model; CPU or GPU fast backend (outputs identical). Example (one model; repeat for each):

```bash
# A2 CHEN11-only training (all)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_chen11_all_s0/<benchmark> --model results/ablations/models/abl_chen11_all_s0.json --ranker all --device cpu --workers 24 --no-points
# A2 CHEN11-only training (small)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_chen11_small_s0/<benchmark> --model results/ablations/models/abl_chen11_small_s0.json --ranker small --device cpu --workers 24 --no-points
# A1 fold-disjoint training (all)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_fold_disjoint_all_s0/<benchmark> --model results/ablations/models/abl_fold_disjoint_all_s0.json --ranker all --device cpu --workers 24 --no-points
# A1 fold-disjoint training (small)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_fold_disjoint_small_s0/<benchmark> --model results/ablations/models/abl_fold_disjoint_small_s0.json --ranker small --device cpu --workers 24 --no-points
# A3 shuffled labels (negative control) (all)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_shuffled_all_s0/<benchmark> --model results/ablations/models/abl_shuffled_all_s0.json --ranker all --device cpu --workers 24 --no-points
# A3 shuffled labels (negative control) (small)
PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict <test inputs: @list|dir> --out results/test/ablations/abl_shuffled_small_s0/<benchmark> --model results/ablations/models/abl_shuffled_small_s0.json --ranker small --device cpu --workers 24 --no-points
```

Then evaluate with the same test evaluation script / protocol as v1 (paired vs v1 on identical structures), reporting the rows as descriptive (ablations are dev-selected, not part of the pre-registered primary comparison).
