# P2Rank retrained on our training set (fair-data control; DEV ONLY)

**What:** P2Rank 2.5.1 `traineval` with its default config (`default.groovy`: features chem/volsite/protrusion/bfactor, FastRandomForest 100 trees, default hyperparameters,
tessellation 2, positives = SAS points <= 2.6 A from relevant ligands), seed 20261026 (SEEDS[0]), on entries of our audit-filtered BioLiP2 train set (`data/processed/train/manifest.csv`, 6,475 entries).
**Training data construction** (`scripts/build_p2rank_trainset.py`): per entry the representative chain (polymer heavy atoms, deposited frame) + every pooled non-ion site ligand
(>= 5 heavy atoms = P2Rank's `min_ligand_atoms`; ions never written = `small`-equivalent) as unique HETATM residues (resnames L01.., chain Z, atom names C1..; none in P2Rank's
`ignore_het_groups`). 4,236 of 6,475 entries have >= 1 such ligand (20,022 ligands) and are the usable set; entries with only ions/tiny ligands were dropped (P2Rank trains on liganded proteins only).
**Subset:** the full 4,236-entry run (19.3M training vectors) and a 50% run were OOM-killed by the container cgroup (~79 GB RSS), so the model was trained on a **random 25% subset: 1,059 entries**
(seed 20261026; list `baselines/p2rank_ours/data/train_q.entries.txt`; 7 min on 24 threads). It is therefore *handicapped in data volume* relative to BURROW (full train set) - differences should be read as a lower bound for "P2Rank + our data".
Caveats: score transformers (probability calibration, z-score) of the stock default model are re-used by P2Rank (only affect `probability`, not ranking); eval inside traineval used 30 training entries only (not informative, model unaffected).
Files: `baselines/p2rank_ours/{train.sh,data/,train_out/,model/FastRandomForest.model,predict_ours.sh}` (git-ignored), `scripts/{build_p2rank_trainset,eval_p2rank_ours}.py`.

**Test-set prediction command (not run; for the eval agent):**
```
.venv/bin/python baselines/run_benchmarks.py --bench <ligysis|coach420_mlig|holo4k_mlig|cryptobench|pocketwatch|af_set|dimer_assembly|dimer_chains> --variant p2rank_ours --jobs 24
# equivalently, a single directory of inputs:
.venv/bin/python baselines/run_p2rank.py <inputs> -o OUT -j 24 --version 2.5.1 --model ../../../p2rank_ours/model/FastRandomForest.model
```
(`run_benchmarks.py` builds preds and scores with the project harness; record the run in results/baselines/RUNS.md automatically. Model sha256: see below.)
2e8f934943214bf749869096da068d649316c1b62fdc222c168da43cb9c9ac9a
