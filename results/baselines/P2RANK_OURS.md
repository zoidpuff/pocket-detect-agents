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

## Results (DEV ONLY; recall % [95% CI], cluster bootstrap on dev (30% clusters) / UniProt units on apo-dev)

| table | method | n_sites | pockets_mean | recall_topn | recall_topn2 | recall_ldca_topn2 | recall_max | P1K |
|---|---|---|---|---|---|---|---|---|
| dev_small | P2Rank stock | 805 | 4.7 | 51.7 [46.5, 57.4] | 58.4 [52.3, 64.6] | 47.8 [42.1, 53.6] | 63.4 [56.6, 69.6] | 41.1 [37.8, 44.8] |
| dev_small | P2Rank-ours | 805 | 3.1 | 49.8 [42.9, 56.9] | 54.9 [47.3, 62.4] | 46.3 [39.7, 52.9] | 56.9 [49.1, 64.3] | 45.9 [42.0, 50.0] |
| dev_small | BURROW-v1 small (default s0) | 805 | 41.9 | 55.2 [50.5, 59.3] | 65.0 [60.3, 69.2] | 60.7 [55.9, 65.1] | 98.9 [98.1, 99.6] | 61.3 [56.9, 67.0] |
| dev_all | P2Rank stock | 1221 | 4.2 | 47.7 [43.6, 52.3] | 54.1 [49.7, 58.8] | 36.0 [32.4, 39.8] | 58.5 [53.9, 63.4] | 46.3 [42.8, 50.2] |
| dev_all | P2Rank-ours | 1221 | 2.7 | 44.0 [38.7, 49.2] | 48.9 [43.3, 54.3] | 34.2 [30.0, 38.8] | 50.6 [44.7, 56.3] | 50.0 [46.5, 54.3] |
| dev_all | BURROW-v1 all (default s0) | 1221 | 40.9 | 56.1 [52.3, 59.9] | 65.4 [61.9, 68.9] | 55.5 [51.8, 59.3] | 99.0 [98.4, 99.5] | 69.7 [65.7, 74.2] |
| apo_cryptic_small | P2Rank stock | 1091 | 4.5 | 51.6 [48.5, 54.9] | 68.5 [65.6, 71.4] | 50.6 [47.7, 53.8] | 76.4 [73.7, 79.1] | 51.1 [48.1, 54.1] |
| apo_cryptic_small | P2Rank-ours | 1091 | 3.0 | 46.7 [43.4, 49.9] | 62.2 [59.3, 65.1] | 46.9 [43.9, 50.0] | 67.4 [64.5, 70.1] | 50.5 [47.6, 53.8] |
| apo_cryptic_small | BURROW-v1 small (default s0) | 1091 | 42.4 | 41.5 [38.0, 44.4] | 62.1 [58.8, 65.1] | 55.3 [51.7, 58.6] | 99.5 [99.1, 99.9] | 58.1 [54.6, 61.9] |
| apo_cryptic_all | P2Rank stock | 1091 | 4.5 | 51.6 [48.5, 54.9] | 68.5 [65.6, 71.4] | 50.6 [47.7, 53.8] | 76.4 [73.7, 79.1] | 51.1 [48.1, 54.1] |
| apo_cryptic_all | P2Rank-ours | 1091 | 3.0 | 46.7 [43.4, 49.9] | 62.2 [59.3, 65.1] | 46.9 [43.9, 50.0] | 67.4 [64.5, 70.1] | 50.5 [47.6, 53.8] |
| apo_cryptic_all | BURROW-v1 all (default s0) | 1091 | 42.3 | 39.2 [36.1, 42.4] | 61.5 [58.1, 64.5] | 52.8 [49.3, 56.0] | 99.7 [99.4, 100.0] | 53.9 [49.8, 57.4] |

## Paired differences (A - B, percentage points)

| table | A | B | metric | diff |
|---|---|---|---|---|
| dev_small | P2Rank-ours | P2Rank stock | recall_topn2 | -3.5 [-6.6, -0.1] |
| dev_small | P2Rank-ours | P2Rank stock | recall_ldca_topn2 | -1.5 [-4.1, +1.3] |
| dev_small | P2Rank-ours | P2Rank stock | P1K | +4.8 [+2.3, +7.2] |
| dev_small | BURROW-v1 small (default s0) | P2Rank-ours | recall_topn2 | +10.1 [+2.9, +18.7] |
| dev_small | BURROW-v1 small (default s0) | P2Rank-ours | recall_ldca_topn2 | +14.4 [+7.6, +22.3] |
| dev_small | BURROW-v1 small (default s0) | P2Rank-ours | P1K | +15.4 [+11.9, +19.4] |
| dev_small | BURROW-v1 small (default s0) | P2Rank stock | recall_topn2 | +6.6 [+0.5, +13.7] |
| dev_small | BURROW-v1 small (default s0) | P2Rank stock | recall_ldca_topn2 | +12.9 [+7.0, +19.4] |
| dev_small | BURROW-v1 small (default s0) | P2Rank stock | P1K | +20.2 [+16.5, +24.3] |
| dev_all | P2Rank-ours | P2Rank stock | recall_topn2 | -5.2 [-8.0, -2.5] |
| dev_all | P2Rank-ours | P2Rank stock | recall_ldca_topn2 | -1.8 [-4.0, +0.3] |
| dev_all | P2Rank-ours | P2Rank stock | P1K | +3.7 [+1.8, +6.1] |
| dev_all | BURROW-v1 all (default s0) | P2Rank-ours | recall_topn2 | +16.5 [+11.0, +22.6] |
| dev_all | BURROW-v1 all (default s0) | P2Rank-ours | recall_ldca_topn2 | +21.3 [+16.3, +26.8] |
| dev_all | BURROW-v1 all (default s0) | P2Rank-ours | P1K | +19.7 [+15.8, +23.4] |
| dev_all | BURROW-v1 all (default s0) | P2Rank stock | recall_topn2 | +11.4 [+6.8, +16.3] |
| dev_all | BURROW-v1 all (default s0) | P2Rank stock | recall_ldca_topn2 | +19.5 [+15.3, +24.3] |
| dev_all | BURROW-v1 all (default s0) | P2Rank stock | P1K | +23.4 [+19.8, +27.2] |
| apo_cryptic_small | P2Rank-ours | P2Rank stock | recall_topn2 | -6.2 [-8.7, -3.9] |
| apo_cryptic_small | P2Rank-ours | P2Rank stock | recall_ldca_topn2 | -3.6 [-5.6, -1.4] |
| apo_cryptic_small | P2Rank-ours | P2Rank stock | P1K | -0.6 [-2.5, +1.5] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank-ours | recall_topn2 | -0.1 [-3.3, +3.1] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank-ours | recall_ldca_topn2 | +8.1 [+5.1, +10.9] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank-ours | P1K | +7.6 [+4.4, +10.8] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank stock | recall_topn2 | -6.3 [-9.2, -3.5] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank stock | recall_ldca_topn2 | +4.5 [+1.8, +7.2] |
| apo_cryptic_small | BURROW-v1 small (default s0) | P2Rank stock | P1K | +7.0 [+4.0, +10.1] |
| apo_cryptic_all | P2Rank-ours | P2Rank stock | recall_topn2 | -6.2 [-8.7, -3.9] |
| apo_cryptic_all | P2Rank-ours | P2Rank stock | recall_ldca_topn2 | -3.6 [-5.6, -1.4] |
| apo_cryptic_all | P2Rank-ours | P2Rank stock | P1K | -0.6 [-2.5, +1.5] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank-ours | recall_topn2 | -0.7 [-3.9, +2.4] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank-ours | recall_ldca_topn2 | +5.7 [+2.8, +8.7] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank-ours | P1K | +3.4 [-0.6, +6.5] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank stock | recall_topn2 | -7.0 [-10.2, -4.0] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank stock | recall_ldca_topn2 | +2.1 [-0.8, +5.0] |
| apo_cryptic_all | BURROW-v1 all (default s0) | P2Rank stock | P1K | +2.8 [-1.1, +5.9] |
