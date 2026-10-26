# BURROW ablations (pre-registered, DESIGN §2.5 / §5) — dev only

All models: FREEZE v1 default-preset recipe (30 rays, 6/10/16 Å, 3 Å sub-lattice, score floor 0.7, 250x63 point model, graded ranker 300 rounds), seed SEEDS[0], trained by `scripts/ablation_train.py` with the frozen code (worktree at 5e13267); predictions with the deployable fast CLI (CPU). Each row is paired against the frozen v1 seed-0 model of the same variant (`burrow_{small,all}_default_s0`) on the same structures: Δ = ablation − v1, percentage points, 95 % paired cluster-bootstrap CI (BioLiP dev: MMseqs2-30 % clusters; apo-dev: UniProt units; 1,000 resamples). DCC12 = closest-pocket centre within 12 Å of the site centre among the top-N(+2); DCA4 = closest-by-DCA pocket within 4 Å of a ligand atom; P1K = pooled precision of the top-1 pocket. Dev-only numbers: the test-set rows for A1–A3 are prepared in `results/ablations/TEST_COMMANDS.md` and are run by the eval agent after the final version is fixed.

**Reproducibility check:** re-training v1 `small` with this script (ablation `none`, 16 LightGBM threads, frozen code) gives a point model and ranker that are byte-identical to the frozen `burrow_small_default_s0.json` (model strings equal), and its predictions on BioLiP dev equal the method developer's frozen predictions. The ablation models were trained with 12 LightGBM threads (2 lanes within the 24-worker budget; `deterministic=True`). Thread count does not change the trees: re-training `none` and 5 ablations (shuffled, fold-disjoint, CHEN11, no-rays, no-hitchem; `small`) with 16 threads gave model strings that differ only in the `[num_threads: N]` parameter line, and the predictions are byte-identical (row `v1 re-trained (sanity) (12 LightGBM threads)`).

**Notes:** A1 keeps 4,839 of 8,184 training inputs (entries with Foldseek max(qTM, tTM) ≥ 0.5 to any test chain removed, `data/processed/train/fold_to_test.tsv`). A2 uses the 251 CHEN11 structures with P2Rank-rule ligands (no ions), so the `small` and `all` CHEN11 models are identical (same trees; only `meta.variant` differs) — their apo-dev rows coincide, and their BioLiP dev rows differ only because the `all` site table contains ion sites. A3 permutes point labels and pocket targets within each protein (OOF scoring and candidate formation as in v1).

Feature groups dropped in A4 (point model and the matching pocket descriptors of the ranker; candidate generation is unchanged because it is not learned):

- `rays`: point features `^(bur(6|10|16|24)|depth)$`; ranker descriptors `^(bur(10|16|24)_(mean|q90|q10)|depth_mean|geo_strength)$`
- `hitchem`: point features `^(hit_\w+|escape)$`; ranker descriptors `^(hit_\w+|escape)$`
- `bbbur`: point features `^burbb\d+$`; ranker descriptors `^burbb\d+_mean$`
- `metal`: point features `^(metal_cand|n_coord28|coord_spread|(n4|f8|f12)_coord)$`; ranker descriptors `^(metal_n|n_coord_max|f8_coord)$`
- `rescomp`: point features `^(n4|f8|f12)_rg_\w+$`; ranker descriptors `^(n4|f8)_rg_\w+$`
- `atomclass`: point features `^(n4|f8|f12)_(C|N|O|S|aromatic|hydrophobic_sc_C|donor|acceptor|backbone|sidechain|positive|negative)$`; ranker descriptors `^f8_(C|N|O|S|aromatic|hydrophobic_sc_C|donor|acceptor|backbone|sidechain|positive|negative)$`


## BioLiP dev

| ablation | variant | train inputs | DCC12 top-N+2: value (Δ vs v1 [95% CI]) | DCC12 top-N: value (Δ vs v1 [95% CI]) | DCA4 top-N+2: value (Δ vs v1 [95% CI]) | P1K: value (Δ vs v1 [95% CI]) |
|---|---|---|---|---|---|---|
| A2 CHEN11-only training | all | 251 | 54.6 (-10.8 [-14.2, -7.6]) | 44.2 (-11.9 [-15.7, -8.1]) | 39.9 (-15.6 [-19.6, -12.1]) | 60.5 (-9.2 [-13.2, -6.0]) |
| A2 CHEN11-only training | small | 251 | 61.7 (-3.2 [-5.9, -0.5]) | 49.9 (-5.2 [-8.9, -1.9]) | 54.4 (-6.3 [-9.5, -3.0]) | 55.2 (-6.1 [-9.8, -3.3]) |
| A4 − atom-class densities | all | 8184 | 63.7 (-1.7 [-3.5, -0.2]) | 53.9 (-2.2 [-4.4, -0.3]) | 54.3 (-1.2 [-3.2, +0.7]) | 67.6 (-2.1 [-4.6, +0.1]) |
| A4 − atom-class densities | small | 8184 | 63.9 (-1.1 [-3.5, +1.4]) | 52.4 (-2.7 [-5.2, -0.7]) | 59.1 (-1.6 [-4.0, +1.0]) | 59.7 (-1.6 [-4.4, +0.3]) |
| A4 − backbone-only buriedness | all | 8184 | 65.0 (-0.4 [-1.9, +1.1]) | 56.3 (+0.2 [-1.3, +1.8]) | 56.4 (+0.9 [-0.6, +2.6]) | 70.4 (+0.7 [-1.2, +2.0]) |
| A4 − backbone-only buriedness | small | 8184 | 64.8 (-0.1 [-1.7, +1.4]) | 55.0 (-0.1 [-2.0, +1.6]) | 60.0 (-0.7 [-2.8, +1.0]) | 62.7 (+1.4 [-1.2, +2.7]) |
| A4 − first-hit chemistry | all | 8184 | 65.2 (-0.2 [-1.9, +1.5]) | 54.4 (-1.7 [-3.5, +0.1]) | 55.2 (-0.3 [-2.0, +1.3]) | 70.0 (+0.3 [-1.2, +2.0]) |
| A4 − first-hit chemistry | small | 8184 | 63.6 (-1.4 [-3.4, +0.8]) | 54.4 (-0.7 [-2.9, +1.3]) | 59.6 (-1.1 [-3.2, +1.3]) | 61.2 (-0.1 [-2.3, +1.4]) |
| A4 − metal features | all | 8184 | 62.9 (-2.5 [-4.4, -0.7]) | 54.1 (-2.0 [-3.6, -0.3]) | 54.6 (-0.9 [-2.5, +0.8]) | 69.3 (-0.4 [-2.3, +1.5]) |
| A4 − metal features | small | 8184 | 64.0 (-1.0 [-3.0, +0.9]) | 54.4 (-0.7 [-2.8, +1.0]) | 59.8 (-1.0 [-2.8, +0.8]) | 60.8 (-0.5 [-2.6, +0.9]) |
| A4 − ray buriedness | all | 8184 | 64.6 (-0.8 [-2.4, +0.7]) | 55.0 (-1.1 [-2.5, +0.3]) | 55.5 (+0.0 [-1.7, +2.0]) | 70.8 (+1.1 [-1.0, +2.6]) |
| A4 − ray buriedness | small | 8184 | 64.3 (-0.6 [-2.3, +1.3]) | 54.2 (-1.0 [-3.2, +1.1]) | 59.5 (-1.2 [-3.3, +0.8]) | 61.8 (+0.5 [-2.0, +2.0]) |
| A4 − residue composition | all | 8184 | 65.1 (-0.3 [-2.0, +1.4]) | 56.0 (-0.1 [-1.7, +1.5]) | 55.0 (-0.6 [-2.1, +1.0]) | 69.4 (-0.3 [-2.4, +1.6]) |
| A4 − residue composition | small | 8184 | 63.9 (-1.1 [-3.3, +1.4]) | 53.2 (-2.0 [-4.4, +0.4]) | 58.4 (-2.4 [-4.8, +0.0]) | 59.9 (-1.4 [-4.0, +0.1]) |
| A1 fold-disjoint training | all | 4839 | 65.1 (-0.3 [-2.0, +1.3]) | 55.2 (-0.9 [-2.3, +0.6]) | 55.5 (+0.0 [-1.7, +1.8]) | 68.3 (-1.4 [-3.6, +0.4]) |
| A1 fold-disjoint training | small | 4839 | 64.3 (-0.6 [-2.4, +1.3]) | 53.9 (-1.2 [-3.4, +0.8]) | 59.5 (-1.2 [-3.5, +1.2]) | 59.2 (-2.1 [-4.8, -0.5]) |
| v1 re-trained (sanity) | small | 8184 | 65.0 (+0.0 [+0.0, +0.0]) | 55.2 (+0.0 [+0.0, +0.0]) | 60.7 (+0.0 [+0.0, +0.0]) | 61.3 (+0.0 [+0.0, +0.0]) |
| v1 re-trained (sanity) (12 LightGBM threads) | small | 8184 | 65.0 (+0.0 [+0.0, +0.0]) | 55.2 (+0.0 [+0.0, +0.0]) | 60.7 (+0.0 [+0.0, +0.0]) | 61.3 (+0.0 [+0.0, +0.0]) |
| A3 shuffled labels (negative control) | all | 8184 | 29.8 (-35.6 [-40.5, -31.4]) | 22.7 (-33.4 [-38.8, -28.8]) | 20.7 (-34.8 [-39.0, -30.9]) | 29.6 (-40.1 [-46.0, -35.5]) |
| A3 shuffled labels (negative control) | small | 8184 | 36.8 (-28.2 [-32.7, -23.7]) | 28.3 (-26.8 [-31.7, -22.4]) | 30.2 (-30.6 [-35.3, -26.2]) | 27.9 (-33.4 [-39.6, -28.3]) |

v1 reference values: all: DCC12 top-N+2 65.4, DCC12 top-N 56.1, DCA4 top-N+2 55.5, P1K 69.7; small: DCC12 top-N+2 65.0, DCC12 top-N 55.2, DCA4 top-N+2 60.7, P1K 61.3 (1221 scored sites).

## apo-dev cryptic

| ablation | variant | train inputs | DCC12 top-N+2: value (Δ vs v1 [95% CI]) | DCC12 top-N: value (Δ vs v1 [95% CI]) | DCA4 top-N+2: value (Δ vs v1 [95% CI]) | P1K: value (Δ vs v1 [95% CI]) |
|---|---|---|---|---|---|---|
| A2 CHEN11-only training | all | 251 | 61.7 (+0.2 [-2.4, +2.8]) | 41.1 (+1.8 [-0.9, +4.5]) | 52.5 (-0.3 [-2.6, +2.2]) | 58.3 (+4.4 [+1.5, +7.6]) |
| A2 CHEN11-only training | small | 251 | 61.7 (-0.5 [-2.8, +2.1]) | 41.1 (-0.5 [-3.1, +2.3]) | 52.5 (-2.7 [-5.1, -0.3]) | 58.3 (+0.2 [-2.9, +2.6]) |
| A4 − atom-class densities | all | 8184 | 61.0 (-0.5 [-3.0, +2.0]) | 38.6 (-0.6 [-2.9, +1.5]) | 51.4 (-1.4 [-3.6, +0.9]) | 53.4 (-0.5 [-2.7, +1.7]) |
| A4 − atom-class densities | small | 8184 | 61.8 (-0.4 [-2.7, +1.9]) | 41.2 (-0.3 [-2.5, +1.7]) | 53.1 (-2.1 [-4.4, +0.2]) | 59.2 (+1.1 [-1.2, +3.1]) |
| A4 − backbone-only buriedness | all | 8184 | 59.9 (-1.6 [-3.4, +0.4]) | 37.3 (-1.9 [-3.7, -0.2]) | 52.1 (-0.6 [-2.3, +1.1]) | 53.8 (-0.1 [-1.6, +1.7]) |
| A4 − backbone-only buriedness | small | 8184 | 62.9 (+0.7 [-1.3, +2.5]) | 42.6 (+1.1 [-0.7, +2.9]) | 56.2 (+0.8 [-0.9, +2.6]) | 59.9 (+1.8 [+0.1, +2.8]) |
| A4 − first-hit chemistry | all | 8184 | 59.6 (-1.9 [-4.0, +0.0]) | 37.9 (-1.3 [-3.4, +0.6]) | 52.1 (-0.7 [-2.9, +1.4]) | 53.3 (-0.6 [-2.3, +1.2]) |
| A4 − first-hit chemistry | small | 8184 | 64.5 (+2.4 [+0.3, +4.5]) | 40.8 (-0.7 [-3.0, +1.5]) | 54.7 (-0.5 [-2.5, +1.4]) | 59.2 (+1.1 [-1.0, +2.6]) |
| A4 − metal features | all | 8184 | 60.0 (-1.5 [-3.5, +0.4]) | 39.1 (-0.1 [-2.0, +1.7]) | 53.0 (+0.2 [-1.5, +2.0]) | 53.6 (-0.3 [-2.4, +1.3]) |
| A4 − metal features | small | 8184 | 61.7 (-0.5 [-2.3, +1.4]) | 42.3 (+0.8 [-1.2, +2.7]) | 54.8 (-0.5 [-2.3, +1.3]) | 58.4 (+0.3 [-1.3, +1.9]) |
| A4 − ray buriedness | all | 8184 | 59.9 (-1.6 [-3.5, +0.5]) | 38.4 (-0.8 [-2.7, +1.0]) | 52.5 (-0.3 [-2.1, +1.5]) | 53.6 (-0.3 [-1.8, +1.6]) |
| A4 − ray buriedness | small | 8184 | 62.2 (+0.1 [-1.9, +2.0]) | 42.2 (+0.6 [-1.4, +2.6]) | 55.2 (-0.1 [-1.9, +1.8]) | 58.5 (+0.4 [-1.6, +1.8]) |
| A4 − residue composition | all | 8184 | 60.9 (-0.6 [-3.1, +1.7]) | 40.3 (+1.1 [-1.2, +3.4]) | 51.9 (-0.9 [-3.0, +1.1]) | 53.3 (-0.6 [-2.4, +1.6]) |
| A4 − residue composition | small | 8184 | 63.9 (+1.7 [-0.5, +4.0]) | 43.4 (+1.8 [-0.5, +3.9]) | 56.0 (+0.6 [-1.6, +2.9]) | 57.5 (-0.6 [-2.9, +1.1]) |
| A1 fold-disjoint training | all | 4839 | 59.7 (-1.8 [-3.9, +0.2]) | 37.5 (-1.7 [-4.0, +0.2]) | 51.9 (-0.9 [-3.0, +1.2]) | 50.7 (-3.2 [-4.8, -0.7]) |
| A1 fold-disjoint training | small | 4839 | 61.2 (-0.9 [-3.1, +1.2]) | 41.2 (-0.4 [-2.4, +1.8]) | 54.4 (-0.8 [-2.8, +1.0]) | 56.7 (-1.4 [-3.5, +0.3]) |
| v1 re-trained (sanity) | small | 8184 | 62.1 (+0.0 [+0.0, +0.0]) | 41.5 (+0.0 [+0.0, +0.0]) | 55.3 (+0.0 [+0.0, +0.0]) | 58.1 (+0.0 [+0.0, +0.0]) |
| v1 re-trained (sanity) (12 LightGBM threads) | small | 8184 | 62.1 (+0.0 [+0.0, +0.0]) | 41.5 (+0.0 [+0.0, +0.0]) | 55.3 (+0.0 [+0.0, +0.0]) | 58.1 (+0.0 [+0.0, +0.0]) |
| A3 shuffled labels (negative control) | all | 8184 | 33.3 (-28.2 [-31.9, -24.5]) | 19.2 (-20.1 [-23.5, -16.7]) | 26.5 (-25.3 [-28.9, -21.8]) | 21.1 (-32.8 [-36.8, -28.3]) |
| A3 shuffled labels (negative control) | small | 8184 | 33.8 (-28.3 [-31.6, -25.0]) | 20.3 (-21.2 [-24.3, -18.0]) | 27.4 (-26.8 [-29.9, -23.7]) | 20.8 (-37.3 [-42.1, -33.8]) |

v1 reference values: all: DCC12 top-N+2 61.5, DCC12 top-N 39.2, DCA4 top-N+2 52.8, P1K 53.9; small: DCC12 top-N+2 62.1, DCC12 top-N 41.5, DCA4 top-N+2 55.3, P1K 58.1 (1091 scored sites).

## apo-dev non-cryptic

| ablation | variant | train inputs | DCC12 top-N+2: value (Δ vs v1 [95% CI]) | DCC12 top-N: value (Δ vs v1 [95% CI]) | P1K: value (Δ vs v1 [95% CI]) |
|---|---|---|---|---|---|
| A2 CHEN11-only training | all | 251 | 54.3 (+1.3 [-1.1, +3.8]) | 45.9 (+2.2 [-0.4, +4.8]) | 54.5 (-2.2 [-5.2, +0.7]) |
| A2 CHEN11-only training | small | 251 | 54.3 (+0.5 [-1.7, +2.8]) | 45.9 (-0.5 [-3.1, +1.9]) | 54.5 (-3.7 [-6.9, -1.5]) |
| A4 − atom-class densities | all | 8184 | 53.2 (+0.3 [-1.4, +1.8]) | 43.9 (+0.3 [-1.5, +2.0]) | 53.8 (-2.9 [-5.0, -0.9]) |
| A4 − atom-class densities | small | 8184 | 54.3 (+0.5 [-1.6, +2.7]) | 45.5 (-0.9 [-3.3, +1.4]) | 56.4 (-1.8 [-4.5, -0.1]) |
| A4 − backbone-only buriedness | all | 8184 | 54.5 (+1.6 [+0.1, +3.2]) | 44.4 (+0.7 [-1.0, +2.5]) | 56.6 (-0.1 [-1.8, +1.5]) |
| A4 − backbone-only buriedness | small | 8184 | 54.1 (+0.3 [-1.4, +2.1]) | 45.5 (-0.9 [-2.6, +0.7]) | 58.4 (+0.2 [-2.0, +1.2]) |
| A4 − first-hit chemistry | all | 8184 | 53.9 (+1.0 [-0.8, +2.8]) | 44.5 (+0.9 [-0.8, +2.5]) | 56.4 (-0.3 [-2.2, +1.4]) |
| A4 − first-hit chemistry | small | 8184 | 56.4 (+2.6 [+0.8, +4.6]) | 47.0 (+0.6 [-1.2, +2.4]) | 58.2 (+0.0 [-2.3, +1.6]) |
| A4 − metal features | all | 8184 | 53.7 (+0.8 [-0.9, +2.6]) | 44.7 (+1.0 [-0.7, +2.7]) | 55.4 (-1.3 [-2.9, +0.3]) |
| A4 − metal features | small | 8184 | 55.2 (+1.4 [-0.2, +3.1]) | 46.4 (+0.0 [-1.8, +1.8]) | 57.9 (-0.3 [-2.5, +1.0]) |
| A4 − ray buriedness | all | 8184 | 54.0 (+1.1 [-0.3, +2.5]) | 44.3 (+0.7 [-0.9, +2.3]) | 56.2 (-0.5 [-2.1, +1.3]) |
| A4 − ray buriedness | small | 8184 | 54.1 (+0.3 [-1.2, +1.8]) | 45.5 (-0.9 [-2.7, +0.9]) | 59.1 (+0.9 [-1.2, +2.3]) |
| A4 − residue composition | all | 8184 | 53.3 (+0.3 [-1.4, +2.1]) | 44.9 (+1.3 [-0.5, +3.1]) | 54.7 (-2.0 [-4.1, +0.2]) |
| A4 − residue composition | small | 8184 | 55.8 (+2.0 [+0.1, +4.1]) | 47.3 (+0.9 [-1.1, +2.9]) | 57.2 (-1.0 [-3.6, +0.8]) |
| A1 fold-disjoint training | all | 4839 | 54.4 (+1.5 [-0.1, +3.2]) | 44.0 (+0.3 [-1.2, +2.1]) | 54.2 (-2.5 [-4.0, -0.2]) |
| A1 fold-disjoint training | small | 4839 | 55.4 (+1.6 [+0.0, +3.3]) | 45.7 (-0.7 [-2.7, +1.2]) | 56.0 (-2.2 [-4.4, -0.7]) |
| v1 re-trained (sanity) | small | 8184 | 53.8 (+0.0 [+0.0, +0.0]) | 46.4 (+0.0 [+0.0, +0.0]) | 58.2 (+0.0 [+0.0, +0.0]) |
| v1 re-trained (sanity) (12 LightGBM threads) | small | 8184 | 53.8 (+0.0 [+0.0, +0.0]) | 46.4 (+0.0 [+0.0, +0.0]) | 58.2 (+0.0 [+0.0, +0.0]) |
| A3 shuffled labels (negative control) | all | 8184 | 36.1 (-16.8 [-20.6, -13.6]) | 28.0 (-15.6 [-19.3, -12.6]) | 26.7 (-30.0 [-34.6, -25.4]) |
| A3 shuffled labels (negative control) | small | 8184 | 37.3 (-16.5 [-20.6, -13.0]) | 28.8 (-17.6 [-21.9, -14.0]) | 26.6 (-31.6 [-37.0, -27.9]) |

v1 reference values: all: DCC12 top-N+2 52.9, DCC12 top-N 43.7, P1K 56.7; small: DCC12 top-N+2 53.8, DCC12 top-N 46.4, P1K 58.2 (1489 scored sites).
