# BURROW phase 1 — dev ranking results (interim data: CHEN11 + JOINED, `p2rank` ligand rules)

All methods scored by the project harness (`scripts/burrow_eval_dev.py` → `pocketdetect.evaluate`),
identical cleaned inputs (`results/dev/manifests/all.csv`), ligand-averaged success, 95 % bootstrap
CI over structures (B = 1000). DCA4 = P2Rank criterion; LDCC12 = LIGYSIS-style (only the pocket
closest to the ligand centroid counts, DCC ≤ 12 Å). Tables: `results/dev/eval_chen11_to_joined.csv`,
`results/dev/eval_cv5.csv`. Reproduce: `scripts/burrow_featurize.py`, `scripts/burrow_train_dev.py
--scheme chen11|cv5 [--geo]`, `results/dev/run_final_eval.sh`.

## Fair comparison: trained on CHEN11 only (= P2Rank's training set), tested on JOINED
P2Rank 2.5.1 default model was trained on CHEN11; JOINED was used by P2Rank's authors during
development (so P2Rank is, if anything, favoured on JOINED). `joined_novel` = JOINED structures
whose MMseqs2 30 % group contains no CHEN11 structure (353/444 structures with ligands).
BURROW: 3 seeds (SEEDS[0:3]); default output (NMS 10 Å, ≤ 50 pockets) and the LDCC operating
point (NMS 12 Å, score ≥ 0.1, keep ≥ 3).

| method | subset | pockets | DCA4 top-n | DCA4 top-n+2 | LDCC12 top-n | LDCC12 top-n+2 |
|---|---|---|---|---|---|---|
| P2Rank 2.5.1 | JOINED | 4.6 | **73.5** [69.9,76.9] | 79.7 [76.1,83.0] | **80.4** [77.0,83.5] | 88.0 [85.4,90.7] |
| fpocket 4.2.3 | JOINED | 18.4 | 39.7 | 53.3 | 43.2 | 57.7 |
| BURROW geometry-only (no learning) | JOINED | 22.9 | 35.4 | 43.4 | 57.3 | 74.1 |
| BURROW s0 / s1 / s2 (default) | JOINED | 37.7 | 71.8 / 71.3 / 71.1 | **83.2 / 83.1 / 82.4** | 69.7 / 67.9 / 69.1 | 86.1 / 84.5 / 85.5 |
| BURROW s0 / s1 / s2 (LDCC point) | JOINED | 4.8 | 72.1 / 71.1 / 70.7 | 82.0 / 80.4 / 81.0 | 77.1 / 74.8 / 75.7 | **90.7 / 89.1 / 89.6** |
| P2Rank 2.5.1 | JOINED-novel | 4.7 | 71.6 | 78.5 | 78.8 | 86.8 |
| BURROW s0 (default) | JOINED-novel | 37.8 | 69.7 | 82.2 | 67.4 | 84.8 |
| BURROW s0 (LDCC point) | JOINED-novel | 4.9 | 70.7 | 80.8 | 75.1 | 89.8 |

(CIs for every row in the CSV; BURROW s0 default DCA4 top-n 71.8 [68.0,75.3], top-n+2 83.2 [79.9,86.3].)

Reading: with the same 251 training proteins, BURROW is within CI of P2Rank at DCA4 top-n
(−1.4 to −2.8 points), better at top-n+2 (+2.7 to +3.5), better at LDCC12 top-n+2 at the LDCC
operating point (+1.1 to +2.7, within CI), worse at LDCC12 top-n (−3 to −6). Unpaired CIs only;
paired bootstrap Δ not computed yet. No geometry-only variant comes close to learned ranking.

## 5-fold group CV over CHEN11 + JOINED (folds = MMseqs2 30 % groups; cross-fitted point scores)
| method | subset | DCA4 top-n | DCA4 top-n+2 | LDCC12 top-n | LDCC12 top-n+2 |
|---|---|---|---|---|---|
| P2Rank (CHEN11 = its training set → optimistic) | CHEN11 | 84.8 | 86.0 | 94.3 | 95.8 |
| BURROW cv5 s0 | CHEN11 | 61.7 | 73.2 | 57.5 | 75.2 |
| P2Rank | JOINED | 73.5 | 79.7 | 80.4 | 88.0 |
| BURROW cv5 s0 | JOINED | 73.0 | 83.6 | 70.0 | 84.8 |

The CHEN11 rows are not a fair comparison (P2Rank trained on CHEN11). More training data (cv5
uses ~630 structures vs 251) gives only +1.2 DCA4 top-n on JOINED.

## Selection log (all on dev; no test set touched)
* G1 candidate variant: 7 variants on all 974 ligands (`results/dev/G1.md`).
* Geometry-only score: 4 variants, chosen on CHEN11 (bur70sq), confirmed on JOINED.
* NMS radius / overlap / score floor: chosen on CHEN11 cv5-OOF (6 variants); an earlier
  exploration on the JOINED chen11-scheme run also looked at 10 variants (mild optimism risk for
  the JOINED post-processing numbers — the chosen setting is the CHEN11-selected one).
* Model hyperparameters: not tuned (defaults in `burrow/model.py`).
