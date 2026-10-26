# BURROW phase 2 — BioLiP2-trained models, dev results (LIGYSIS-like dev set)

Owner: method developer. Data: Track A train/dev (docs/TRAINSET.md), **audited per decision D6**
(train 6,475 entries; dev 579 representative chains, 1,221 sites of which 416 ion-only; `small`
table: 400 chains, 805 non-ion sites). Snapshot used: `results/dev/cache/trainset_snapshot/audited_d6`
(manifests identical to data/processed/{train,dev}/manifest.csv after D6; entries unchanged).
No test set was touched.

## Protocol
* Inputs: polymer-only representative chain, written once to `results/dev/biolip/inputs/<entry>.cif`
  (`scripts/burrow_biolip_prep.py`); **every method gets the same file** (BURROW, P2Rank 2.5.1
  default, fpocket 4.2.3 via the baseline wrappers).
* Ground truth: Track A pooled sites → site tables `dev_sites_all.tsv` (all sites; centre = Track A
  site centre) and `dev_sites_small.tsv` (non-ion sites; ion atoms removed; centre = centroid of
  the remaining ligand atoms). `n_residues_structure` = residues in the input file.
* Metrics (`pocketdetect.site_eval`, LIGYSIS protocol): `recall_topn2` = LIGYSIS top-N+2
  (closest pocket by DCC ≤ 12 Å, rank ≤ N+2); `recall_ldca_topn2` = DCA ≤ 4 Å (closest-by-DCA pocket,
  rank ≤ N+2; C1-strict proxy); `recall_topn`; `P1K` pooled precision of the top 1,000 pockets over
  the whole dev set (and one-to-one `P1K_1to1`). Paired cluster bootstrap over Track A `cluster30`
  (B = 1000), P1K recomputed per resample (`scripts/burrow_eval_biolip.py`).
* BURROW: trained on train (chain inputs + stored assembly-1 inputs), point scorer cross-fitted
  over 5 folds of cluster30, ranker on out-of-fold candidates; dev predicted by the final models.
  Final numbers below come from the deployable CLI (`burrow predict`, `scripts/burrow_dev_predict.sh`).

## Final dev results (audited data, 3 seeds)

### ranker `all` on dev_sites_all (1221 sites)

| method | pockets | top-N | top-N+2 (DCC12) | DCA4 top-N+2 | P1K | Δ top-N+2 vs P2Rank | Δ DCA4 | Δ P1K |
|---|---|---|---|---|---|---|---|---|
| p2rank | 4.2 | 47.7 [43.6,52.3] | 54.1 [49.7,58.8] | 36.0 [32.4,39.8] | 46.3 [42.8,50.2] | – | – | – |
| fpocket | 20.7 | 33.3 [29.2,38.1] | 43.5 [39.6,47.7] | 26.5 [23.0,30.8] | 29.5 [26.9,32.3] | -10.6 [-16.5,-3.6] | -9.5 [-14.3,-3.7] | -16.8 [-20.9,-12.7] |
| burrow_geo | 24.1 | 42.8 [38.6,47.3] | 53.3 [49.6,57.2] | 23.0 [19.4,27.5] | 32.9 [29.6,36.4] | -0.7 [-6.3,+5.5] | -13.0 [-18.3,-6.6] | -13.4 [-16.3,-10.3] |
| **BURROW default** (seeds 0, 1, 2; mean [min–max]) | 40.9 | 55.7 [55.3–56.1] | 65.0 [64.2–65.4] | 55.7 [55.5–55.8] | 70.2 [69.7–71.0] | +11.4 [+6.8,+16.3]; +10.2 [+5.8,+14.5]; +11.3 [+6.8,+15.5] | +19.5 [+15.3,+24.3]; +19.7 [+15.4,+24.5]; +19.7 [+15.6,+24.0] | +23.4 [+19.8,+27.2]; +23.6 [+20.0,+26.9]; +24.7 [+21.0,+28.0] |
| **BURROW accurate** (seeds 0, 1, 2; mean [min–max]) | 37.1 | 55.8 [55.5–56.2] | 64.8 [64.6–65.1] | 56.1 [55.8–56.2] | 70.7 [70.5–70.9] | +11.1 [+6.3,+16.0]; +10.6 [+5.8,+15.3]; +10.6 [+5.6,+15.5] | +19.7 [+15.7,+24.1]; +20.1 [+15.5,+25.2]; +20.1 [+15.9,+24.7] | +24.6 [+20.7,+28.1]; +24.3 [+20.5,+28.0]; +24.2 [+20.2,+28.4] |

### ranker `small` on dev_sites_small (805 sites)

| method | pockets | top-N | top-N+2 (DCC12) | DCA4 top-N+2 | P1K | Δ top-N+2 vs P2Rank | Δ DCA4 | Δ P1K |
|---|---|---|---|---|---|---|---|---|
| p2rank | 4.7 | 51.7 [46.5,57.4] | 58.4 [52.3,64.6] | 47.8 [42.1,53.6] | 41.1 [37.8,44.8] | – | – | – |
| fpocket | 21.9 | 34.9 [29.0,41.1] | 44.8 [39.1,49.9] | 35.2 [30.3,39.6] | 28.1 [25.2,30.7] | -13.5 [-22.3,-4.0] | -12.7 [-20.3,-4.6] | -13.0 [-17.0,-9.7] |
| all_default_s0 | 41.9 | 52.0 [47.1,56.3] | 62.9 [58.1,67.2] | 58.6 [53.6,63.0] | 58.4 [54.0,63.4] | +4.5 [-1.1,+11.3] | +10.8 [+4.9,+17.4] | +17.3 [+13.4,+21.1] |
| burrow_geo | 25.0 | 48.3 [42.8,54.0] | 59.0 [53.8,63.5] | 32.3 [27.0,37.9] | 32.0 [29.0,35.5] | +0.6 [-6.7,+8.7] | -15.5 [-24.0,-6.7] | -9.1 [-12.1,-6.1] |
| **BURROW default** (seeds 0, 1, 2; mean [min–max]) | 41.9 | 54.5 [53.8–55.2] | 64.2 [62.7–65.0] | 60.3 [59.8–60.7] | 62.0 [61.3–62.5] | +6.6 [+0.5,+13.7]; +4.3 [-1.1,+10.7]; +6.5 [+1.3,+12.4] | +12.9 [+7.0,+19.4]; +11.9 [+6.3,+17.8]; +12.7 [+7.4,+18.2] | +20.2 [+16.5,+24.3]; +21.2 [+17.4,+24.4]; +21.4 [+17.1,+24.4] |
| **BURROW accurate** (seeds 0, 1, 2; mean [min–max]) | 38.3 | 55.4 [54.4–56.5] | 65.4 [64.2–67.0] | 62.1 [61.7–62.4] | 64.4 [64.0–64.8] | +5.8 [-0.1,+12.8]; +6.7 [+0.3,+14.4]; +8.6 [+2.5,+15.3] | +14.5 [+8.6,+21.5]; +13.9 [+8.2,+20.1]; +14.4 [+8.2,+21.1] | +22.9 [+18.5,+26.5]; +23.7 [+19.2,+27.3]; +23.3 [+18.9,+26.4] |

Δ columns: paired cluster-bootstrap difference vs P2Rank 2.5.1 (95 % CI), one entry per seed.
`burrow_geo` = geometry-only BURROW (no learning; accurate feature design, buried-volume ranking).
`all_default_s0` in the `small` table = the `all` ranker scored on non-ion sites (the pre-registered
mapping uses `small` there).

**Preset decision (PI rule).** On the C1-relevant `all` sites the default preset is within 0.5 pt of
accurate on top-N+2 (65.0 vs 64.8) and DCA4 (55.7 vs 56.1; −0.4), P1K −0.5 ⇒ **default = shipped
default** (est. ≈ 80 ms/structure/core in the fast backend vs ≈ 200 ms for accurate). On non-ion sites
(`small`) accurate is meaningfully better (top-N+2 +1.2, DCA4 +1.8, P1K +2.4) ⇒ **also ship
`--preset accurate`**; pre-registered claims run on default, accurate reported alongside.
Both presets beat P2Rank on every metric on `all` with all CIs excluding 0 (top-N+2 +10.2…+11.4,
DCA4 +19.5…+20.1, P1K +23.4…+24.7); on `small` DCA4 and P1K CIs exclude 0 for every seed, top-N+2
+4.3…+8.6 with the lower CI bound slightly below 0 for 1 of 3 default seeds and 1 of 3 accurate seeds.


## Speed/accuracy sweep (ranker `all`, seed 0; exploration on the pre-audit train set, evaluated on the audited dev set)
Estimated fast-backend ms/structure/core are scaled from the fast-backend engineer's per-stage
measurements of the reference design (grid+PSP 30, rays 80, other features 15, point model 55,
post-feature 24 ms); see `scripts/burrow_sweep_summary.py`. Table: `speed_accuracy_sweep.csv`.

| setting | rays | max_len | subsample | trees | leaves | score_floor | est_ms | topn | topn2 | dca4_topn2 | P1K | P1K_1to1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | 60 | 24 | 2 | 250 | 63 | – | 204 | 55.4 | 65.2 | 56.6 | 70.9 | 58.1 |
| B | 60 | 24 | 2 | 100 | 31 | – | 167 | 55.0 | 64.9 | 55.6 | 70.3 | 57.3 |
| C | 60 | 24 | 2 | 100 | 31 | 0.7 | 155 | 56.5 | 66.1 | 55.6 | 69.0 | 57.5 |
| F1 | 30 | 24 | 2 | 100 | 31 | – | 127 | 55.9 | 65.9 | 55.2 | 70.1 | 56.8 |
| F2 | 60 | 16 | 2 | 100 | 31 | – | 140 | 55.7 | 65.4 | 55.3 | 69.7 | 57.8 |
| F3 | 60 | 24 | 3 | 100 | 31 | – | 109 | 54.2 | 64.5 | 55.0 | 68.5 | 56.4 |
| F4 | 30 | 16 | 3 | 100 | 31 | 0.7 | 72 | 56.0 | 65.8 | 55.3 | 68.6 | 56.9 |
| F5/default s0 | 30 | 16 | 3 | 250 | 63 | 0.7 | 81 | 56.1 | 65.4 | 55.5 | 69.7 | 58.0 |
| REF/accurate s0 | 60 | 24 | 2 | 250 | 63 | – | 204 | 56.2 | 65.1 | 55.8 | 70.9 | 58.4 |

(Settings REF…F4 trained on the pre-audit train set; F5/default and REF/accurate rows are the audited seed-0 models. est_ms: sub-lattice 3 keeps 60 % of the points; with the 0.7 floor 45 % of those are scored.)

Reading: no single speed lever costs more than ~1 pt top-N+2; the 100×31 point model costs
~1–1.5 pt DCA4 (B vs REF), so the default keeps the full point model and buys speed with
30 rays, 16 Å rays, the 3 Å sub-lattice and a score floor (only points with b ≥ 0.7 are scored).

## NMS sensitivity (ranker `all`, REF model, pre-audit seed-0 run, evaluated on the pre-audit 596-chain dev; `nms_sweep_burrow_all_s0.csv`)

All 21 evaluated settings (the complete grid of `scripts/burrow_nms_sweep.py`; an earlier pass of the same grid on the CHEN11/JOINED interim data and 6 CHEN11 cv5 settings are logged in results/dev/DEV_RESULTS.md):

| radius Å | overlap | min score | pockets | top-N | top-N+2 | DCA4 top-N+2 | DCA4 any top-N+2 | max recall | P1K | P1K 1:1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.3 | – | 49.9 | 40.1 | 52.1 | 47.6 | 56.5 | 94.8 | 84.5 | 53.1 |
| 4 | 0.3 | 0.05 | 9.6 | 45.3 | 59.1 | 50.2 | 56.1 | 75.2 | 84.5 | 53.1 |
| 4 | 0.3 | 0.2 | 4.1 | 48.5 | 61.5 | 48.8 | 51.3 | 64.5 | 84.5 | 53.1 |
| 6 | 0.3 | – | 49.1 | 44.8 | 56.4 | 51.2 | 57.2 | 96.1 | 82.4 | 55.4 |
| 6 | 0.3 | 0.05 | 7.7 | 50.2 | 63.7 | 52.8 | 56.1 | 75.1 | 82.4 | 55.4 |
| 6 | 0.3 | 0.2 | 3.6 | 51.9 | 63.1 | 50.4 | 51.3 | 64.4 | 82.4 | 55.4 |
| 8 | 0.3 | – | 44.8 | 50.6 | 61.1 | 54.6 | 57.9 | 97.9 | 77.7 | 57.9 |
| 8 | 0.3 | 0.05 | 6.1 | 55.9 | 67.9 | 54.7 | 56.5 | 74.8 | 77.7 | 57.9 |
| 8 | 0.3 | 0.2 | 3.3 | 54.4 | 64.4 | 51.2 | 51.5 | 64.7 | 77.7 | 57.9 |
| 10 | 0.3 | – | 36.9 | 55.6 | 65.3 | 56.2 | 57.7 | 98.5 | 72.1 | 59.5 |
| 10 | 0.3 | 0.05 | 5.0 | 60.1 | 70.6 | 55.2 | 55.7 | 74.5 | 72.1 | 59.5 |
| 10 | 0.3 | 0.2 | 3.1 | 57.7 | 65.3 | 50.8 | 51.0 | 65.6 | 72.0 | 59.4 |
| 12 | 0.3 | – | 28.6 | 59.6 | 69.7 | 56.7 | 57.4 | 97.7 | 65.3 | 58.6 |
| 12 | 0.3 | 0.05 | 4.4 | 61.5 | 71.0 | 53.8 | 53.9 | 73.1 | 65.3 | 58.6 |
| 12 | 0.3 | 0.2 | 3.1 | 59.2 | 65.7 | 49.6 | 49.6 | 65.8 | 66.0 | 59.1 |
| 14 | 0.3 | – | 21.6 | 61.8 | 72.6 | 55.8 | 55.9 | 95.6 | 60.1 | 57.7 |
| 14 | 0.3 | 0.05 | 3.9 | 61.6 | 69.3 | 51.5 | 51.5 | 71.0 | 60.1 | 57.7 |
| 14 | 0.3 | 0.2 | 3.1 | 60.1 | 66.1 | 48.9 | 48.9 | 66.1 | 60.7 | 58.2 |
| 8 | 0.1 | – | 44.5 | 51.1 | 61.1 | 54.7 | 57.9 | 97.7 | 76.5 | 57.7 |
| 8 | 0.5 | – | 44.8 | 50.5 | 61.1 | 54.8 | 58.0 | 98.0 | 77.6 | 57.7 |
| 8 | – | – | 45.2 | 49.4 | 60.7 | 54.0 | 57.5 | 97.9 | 78.2 | 57.2 |

LIGYSIS top-N+2 increases monotonically with the suppression radius (fewer, more spread pockets:
the closest-pocket rule and 12 Å tolerance reward spreading), raw P1K decreases monotonically
(small radii keep duplicate hits on the same site), DCA4 top-N+2 is flat between 10 and 12 Å and
one-to-one P1K peaks at 10 Å. **Chosen: 10 Å, no score floor** — pre-selected on CHEN11 cv5-OOF in
phase 1, maximal one-to-one P1K, within 0.5 pt of the best DCA4, and below the 12 Å evaluation
tolerance. (A 12 Å radius would add +4.4 top-N+2 on dev for −0.9 P1K_1to1 / −6.8 P1K; we do not take it.)

## Ablations (ranker `all`, REF model seed 0, pre-audit run, pre-audit 596-chain dev)
| method | pockets | top-N | top-N+2 | DCA4 top-N+2 | P1K |
|---|---|---|---|---|---|
| BURROW (own candidates) | 36.9 | 55.6 | 65.3 | 56.2 | 72.1 |
| fpocket candidates + BURROW ranker (no retraining) | 15.6 | 53.1 | 64.8 | 39.7 | 59.4 |
| fpocket candidates, fpocket order (+ BURROW NMS) | 15.6 | 35.6 | 47.7 | 27.3 | 32.5 |
| fpocket 4.2.3 (own output) | 20.7 | 32.7 | 43.1 | 25.9 | 29.4 |
| BURROW, binary ranker target (`_hit`) | 37.1 | 53.9 | 64.6 | 54.4 | 74.5 |
| P2Rank 2.5.1 | 4.2 | 47.3 | 53.6 | 35.7 | 46.6 |

The ranker carries most of the LIGYSIS (DCC 12 Å) gain — it lifts fpocket's own candidates from
47.7 to 64.8 top-N+2 — while BURROW's volume-element candidates are what make the DCA4 and P1K
gains (+16.5 DCA4, +12.7 P1K over fpocket candidates with the same ranker).
(fpocket-candidate ranker not retrained on fpocket candidates — a conservative variant.)

## Caveats
* Dev set = the selection set for NMS radius and preset; dev numbers are optimistic by the
  selection effect (small: ~10 settings compared).
* Ion-only sites are in the `all` table; P2Rank is not trained for ions — the `small` table is the
  fair comparison for P2Rank's own ligand definition.
* Speed numbers are estimates for the fast backend, not measurements.
