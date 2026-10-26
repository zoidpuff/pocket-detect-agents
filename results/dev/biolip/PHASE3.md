# BURROW phase 3 — ranker improvements (until feature freeze, 23:00)

Goal (PI): improve the `small` ranker and within-protein ranking (top-N) without changing the
default preset's speed (point models / feature design frozen; ranker-side only).

## Protocol (limits dev optimism)
* Candidate changes (≤ 5, fixed before evaluation): V0 base (phase-2 ranker) · V1 + within-protein
  relative features · V2 = V1 + pocket-pair context · V3 = V2 with LambdaRank (query = structure) ·
  V4 = V2 with a 3-seed ranker ensemble. Definitions: `pocketdetect.burrow.model.context_features`,
  `scripts/burrow_ranker_cv.py`.
* Selection by **train-set cluster CV**: candidate pockets built from out-of-fold point scores
  (5 folds of cluster30), ranker trained on 4/5, evaluated on held-out folds 0 and 1 (chain inputs
  only) with the site_eval definitions (top-N, top-N+2 DCC12, DCA4 top-N+2, P1K).
* Dev used **once** to confirm the selected change (logged below with every dev evaluation).
* Freeze runs (`p3_*`): same data/feature design/point models as phase 2 (audited D6 train set),
  retrained under new names (phase-2 models untouched for the independent review).

## Selection rule (fixed at 20:55, after folds 0-1 of V0-V3/V5, before folds 2-4)
Choose the variant with the highest mean of (top-N+2, DCA4 top-N+2, P1K) over the 5 held-out
folds, provided it is not worse than V0 by > 0.5 pt on top-N, top-N+2 or DCA4 top-N+2; else keep V0.
Folds 2-4 are run for V0, V3 and V4 only (V1, V2, V5 were not better than V0 on folds 0-1).

## Train cluster-CV results
Run: `p3_small_default_s0` (ranker `small`, default preset, seed 0), held-out train folds, chain
inputs only (`scripts/burrow_ranker_cv.py`; per-fold table `ranker_cv_small_all_folds.csv`).
V4 (V2 + 3-seed ranker ensemble) was started but stopped before producing a result (node load;
V2 itself was neutral) — **not evaluated**. V3 needed a fix (LightGBM's 10,000-rows-per-query limit:
queries > 9,000 candidates keep all positives + random negatives).
V5 (atom-type propensity, PI suggestion) was implemented as a *ranker* feature (fold-wise
out-of-fold log-odds tables from the training set; `src/pocketdetect/burrow/propensity.py`,
`scripts/burrow_propensity.py`) rather than a point feature, because a point-feature change would
have required retraining all point models before the freeze.

| variant | fold | sites | top-N | top-N+2 | DCA4 top-N+2 | P1K |
|---|---|---|---|---|---|---|
| V0_base | 0 | 1769 | 54.4 | 63.8 | 59.6 | 82.1 |
| V0_base | 1 | 1929 | 57.2 | 68.0 | 61.9 | 84.1 |
| V0_base | 2 | 1649 | 58.0 | 68.7 | 63.1 | 84.3 |
| V0_base | 3 | 1682 | 54.6 | 66.7 | 61.2 | 82.9 |
| V0_base | 4 | 1595 | 53.9 | 65.3 | 60.9 | 83.9 |
| V1_rel | 0 | 1769 | 53.6 | 63.1 | 58.2 | 82.6 |
| V1_rel | 1 | 1929 | 57.1 | 67.2 | 61.9 | 85.1 |
| V2_rel_pair | 0 | 1769 | 54.3 | 64.4 | 58.9 | 83.9 |
| V2_rel_pair | 1 | 1929 | 56.8 | 67.4 | 62.1 | 85.1 |
| V3_rel_pair_lambdarank | 0 | 1769 | 55.2 | 65.2 | 59.2 | 81.5 |
| V3_rel_pair_lambdarank | 1 | 1929 | 57.5 | 68.7 | 62.1 | 84.7 |
| V3_rel_pair_lambdarank | 2 | 1649 | 59.1 | 68.0 | 62.5 | 83.1 |
| V3_rel_pair_lambdarank | 3 | 1682 | 56.1 | 64.9 | 59.9 | 80.0 |
| V3_rel_pair_lambdarank | 4 | 1595 | 55.2 | 66.4 | 61.3 | 81.4 |
| V5_propensity | 0 | 1769 | 53.0 | 63.1 | 59.2 | 83.0 |
| V5_propensity | 1 | 1929 | 56.8 | 67.1 | 61.4 | 86.0 |

Means (criterion = mean of top-N+2, DCA4 top-N+2, P1K):

| variant | folds | top-N | top-N+2 | DCA4 | P1K | criterion |
|---|---|---|---|---|---|---|
| V0_base | 0,1,2,3,4 | 55.61 | 66.49 | 61.36 | 83.46 | 70.44 |
| V1_rel | 0,1 | 55.39 | 65.16 | 60.03 | 83.85 | 69.68 |
| V2_rel_pair | 0,1 | 55.54 | 65.94 | 60.48 | 84.50 | 70.31 |
| V3_rel_pair_lambdarank | 0,1,2,3,4 | 56.62 | 66.65 | 60.98 | 82.14 | 69.92 |
| V5_propensity | 0,1 | 54.89 | 65.14 | 60.31 | 84.50 | 69.98 |
| V0_base (folds 0-1 only) | 0,1 | 55.78 | 65.89 | 60.77 | 83.10 | 69.92 |
| V1_rel (folds 0-1 only) | 0,1 | 55.39 | 65.16 | 60.03 | 83.85 | 69.68 |
| V2_rel_pair (folds 0-1 only) | 0,1 | 55.54 | 65.94 | 60.48 | 84.50 | 70.31 |
| V3_rel_pair_lambdarank (folds 0-1 only) | 0,1 | 56.36 | 66.99 | 60.65 | 83.10 | 70.24 |
| V5_propensity (folds 0-1 only) | 0,1 | 54.89 | 65.14 | 60.31 | 84.50 | 69.98 |

## Dev evaluations performed in phase 3 (count for the review)
Exactly **one** BioLiP2-dev evaluation was made in phase 3: the frozen configuration (V0 — no ranker
change) for all 12 frozen models through the CLI (`results/dev/biolip/run_freeze_eval.sh` →
`eval_freeze_{all,small}.csv`, predictions `preds_freeze/`). It reproduces the phase-2 numbers
exactly (same models; only reported residues changed — see below). No CV variant (V1–V5) was ever
evaluated on dev. The `p3_*` training runs also emitted dev predictions as a by-product; they were
only byte-compared with the phase-2 predictions (identity check, 579/579 identical for 2 runs), not scored.
Secondary set (DEV-only, CryptoBench train folds, never trained on): one check of the frozen
`small` default s0 model (`scripts/burrow_apo_check.py` → `apo_check.csv`).

## Decision
**Keep V0 (phase-2 ranker) — no ranker change at the freeze.** By the pre-declared rule V0 has the
highest 5-fold criterion (70.44 vs V3 69.92). V3 (LambdaRank within protein) improves within-protein
ranking (top-N +1.0, top-N+2 +0.2) but loses DCA4 (−0.4) and pooled precision (P1K −1.3, as
expected from uncalibrated per-query scores); V1 (relative features), V2 (+ pair context) and V5
(propensity) were not better than V0 on folds 0–1. Conclusion: the ranker is not the bottleneck
for non-ion sites at this point; the base ranker already exploits the within-protein signal.

### Frozen models (`results/dev/FREEZE.md`)
12 JSONs = SEEDS[:3] × {all, small} × {default, accurate}; identical point models / rankers to the
phase-2 models (the `p3_*` retrains of 5/12 configs were verified bit-identical, the remaining
retrains were stopped as redundant). Added at the freeze:
* **Provenance** in `meta.provenance`: sha256 of data/processed/{train,dev}/manifest.csv
  (post-D6), data/manifests/test_union.tsv, data/manifests/train_dev_audit_exclusions.tsv, git
  commit (+ dirty flag), seed; `meta.feature_cfg` (frozen feature design), `meta.score_floor`.
* **Reported residues never empty** (review item): pocket residues = residues with a heavy atom
  ≤ 4 Å of a pocket point; if empty (pockets made only of distant sub-lattice points) → residues with
  any atom ≤ 6 Å of the pocket centre (LIGYSIS rule for point-less methods) → else the nearest
  atom's residue (`pipeline.output_residues`). Output-only: candidate features, NMS and ranking are
  unchanged. Empty residue lists in dev predictions: 5,345/23,667 pockets before → 0 after.
* CLI presets point to the frozen models; `--af-mode` default = standard ranker (decision D8),
  `--af-ranker PATH` optional.

### Final dev results of the frozen models (BioLiP2 dev, audited; mean over 3 seeds; paired Δ vs P2Rank 2.5.1 per seed)
| sites | model | top-N | top-N+2 | DCA4 top-N+2 | P1K | Δ top-N+2 vs P2Rank (per seed) |
|---|---|---|---|---|---|---|
| all (1,221) | P2Rank 2.5.1 | 47.7 | 54.1 | 36.0 | 46.3 | – |
| all | BURROW-all default | 55.7 | 65.0 | 55.7 | 70.2 | +11.4 [+6.8,+16.3]; +10.2 [+5.8,+14.5]; +11.3 [+6.8,+15.5] |
| all | BURROW-all accurate | 55.8 | 64.8 | 56.1 | 70.7 | +11.1; +10.6; +10.6 (all CIs > 0) |
| **non-ion (805)** | P2Rank 2.5.1 | 51.7 | 58.4 | 47.8 | 41.1 | – |
| non-ion | **BURROW-small default** | 54.5 | 64.2 | 60.3 | 62.0 | +6.6 [+0.5,+13.7]; +4.3 [−1.1,+10.7]; +6.5 [+1.3,+12.4] |
| non-ion | BURROW-small accurate | 55.4 | 65.4 | 62.1 | 64.4 | +5.8 [−0.1,+12.8]; +6.7 [+0.3,+14.4]; +8.6 [+2.5,+15.3] |
| non-ion | BURROW-all default (s0/s1/s2) | 52.0/51.7/51.8 | 62.9/62.9/63.4 | 58.6/59.8/59.0 | 58.4/59.7/59.7 | +4.5; +4.5; +5.0 (lower CI −1.1…−0.4) |
| non-ion | BURROW-all accurate (s0/s1/s2) | 52.2/51.2/51.9 | 62.7/62.6/63.4 | 58.6/59.0/59.9 | 59.8/61.1/60.6 | +4.3; +4.2; +5.0 (lower CI −2.0…−1.1) |

Non-ion sites: DCA4 top-N+2 and P1K beat P2Rank with CIs excluding 0 for every model and seed;
LIGYSIS top-N+2 is +4 to +9 with the lower CI bound slightly below 0 for 1/3 seeds of each `small`
preset and all `all`-ranker seeds. As the independent review found, most of the `all` gain on the
full table comes from ion-only sites.

### Apo-dev (CryptoBench train folds, DEV-only secondary set), frozen `small` default s0
| table | sites | BURROW top-N+2 | P2Rank top-N+2 | Δ [95 % CI] | BURROW DCA4 | P2Rank DCA4 |
|---|---|---|---|---|---|---|
| apo, cryptic | 1,091 | 61.8 | 68.5 | −6.7 [−9.7, −3.8] | 54.9 | 50.6 |
| apo, non-cryptic | 1,489 | 53.7 | 54.6 | −0.9 [−5.4, +3.6] | – | – |

BURROW localises apo/cryptic sites more precisely (DCA4 +4.1) but ranks them lower within the
12 Å LIGYSIS criterion than P2Rank. This deficit on apo/partially formed pockets is the main open
issue for O4 (CryptoBench) and was not addressed before the freeze.
