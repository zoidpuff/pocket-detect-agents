# BURROW v2 research cycle — apo / partially-formed (cryptic) pocket ranking

Started 2026-10-26 21:30 after FREEZE v1 (commit 5e13267; models results/dev/freeze/, untouched).

## Gate for adopting v2 (fixed by the PI before any v2 result)
`small` default preset, vs v1 (`burrow_small_default_s0` frozen), paired:
* apo-dev cryptic top-N+2 (DCC12, closest-pocket protocol) **≥ +3.0 pt with paired CI > 0**;
* BioLiP2-dev (small sites) top-N+2 and DCA4 top-N+2 each ≥ v1 − 0.5; P1K ≥ v1 − 1;
* the same check for `all` must not be worse; default-preset speed must not regress.
Dev sets: BioLiP2 dev (primary guard), apo-dev = CryptoBench TRAIN folds (target; never trained on),
AF dev (secondary guard; prune_soft input variant). Evaluation driver: `scripts/burrow_v2_eval.py`
(every evaluated config appends to `v2_eval.csv`).

## Variants (≤ 5, fixed at 21:35)
* **H1** AF models of training proteins as extra training inputs (`<entry>__af`; data/processed/af_train
  train split, 2,844 proteins with transferred pooled sites; low-pLDDT spaghetti pruned exactly as
  `--af-mode`; no soft-occlusion weights; pLDDT never a feature) — point model and ranker.
* **H2** side-chain dropout copies of every training chain (`<entry>__scd`): side-chain atoms beyond
  Cβ deleted with p = 0.5 for residues ≤ 8 Å of a pooled ligand atom and p = 0.2 elsewhere
  (deterministic per entry) — point model and ranker.
* **H1+H2** both.
* **H3** ranker-only apo-oriented pocket aggregates (`model.APO_NAMES`: backbone+Cβ buriedness means
  and quantiles, side-chain enclosure deficit, fraction of backbone-enclosed points, mouth openness)
  — first on the v1 point model, then stacked on the best of H1/H2/H1+H2.
* **H4** apo PDB chains of training proteins — only if data/processed/apo_train/ lands in time.
All variants keep the frozen default feature design (no speed change).

## Evaluations
All rows: `v2_eval.csv` (paired vs frozen v1 `small` default s0; v1 self-comparison rows = 0 as a
sanity check). Sets: biolip_dev_small (805 non-ion sites), apo_cryptic (1,091) / apo_noncryptic
(1,489) = CryptoBench train-fold apo structures, apo_track_dev_small (Track A2 apo chains of the
160 dev proteins, 260 non-ion sites), af_dev_all (AF-dev models, prune_soft, all sites).

v1 reference (small default s0): BioLiP top-N+2 65.0 / DCA4 60.7 / P1K 61.3; apo cryptic 61.8
(DCA4 54.9); apo non-cryptic 53.7; apo-track-dev 61.5; AF dev 58.8.

| variant | set | top-N | Δ | top-N+2 | Δ [95 % CI] | DCA4 top-N+2 | Δ | P1K | Δ |
|---|---|---|---|---|---|---|---|---|---|
| H3_small_default_s0 | biolip_dev_small | 53.5 | -1.6 [-3.4,+0.1] | 63.4 | -1.6 [-3.6,+0.3] | 60.4 | -0.4 [-1.9,+1.3] | 61.9 | +0.6 [-1.1,+1.8] |
| H3_small_default_s0 | apo_cryptic | 41.6 | +0.4 [-1.1,+1.8] | 62.1 | +0.3 [-1.4,+1.8] | 55.2 | +0.3 [-1.3,+1.8] | 58.8 | +0.9 [-0.5,+2.1] |
| H3_small_default_s0 | apo_noncryptic | 46.3 | +0.1 [-1.2,+1.2] | 54.3 | +0.7 [-0.7,+2.1] | – | – | 58.4 | -0.1 [-1.8,+1.1] |
| H3_small_default_s0 | af_dev_all | 47.9 | -0.7 [-2.7,+1.6] | 56.7 | -2.1 [-4.1,-0.2] | 47.1 | -0.7 [-2.3,+1.0] | 46.1 | +0.3 [-1.3,+1.8] |
| H3_small_default_s0 | apo_track_dev_small | 48.8 | -0.8 [-3.3,+1.7] | 60.0 | -1.5 [-4.0,+0.8] | 52.3 | +0.0 [-2.7,+2.2] | 28.8 | -0.5 [-1.4,+0.7] |
| v2_H2scd_small_default_s0 | biolip_dev_small | 55.3 | +0.1 [-2.0,+2.2] | 66.1 | +1.1 [-0.9,+3.1] | 61.2 | +0.5 [-1.9,+2.9] | 62.0 | +0.7 [-2.2,+2.1] |
| v2_H2scd_small_default_s0 | apo_cryptic | 41.5 | +0.3 [-1.9,+2.3] | 61.8 | +0.0 [-2.4,+2.2] | 55.4 | +0.5 [-1.6,+2.6] | 57.9 | +0.0 [-1.9,+1.8] |
| v2_H2scd_small_default_s0 | apo_noncryptic | 45.9 | -0.4 [-2.2,+1.4] | 54.3 | +0.7 [-1.2,+2.5] | – | – | 57.6 | -0.9 [-3.3,+0.8] |
| v2_H2scd_small_default_s0 | apo_track_dev_small | 49.2 | -0.4 [-4.1,+2.8] | 61.2 | -0.4 [-5.0,+4.0] | 49.6 | -2.7 [-9.6,+3.2] | 30.5 | +1.2 [-1.3,+5.7] |
| v2_H2scd_small_default_s0 | af_dev_all | 49.1 | +0.5 [-1.9,+3.1] | 57.7 | -1.2 [-3.4,+1.3] | 47.3 | -0.5 [-3.0,+2.1] | 46.4 | +0.6 [-1.6,+2.4] |
| v2_H12_small_default_s0 | biolip_dev_small | 55.5 | +0.4 [-2.0,+2.7] | 65.0 | +0.0 [-1.8,+1.9] | 60.9 | +0.1 [-2.3,+2.9] | 61.9 | +0.6 [-1.9,+2.1] |
| v2_H12_small_default_s0 | apo_cryptic | 41.8 | +0.5 [-1.5,+2.6] | 59.9 | -1.9 [-4.2,+0.3] | 53.6 | -1.3 [-3.4,+0.9] | 59.6 | +1.7 [-0.0,+3.5] |
| v2_H12_small_default_s0 | apo_noncryptic | 44.2 | -2.1 [-4.0,-0.1] | 52.7 | -1.0 [-2.8,+0.8] | – | – | 58.0 | -0.5 [-2.2,+1.5] |
| v2_H12_small_default_s0 | apo_track_dev_small | 51.2 | +1.5 [-3.5,+5.8] | 62.3 | +0.8 [-3.7,+5.4] | 51.9 | -0.4 [-5.6,+4.1] | 31.9 | +2.6 [-0.1,+6.0] |
| v2_H12_small_default_s0 | af_dev_all | 49.4 | +0.8 [-1.4,+3.1] | 58.3 | -0.5 [-2.8,+1.9] | 48.9 | +1.2 [-1.4,+3.8] | 46.2 | +0.4 [-1.5,+2.6] |
| v2_H1af_small_default_s0 | biolip_dev_small | 54.9 | -0.2 [-2.3,+1.6] | 63.9 | -1.1 [-2.8,+0.5] | 60.4 | -0.4 [-2.4,+1.6] | 61.2 | -0.1 [-2.0,+1.4] |
| v2_H1af_small_default_s0 | apo_cryptic | 43.4 | +2.2 [+0.4,+3.9] | 63.7 | +1.9 [-0.0,+3.8] | 55.1 | +0.2 [-1.7,+2.0] | 59.2 | +1.3 [-0.7,+3.1] |
| v2_H1af_small_default_s0 | apo_noncryptic | 46.0 | -0.3 [-1.8,+1.4] | 54.5 | +0.8 [-0.6,+2.3] | – | – | 58.8 | +0.3 [-1.4,+2.0] |
| v2_H1af_small_default_s0 | apo_track_dev_small | 48.8 | -0.8 [-4.2,+2.8] | 60.0 | -1.5 [-5.7,+2.5] | 52.3 | +0.0 [-3.7,+3.2] | 28.1 | -1.2 [-2.2,+0.6] |
| v2_H1af_small_default_s0 | af_dev_all | 48.8 | +0.2 [-2.2,+2.5] | 58.2 | -0.7 [-3.0,+1.5] | 47.3 | -0.5 [-2.7,+1.7] | 46.7 | +0.9 [-0.6,+2.6] |
| H5_small_default_s0 | biolip_dev_small | 60.6 | +5.5 [+3.4,+7.5] | 71.2 | +6.2 [+4.4,+8.2] | 62.1 | +1.4 [+0.1,+2.8] | 61.3 | +0.0 [+0.0,+0.0] |
| H5_small_default_s0 | apo_cryptic | 47.3 | +6.0 [+4.6,+7.6] | 71.5 | +9.7 [+8.0,+11.5] | 57.2 | +2.2 [+1.3,+3.2] | 57.9 | +0.0 [+0.0,+0.0] |
| H5_small_default_s0 | apo_noncryptic | 51.2 | +4.9 [+0.6,+8.7] | 59.2 | +5.6 [+1.1,+9.6] | – | – | 58.5 | +0.0 [+0.0,+0.0] |
| H5_small_default_s0 | apo_track_dev_small | 52.3 | +2.7 [-5.5,+9.1] | 67.3 | +5.8 [-5.1,+13.5] | 50.4 | -1.9 [-11.9,+4.7] | 32.3 | +3.0 [+0.6,+5.9] |
| H5_small_default_s0 | af_dev_all | 55.2 | +6.6 [+4.1,+9.1] | 66.1 | +7.2 [+4.6,+9.8] | 47.9 | +0.2 [-0.9,+1.2] | 45.8 | +0.0 [+0.0,+0.0] |
| v2_H4apo_small_default_s0 | biolip_dev_small | 54.9 | -0.2 [-2.4,+1.8] | 64.1 | -0.9 [-2.5,+0.8] | 61.1 | +0.4 [-1.3,+2.3] | 62.1 | +0.8 [-1.8,+2.3] |
| v2_H4apo_small_default_s0 | apo_cryptic | 42.8 | +1.6 [-0.2,+3.3] | 62.3 | +0.5 [-1.2,+2.3] | 55.6 | +0.6 [-1.2,+2.6] | 59.5 | +1.6 [-0.3,+2.8] |
| v2_H4apo_small_default_s0 | apo_noncryptic | 46.7 | +0.4 [-1.0,+1.8] | 54.4 | +0.7 [-0.8,+2.2] | – | – | 58.4 | -0.1 [-1.5,+1.7] |
| v2_H4apo_small_default_s0 | apo_track_dev_small | 51.9 | +2.3 [-0.4,+5.4] | 61.5 | +0.0 [-3.1,+3.2] | 55.4 | +3.1 [-0.4,+6.3] | 30.0 | +0.7 [-0.8,+2.2] |
| v2_H4apo_small_default_s0 | af_dev_all | 49.1 | +0.5 [-1.5,+2.8] | 58.5 | -0.3 [-2.4,+2.0] | 48.3 | +0.5 [-1.7,+2.8] | 46.1 | +0.3 [-1.3,+1.9] |
| H5_all_default_s0 | biolip_dev_all | 60.4 | +4.3 [+2.9,+5.7] | 70.8 | +5.4 [+3.9,+7.0] | 56.1 | +0.6 [+0.0,+1.2] | 69.7 | +0.0 [+0.0,+0.0] |
| H5_all_default_s0 | apo_cryptic | 42.6 | +3.8 [+2.6,+5.1] | 68.5 | +7.4 [+5.8,+9.1] | 54.7 | +2.4 [+1.4,+3.5] | 53.8 | +0.0 [+0.0,+0.0] |
| H5_all_default_s0 | apo_noncryptic | 49.1 | +5.6 [+1.1,+9.4] | 59.0 | +6.2 [+1.4,+10.4] | – | – | 56.7 | +0.0 [+0.0,+0.0] |
| H5_all_default_s0 | apo_track_dev_all | 47.1 | +4.6 [-4.9,+11.8] | 59.4 | +7.7 [-2.7,+15.7] | 42.2 | -2.2 [-9.0,+2.3] | 31.9 | +0.0 [+0.0,+0.0] |
| H5_all_default_s0 | af_dev_all | 62.1 | +6.1 [+3.8,+8.7] | 71.2 | +6.9 [+4.6,+9.3] | 57.7 | +1.3 [+0.5,+2.3] | 49.2 | +0.0 [+0.0,+0.0] |

Notes: H1 (AF inputs) was first OOM-killed on the shared node and rerun with lighter pocket caching
(results identical in design). H1+H2 and H2 rows are the only runs of those configs. H3 additionally
evaluated on apo_track_dev (row above).

### H5 — output score floor (sixth variant, approved by the PI as decision D12 at ~22:45)
tau chosen on TRAIN cluster-CV only (`scripts/burrow_tau_cv.py`; grid of 18 settings: tau in
{none, .01, .02, .03, .05, .08, .12, .20} × min_keep {1, 3} and max-pockets caps {5, 10, 20};
tables `tau_cv_p3_{small,all}_default_s0.csv`). Rule (D12): maximise mean(closest top-N+2,
closest-DCA4 top-N+2, one-to-one P1K), no loss > 0.5 pt in those nor in top-N vs v1. Admissible
best: small tau = .02/min_keep 3 (76.58), all tau = .03/min_keep 3 (75.97); one tau for both rankers
⇒ mean of the two criteria: tau .03 (76.21) vs .02 (76.18) — a near tie, **tau = 0.03, min_keep 3**
selected mechanically. (Train-CV P1K is saturated — the train pool has ~10k sites — so the P1K term
did not discriminate there; on dev P1K is unchanged by construction because the floor only removes
pockets below the pooled top 1,000.)
Single dev confirmation of H5 (rows `H5_*` above) and supplementary metrics (`h5_extra.csv`):

| ranker | set | model | top-N+2 (closest) | any-pocket top-N+2 | max recall | P1K | P1K 1:1 | pockets/structure |
|---|---|---|---|---|---|---|---|---|
| small | biolip_dev_small | v1 | 65.0 | 74.9 | 98.9 | 61.3 | 43.2 | 41.9 |
| small | biolip_dev_small | H5 | 71.2 | 73.7 | 80.1 | 61.3 | 43.2 | 6.9 |
| small | apo_track_dev_small | v1 | 61.5 | 73.1 | 99.2 | 29.3 | 17.7 | 40.9 |
| small | apo_track_dev_small | H5 | 67.3 | 68.5 | 75.4 | 32.3 | 20.0 | 6.3 |
| small | af_dev_all | v1 | 58.8 | 71.2 | 96.5 | 45.8 | 32.1 | 43.3 |
| small | af_dev_all | H5 | 66.1 | 68.5 | 73.0 | 45.8 | 32.1 | 6.9 |
| small | apo_cryptic | v1 | 61.8 | 77.8 | 98.7 | 57.9 | 49.9 | 42.1 |
| small | apo_cryptic | H5 | 71.5 | 77.5 | 84.7 | 57.9 | 49.9 | 6.2 |
| small | apo_noncryptic | v1 | 53.7 | 73.5 | 97.4 | 58.5 | 47.0 | 42.6 |
| small | apo_noncryptic | H5 | 59.2 | 61.7 | 65.1 | 58.5 | 47.0 | 6.4 |
| all | biolip_dev_all | v1 | 65.4 | 75.3 | 99.0 | 69.7 | 58.0 | 40.9 |
| all | biolip_dev_all | H5 | 70.8 | 74.3 | 81.6 | 69.7 | 58.0 | 8.1 |
| all | apo_track_dev_all | v1 | 51.7 | 68.6 | 99.1 | 31.9 | 20.6 | 39.4 |
| all | apo_track_dev_all | H5 | 59.4 | 64.0 | 75.4 | 31.9 | 20.6 | 7.6 |
| all | af_dev_all | v1 | 64.3 | 73.6 | 98.4 | 49.2 | 35.1 | 43.3 |
| all | af_dev_all | H5 | 71.2 | 73.0 | 78.3 | 49.2 | 35.1 | 8.0 |
| all | apo_cryptic | v1 | 61.0 | 77.5 | 98.9 | 53.8 | 46.9 | 42.0 |
| all | apo_cryptic | H5 | 68.5 | 77.2 | 87.4 | 53.8 | 46.9 | 7.5 |
| all | apo_noncryptic | v1 | 52.8 | 73.9 | 97.2 | 56.7 | 46.6 | 42.6 |
| all | apo_noncryptic | H5 | 59.0 | 63.3 | 68.0 | 56.7 | 46.6 | 7.7 |

H5 raises closest-pocket top-N / top-N+2 on every set and both rankers and slightly raises
closest-DCA4 top-N+2, leaves P1K unchanged, and costs **coverage**: pockets/structure 41→6–8, max
recall ~99 % → 73–87 % (apo non-cryptic 97→65), any-pocket top-N+2 −0.3 to −4.6. The gain is a
consequence of the closest-pocket rule penalising long output lists (benchmark finding, D12), not
of better ranking. The full list remains available (`burrow predict --min-score none` reproduces v1
exactly; verified 20/20 files).

### Diagnostic (not a variant): why v1 trails P2Rank on apo cryptic sites
| method | closest-pocket top-N+2 | any-pocket top-N+2 | max recall | pockets / structure | misses with closest pocket ≤ 12 Å but rank > N+2 | misses with no pocket ≤ 12 Å |
|---|---|---|---|---|---|---|
| BURROW v1 small default s0 | 61.8 | **77.8** | 98.7 | 42.1 | 403 | 5 |
| P2Rank 2.5.1 | **68.5** | 72.3 | 76.4 | 4.5 | 86 | 205 |

BURROW places a pocket within 12 Å of the cryptic site in its top-N+2 more often than P2Rank
(any-pocket rule), but under the LIGYSIS closest-pocket rule a low-ranked BURROW pocket is often
closer to the site centre than the ranked hit, because BURROW outputs ~10× more pockets. The apo
"deficit" is therefore mainly an output-size × metric-rule effect, not a ranking-quality failure;
training-data and ranker-feature changes (H1–H4) do not address it. Reported to the PI at 22:35
(an output cap / score floor would be a sixth change and needs approval).

## Decision
**Gate (fixed before results), `small` default s0 vs v1:** apo-dev cryptic closest top-N+2 +9.7
[+8.0, +11.5] (≥ +3, CI > 0 ✓); BioLiP-dev top-N+2 +6.2 (≥ −0.5 ✓), DCA4 +1.4 (≥ −0.5 ✓), P1K +0.0
(≥ −1 ✓); `all` not worse on any set ✓ (top-N+2 +5.4 BioLiP, +7.4 apo cryptic, +6.9 AF dev);
default-preset speed unchanged (identical computation, NMS stops earlier) ✓.
⇒ **H5 passes and becomes v2** = v1 models + output score floor tau = 0.03 (min_keep 3), all 12
models (`results/dev/FREEZE_v2.md`, `results/dev/freeze_v2/models/`). The same tau is applied to the
accurate preset (selected on default-preset train-CV; not separately validated on dev).
H1–H4 (training-data / ranker-feature changes) all **failed** the gate: best H1 (AF inputs) apo
cryptic +1.9 [−0.0, +3.8] with BioLiP top-N+2 −1.1; H2 +0.0; H1+H2 −1.9; H3 +0.3; H4 (apo PDB chains)
+0.5. Dev-set use in this cycle: 8 configurations evaluated on the four dev sets (v1 self-check,
H1, H2, H1+H2, H3 [+ apo-track-dev], H4, H5 small, H5 all); τ selection never touched dev.
