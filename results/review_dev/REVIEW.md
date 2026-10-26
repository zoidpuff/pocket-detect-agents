# Independent review: BURROW phase-2 dev claims (BioLiP2 dev, 579 chains)

Reviewer: independent agent (did not produce the results). Date 2026-10-26. Scope: `results/dev/biolip/PHASE2.md`,
`final_table.md`. Nothing outside `results/review_dev/` and `scripts/review/` was modified. No test set was touched.

**Claim under review** (ranker `all`, default preset, mean of 3 seeds): top-N+2 DCC≤12 65.0 vs P2Rank 2.5.1 54.1;
DCA≤4 top-N+2 55.7 vs 36.0; P1K 70.2 vs 46.3; ranker `small` on non-ion sites 64.2 vs 58.4; NMS 10 Å.

## Verdicts

| # | check | verdict |
|---|---|---|
| 1 | Recompute the headline numbers (independent code + shared harness); same denominators, ground truth and chains | **CONFIRMED** |
| 2 | Regenerate 30 random dev chains with the CLI | **CONFIRMED** (30/30 bit-identical) |
| 3 | Leakage / dev-selection optimism | **CONFIRMED**, minor notes (3 small issues, none affects the conclusion) |
| 4 | Gaming (pocket count/size, 1:1 matching, DCA4 closest pocket, surface tiling) | **CONFIRMED** no gaming. **ISSUE (framing):** about 2/3 of the `all` gain comes from ion sites; the non-ion gain is about +5 pts and its CI touches 0 |
| 5 | P2Rank run correctly | **CONFIRMED** |

## 1. Recomputation: CONFIRMED

Two independent routes:
* **My own implementation** `scripts/review/indep_eval.py`. It does not import `pocketdetect`. It implements closest-pocket DCC
  (rounded to 0.01, ties go to the larger IRel and then the better rank), closest-by-DCA, any-pocket, Hungarian 1:1 and pooled P1K / P1K-1:1. It runs a paired
  cluster bootstrap over the site-table `group` (cluster30, 500 units) with its own RNG seed (12345). Output:
  `indep_eval_all.csv`, `indep_eval_small.csv`, with per-site dumps in `indep_all/` and `indep_small/`.
* **Shared harness** `scripts/run_dev_eval.py`, which wraps `pocketdetect.site_eval`, run on P2Rank and the 3 default seeds (`harness_all_*`).

| `all` sites (1,221 sites, 579 chains) | top-N+2 | DCA4 top-N+2 | P1K | Δ top-N+2 vs P2Rank [95% cluster CI] |
|---|---|---|---|---|
| P2Rank 2.5.1 | 54.1 | 36.0 | 46.3 | – |
| BURROW default s0 / s1 / s2 | 65.4 / 64.2 / 65.4 (mean **65.0**) | 55.5 / 55.7 / 55.8 (mean **55.7**) | 69.7 / 69.9 / 71.0 (mean **70.2**) | +11.4 [+6.9,+16.1] / +10.2 [+6.0,+14.7] / +11.3 [+7.2,+15.7] |
| `small` sites (805 sites, 400 chains): P2Rank / BURROW-small default s0, s1, s2 | 58.4 / 65.0, 62.7, 64.8 (mean **64.2**) | 47.8 / 60.7, 59.8, 60.5 | 41.1 / 61.3, 62.3, 62.5 | +6.6 [+1.0,+13.5] / +4.3 [**−0.9**,+10.2] / +6.5 [+1.2,+12.3] |

* My code and the harness give the same point estimates for every reported number (identical counts). An initial 0.1-pt DCA
  difference went away once DCA was rounded to 0.01 Å, as the harness does. The bootstrap CIs agree with the developer's
  within about 0.5 pt.
* Ground truth: there is a single site table per variant (`dev_sites_{all,small}.tsv`) for every method. The chain set is the same 579 chains for all methods,
  equal to `inputs_audited.list`. There are 0 missing prediction files for any method. P2Rank returns an empty output (header only) for 52 chains;
  these chains (80 sites) are counted as misses, and the denominators never shrink.
* The developer's own caveat is reproduced: for `small`, seed 1's top-N+2 CI includes 0.

## 2. CLI regeneration: CONFIRMED

I took a seeded random sample of 30 audited dev chains (`regen_list.txt`) and ran
`python -m pocketdetect.burrow predict @list --ranker all --preset default` with no `--model`, so the CLI resolves its own default
to `burrow_all_default_s0.json`. **30/30 files were bit-identical** to `preds_cli/burrow_all_default_s0`: the same number of pockets, centres
and scores (max |Δ| = 0), and identical residue strings. Outputs are in `regen/`.

## 3. Leakage and dev-selection optimism: CONFIRMED (minor notes)

**Train/dev overlap** (training snapshot `results/dev/cache/trainset_snapshot/audited_d6`, which is the snapshot used by `final_run.sh`):
* There is no overlap of entry ID, UniProt, cluster30 or group between the 6,475 train entries and the 579 dev chains. The training logs show
  `train inputs 8184 (assemblies 1709), dev 579` (= 6,475 + 1,709), drawn only from the train manifest (`burrow_train_biolip.py`).
* 36 PDB IDs contain both a train chain and a dev chain (different proteins in large complexes such as ribosomes and photosystems).
  None of those train entries has a stored assembly-1 input: 21 were `dropped_partner_leak`, 9 `too_large` and 72 `not_selected`. So no dev chain enters
  training as an assembly partner.
* My own MMseqs2 search (`-s 7.5`, which is more sensitive than the builder's) of the 579 dev chains against the 6,474 train representatives plus all 4,092
  stored assembly chains found **2 dev chains with borderline rule hits** (identity 30.0% and 30.3%, E ≤ 1e-5;
  A0ABF7SXA6__4rjj_A, Q74BH2__4rn3_A). The audit had passed them, probably because of the lower search sensitivity. The impact is negligible (2/579 chains).
* Remote homology is common. 194/579 dev chains have an E < 1e-3 train hit (19–53% identity), and Foldseek gives 225/579 with
  max TM > 0.7 to a train representative (TM < 0.5: 197). **The gain does not come from similarity**: Δ top-N+2 is **+19.4** [+10.3,+28.3] on
  TM < 0.5 chains, compared with +5.7 [+1.2,+10.4] on TM > 0.7 chains (s0; `strata_similarity.csv`). The gain is also positive on chains with no sequence hit
  (+9.7 [+5.2,+14.3]).
* Minor provenance gap: the model JSON `meta` records neither the training-manifest nor the test-union checksum. It records only
  `n_train_inputs` and the feature-cache path, while DESIGN §9 says training artifacts record the test-union checksum.

**Selection optimism**
* **NMS 10 Å was not chosen on the BioLiP dev set.** `nms_dist = 10.0` is already in commit 0193f6d (16:35), with the comment
  "selected on CHEN11 cv5-OOF". The first BioLiP dev artefacts appeared at 16:38 or later, and the audited dev table at 18:26. The BioLiP sweep
  (`nms_sweep_burrow_all_s0.csv`, **21 settings**, not the 8 shown in PHASE2.md) only confirms the choice. The settings that were not taken would have
  *raised* top-N+2: 14 Å gives 72.6, and 10 Å with a 0.05 score floor gives 70.6. So the chosen value is not the maximum of the headline metric. Small inaccuracy:
  METHOD.md §7 says 10 Å "maximis[es] DCA4 top-N+2", but 12 Å is higher (56.7 vs 56.2). PHASE2.md's justification, maximal 1:1 P1K, is the correct one.
* The settings compared on the audited dev set for `all` were 9 speed/accuracy settings (top-N+2 64.5–66.1), 2 ranker targets (graded/hit), 2 presets
  and 3 seeds. The shipped default (65.4 for s0) is mid-range, not the maximum (C = 66.1). Optimism from choosing the best of these is at most
  about 0.7–1 pt. Seed-to-seed SD is about 0.7 pt. Even a pessimistic 2-pt optimism is small next to Δ ≈ +10.9 (CI half-width about 4.5, lower bound ≥ +5.8).
  **Selection cannot explain the dev gap.** The phase-2 design as a whole was still developed while looking at this dev set, and that part of the optimism cannot be
  quantified. The pre-registered test run is the arbiter.
* The sweep models REF and B–F4 were trained on the *pre-audit* train set, which still included the 169 entries D6 later dropped. They informed only the preset
  choice; the headline models were trained on the audited set.

## 4. Gaming: CONFIRMED no gaming; ISSUE with the framing (ion share)

**Pocket count and size** (`gaming_pocket_stats.csv`):

| method | pockets/chain (median; % at cap 50) | residues/pocket all / top-N+2 | centre → nearest atom, top-N+2 | top-N+2 pockets with empty residue list |
|---|---|---|---|---|
| P2Rank | 4.2 (3; 0%) | 13.3 / 14.7 | 3.36 Å | 0% |
| fpocket | 20.7 (17; 5%) | 9.1 / 12.2 | 3.34 Å | 0% |
| BURROW default s0 | 40.9 (46; 44%) | 4.4 / 11.4 | 3.36 Å | **8.3%** (21.8% of all pockets) |

* Many pockets do **not** help on the LIGYSIS closest-pocket metric; they hurt. A lower-ranked pocket that lies closer to a site "steals" the match. Any-pocket top-N+2 is
  75.3 compared with closest-pocket 65.4, and the developer's own score-floor variant with about 5 pockets scores higher (70.6). Pocket count therefore
  inflates only `recall_max` (99.0 vs P2Rank 58.5), which is not claimed.
* **The gain does not depend on the 12 Å tolerance.** Δ top-N+2 is +16.4 at DCC ≤ 4 Å (39.3 vs 22.9), +22.2 at 6 Å, +20.6 at 8 Å and +11.4 at 12 Å
  (`gaming_dcc_threshold.csv`). The median DCC of BURROW's successful matches is 3.3 Å, against 4.7 Å for P2Rank. Only 1.8% of BURROW hits lie at 8–12 Å, compared with 19% for P2Rank.
  BURROW's top-N+2 curve saturates by 8 Å.
* **One-to-one (Hungarian) top-N+2:** 71.1 vs 46.3 (+24.8 [+19.9,+30.0]). **P1K 1:1:** 57.8 vs 42.0. **DCA4 closest pocket:** +19.5
  [+15.3,+24.0]. **DCA4 any pocket:** +21.1. All of these hold for every seed and for `small` (1:1 +19.6, DCA4 +12.9).
* **Surface-tiling control** (`scripts/review/tiling_control.py`, `indep_eval_controls.csv`). Pockets are SAS points (1.4 Å probe) tiled
  with a 12 Å Poisson-disk spacing, about 39 per chain (similar to BURROW's 41), capped at 50:

  | control | top-N+2 DCC12 | DCA4 top-N+2 | P1K |
  |---|---|---|---|
  | tiling 12 Å, random rank (3 seeds) | 15.2 / 16.2 / 15.8 | 7.0–7.8 | 9–10 |
  | tiling 12 Å, ranked by buriedness (#atoms ≤ 10 Å) | 37.8 | 22.6 | 21.8 |
  | tiling 10 Å, buriedness | 26.8 | 18.5 | 18.7 |
  | greedy buriedness-NMS over SAS points (10 / 12 Å) | 18.9 / 27.2 | 14.7 / 16.5 | 18 |
  | **BURROW s0 pockets with randomly shuffled ranks** | **14.6** | 11.1 | 12.1 |
  | BURROW geometry-only (developer's) | 53.3 | 23.0 | 32.9 |
  | P2Rank / BURROW s0 | 54.1 / 65.4 | 36.0 / 55.5 | 46.3 / 69.7 |

  Surface coverage alone gives about 15%, and about 38% with a buriedness ranking. With their ranks shuffled, BURROW's own candidates fall to the tiling level (14.6%). **The gain comes from the
  learned ranking, not from spreading pockets over the surface.**
* **ISSUE (framing, not an error):** 416/1,221 `all` sites (34%) are ion-only. P2Rank is not trained for these sites. Stratified Δ top-N+2
  (`strata_ion_p2zero.csv`): **ion sites +22.4 / +20.7 / +21.4**; **non-ion sites +5.7 [−0.1,+12.9] / +4.7 [−0.8,+10.4] / +6.1 [+0.6,+12.1]**.
  About two thirds of the headline +10.9 therefore comes from ions. On P2Rank's own ligand definition the advantage is about +5 pts on DCC12 and
  is not robustly significant (the `small` ranker shows the same). DCA4 and P1K stay clearly significant on non-ion sites (+11.9 to +12.5 DCA4). Also, 80
  sites sit on the 52 chains where P2Rank outputs nothing, and BURROW gains +45 pts there. Excluding those chains, Δ top-N+2 = +9.0 / +7.5 / +8.7
  (CIs exclude 0). PHASE2.md's caveat mentions the ion point, but the headline sentence should state it.
* Minor output defect: 21.8% of BURROW pockets (8.3% of top-N+2 pockets) have an empty `residues` field, and 21% of top-N+2 centres are more than 4 Å
  from any protein atom. This does not affect DCC or DCA metrics, but it lowers residue-level metrics and is visible to users. It probably comes from the 3 Å sub-lattice
  points lying more than 4 Å from atoms.

## 5. P2Rank execution: CONFIRMED

* Version: `prank -v` reports **P2Rank 2.5.1** (`baselines/tools/p2rank`). The wrapper uses the default config, which includes the B-factor
  feature. B-factors are kept in the polymer-only inputs (`write_chain_input` copies them through gemmi).
* Inputs: the same `results/dev/biolip/inputs/<id>.cif` files as BURROW, converted by the wrapper's gemmi cleaner (no ligands or waters; modified
  residues such as MSE are kept). Timing log: 1,266/1,266 `ok`, 0 failures.
* Spot check: I ran P2Rank 2.5.1 directly (`prank predict`, no wrapper, raw .cif) on 5 random dev chains. Two of them contain MSE/HETATM
  polymer residues. Pocket count, centres (max |Δ| 0.001 Å), scores and residue sets were identical to `preds/p2rank` on 5/5. I also re-ran 3 of the 52
  zero-pocket chains directly, and P2Rank again produced an empty prediction file. The empty outputs are genuine P2Rank behaviour on small chains (median 122 residues),
  not conversion failures. Files: `p2rank_spot/`.

## Files
Scripts: `scripts/review/{indep_eval.py, tiling_control.py, gaming_stats.py, leak_seqs.py}`. Results (all in
`results/review_dev/`): `indep_eval_{all,small,controls}.csv`, `harness_all_*`, `gaming_*.csv`, `strata_*.csv`,
`leak/dev_vs_train{,_fs}.m8`, `regen/`, `p2rank_spot/`, `controls/`, `logs/`.
