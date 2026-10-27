# External review of REPORT.md (FREEZE v2) — 2026-10-27 ~08:20 CET

Reviewer: independent agent (did not produce any result). Saved verbatim-in-substance by the PI because the reviewer could not write files. Reviewer recomputations used the reviewer's own bootstrap RNG; the post-hoc numbers quoted in the report are regenerated with the project scripts and `BOOTSTRAP_SEED` (results/test/v2/POSTHOC.md).

## Overall
Honest and traceable (checker 1,408/1,408). Problems: framing, one mislabelled table, two reproducibility statements. No restricted data committed (only data/manifests/scpdb.tsv checksum + download script).

## Reviewer recomputations (own RNG; to be regenerated)
| comparison | n | BURROW | comparator | Δ [95% CI] |
|---|---|---|---|---|
| O5 ion sites vs P2Rank-AF | 2,422 | 68.7 | 36.6 | +32.1 [28.0, 36.2] |
| O5 non-ion sites vs P2Rank-AF | 4,080 | 62.1 | 60.2 | +1.9 [−0.2, 4.1] |
| LIGYSIS_NI v2 s0 vs fpocket+PRANK | 4,141 | 66.4 | 63.4 | +3.0 [1.1, 4.8] |
| LIGYSIS_NI v2 s0 vs P2Rank | 4,141 | 66.4 | 60.9 | +5.4 [3.4, 7.4] |
| LIGYSIS_NI v1 s0 vs fpocket+PRANK | 4,141 | 58.7 | 63.4 | −4.6 [−6.3, −2.8] |
| LIGYSIS_NI v1 s0 vs P2Rank | 4,141 | 58.7 | 60.9 | −2.2 [−4.4, 0.0] |
| LIGYSIS_NI DCA4 v2 vs fpocket+PRANK / P2Rank | 4,141 | 51.3 | 48.2 / 44.6 | +3.1 / +6.7 |

## MUST
- **M1** Goal-by-goal verdict vs PROMPT at the top of §1 ("clearly beats": partly — clear on LIGYSIS incl. ions and DCA4; non-ion modest; COACH420/HOLO4K non-inferior only; O4/O6 at par or below. "Ultra-fast": fast, ~1.9× P2Rank per core incl. parsing; GPU below 32 CPU workers. Generalisation: unseen folds supported by fold-disjoint retraining (8/9 within noise, COACH420 −3.2); apo/cryptic not shown; allosteric not evaluated. AF mode: removes hallucinations, not shown to beat pLDDT<70 filter, PAE not used (403), side-chain tolerance off; O5 superiority is an ion effect. Dimers: native input; better on non-interface sites in assemblies; not better on interface sites; hetero interfaces −6.1 (n=66)).
- **M2** O5/AF-mode attribution: O5 gain concentrated on ion sites; AF mode not shown to beat the pLDDT<70 filter (not run on test); PocketWatch B=100 ⇒ AF mode inert on O3.
- **M3** §4.3: O6 fold-novelty rows are all-site (807), not the 243-site interface endpoint — relabel/recompute; base the fold-memorisation claim on the A1 fold-disjoint retraining ablation.
- **M4** One "deviations" caveat in §1: A1–A3 (pre-test; PREREG hash changed 582b8b56→59c7e41e…), G1 failed/fallback not taken (D7), v2 floor frozen 70 min after v1 predictions existed (before scoring), v3 motivated by O6 test result (no change), `--af-ranker` advice test-informed; v1 O3 significantly worse (−2.7 [−4.9, −0.5]).
- **M5** Missing numbers: LIGYSIS_NI paired Δs; P2Rank on LIGYSIS (52.6 top-N+2, 29.8 DCA4); pockets/chain next to 72.5 (9.1 vs fpocket+PRANK ~20.9).
- **M6** Say why scPDB, AF2BIND, HOTPocket were not test benchmarks.
- **M7** Reproducibility: branch `burrow-final` does not exist yet; reconcile scoring-commit statements (§8 vs §6.12 vs summary.json: v1 20421b6, v2 9521756, dirty tree) — verify headline numbers unchanged.

## SHOULD
S1 "analytical" framing (geometry-only ranking 53.3 vs P2Rank 54.1 on dev; gain is learned; name what's new). S2 speed paragraph like-for-like first; GPU below CPU path; proteome comparison different setting. S3 "Pass" → "Non-inferior"; O3 Holm p near resolution. S4 leakage audit initial FAIL + D6 fix; no full re-audit after. S5 no list-length policy tuned for baselines; C1-strict is the robust evidence. S6 C2 vs best published P1K (IF-SitePred-rescored-NR 94.3) margin +1.8, untested. S7 dimer counts (400 vs 283 scored assemblies; 70 minor-interface sites outside endpoint; isolated-chain BURROW untested). S8 time zones. S9 `--af-ranker` test-informed; lowered PocketWatch −3.4. S10 CHEN11-only model AF-set P1K 78.2 > P2Rank-AF 73.9 > v2 58.9 → low AF precision likely from ion-heavy training.

## COULD
Compress Family-1 table; define metrics once in §1; explain PRANK 2.4.2 vs 2.5.1; PREREG "five" vs six endpoints; DCA4-from-centre favours compact pockets; speed×accuracy table; PocketMiner 38 vs 35 chains.
