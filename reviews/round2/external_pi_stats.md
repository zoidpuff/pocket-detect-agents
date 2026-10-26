# Review round 2: external PI (statistics & benchmarking), DESIGN v2

Written 2026-10-26 15:40 CET. Verdict: v2 addresses most round-1 MUSTs. Five internal contradictions would undermine the pre-registration, and a few round-1 MUSTs were dropped silently. Every fix is a wording change or a few lines of harness code.

## 1. Round-1 MUSTs not (fully) addressed
- **Pre-registration not frozen.** `docs/EVALUATION.md` does not exist (docs/ is empty). §1 is only "mirrored" in it. **MUST:** write and commit it, and record its sha256 in the ledger, before any test-set run.
- **Huge-input handling** (algorithms MUST: float32/int32 grids, ~96 Å bricks with halo, voxel-budget GPU batching, smoke test on the largest HOLO4K entries, dimer assemblies and 2,700-aa AF models) is absent from v2. **MUST.**
- **One endpoint per secondary** (stats MUST). O2 has four endpoints (two sets × two cut-offs), and O5 and O6 have none. **MUST:** name one endpoint per objective. Suggested:
  - O2: HOLO4K-mlig top-(n+2) DCA 4 Å.
  - O5: top-N+2 DCC 12 Å recall vs "naive + pLDDT<70 filter".
  - O6: interface-site top-N+2 on assembly input.
- **PocketWatch `--af-mode` on/off** (data MUST 7) is not declared. **MUST:** declare it.
- **Exclusion fallback** (data MUST 2) is missing: report retained clusters, and pre-declare per-benchmark exclusion lists if >50% is removed. Same for the ECOD/CATH fallback when the TM ≥ 0.5 fold-disjoint set is near-empty. **SHOULD.**
- **Reproduction tolerances** are not stated. **SHOULD:** ≤0.5 pp on LIGYSIS from the authors' predictions; ±2 pp for P2Rank on COACH420/HOLO4K. Stop and debug otherwise.

## 2. Internal inconsistencies
1. **Test v1 at T+10 h but feature freeze at T+12 h.** Test numbers would then steer features. **MUST:** no test-set metric is seen before the freeze. A pre-freeze run is a pipeline smoke test only: done by a non-developer agent, reporting n_expected/n_parsed/n_failed and controls, with metrics sealed.
2. **"Holm-adjusted, fixed order"** mixes two procedures. **MUST:** pick one. I recommend Holm over O2–O6, using one-sided bootstrap p-values for H0: Δ ≤ −δ. Also fix the gatekeeping order C1 → C1-strict → C2 → {O2–O6}, all at α = 0.05.
3. **C1 bar vs pairing.** The margin uses max(published, re-run), but the CI is paired against the authors' predictions. **MUST:** run the paired test against *both* prediction sets (published and our re-run). Require ≥ 2.0 points and CI > 0 for each. Also report on the intersection of chains covered by all methods (sensitivity check; failures still count as misses in the primary).
4. **C2 "best baseline" is undefined.** If it includes P2Rank-CONS (P1K 90.7), the claim is near-unwinnable for a structure-only method. **MUST:** name the comparator now: fpocket+PRANK (and P2Rank no-cons as a descriptive row).
5. **AF no-op check vs the AF-trained ranker.** A separate ranker with pLDDT features cannot reproduce standard-mode output by construction. **MUST:** scope the no-op identity to mechanisms 1–3 (identical points and features). Report the AF ranker on PDB input separately.

## 3. New problems introduced by v2
- **Dimer symmetric-copy rule.** Copies still consume top-N+2 slots while N counts unique sites, which penalises methods in proportion to symmetry. The "not FP" exception is also label-dependent. **MUST:** collapse predictions label-free and identically for all methods: map centres through the assembly operators and run NMS at 5 Å across copies. Then N = unique sites. Isolated-chain hetero sites: a hit on either chain counts. Classes with < 100 sites are descriptive only.
- **Two rankers.** It is not stated whether the *point scorer* is trained with ion labels for `small`, or which model is the CLI default. **SHOULD:** state both. Suggested: the point scorer is shared with ion labels, and the default is `all`.
- **Pooled superposed labels.** Pooled ligands can clash with the representative chain or sit in a closed conformation. **SHOULD:** drop ligands with heavy-atom distance < 2.2 Å to the rep chain, and log the count. Exclude BioLiP peptide/DNA/RNA. Fallback if the builder misses T+4 h: single-entry labels on the dev set, documented.
- **Metal candidates "≥ 2 coordinating atoms incl. backbone O"** will flood points. **SHOULD:** require ≥ 1 side-chain donor, and gate the point count on dev.
- **Seeds.** **SHOULD:** state that the shipped/headline model is `SEEDS[0]`, and enable bagging/feature_fraction. Otherwise the three LightGBM seeds are identical and the "every seed" requirement tests nothing.
- **Scope vs time.** There are ~10 ablations and 6 benchmarks in 21 h. **MUST:** pre-declare the cut order now, for example PocketWatch → CHEN11-only → fold-disjoint → PAE/surrogate. Any objective not run is reported as "not run". Nothing is dropped after its test number has been seen.

## 4. Minor
- SHOULD: 10k bootstrap resamples for C1/C2.
- SHOULD: chance level by protein-size tertile for the controls.
- SHOULD: report the stratum × method interaction, not per-stratum wins.
- Note in EVALUATION.md that the uniform −2 pt margin makes O3/O4/O6 likely "inconclusive" (MDE ~6–9 pp).
