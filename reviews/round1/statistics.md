# Round 1 review — Statistics & evaluation

**Scope reviewed:** DESIGN v1 §1, §3 and §5, plus `src/pocketdetect/evaluate.py` and `seeds.py`.

**Verdict:** the overall plan is sound. Three problems need fixing:
- the primary hypothesis is underspecified;
- the "no significant loss" rule is anti-conservative;
- re-use of the test set is uncapped.

## 1. "Clearly beat"

- **MUST: pre-register one primary hypothesis** in `docs/EVALUATION.md`, committed before any test evaluation.
  - **Dataset and model.** Pick one: LIGYSIS full with the `all`-label model, or LIGYSIS_NI with the default model. As written, the default model ignores ions while O1 is scored on a set that is about 40% ions.
  - **Metric.** Site-micro top-N+2 recall at DCC ≤ 12 Å.
  - **Comparator.** fpocket+PRANK.
  - **Decision rule.** The paired 95% CI lower bound is > 0 and the point estimate is ≥ +2 pp.
- **MUST: add Precision1K non-inferiority as a co-primary.** The margin is δ = 3 pp versus the best baseline. Without it, a recall gain could be bought by a precision collapse.
- **MUST: replace "no significant loss on O2–O6" with non-inferiority.** The CI lower bound must be > −δ: 3 pp in general, 5 pp for sets with fewer than 500 sites. A CI wider than the margin is reported as *inconclusive*, not as a pass. Under the current rule, small sets (PocketMiner, n = 38) pass automatically.
- **SHOULD: pair against published predictions.** For baselines we cannot re-run (P2Rank-CONS), use the per-chain predictions in the LIGYSIS Zenodo files (`data/raw/ligysis/lbs_comparison_results`). The inputs are identical, so the comparison stays paired; a comparison against published scalars would not be.

## 2. Resampling unit and paired tests

- **MUST: bootstrap over redundancy clusters, not chains.** Sites are correlated within a chain, and chains are correlated within a UniProt or family (LIGYSIS has 4,037 segments from 3,513 UniProts).
  - Primary unit: MMseqs2 30% clusters within the test set.
  - Sensitivity unit: UniProt.
  - Implementation: add a `cluster_id` argument to `_bootstrap_weights` and its callers.
  - The same applies to HOLO4K, the dimer set and O5 (unit = UniProt).
- **MUST: score failed or empty outputs as misses.** `compare_methods` currently calls `dropna`. Log the failure rate for each method.
- **SHOULD: drop the exact McNemar test from the reported results**, because it assumes sites are independent. Use 10k resamples for headline CIs; DESIGN currently says 1k.

## 3. Multiplicity

- **MUST: test in a fixed hierarchy.**
  1. O1 at α = 0.05.
  2. Then the Precision1K co-primary.
  3. Then one pre-named endpoint and comparator per objective for O2–O6, Holm-adjusted as one family.
- Everything else is descriptive and labelled unadjusted: other metrics, strata, ablations, DL baselines.
- For generalization, report the stratum × method interaction (how the gain trends with novelty), not per-stratum "wins".

## 4. Seeds

- **MUST: reconcile the two seed lists.** DESIGN says `[0,1,2]` with bootstrap seed 0; `seeds.py` says `(20261026, 1, …)`.
- **MUST: declare the headline model now.** Either the model trained with `SEEDS[0]` or the 3-seed mean ensemble — never the best seed.
- **MUST: report the paired CI for each seed.** The claim must hold for every seed.

## 5. Precision1K and calibration

- **MUST: recompute the pooled top-1000 inside each bootstrap resample.** A fixed-set ratio CI is invalid for a rank-cut statistic.
- **MUST: report a one-to-one Precision1K**, computed after NMS, with duplicates counted as FP (the greedy matcher already exists). Otherwise near-duplicate pockets around one large site inflate the score.
- **SHOULD: back the "calibrated" claim with test-set evidence.** Global isotonic calibration is monotone, so it cannot change Precision1K or #TP@100FP. Report Brier score, ECE, a reliability diagram, and pooled AP.

## 6. Test-set reuse (TEST_LEDGER)

- **MUST: enforce the ledger.**
  - **Cap:** at most 3 frozen model versions may touch any test set before E9. All of them appear in REPORT.md. Baselines are exempt.
  - **Selection on dev only:** variants, τ, β₀, NMS radius and the pocket-centre definition are all chosen on dev.
  - **Separation:** test evaluation is run by an agent that did not develop the model. Training and tuning code never reads the test-label paths; add a guard test.
  - **Lock:** the sha256 of `docs/EVALUATION.md` is recorded in the ledger before the first test run.
- **SHOULD: build dev the LIGYSIS way.** Use pooled UniProt sites, one representative chain, and the same ion policy, so that dev-optimal settings transfer. Report the dev→test gap.

## 7. 12 Å DCC leniency and gaming

- **MUST: add two adversarial controls.**
  - **Surface tiling:** N+2 farthest-point-sampled surface points, ranked by buriedness.
  - **Protein centroid** as the prediction.
  - Report chance level per protein-size tertile, and our gain over these controls. On small proteins, 12 Å covers much of the surface.
- **MUST: the primary claim must also hold under one stricter criterion, fixed now.** Either residue overlap I_rel ≥ 0.5 or DCC ≤ 8 Å.
- Report pocket size (residues and volume) and redundancy % for every method.

## 8. Strata and power

Approximate minimum detectable effect (MDE), paired, 80% power:

| Benchmark | Size | MDE |
|---|---|---|
| LIGYSIS | 6,882 sites; design effect about 1.5 | ~2 pp |
| COACH420 | — | ~5–6 pp |
| CryptoBench-test | ~300 sites | ~9 pp |
| PocketMiner | 38 pairs | ~20 pp — descriptive only |

- **MUST: fix stratum boundaries from the novelty distribution before evaluation**, e.g. TM < 0.5 / 0.5–0.7 / ≥ 0.7, or tertiles.
  - Merge any stratum with fewer than 100 sites.
  - Sequence-identity strata are degenerate (everything is < 30% by construction), so stratify by TM and by ligand class instead.
  - Also report overlap with the baselines' training sets (CHEN11, JOINED, sc-PDB).
- **SHOULD: require ≥ 100 sites per O6 interface class.** Classes below that are descriptive only.

## 9. Controls

- **SHOULD: specify the controls.**
  - Shuffled labels are permuted within each protein.
  - The oracle should score about 100%; anything less indicates a harness bug.
  - Agreement with the authors' LBS-comparison code is accepted at within 0.5 pp aggregate and exact match on ≥ 99% of sites.
- **COULD: report the train / dev / test performance gap** as a memorization canary.
