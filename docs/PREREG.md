# PRE-REGISTRATION — claims, endpoints, comparators, tests (frozen 2026-10-26 ~16:10 CET, before any test-set run)

This file is frozen. Its sha256 is recorded in RESEARCH_TREE.md and in every test-ledger entry. Any later change is an *amendment* appended at the bottom with date + reason, and results reported under both the original and amended wording.

## Gatekeeping family 1 (fixed-sequence; stop at first failure; α = 0.05 two-sided CIs)
1. **C1** — LIGYSIS full (ions included; 2,775 chains; LBS-comparison protocol). Method: BURROW with ranker `all`, standard mode. Endpoint: top-N+2 recall, DCC ≤ 12 Å (LIGYSIS definition: predicted pocket centroid to site centroid; N = #sites of the chain). Comparator: fpocket+PRANK. Bar = max(published 60.4, our re-run). Requirement: Δ ≥ +2.0 points AND paired cluster-bootstrap 95% CI of Δ excludes 0, against **both** (a) the authors' published per-chain predictions and (b) our own re-run, for **each** of the 3 training seeds (`SEEDS[:3]`). Failures/empty outputs count as misses; denominators matched (identical chain set; if published predictions miss chains, those chains count as misses for that method, and a sensitivity analysis on the common subset is reported).
2. **C1-strict** — same pairing, criterion DCA ≤ 4 Å (min distance from predicted pocket centre to any heavy atom of any ligand that defines the site, after mapping the site's ligands onto the representative chain), top-N+2: CI of Δ excludes 0 in our favour.
3. **C2** — Precision1K (pooled top-1,000 predictions over the dataset, DCC ≤ 12 Å, recomputed inside each bootstrap resample): BURROW ≥ fpocket+PRANK − 3 points (non-inferiority; lower CI bound of Δ > −3).

## Family 2 (Holm–Bonferroni across the five; non-inferiority margin −2.0 points vs the best of P2Rank-2.5 / fpocket+PRANK re-run on identical inputs; lower CI bound > −2 = pass; CI spans −2 = inconclusive; also report superiority)
- **O2** COACH420(mlig) **and** HOLO4K(mlig) (two endpoints, both in the Holm family): DCA ≤ 4 Å top-n success (n = #relevant ligands per structure, P2Rank rules), ranker `small`.
- **O3** PocketWatch temporal test: top-N+2 recall DCC ≤ 12 Å on scored (unmasked) sites, following the PocketWatch mask; inputs are their predicted structures; `--af-mode` ON (pre-registered: predicted-structure input ⇒ AF mode), ranker `small` (their ligand rules exclude ions — verify; if ions included, ranker `all`).
- **O4** CryptoBench test (222 apo): top-N+2 recall, DCC ≤ 12 Å to the site-residue centroid on the apo structure, ranker `small`.
- **O5** AF set (LIGYSIS UniProts on AFDB v6): top-N+2 recall DCC ≤ 12 Å; BURROW `--af-mode` vs P2Rank `-c alphafold` (and fpocket) on the identical full-length model; ranker `all`.
- **O6** Dimer benchmark, interface sites (homo + hetero pooled): top-N+2 recall DCC ≤ 12 Å with full-assembly input, all methods on identical inputs, symmetric-copy rule below; ranker `all`.

## Descriptive only
Strata (fold novelty by max Foldseek TM to training: <0.5 / 0.5–0.7 / >0.7; ligand class ion / nucleotide-cofactor / other; AF pocket pLDDT bands; dimer site class), LIGYSIS_NI, one-to-one Precision1K, residue-level F1/MCC/AUPRC, PocketMiner-38, top-N, max recall, RRO/RVO, runtime.

## Fixed rules
- Bootstrap: 1,000 resamples over 30% sequence clusters (UniProt-level as sensitivity), seed `BOOTSTRAP_SEED` from src/pocketdetect/seeds.py.
- Residue-level rule (all methods): pocket residues = residues with any heavy atom ≤ 4 Å of any pocket point (methods without points: residues with any atom ≤ 6 Å of the centre, as in LIGYSIS); per-residue score = max score of pockets containing it.
- Dimer symmetric copies: GT sites with identical entity-level residue sets collapse to one unique site (any copy may be hit); before scoring, **every method's** predictions are collapsed label-free: predictions whose entity-level residue sets have Jaccard ≥ 0.5 are merged keeping the highest-ranked one.
- Test gating: test sets are run only via scripts/run_test_eval.py (clean tree, model checksum, auto-appends results/TEST_LEDGER.md). No test-set run before feature freeze; ≤ 3 frozen versions on test before the final run; the final run reports all versions.
- Reproduction gates (before any comparison is trusted): our harness on the authors' LIGYSIS predictions reproduces published top-N, top-N+2, max recall within ±0.5 points; our P2Rank 2.4.2 and fpocket 4.0 re-runs within ±2.0 points of the published rows.

## Amendments
- **A1 (2026-10-26 16:40, before any method output or test run exists).** Dimer GT symmetric-copy grouping changed from "identical entity-level residue sets" to "entity-level residue-set Jaccard ≥ 0.5" — the same criterion as the label-free prediction collapse. Reason: Track D found the strict rule under-collapses true symmetric copies (e.g. 4HHB HEM copies differ by one residue: 1,063 strict vs 807 J≥0.5 unique sites), and combined with the J≥0.5 prediction collapse it caps achievable recall below the oracle. Strict grouping reported as sensitivity; oracle ceilings reported for both.
- **A2 (2026-10-26 ~17:15, before any BURROW prediction on any test set).** C1-strict operationalisation: site ligand atoms are placed on the representative chain via SIFTS/UniProt pairing + local superposition (scripts/build_ligysis_ligands.py); placed copies must lie within 4.5 Å of the site residues. 6,387/6,721 scored sites receive ligand atoms; the remaining 334 (5%) are excluded from the C1-strict denominator for **every** method. Endpoint = `recall_ldca_topn2` (pocket closest by DCA must have DCA ≤ 4 Å and rank ≤ N+2); "any pocket in top-N+2 with DCA ≤ 4 Å" reported alongside. Oracle checks: real ligand atom 100%, ligand centroid 95.2%, site Cα centroid 76.8%.
