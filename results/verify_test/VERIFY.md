# Independent verification of BURROW TEST headline numbers (v1, v2)

Verifier: independent agent, 2026-10-27. No `pocketdetect` imports in any scorer. All scripts are in
`scripts/verify_test/`; raw outputs are in `results/verify_test/`. Third-party pickles were loaded only through a copy
of `scripts/data/safe_pickle.py` (`vt_safe_pickle.py`).

## Method

* **C1 (authors' code).** `ligysis_authors_eval.py` runs the bartongroup LBS-comparison v2.0.0 functions
  `filter_redundant_pockets2` and `compare_ALL_predictions`, pulled from `benchmark_lib.py` with `ast`, and
  then applies notebook 12's selection: rounding to 0.01, closest pocket by DCC with ties broken by higher IRel,
  `distance <= 12 & RANK <= N_SITES + 2`, denominator `len(LIGYSIS_NR)`. Everything runs in the authors' TRANS
  frame:
  * Sites come from `ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl`.
  * BURROW and the re-run are read from their TRANS-frame CSVs (`preds/raw`).
  * The BURROW inputs (`baselines/outputs/inputs/ligysis`) are byte-identical to the authors' `trans_rep_chains`
    PDBs (checked with `cmp`).
* **C1 / C1-strict / C2 (own minimal scorer).** `ligysis_dca_check.py` scores in the wwPDB frame, using
  `preds_orig` and `data/processed/gt/ligysis_lbs/sites.tsv` (centres and `ligand_atoms`):
  * C1: closest pocket by DCC (LDCC12).
  * C1-strict: the pocket with the smallest DCA must have DCA ≤ 4 Å and rank ≤ N+2.
  * C2: pooled top-1000 by score; a pocket is a TP if its DCC ≤ 12 Å to any scored site of its chain.
  * Frame check: each raw TRANS-frame centre is mapped back with the authors' `PDB_rot_matrices` and
    `PDB_orig_centroids` (x = Rᵀx_T + c0) and compared with `preds_orig`.
* **Bootstrap.** `paired_boot.py` runs a paired cluster bootstrap over the 30 % clusters (1,000 resamples) on my
  own per-site tables.
* **O2b.** `holo4k_dca4.py` is my own minimal ground truth from the explicit codes in `holo4k(mlig).ds`: HETATM
  groups clustered at 1.7 Å, H atoms kept as in P2Rank, first record per atom name. Score = DCA4 top-n.
* **O5.** `ligysis_dca_check.py` run against `data/processed/gt/af_set/sites.tsv`.

## Results

| claim | reported | recomputed | agree (±0.2)? | notes |
|---|---|---|---|---|
| fpocket+PRANK published, LIGYSIS top-N+2 (sanity anchor) | 60.42 | **60.42** (4061/6721) with the authors' code; 60.42 with my own scorer | yes | Also matches top-N 48.8, max 91.3 and P1K 81.7 exactly. 16 chains have no published predictions and count as misses. |
| fpocket 4.2.3+PRANK 2.4.2 re-run, top-N+2 | 59.50 | 59.50 (authors' code), 59.50 (own) | yes | |
| C1 v1 s0 / s1 / s2 | 64.29 / 64.51 / 64.90 | 64.31 / 64.54 / 64.90 (authors' code); the same values from my own scorer | yes | The 1–2 site differences are exact-DCC ties that the harness breaks by IRel, as the authors' rule says. |
| C1 v2 s0 / s1 / s2 | 72.47 / 72.01 / 72.28 | 72.47 / 72.01 / 72.28 (both scorers) | yes | Per site, identical to the harness for all 6,721 sites. |
| C1 Δ and 95 % CI (v2 s0 vs published / re-run) | +12.05 [9.92, 14.25] / +12.97 [10.86, 15.17] | +12.05 [9.92, 14.25] / +12.97 [10.86, 15.17] | yes | Every seed and version has Δ ≥ 2 with the CI excluding 0. The tightest case is v1 s0 vs published: +3.88 [2.00, 6.04]. |
| C1-strict v1 s0 / s1 / s2 | 54.44 / 54.13 / 54.52 | 54.44 / 54.13 / 54.52 | yes | n = 6,387 sites for every method. 0 per-site differences from the harness. |
| C1-strict v2 s0 / s1 / s2 | 54.19 / 54.16 / 54.00 | 54.19 / 54.16 / 54.00 | yes | |
| C1-strict comparators (published / re-run) | 31.58 / 31.80 | 31.58 / 31.80 | yes | CIs reproduced, e.g. v2 s0 vs published: +22.61 [20.00, 25.46]. |
| C2 P1K v1 s0 / s1 / s2 | 95.7 / 96.4 / 96.9 | 95.7 / 96.4 / 96.9 | yes | |
| C2 P1K v2 s0 / s1 / s2 | 96.1 / 96.5 / 96.9 | 96.1 / 96.5 / 96.9 | yes | |
| C2 P1K comparator | 81.7 | 81.7 (published, using the authors' P1K rule), 81.7 (re-run) | yes | |
| O5 AF set, v2 s0 vs P2Rank `-c alphafold` | 64.55 vs 51.41 (Δ +13.13) | 64.55 vs 51.41 (Δ +13.14) | yes | v1 s0 is 59.20 (reported 59.18). P2Rank without AF mode scores 47.32 and fpocket+PRANK 48.60. |
| O2b HOLO4K(mlig) DCA4 top-n, s0 | v1 78.40 / v2 78.21 vs P2Rank 2.5.1 76.58 | v1 78.32 / v2 78.13 vs 76.49 | yes | My ground truth has 6,885 ligands (harness 6,874, P2Rank 6,877). Δ for v2 is +1.64 (reported +1.63). |

## Provenance and sanity checks

* **Models.** In all 96 test `run.json` files (`results/test/v{1,2}/preds/*/*/run.json`), the model sha256 matches
  both the on-disk file and FREEZE.md (v1) or FREEZE_v2.md (v2). Every run has rc = 0 and
  n_pred_files = n_inputs. The ledger's sha256 prefixes match these.
* **Timeline.** FREEZE v2 was committed at 22:09Z (0d925b2). The v1 test scoring (first ledger row) ran at 22:41Z,
  so v2 was frozen before any v1 test score was computed.
* **Input files.** The benchmarks differ in whether ligands are present in the inputs:
  * LIGYSIS inputs: 0 HETATM records across all 2,775 files.
  * AF inputs: 0 HETATM records in a 50-file sample.
  * HOLO4K and COACH420 inputs do contain HETATM ligands, and the baselines got these same files.
  * BURROW drops them. I re-ran the frozen v2 `small` s0 model on CPU for 3 HOLO4K files (1hdx, 1jiu, 1axe),
    once as given and once with HETATM records stripped. Rank, score and centre columns were identical between the
    two runs, and identical to the test predictions.
* **Pockets per chain on LIGYSIS:**

  | method | mean | median | range |
  |---|---|---|---|
  | BURROW v1 | 32.9 | 33 | 4–50 |
  | BURROW v2 | 9.1 | 8 | 3–50 |
  | fpocket+PRANK | 20.9 | 17 | 0–349 |

  No BURROW chain has zero predictions, and no score is NaN.
* **Coordinate frames.**
  * Mapping the TRANS-frame predictions back to the wwPDB frame reproduces `preds_orig` exactly (maximum deviation
    0.000 Å; 0.0005 Å for the re-run).
  * In 5 random chains, the top-3 pocket centres lie 1.9–4.3 Å from the nearest protein heavy atom, in both the
    wwPDB and TRANS frames.
  * C1 computed in the two frames gives the same number.
* **C1-strict ligand atoms.** In 9 of the 12 sampled rep-entry placements, every real heavy atom from the mirror
  mmCIF is present in `sites.tsv`, matched within 0.000 Å. In the other 3, my lookup could not find the residue
  because of two-letter label chains, so they are a limitation of my lookup rather than a mismatch. Placements
  superposed from other PDB entries were not re-derived.
* **Leakage.** In 20 random LIGYSIS test chains, no UniProt appears in the train manifest (6,475 rows) or the dev
  manifest (579 rows). Extending the check to all 2,565 LIGYSIS UniProts and all PDB IDs also finds 0 overlap.
  30 % sequence-cluster overlap was not checked.

## Caveats

These do not affect agreement but matter for interpretation.

1. **v2's C1 gain comes from shortening the output list.** v2 is v1 plus a score floor (τ = 0.03).
   * Under the LIGYSIS closest-pocket rule this raises C1 from 64.3 to 72.5.
   * It lowers max recall from 99.3 to 84.2 and any-pocket top-N+2 from 80.1 to 76.9.
   * It leaves C1-strict unchanged (54.4 vs 54.2).
   * v1 already passes C1 on its own. Its smallest margin is +3.9 [2.0, 6.0] vs the published comparator, and v1 is
     still 10 points ahead on the any-pocket variant.
2. **v2 was frozen while v1 test predictions already existed.** The v1 test prediction files were on disk from
   20:58Z, 70 minutes before FREEZE v2. Scoring happened later, but the files do not let me rule out that someone
   looked at them.
3. **O5 precision is lower for BURROW.** On the AF set, BURROW's P1K is 58.9, below P2Rank-AF (73.9) and
   fpocket+PRANK (67.0). This is descriptive only.

## Verdict

**I found no errors.** Every headline number I recomputed (C1, C1-strict and C2 for 3 seeds and both versions; O5 and
O2b for seed 0) agrees within ±0.2 points. That includes the reproduction from the LIGYSIS authors' own code, which
also gives exactly the published 60.4 for fpocket+PRANK. Paired CIs reproduce, and provenance and leakage checks pass.
The claims stand as reported, subject to the interpretation caveats above.
