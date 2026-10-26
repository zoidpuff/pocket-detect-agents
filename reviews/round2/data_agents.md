# Review round 2: data/leakage and agent failure modes (DESIGN v2)

**Verdict.** Most round-1 MUSTs are now addressed: the model is pre-registered per benchmark; the input policy and invariance test are in place; there are metal candidates and labels from superposition; the test gate and ledger exist; seeds are reconciled; there is a symmetric-copy rule and an AF no-op check. Six gaps remain where an agent building §2.5–2.7 would have to choose for itself. There is also one new path by which the test set could leak into development.

Checked locally:
- The BioLiP2 date (2025-11-12) matches `LAST_MODIFIED.txt`.
- LIGYSIS is human-only, so the AFDB human tar is the right source.
- The PocketWatch test cutoff is 2024-01-01.
- `docs/EVALUATION.md` does not exist yet.
- DATA.md lists only PDB and AFDB, although `data/manifests/*.tsv` hold checksums for the rest.

## MUST

1. **Test view before freeze (new risk).** In §4, test v1 runs at T+10 h but the feature freeze is at T+12 h. Pick one:
   - move test v1 to after the freeze; or
   - state that changes between T+10 and T+12 h may use dev only, and label any model trained in that window "post-hoc".
2. **Freeze the exclusion union before training (new risk).** The §2.5 exclusion list needs the AF and dimer sets, but tracks C and D build those in parallel with A.
   - Commit `data/manifests/test_union.tsv` (ID, sequence, source set) and its sha256 before the training build. The training artifacts record that checksum.
   - Any test set added later (e.g. allosteric) is checked against the frozen training set and its overlap reported. It never joins silently.
3. **Complete the leakage rule (round-1 data MUST 2).**
   - Add E < 1e-3 to "≥ 30% identity over ≥ 80 aligned residues".
   - Use Foldseek max(qTM, tTM).
   - Include all polymer entities of HOLO4K and COACH420, and all dimer partner chains.
   - Apply the same rule between train and dev: 30% clusters alone leak shared domains into dev.
   - Pre-declare the fallback if more than 50% of clusters are removed: per-benchmark exclusion models.
   - Pre-declare the fallback if the TM ≥ 0.5 fold-disjoint set keeps less than 20%: ECOD/CATH topology exclusion.
4. **Numeric reproduction gates.** "Must reproduce" has no tolerance. Set these:
   - LIGYSIS table from the authors' pickles: ±0.5 pp.
   - P2Rank 2.4.2 on COACH420/HOLO4K: ±2 pp of 72.0/68.6.
   - P2Rank `alphafold` config on PocketWatch: per-target AUPRC 0.477 ± 0.03.
   
   If a gate fails, stop and debug. Use the authors' `*_DEF_TRANS` centroids.
5. **Define metrics for O3, O4 and O6.** They have margins but no metric.
   - O3 and O4: per-target residue AUPRC, macro primary, pooled also reported.
   - Residue score = max calibrated score of the pockets that contain the residue.
   - O3 masked sites are neither positive nor negative.
   - O3 runs in standard mode. ESMFold B-factors are pLDDT, so `--af-mode` results there are descriptive only.
   - O6: top-N+2 DCA ≤ 4 Å per site class, with N = unique sites.
6. **Make §2.7 buildable.** Specify:
   - Source: PDB entries of LIGYSIS UniProts, plus PocketWatch `interface_only` val/test sites if wanted.
   - Ligands: BioLiP-relevant, per entry, not pooled.
   - Contact: heavy atom ≤ 4 Å.
   - Partner chains must have ≥ 50 residues.
   - Isolated-chain input: run every lining chain; a hit on any one counts.
   - A minimum number of unique sites per class, below which the class is descriptive only.

## SHOULD

1. **§2.5 determinism.**
   - Pool from up to 20 entries, ordered by resolution, then PDB ID.
   - Exclude BioLiP peptide, DNA and RNA ligands.
   - Cluster sites with the function in `data/raw/ligysis/ligysis_pipeline`, using explicit parameters.
   - Choose the "~30% assembly-1" inputs with `SEEDS[0]` from `author_and_software_defined` assemblies of ≥ 2 chains, labelled with ligands from all chains.
   - Log the count dropped at each filter, including chains with no UniProt mapping.
2. **§2.6 precision.**
   - Define "low-weight" as w ≤ 0.5.
   - Add the round-1 oracle-crop control (crop to PDB-observed residues; report gap closure).
   - Drop sites with < 95% residue-type match, and count them.
   - Project centroids with the PocketWatch Kabsch recipe.
   - Report AFDB coverage of the training UniProts and define the fallback if coverage is low.
3. **C1 pairing.** If our re-run beats 60.4, compute the paired bootstrap against the re-run predictions.
4. **DATA.md.** Fill it now from the manifests. Add a PDB-mirror cutoff, defined as its maximum `initial_release_date`.
5. **Control flags.** Flag a run if the method scores below random-surface + 5 pp, or the oracle scores below 95%.

## COULD

- Re-score the baselines on LIGYSIS_NI from the authors' pickles (free).
