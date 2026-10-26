# Review round 1: data, benchmarks, leakage, prior work

**Verdict.** The plan is right in direction: LIGYSIS as the primary benchmark, cluster hold-out, and baselines we re-run ourselves. Several points are under-specified, and as written they would either leak test information or give methods different inputs. Every fix below stays within scope.

## Facts from data/raw (inspected read-only)

- **PocketWatch is MIT-licensed** (stated in its Hugging Face README), so O3a is feasible.
  - Its inputs are **ESMFold2 predicted folds**, not experimental structures.
  - A `site_policy` mask is included; 3,318 sites are `interface_only`.
  - `projection/` provides a recipe that projects site centroids onto any structure (Kabsch on CAs matched by canonical position).
- **The BioLiP2 snapshot is dated 2025-11-12**, not "current weekly". DATA.md must record this date.
- **LIGYSIS sites use UniProt numbering (SIFTS, April 2024). The PDB mirror has no SIFTS**, so a residue-level SIFTS file has to be downloaded.
- **AFDB v6 human contains fragmented models** (590 F2+ files) and no PAE. `alphafold/v6/msa_depths.csv` exists and can serve as a template-richness stratifier.
- **The LIGYSIS published predictions cover only some chains:** P2Rank 2,402, P2Rank-CONS 2,579, fpocket 2,759, out of 2,775. Our re-runs must use the identical denominator, with chains that have no prediction counted as misses.

## MUST

1. **Pre-register which model produces each headline number.**
   - About 40% of full-LIGYSIS sites are ions, and the 60.4 bar is measured on full LIGYSIS. A `small-molecule` default model is therefore mis-matched to the primary benchmark.
   - Fix the default before E1. I recommend `all` labels, which matches LIGYSIS relevance.
   - Report both label regimes everywhere, and make LIGYSIS_NI co-primary (with baselines re-scored on NI).
2. **Make leakage control domain-aware, and measure what it removes.**
   - MMseqs2 `cov ≥ 0.5` misses a shared domain inside a long multi-domain chain. Instead, exclude any hit with ≥ 30% identity over ≥ 50 aligned residues and E < 1e-3.
   - For Foldseek, use max(qTM, tTM).
   - Include every entity of HOLO4K asymmetric units and every dimer-benchmark partner chain.
   - Report the clusters retained before E2. If the union exclusion removes more than 50%, pre-declare per-benchmark exclusion models: same frozen recipe, only the exclusion list changes, and GBM retraining is cheap.
   - Expect the TM ≥ 0.5 fold-disjoint ablation to be nearly empty, since tests include Rossmann and TIM-barrel folds. Pre-specify a fallback (ECOD/CATH topology).
3. **Define 3D point labels under pooling.**
   - Pooled ligands from other entries need coordinates on the representative chain. Superpose them by Kabsch on the CAs of the contact residues, and keep a ligand only if the site RMSD is ≤ 2 Å.
   - Exclude BioLiP `peptide`/`dna`/`rna` entries; LIGYSIS uses HETATM ligands only.
   - Log chains that have no UniProt mapping.
4. **Stay faithful to the LIGYSIS protocol.**
   - First reproduce the published table to ±0.1 from the authors' pickled predictions.
   - Then re-run **P2Rank 2.4.2 and fpocket 4.0** (the paper's versions) on the identical 2,775 cleaned files before switching to 2.5.
   - Use the authors' `*_TRANS` site centroids.
5. **AF benchmark numbering.**
   - Map sites onto AFDB v6 only where the SIFTS UniProt residue type matches the AF residue type. Drop any site with < 95% matching residues, and report how many were dropped.
   - Handle fragment offsets ((k−1)·200), or exclude proteins longer than 2,700 residues.
   - The unit is the UniProt entry, with N = the union of its segments' sites.
   - Project ligand centroids onto the AF model with the PocketWatch recipe.
   - Compute the **same metric on the PDB rep chains of the same subset**, so the PDB→AF delta is like-for-like.
6. **Dimer benchmark integrity.**
   - Assembly-1 is not the same as the PISA-preferred assembly. Require `author_and_software_defined_assembly` or cross-entry support (the same interface site in ≥ 2 entries) to exclude crystal contacts.
   - Exclude partners shorter than 50 residues.
   - Fix the symmetry rules in advance: a match to any symmetric copy counts, N = unique sites, and extra hits on other copies are redundant, not FP.
   - For hetero-interface sites given isolated-chain input, run each chain and count a hit on either.
7. **PocketWatch protocol.**
   - Masked sites are neither positives nor negatives.
   - Use their centroids, their ESMFold2 inputs and their strata.
   - Run P2Rank with the `alphafold` config, using the paper's per-target AUPRC of 0.477 as the reproduction check.
   - Pre-declare whether `--af-mode` is used.

## SHOULD

1. **Cheap fairness controls.**
   - Score the authors' LIGYSIS predictions (DL methods, P2Rank-CONS, fpocket+PRANK+CONS) through our harness instead of installing those tools. This gives identical inputs by construction and saves hours.
   - Add the reverse data control: **our model trained on CHEN11 only** (P2Rank's training set).
2. **LIGYSIS-clean subset.** Drop sites that overlap sc-PDB, MOAD, PDBbind, CHEN11 or JOINED (from the authors' overlap analysis). This removes the DL baselines' training advantage.
3. **Bootstrap over 30% sequence clusters**, not chains. LIGYSIS is family-redundant.
4. **Add stratifications** (LITERATURE §6.5 lists these; DESIGN omits them):
   - ligand class: ion, cofactor, drug-like, sugar, lipid;
   - apo vs holo representative chain;
   - pocket novelty: maximum coverage of a site's residues by training sites (PocketWatch's `max_qcov` idea).
5. **Apo→holo drop.**
   - Run all methods on the CryptoBench holo partners too; their CIFs are already local.
   - Use the CryptoBench train folds as the apo dev set.
   - On PocketMiner, apply its residue mask and report macro and pooled AUC together.
6. **Control for AF holo-memorisation.**
   - Add AFDB v6 models of PocketWatch test UniProts, whose sites first appear in 2024 or later. This is valid only if the AFDB v6 template cutoff predates 2024; verify that first.
   - Stratify by `msa_depths.csv`.
   - Count pockets in PDB-uncovered regions as "unsupported", not as FP: those regions are unlabelled.
7. **Dimer seed set.** Take PocketWatch `interface_only` val/test sites (temporal, already pooled) and build their assemblies from the PDB mirror.
8. **O2 like-for-like.** The published P2Rank 72.0 / 68.6 refer to the non-mlig COACH420/HOLO4K. Re-run both variants and cross-check against P2Rank's `eval-predict`. Report HOLO4K with duplicate interfaces removed.
9. **Realistic dev set.** Build dev with the LIGYSIS recipe (I_rel site clustering, single representative chain), so that β₀, τ and NMS thresholds transfer to the test benchmark.

## COULD

- Add the AF2BIND 67-protein test (already local) as a residue-level benchmark.
- Analyse overlap with HOTPocket's 7-method pockets on human AF models.
- State why sc-PDB is not a test set: it overlaps every benchmark and has one site per entry.
- Train the AF-mode ranker on PocketWatch train folds (ESMFold2), since local AFDB covers only 48 proteomes plus SwissProt.

## Prior-work gaps

- **Allosteric sites.** PROMPT names them, but LITERATURE and DESIGN barely cover them. Add an ASD/ASBench or PASSer-test allosteric stratum after leakage filtering (SHOULD).
- **Pocket-level cross-fold leakage** (recurrent P-loop and Rossmann cofactor sites). Apply PocketWatch's coverage and PLINDER's pocket-similarity methodology (SHOULD 4).
- **Positioning claims.** Co-folding finders (AF3, Boltz-2, Chai-1) are correctly out of scope; REPORT should still name them. Seq2Pocket (≈69%) is correctly framed as the ceiling, so do not claim SOTA over it without a like-for-like run.
