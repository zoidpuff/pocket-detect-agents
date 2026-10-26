# Review round 1 — biophysics, ligand binding, AF model quality

Reviewer: biophysics/AF agent. Scope: DESIGN.md v1 §2.1 features, §2.3, §2.4, O4–O6. Labels: **MUST** / **SHOULD** / **COULD**.

## 1. Physical/analytical feature set

Buriedness-first design is right (LIT §5). Gaps and mis-specifications:

- **MUST: say what the feature input contains.** Features must be computed on polymer atoms only: no waters, ligands, metal ions or glycans; MSE is mapped to MET; the highest-occupancy altloc is kept; no H. Use the same input for every baseline. Without this, a retained cofactor or metal becomes a label leak. `structure.py` classifies these atoms, but DESIGN does not state the rule.
- **MUST: make metal sites reachable by candidate points.** Ions sit 2.0–2.4 Å from coordinating O/N/S atoms. The solvent-shell rule (≥ vdW + 1.4 Å ≈ 3.1 Å) removes exactly those positions.
  - Add a third candidate class: points ≥ 1.8 Å from atoms that have ≥ 3 coordinating atoms (His ND1/NE2, Cys SG, Asp/Glu carboxylate O, Met SD, Asn/Gln OD1/OE1, backbone O) within 2.9 Å.
  - Give each point features: coordinating-atom count, the distance spread of those atoms, and their angular dispersion (mean resultant length of the unit vectors; tetrahedral/octahedral ≈ 0).
  - Use a 3.5 Å "tolerant" variant for apo and AF input, where rotamers are off.
  - The "all" regime cannot reach high recall without this. LIT notes P2Rank's 57% maximum recall.
- **SHOULD: switch hydrophobicity from a residue scale to atom types.** "Fauchère–Pliska atom-projected" smears a residue scale onto atoms: a Lys Cε counts as hydrophobic, and a Tyr OH counts as aromatic-apolar. Use atom classes instead (carbon bonded only to C, versus polar N/O/S-H), weighted by exposure. These are rotamer-robust (LIT §5).
- **SHOULD: add a desolvation / hydrophobic-enclosure proxy at little extra cost.** In the existing ray cast, record the class of the *first atom hit*. Per point, output the fraction of rays hitting apolar C, polar N/O, and charged atoms. Apolar enclosure from several directions is the WScore/SiteMap-style signal, and atom-density counts within a sphere do not capture it.
- **SHOULD: refine the electrostatics.**
  - Use Debye-screened Coulomb (κ ≈ 0.1 Å⁻¹) with ε = 4r and a 12 Å cutoff.
  - Fix protonation at pH 7: His neutral, termini charged.
  - Expose both the signed φ and |φ|. Phosphate and nucleotide sites are positive and cation/metal sites negative, so a monotone split on signed φ alone wastes trees.
  - Coulomb with ε = 4r misses focusing in clefts. That is acceptable, because GBM interactions with buriedness capture it.
- **SHOULD: replace "H-bond capacity" with unsatisfied polar atoms.** Count donors and acceptors (including backbone N/O) that are exposed to the point and have no intra-protein partner within 3.5 Å. Hot spots are enriched in these.
- **COULD: add aromatic/cation-π features.** For each point, count aromatic ring centroids (Phe/Tyr/Trp/His) within 6.5 Å whose ring normal is within 35° of the centroid-to-point vector (stacking/cation-π geometry), plus Arg guanidinium planes. Composition features only partly capture this.
- **MUST: add a flexibility proxy usable on PDB, AF and apo input.** B-factor is rightly dropped. Replace it with:
  - weighted contact number per residue (WCN = Σ 1/r² over Cα; O(N), correlates with B-factors about as well as GNM);
  - SHOULD: GNM mean-square fluctuation from the 20 slowest modes (sparse `eigsh` on the Cα Kirchhoff matrix, 7.3 Å cutoff; under 0.3 s for N ≤ 2,700). It is the same quantity on all three input types, so it does not create an AF-versus-PDB distribution shift.
- **SHOULD: add a side-chain-gating feature and side-chain augmentation.**
  - Add a gating feature: Δ = backbone+Cβ buriedness − full-atom buriedness. A large Δ means "pocket occluded by side chains".
  - Augment training by truncating 10–20% of side chains to Cβ, chosen uniformly at random over *all* residues, never site-biased, to avoid a label shortcut.
  - Training only on holo rotamers otherwise teaches "exact holo enclosure".

## 2. AF mode — mechanisms and validation

- **MUST: candidate generation must respect soft occlusion.** If pruned or low-weight atoms still exclude grid/SAS points, an occluded real pocket has no points at all, and soft weights only change buriedness. Exclude points only near atoms with w > 0.5. Ablate this.
- **MUST: define the hallucination rate on the unpruned model.** Compute pocket pLDDT from all original residues within 4 Å of the pocket points, pruned ones included. Otherwise pruning gives 0% by construction. Report it alongside precision of the top-N+2 pockets. Unmatched ≠ false (LIGYSIS is incomplete), so report it as a rate and do not optimise it.
- **MUST: run controls showing the gain is not just "fewer atoms".**
  1. *Matched-random pruning*: remove random segments of high-pLDDT surface loops with the same length and count distribution. The gain should vanish or be negative.
  2. *Same preprocessing for baselines*: run P2Rank and fpocket on our pruned model. Report the AF-mode gain over naive input and over pruned-input baselines separately.
  3. *Oracle crop*: crop the AF model to the residues observed in the holo PDB entries (SIFTS). Report gap closure = (AF-mode − naive)/(oracle − naive).
  4. *Site-loss cost*: report the fraction of true-site residues pruned. Some true sites lie in loops that close over the ligand ("lid" loops), or in conditionally folded regions.
  5. *No-op check*: on PDB input (pLDDT absent, or set to 100), AF mode must give identical output.
- **SHOULD: tune thresholds on dev AF models (training UniProts), never on O5.** Search a small grid (prune cut-off 40/50/60, internal minimum length 5/7/10, soft-weight ramp). AF2BIND's 50/7 is a convention (LIT §4). AFDB stores per-residue pLDDT; AF3 stores per-atom pLDDT, so average per residue for consistency.
- **SHOULD: make the side-chain ×0.5 weight learned instead of global.** High-pLDDT AF side chains are mostly correct, and a blanket ×0.5 erodes enclosure everywhere. Supply both full-atom and backbone+Cβ buriedness, plus the Δ above, and let the AF-aware ranker weight them. Keep ×0.5 only as an ablation arm.
- **MUST: make PAE consistent between training and test.** PAE is not in the local tars. A feature present only at test time cannot be learned, and "download PAE for test only" makes it an untuned rule. Choose one of:
  - (a) download PAE for the AF-aware training UniProts as well (compressed, a few GB), or
  - (b) SHOULD, structure-only surrogate that is always available: split domains by spectral clustering of the pLDDT-weighted Cα contact graph, then flag pockets lined by ≥ 2 domains with low inter-domain contact density. Validate the surrogate against real PAE on the test downloads (AUROC for "median PAE > 10 Å").
- **MUST: transfer AF-aware training labels by local superposition, not residue numbers alone.** Point labels need ligand coordinates. Superpose holo-site backbone atoms (within 6 Å of the ligand) onto the model, AlphaFill-style, and record the local RMSD. Keep high-RMSD sites, because they are the realistic case, and stratify O5 by local RMSD. Apply the 30% leakage filter at UniProt level against the O5 proteins.

## 3. Dimers / interface sites

- **MUST: handle symmetric copies consistently.** In assembly input, each copy of a homodimer site is physically real. Count each copy as a ground-truth site, *or* collapse both the ground truth and the predictions by symmetry. Never collapse only one side, because that inflates or deflates top-N+2.
- **SHOULD: add per-point interface features.** These are the number of distinct chains within 8 Å, the minority-chain atom fraction, and inter-chain gap buriedness. Packing gaps at interfaces are a classic false-positive source, so the model needs to see them explicitly.
- **SHOULD: require cross-entry support** (the same ligand at the same interface in ≥ 2 entries) to exclude crystal-contact sites (LIT §2.8).
- **COULD: evaluate AF-Multimer/AF3 dimer models** as a third input type, with ipTM stratification.

## 4. Apo and cryptic sites

- **SHOULD: add GNM/ANM-derived signals** (shared with §1). These are residue fluctuation, **hinge score** (sign changes of the two slowest-mode eigenvectors between contacting residues), and distance to the nearest hinge. Cryptic sites are enriched at hinges and in flexible but partially buried regions (CryptoSite: 0.74 AUC without MD).
- **SHOULD: add partially-formed-pocket features.** These are the side-chain-gating Δ, large-scale (16–24 Å) buriedness combined with low small-scale buriedness (a groove that could close), and exposed apolar patch size within 8 Å.
- **COULD: add a Lacuna-like openable buriedness.** Generate about 6 ANM conformers along modes 1–3 (±1.5 Å RMSD), recompute backbone+Cβ buriedness, and take the per-point maximum. It costs about 6× buriedness time, so make it optional (`--cryptic`) and report its runtime separately.
- **SHOULD: add apo chains to training.** Use apo chains from training clusters (AHoJ/CryptoBench train split), with holo sites mapped by residue, and apply leakage control against the CryptoBench and PocketMiner test sets. For O4, report per-protein macro *and* pooled AUPRC (LIT §1.6 shows 0.50 vs 0.76 for the same method), and report the apo-versus-holo gap on paired structures.

None of these change the project scope. Every addition is analytical and runs on CPU and GPU.
