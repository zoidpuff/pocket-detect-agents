@@@CAPTION fig1_pipeline
title: The BURROW pipeline.
text: Protein atoms are read from a PDB, mmCIF or AlphaFold file; in `--af-mode`, low-confidence segments are removed and the remaining atoms are weighted by pLDDT. (1) Candidate volume elements are placed on a 1 Å solvent-shell grid, with extra metal-coordination candidates. (2) Each point is described by ray-cast buriedness through a full-atom and a backbone-only grid, the chemistry of the first atom hit and local composition. (3) A LightGBM model scores the points. (4) Pockets are formed on several scales. (5) A second LightGBM model, trained on cross-fitted point scores, ranks the pockets. (6) Suppression at 10 Å and (7, v2 only) an output score floor give the final ranked list.
@@@END

@@@CAPTION fig2_ligysis
title: LIGYSIS results.
text: Top-N+2 recall at DCC ≤ 12 Å (C1), at DCA ≤ 4 Å (C1-strict) and pooled Precision1K (C2) for BURROW v1 and v2 (three seeds), the published fpocket+PRANK and P2Rank predictions, and published deep-learning methods scored with the same harness. Error bars are 95% cluster-bootstrap CIs where paired predictions exist.
@@@END

@@@CAPTION fig3_prereg
title: Pre-registered secondary endpoints (family 2).
text: Differences between BURROW and the best comparator (percentage points, 95% paired cluster-bootstrap CIs) for test versions v1 and v2 and three seeds. The dashed line marks the non-inferiority margin of −2 points.
@@@END

@@@CAPTION fig4_composition_tradeoff
title: Where the LIGYSIS margin comes from.
text: Differences to fpocket+PRANK by ligand class (ion, nucleotide/cofactor, other small molecule), and the trade-off between output length and recall from v1 to v2: closest-pocket top-N+2 rises as the list shortens, while maximum recall and any-pocket recall fall and DCA4 recall is unchanged.
@@@END

@@@CAPTION fig5_speed
title: Runtime.
text: Time per structure and throughput measured in one session on the same 200 LIGYSIS chains for every tool (shared node). BURROW presets fast (optional), default (pre-registered) and accurate, P2Rank 2.5.1 and fpocket 4.2.3, and throughput on 24 CPU workers and on GPUs.
@@@END

@@@CAPTION fig6_af_dimers
title: AlphaFold mode and dimers.
text: (A) AlphaFold test set: recall and hallucinated pockets for BURROW in AlphaFold mode, BURROW on the unmodified model, BURROW with a pLDDT < 70 pocket filter and P2Rank. (B) Dimer benchmark: recall of interface and single-chain sites with full-assembly and isolated-chain input.
@@@END

@@@CAPTION figS1_strata
title: Stratified test results.
text: Differences to the comparator by fold novelty (maximum Foldseek TM-score to training), ligand class, site pLDDT band and dimer site class (descriptive; seed 0).
@@@END

@@@CAPTION figS2_ablations
title: Ablations and controls.
text: Dev and test differences for fold-disjoint training, training on CHEN11 only, shuffled labels, removal of feature groups and P2Rank retrained on our training data, together with the oracle, protein-centroid, random-surface and surface-tiling controls.
@@@END
