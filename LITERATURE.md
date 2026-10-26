# LITERATURE — ligand-binding pocket detection & ranking (state of the art as of 2026-10-26)

Scope: the literature and datasets that bear on our goal: a fast, mostly analytical/lightweight-learned pocket detector and ranker that beats P2Rank, generalises to apo, cryptic and unseen-fold proteins, handles homo/heterodimers, and has an AlphaFold mode.

**Evidence tags.** Every number below is tagged.
- **[V]** We read it in primary full text during this review (Europe PMC XML, arXiv/bioRxiv HTML, GitHub/Zenodo).
- **[S]** It comes from an abstract or secondary source only.
- **[NV]** Background knowledge that we did not re-verify. Do not quote these without checking.
- "Authors" means the method's own paper reported the number. "Independent" means a third party re-ran the method.

Numbers measured under different protocols are **not comparable**. Section 3 explains why.

---

## 0. Executive summary (one paragraph)

**The P2Rank family is still the bar to beat on leakage-controlled, multi-site benchmarks.**
- On LIGYSIS (Utgés & Barton 2024), top-N+2 recall at DCC ≤ 12 Å is:
  - fpocket+PRANK 60.4%; DeepPocket-RESC 58.1%; P2Rank-CONS 53.9%; P2Rank 51.9%.
  - Every newer deep-learning (DL) model scores lower: GrASP 49.9, PUResNet 41.1, VN-EGNN 40.9, IF-SitePred 25.7 [V].
- DL methods win on **precision and pocket shape**, but output too few pockets or redundant ones.

**Two observations set the realistic target.**
1. fpocket's candidate set already contains about 91% of LIGYSIS sites; P2Rank's contains only 57% [V].
   - So the headroom lies mainly in (a) **high-recall candidate generation** and (b) a **better ranker and deduplicator**, not in new cavity geometry.
2. The only credible claims above about 60% top-N+2 on LIGYSIS use large protein language models (PLMs):
   - Seq2Pocket, a fine-tuned ESM-2 3B, claims about 69% (author-reported) [S/V-partial].
   - zPocket, on frozen co-folding features, is evaluated on a new temporal benchmark instead [V].

**Ground-truth construction matters as much as the method.** Methods move by 5–20 points depending on:
- whether ions are included;
- whether sites are pooled across PDB entries;
- the asymmetric unit vs the biological assembly;
- DCA vs DCC thresholds;
- whether matching is one-to-one.

---

## 1. Methods

### 1.1 Geometric methods

These detect concavities from atom coordinates alone. Most are fast (well under 1 s per protein), have no training-set leakage, and recall well with poor ranking.

Citations and mechanisms are [NV] background knowledge unless tagged.

| Method | Core idea | Ranking / score | Notes |
|---|---|---|---|
| **SURFNET**: Laskowski 1995, *J Mol Graph* 13:323, doi:10.1016/0263-7855(95)00073-9 | "Gap spheres" between atom pairs, shrunk until empty; clusters of spheres form clefts. | Cleft volume | Re-implemented as Surfnet+ in ConCavity. LIGYSIS: top-N+2 45.8, Precision1K 39.5, RRO 71.7 [V]. Predicts large clefts. |
| **LIGSITE**: Hendlich et al. 1997, *J Mol Graph Model* 15:359, doi:10.1016/S1093-3263(98)00002-3 | Grid. Counts protein–solvent–protein (PSP) events along x, y, z and the 4 cube diagonals; points with PSP ≥ threshold are pocket points. | Size / buriedness | Ligsite+ on LIGYSIS: top-N+2 48.4, Precision1K 52.3, best RRO 77.6 / RVO 77.0 [V]. Simple, GPU-friendly, strong baseline. |
| **LIGSITEcs / LIGSITEcsc**: Huang & Schroeder 2006, *BMC Struct Biol* 6:19, doi:10.1186/1472-6807-6-19 | PSP events on the Connolly surface (surface–solvent–surface). The "csc" variant re-ranks the top-3 pockets by conservation. | Size, then conservation | One of HOTPocket's 7 constituent methods. It produced no pockets on AF2 models in HOTPocket's pipeline [V]. |
| **PocketFinder**: An, Totrov & Abagyan 2005, *Mol Cell Proteomics* 4:752, doi:10.1074/mcp.T400010-MCP200 | Smoothed Lennard-Jones-like "ligandability" potential on a grid, contoured. | Volume | PocketFinder+ on LIGYSIS: top-N+2 47.8 [V]. |
| **PASS**: Brady & Stouten 2000, *JCAMD* 14:383, doi:10.1023/A:1008124202956 | Iteratively coats the surface with layers of probe spheres and keeps buried probes; "active-site points" are probes with high probe weight. | Probe weight (buriedness) | Fast; superseded. |
| **CASTp**: Liang, Edelsbrunner & Woodward 1998, *Protein Sci* 7:1884; CASTp 3.0: Tian et al. 2018, *NAR* 46:W363, doi:10.1093/nar/gky473 | Alpha shapes / weighted Delaunay with discrete flow. Gives analytic pocket area and volume, and mouth openings. | Volume / area | Web server with precomputed PDB results; no AF2 output in HOTPocket [V]. Tends to give few, large pockets [V]. |
| **fpocket**: Le Guilloux, Schmidtke & Tufféry 2009, *BMC Bioinf* 10:168, doi:10.1186/1471-2105-10-168. fpocket 4.x (conda-forge, MIT). | Voronoi tessellation gives alpha spheres (min/max radius thresholds, about 3.4–6.2 Å by default [NV]), clustered in several steps. | Empirical pocket score; separate druggability score (Schmidtke & Barril 2010) | About 0.2 s per 2,500-atom protein [V, P2Rank paper Table 2]. On LIGYSIS: highest max recall, 91.3%, but Precision1K only 47.3 [V]. Ranking collapses on AF2 models (HAP top-1 0.75 on PDB vs 0.50 on AF2) [V]. **The canonical high-recall candidate generator.** |
| **Ghecom**: Kawabata 2010, *Proteins* 78:1195, doi:10.1002/prot.22639 | Mathematical morphology on a grid: closing with a large probe minus the protein. Multiscale probe radii give a continuous "pocketness" / buriedness. | Pocketness × volume | Grid-native, so easy to put on GPU. Morphological closing is a robust, parameter-light buriedness measure. |
| **KVFinder / parKVFinder / pyKVFinder**: Oliveira et al. 2014, *BMC Bioinf* 15:197; Guerra et al. 2021, *BMC Bioinf* 22:607, doi:10.1186/s12859-021-04519-4 [V citation] | Dual-probe grid. A small "Probe In" and large "Probe Out" define two surfaces; cavities are the space between them. Computes volume, area, depth and hydropathy. | None or size; descriptor output | pyKVFinder is pip-installable and NumPy-native, with C extensions. License: GPL-3.0 [NV]. Same principle as LIGSITE and Ghecom. |
| **PocketPicker**: Weisel, Proschak & Schneider 2007, *Chem Cent J* 1:7, doi:10.1186/1752-153X-1-7 | Grid buriedness index from rays cast in about 30 directions per point. | Buriedness / size | Buriedness by ray-casting is cheap on GPU. |
| **ConCavity**: Capra et al. 2009, *PLoS Comput Biol* 5:e1000585, doi:10.1371/journal.pcbi.1000585 | Combines evolutionary conservation (Jensen–Shannon divergence) with LIGSITE, SURFNET or PocketFinder grids. | Conservation-weighted grid | Its re-implementations (the "+" methods) were used in LIGYSIS [V]. Classic evidence that conservation helps. |
| **DoGSite / DoGSiteScorer / DoGSite3**: Volkamer et al. 2010, *JCIM* 50:2041 (10.1021/ci100241y); 2012, *JCIM* 52:360 (10.1021/ci200454v); *Bioinformatics* 28:2074 (2012); Graef et al. 2023, *JCIM* (DoGSite3) [NV DOI] | Difference-of-Gaussian (image edge) filter on a grid finds pockets and sub-pockets. Descriptors (volume, depth, apolar ratio, element counts) feed an SVM druggability classifier. | Druggability score | 88% druggable/non-druggable accuracy on NRDLD [S]. Runs as a ProteinsPlus server (University of Hamburg); not open source. |
| **SiteMap**: Halgren 2007, *Chem Biol Drug Des* 69:146; Halgren 2009, *JCIM* 49:377, doi:10.1021/ci800324m | Site points with enclosure (buriedness) and hydrophobic/hydrophilic maps. | SiteScore = 0.0733√n + 0.6688·e − 0.20·p; Dscore = 0.094√n + 0.60·e − 0.324·p (n = site points capped at 100, e = enclosure, p = hydrophilicity) [S, consistent across 2 secondary sources] | Known site ranked top-1 in 86% of cases (abstract) [S]. Commercial (Schrödinger). The score form, "size + enclosure − polarity", is a strong analytical prior. |
| **MSPocket**: Zhu & Pisabarro 2011, *Bioinformatics* 27:351, doi:10.1093/bioinformatics/btq672 | Orientation of molecular-surface normals: pocket regions are where normals converge. | Size | Designed to be robust to side-chain differences between apo and holo [NV]. |
| **Others** | **AutoSite** (Ravindranath & Sanner 2016, *Bioinformatics* 32:3142): AutoDock affinity maps plus clustering of high-affinity points; used in Akdel 2022 and HOTPocket [V]. **CAVITY / CavityPlus** (Yuan, Pei & Lai; CavityPlus *NAR* 2018; 2022 update) [NV]: geometric detection plus a ligandability/druggability score; CavitySpace is built on it (§4). **VolSite** (Desaphy 2012, *JCIM*) [NV]: pharmacophore-annotated cavities, used to define sc-PDB pockets. **CAVIAR**: subcavity decomposition [NV]. **SiteHound** (Hernandez et al. 2009, *NAR*): probe affinity maps. **MetaPocket 2.0** (Zhang 2011): consensus of 8 finders. | | |

**Takeaways from the geometric methods**
- **Recall is cheap.** fpocket finds about 91% of LIGYSIS sites somewhere in its list [V].
- **What fails is scoring.** Ligsite+ and fpocket have Precision1K of 40–52%, against 80–93% for ML-scored methods [V].
- **Grid methods (LIGSITE, Ghecom, KVFinder, PocketPicker, DoGSite) suit GPU tensors.** They are dilation/erosion, ray-casting or convolution on a voxel grid.
- **Alpha-sphere methods (fpocket, CASTp) give tighter pockets** but are harder to vectorise.

### 1.2 Energy / probe methods

- **Q-SiteFinder** (Laurie & Jackson 2005, *Bioinformatics* 21:1908, doi:10.1093/bioinformatics/bti315).
  - Computes van der Waals interaction energies of a methyl probe on a grid, clusters favourable probe positions, and ranks clusters by total energy.
  - Reported top-1 success of about 71% on 134 complexes [NV].
- **FTMap / FTSite.**
  - FTMap: Brenke et al. 2009, *Bioinformatics* 25:621, doi:10.1093/bioinformatics/btp036; Kozakov et al. 2015, *Nat Protoc* 10:733, doi:10.1038/nprot.2015.043.
  - FTMap method: about 16 small organic probe molecules are docked globally by FFT (PIPER), minimised with CHARMM plus ACE solvation, and clustered. "Consensus sites" are ranked by the number of probe clusters they hold. The strongest consensus site is the "hot spot".
  - Druggability rules: Kozakov et al. 2015, "New Frontiers in Druggability", *J Med Chem* 58:9063, doi:10.1021/acs.jmedchem.5b00586.
    - The commonly cited criterion is that a druggable site needs a hot spot of at least 16 probe clusters, plus nearby secondary hot spots [S; the exact threshold source was not verified].
  - FTSite (Ngan et al. 2012, *Bioinformatics* 28:286, doi:10.1093/bioinformatics/btr651) ranks FTMap sites for site identification. It reports strong top-3 success on apo structures [NV numbers].
  - Runtime is about 39 min per protein on the web server [V, P2Rank paper Table 2].
- **Mixed-solvent MD** (MixMD, SILCS, cosolvent MD).
  - The most physical approach to finding hot spots and cryptic sites, but it costs hours to days per target.
  - CrypToth (Koseki et al. 2025, *JCIM*, doi:10.1021/acs.jcim.4c02111) uses 6 probes plus topological data analysis. It ranked the cryptic hot spot first in 7 of 9 proteins [V-abstract].
- **AutoSite and SiteHound** are energy-map hybrids; see §1.1.

**Lesson for us.** Probe-energy ideas can be done cheaply:
- precomputed per-atom-type probe potentials on a grid, evaluated by FFT or sparse convolution;
- a desolvation proxy (hydrophobic contact density);
- a "probe-cluster count" style hot-spot score.

Expensive docking of 16 probes is not required to capture most of the signal. Full FTMap fidelity is too slow for the goal.

### 1.3 Knowledge-based ML: PRANK and P2Rank (the main baseline)

**PRANK** (Krivák & Hoksza 2015, *J Cheminform* 7:12, PMC4414931)
- A random forest (RF) on local features of pocket points rescores fpocket or ConCavity candidates.
- fpocket top-n success, before → after rescoring [V]: CHEN11 47.9 → 58.8; ASTEX 58 → 63.6; DT198 37.5 → 56.2.
- On LIGYSIS, **fpocket+PRANK is the best method overall**: top-N+2 60.4, max recall 91.3 [V].

**P2Rank** (Krivák & Hoksza 2018, *J Cheminform* 10:39, doi:10.1186/s13321-018-0285-8, PMC6091426) [V]

*Algorithm*
1. Place points on the solvent-accessible surface (SAS), using the CDK numerical algorithm.
2. Describe each point with 35 features: distance-weighted (w = 1 − d/6) properties of atoms within 6 Å, plus residue-level features.
3. Score each point's ligandability with an RF: 200 trees, unlimited depth, 6 features per split.
4. Cluster positive points by single linkage at 3 Å.
5. Rank pockets by the sum of squared point scores.

*Features.* The most important feature is **protrusion**: the number of protein atoms within 10 Å of the SAS point, which is a buriedness proxy. The default model also uses B-factor.

*Training and tuning.* Trained on **CHEN11** (251 proteins, 476 ligands; at most one protein per SCOP family) and tuned on **JOINED**.

*Results.* DCA ≤ 4 Å success, top-n / top-(n+2) [V]:

| Method | COACH420 top-n | COACH420 top-(n+2) | HOLO4K top-n | HOLO4K top-(n+2) |
|---|---|---|---|---|
| P2Rank | 72.0 | 78.3 | 68.6 | 74.0 |
| P2Rank, protrusion only | 64.2 | 73.0 | 59.3 | 67.7 |
| fpocket | 56.4 | 68.9 | 52.4 | 63.1 |
| fpocket+PRANK | 63.6 | 76.5 | 62.0 | 71.0 |
| MetaPocket 2.0† | 63.4 | 74.6 | 57.9 | 68.6 |
| SiteHound† | 53.0 | 69.3 | 50.1 | 62.1 |
| DeepSite† | 56.4 | 63.4 | 45.6 | 48.2 |

† Computed only on the inputs where the method ran.

*The "protrusion only" row matters.* A single buriedness feature already reaches 64.2% top-n on COACH420. This is strong evidence for analytical buriedness.

*Pocket counts.* Mean true sites per protein are 1.2 (COACH420) and 2.4 (HOLO4K). Mean predicted pockets: P2Rank 6.3 and 12.6; fpocket 14.6 and 27 [V].

*Naming trap.* The paper calls its criterion "DCC" but defines it as pocket centre to the **closest ligand atom**, i.e. DCA [V].

*Runtime* [V], on one 3.7 GHz core for a 2,500-atom protein:
- P2Rank: 0.9 s warm, 6.8 s cold (JVM start-up).
- fpocket: 0.2 s.
- P2Rank 2.5 is about 2× faster [V, README / PrankWeb 4].
- Memory is about 1 GB; requires Java ≥ 17 [V].

*Configs* [V, README]:
- `default`: uses B-factor; for X-ray structures.
- `alphafold`: drops B-factor, because that column holds pLDDT; also for NMR and cryo-EM.
- `rescore_2024`: experimental; no B-factor; recommended for AF, NMR and EM.
- Conservation models (`conservation_hmm`, `alphafold_conservation_hmm`): HMM-based conservation that can be fetched from a conservation server [S, release notes].
- P2Rank-CONS is P2Rank plus a conservation feature (PrankWeb: Jendele et al. 2019, *NAR* 47:W345; PrankWeb 3: Jakubec et al. 2022, *NAR* 50:W593, doi:10.1093/nar/gkac389).

*Release and code.* PrankWeb 4 (Polák et al. 2025, *NAR* 53:W466, doi:10.1093/nar/gkaf421) adds pLDDT colouring and a modular pipeline. It does **not** describe any masking of low-pLDDT regions and reports no new accuracy numbers [V]. Code: github.com/rdk/p2rank, MIT. The latest stable release is 2.5.1; 2.6 is in alpha [V].

**Independent P2Rank numbers** [V]
- LIGYSIS: top-N 46.7, top-N+2 51.9, max 57.0, Precision1K 79.2, residue MCC 0.26, residue AUC 0.74.
- P2Rank-CONS on LIGYSIS: 48.8, 53.9, 57.0 and 90.7, MCC 0.30.
- HAP (IF-SitePred): top-1 DCA 0.81 on both PDB structures and AF2 models.
- HOLO4K-sc DCA top-n: 0.8188, best in UniSite's re-run.
- SwinSite re-run, top-1 DCA: COACH420 73.63; HOLO4K 66.71; PDBbind2020 70.68. This beats every CNN baseline (DeepPocket, PUResNet, Kalasanty) in that paper.
- CryptoBench apo, residue level: AUC 0.81, AUPRC 0.21. On the holo counterparts: 0.89 and 0.34.
- Membrane-interface benchmark (Pliushcheuskaya & Künze 2025), top-7 DCC / DVO: GPCR 31/23, ion channel 34/22.

**Why P2Rank is hard to beat**
- It is trained on only 251 proteins (CHEN11), so it has little memorisation.
- Its local features generalise.
- It outputs every pocket, which gives good recall at top-N+2.

**Its weaknesses** [V, LIGYSIS / UniSite / membrane benchmark]
- Its max recall is capped at about 57% on LIGYSIS. It misses shallow sites and ions, because its SAS-point positive threshold cuts them out.
- It gives low-score tails, i.e. many extra pockets.
- Its pocket shapes are blobby (low DVO on membrane proteins).
- It is single-chain-oriented in the evaluations that used it, although it can run on multi-chain input.

### 1.4 Deep-learning structure-based methods

The dominant pattern: training on sc-PDB plus testing on COACH420/HOLO4K gives inflated numbers, and on the independent LIGYSIS these methods fall below P2Rank.

**DeepSite** (Jiménez et al. 2017, *Bioinformatics* 33:3036, doi:10.1093/bioinformatics/btx350)
- A 3D CNN on 16 Å voxel subgrids with 8 atom-property channels.
- Trained on sc-PDB (2013), 7,622 proteins.
- P2Rank paper: COACH420 56.4 / 63.4, HOLO4K 45.6 / 48.2 [V].
- Kalasanty paper: on the Chen apo/holo set, 23.8% DCC < 4 Å; 12 test proteins were in DeepSite's training data [V].
- Web server only.

**Kalasanty** (Stepniewska-Dziubinska et al. 2020, *Sci Rep* 10:5035, doi:10.1038/s41598-020-61860-z)
- A 3D U-Net segmenting a 70 Å grid at 2 Å resolution, with 18 atom features.
- Trained on sc-PDB 2017 with 481 benchmark-related structures removed.
- Authors: on the Chen apo+holo set, DCC < 4 Å for 44.6% of sites; F1 0.45 [V].
- DeepPocket's re-run (DCA top-n): COACH420 63.51, HOLO4K 61.21. It failed on 340 HOLO4K proteins [V].
- About 1 s per protein on CPU plus a 5 s model load. BSD-3.

**PUResNet v1** (Kandel et al. 2021, *J Cheminform* 13:65)
- A ResNet U-Net trained on a sc-PDB subset of 5,020 structures.
- The authors removed **122 of the 420 COACH420 structures because they were in their training set** [V].
- Returns one pocket for about 90% of chains, so LIGYSIS max recall is 41.1% [V].
- Residue-level F1/MCC on LIGYSIS is the best of all methods: 0.41 / 0.39 [V].
- Fails on multimers: ion channels 2 / 7 [V].
- MIT.

**PUResNet v2.0** (Kandel et al. 2024, *J Cheminform* 16:66, doi:10.1186/s13321-024-00865-6)
- A sparse Minkowski CNN on atoms.
- Trained on BioLiP-derived data: 61,691 complexes in 4,729 families.
- After removing training-set overlap, **only 100 of the 420 COACH420 structures remained** ("Coach100").
- Authors, top-1 DCA / DCC [V]:
  - Coach100: 59.0 / 38.0, vs P2Rank 44.0 / 29.0.
  - Holo801: 85.4 / 53.7, vs P2Rank 74.0 / 48.2.
- Membrane benchmark: second-best overall [V].
- MIT.

**DeepPocket** (Aggarwal et al. 2022, *JCIM* 62:5069, doi:10.1021/acs.jcim.1c00799)
- A 3D CNN (libmolgrid / gnina atom types) rescores fpocket centres; a second CNN segments pocket shape.
- Training proteins with > 50% identity (or > 30% identity with ligand similarity > 0.9) to the test sets were removed from sc-PDB. **This removed 2,418 structures for COACH420 and 7,951 for HOLO4K**, which quantifies the sc-PDB overlap [V].
- Authors, DCA top-n / top-n+2 [V]:
  - COACH420: 67.96 / 79.94, vs P2Rank 68.24 / 75.48.
  - HOLO4K: 73.36 / 82.97, vs P2Rank 70.6 / 80.05.
  - These test sets were reduced: COACH420 to 291 proteins, HOLO4K to 3,413.
- LIGYSIS: RESC mode 58.1 top-N+2 (#2); SEG mode 43.8 with 31% redundancy [V].
- Membrane benchmark: best DVO [V].
- MIT; needs a GPU in practice.

**GrASP** (Smith et al. 2024, *JCIM* 64:2637, doi:10.1021/acs.jcim.3c01698)
- A graph attention network segments surface atoms (17 features). Atoms scoring > 0.3 are clustered by average linkage at 15 Å; pocket score is the sum of squared atom scores.
- Trained on a relabelled multi-site sc-PDB (26,196 sites) with UniProt-aware CV.
- Authors [S]: COACH420(Mlig+) +2.6 points top-N DCA over P2Rank; HOLO4K(Mlig+) −2.2 points at top-N+2; much higher precision. Exact values are only in figures.
- LIGYSIS [V]: top-N+2 49.9; **Precision1K 92.5 (best)**; max recall 50.0; residue MCC 0.34.
- MIT.

**EquiPocket** (Zhang et al., ICML 2024; arXiv 2302.12177)
- An E(3)-equivariant GNN on surface atoms.
- **The arXiv v4/v5 (Aug 2026) were withdrawn by the authors**: "experimental design need[s] substantial revision" [V].
- Its numbers, and baseline numbers copied from it (e.g. into VN-EGNN), should be treated as unreliable.

**VN-EGNN** (Sestak et al., *J Cheminform* 18:11, online 2025-12-15, doi:10.1186/s13321-025-01127-9; arXiv 2404.07194)
- An EGNN with 8 virtual nodes that migrate to pocket centres, plus ESM-2 residue embeddings. Outputs centres only, no residues.
- Authors, DCC / DCA top-M [V]:
  - COACH420: 0.605 / 0.750, vs P2Rank 0.464 / 0.728.
  - HOLO4K: 0.532 / 0.659, vs P2Rank 0.474 / 0.787.
  - It wins DCC and loses DCA on HOLO4K. Many baselines were copied from EquiPocket.
- LIGYSIS [V]: top-N+2 40.9 (#12 of 13); **67% of predictions redundant**; at most 8 pockets.
- MIT code (paper CC BY-NC-ND).

**IF-SitePred** (Carbery et al. 2024, *J Cheminform* 16:32, doi:10.1186/s13321-024-00821-4)
- ESM-IF1 residue embeddings feed 40 LightGBM models; DBSCAN-clustered PyMOL point clouds; pockets ranked by point count.
- Trained on Binding MOAD family leaders.
- Authors (HAP set, §2.7) [V]: top-1 DCA 0.76 on PDB and 0.77 on AF2, vs P2Rank 0.81 / 0.81. The authors state P2Rank was best at top-1.
- LIGYSIS [V]: top-N+2 25.7 (last); 49% redundant.
- BSD-3.

**PGBind / PGpocket**
- PGBind (Shen et al. 2024, *Brief Bioinform* 25:bbae455) is a pocket-guided **blind-docking** module, not a pocket finder [V].
- PGpocket (Zhao et al. 2024, *IJMS* 25:9280) is a point-cloud GNN trained and tested on a random sc-PDB split, so leakage is likely. Authors' DCC success 66.2 vs P2Rank 64.2 [V].

**Other structure-based DL methods, 2025–2026** [V unless noted]
- **SwinSite** (Kim & Lee 2026, *JCIM* 66:2551)
  - 3D CNN plus Swin transformer. Below P2Rank on COACH420, HOLO4K and PDBbind2020 top-1 DCA; above it on Astex.
  - Useful because it **re-ran** all baselines with released weights.
- **UniSite** (Fan et al., arXiv 2506.03237)
  - A set-prediction model with a UniProt-centric dataset (UniSite-DS) and an IoU-based AP metric.
  - It re-ran baselines: on HOLO4K-sc DCA top-n, **P2Rank 0.8188 is best**; DeepPocket 0.7369; VN-EGNN 0.6999.
  - On UniSite-DS, AP@0.5: UniSite-3D 0.3835; GrASP 0.2848; P2Rank 0.2157.
  - It reports DCC/DCA "double counting" (one prediction matching several ligands) in about 9–19% of proteins.
- **YuelPocket** (Wang & Dokholyan, *PNAS* 2026)
  - Ligand-conditioned GNN trained on PLINDER.
  - PLINDER test top-1 DCA about 62% vs P2Rank about 55%.
  - COACH420 was dropped because it largely overlaps PLINDER training.
- **RAPID-Net** (Balytskyi et al., *JCIM* 2025; arXiv 2502.02371)
  - Ensemble 3D U-Net for blind docking. 91.4% pocket–ligand intersection on PoseBusters-308.
  - No P2Rank comparison and no leakage filtering described.
- **GENEOnet** (Bocchi et al. 2025, *Sci Rep* 15:34597)
  - Group-equivariant non-expansive operators: very few parameters, trained on about 200 proteins.
  - Top-1 correct 0.764 vs P2Rank 0.702 on a PDBbind-derived test set.
  - A notable "small, geometric, learned-operator" precedent. Web service only.
- **GDEGAN** (arXiv 2603.19817): claims +37–66% relative DCC; not verified.
- **PeSTo** (Krapp et al. 2023, *Nat Commun* 14:2175): parameter-free geometric transformer; per-residue ligand-interface ROC-AUC 0.86. Residue scores, not pockets. CC BY-NC-SA.
- **ScanNet**: protein–protein interfaces only.

### 1.5 PLM- and co-folding-feature methods (2023–2026)

**Seq2Pocket** (Škrhák, Polák, Novotný, Hoksza; bioRxiv 10.64898/2026.01.28.702257; *Bioinformatics* 2026, btag778 [S])
- From the P2Rank group. Fine-tuned ESM-2 3B residue classifier, an embedding-based "pocket smoothing" step, and mean-shift clustering on surface points.
- Training: sc-PDB "enhanced" (adds ion and small-molecule sites from AHoJ-DB), with UniProt overlap to LIGYSIS removed. That is 56,968 pockets before and 38,578 after filtering; Zenodo 18271517.
- LIGYSIS, DCC ≤ 12 Å, authors [V-text]:
  - About +12 points top-N and +8 points top-N+2 over the previous best.
  - Mean-shift clustering: top-N+2 68.8%, fragmentation index 2.12.
  - Default DBSCAN: 69.2%, but heavily fragmented (index 4.68).
  - The residue-level baselines were "adopted from existing benchmarks" rather than re-run.
- **This is currently the strongest claim to beat P2Rank on LIGYSIS, and it uses a 3B-parameter PLM.**
- Code: github.com/skrhakv/seq2pocket, MIT.

**zPocket and the PocketWatch dataset** (Portnoi, Keys, DiFrancesco, Gruver; arXiv 2610.17696, Oct 2026) [V]
- Small per-residue heads on frozen co-folding trunk features: ESMFold2 pair representation plus ESMC-6B embeddings.
- **PocketWatch** is new (§2.3): Q-BioLiP/RCSB ligands up to 2026-01-02, LIGYSIS/BioLiP2 relevance criteria, pooled per UniProt, **temporal split** (test pockets first seen ≥ 2024-01-01).
- Per-target AUPRC: zPocket 0.745; retrained VN-EGNN 0.598; UniSite-1D 0.595; **P2Rank 0.477**.
- Re-ranking fpocket candidates with zPocket raised DCC-N+2 from 0.259 to 0.391 on fragment screens.
- **Key generalisation evidence: per-pocket AUPRC falls from 0.803 for pockets well covered by earlier structures to 0.327 below 20% coverage.** 71.6% of test pockets are in the best-covered bin.
- Interface-only pockets are masked, so single-chain methods are not penalised.
- Code: github.com/genesistherapeutics/zpocket. Data on Hugging Face: genesisml/PocketWatch.

**hotpocketNN / HOTPocket** (Carpenter & Altman 2025, *J Cheminform* 17:180, doi:10.1186/s13321-025-01125-x)
- A one-hidden-layer MLP on ESM-2 (3B) embeddings of a 15-residue pocket. It filters 15.6 M candidate pockets from 7 finders.
- Top-N_L / N_L+2 "precision" (pocket centre within 4 Å of a relevant ligand) [V]:

| Set | hotpocketNN | P2Rank | fpocket |
|---|---|---|---|
| Astex | 53.1 / 50.4 | 46.4 / 34.7 | — |
| PoseBusters | 53.7 / 51.0 | 47.6 / 36.9 | — |
| Human BioLiP | 33.7 / 29.0 | **50.6 / 38.4** | 35.8 / 27.2 |

- On the large, multi-site human BioLiP set, P2Rank wins clearly.
- The metric is precision-like, so it rewards emitting fewer pockets. The preprint says "DCC criterion" and the journal says "DCA" for the same thing.

**Per-residue PLM methods (sequence-only)**
- **CLAPE-SMB** (*J Cheminform* 2024): MCC 0.529 on SJC. No pocket-level comparison.
- **GPSite** (Yuan et al. 2024, *eLife*): ProtT5 plus ESMFold structure. Specific ligand classes only (ATP, HEM, metals …), no generic pocket mode.
- **LABind** (*Nat Commun* 2025): ligand-aware (needs the SMILES). COACH64: DCC/DCA 53.1 / 85.9 vs P2Rank 46.9 / 82.8.
- **LaMPSite** (arXiv 2312.03016): needs the ligand. COACH420 DCA top-n 66.02, **below P2Rank's 68.24**.
- **PickPocket** (bioRxiv 2025): ESM-2 plus GearNet. LIGYSIS top-N+2 53.3, i.e. **below** fpocket+PRANK and P2Rank-CONS.
- **AstraBIND** (2025): closed models.
- All [V] from the subagent reading primary text.

**Hybrid PLM + GNN** (Gamouh, Novotný & Hoksza 2025, *Bioinformatics* btaf431) [V]
- Adding 3D structure via a GAT raises MCC by about 0.06 with weak features, but only by about 0.014 with ProtT5.
- Large PLMs already encode most of the structural signal. This bears on what an "analytical + small learned" model can capture.

**AF2BIND** (Gazizov, …, Ovchinnikov, Polizzi; *Nat Methods* 23:626, 2026, doi:10.1038/s41592-026-03011-2) [V]
- Runs AF2 once (no MSA, target as template, side chains beyond Cβ masked) with 20 free "bait" amino acids. A logistic regression is applied to the 5,120 pair-representation features per residue.
- Test set: 67 proteins with no sequence, structure or pocket similarity to training. ROC-AUC 0.936, AP 0.728; allosteric subset AUC 0.91.
- No head-to-head comparison with P2Rank. On the human proteome, AF2BIND and P2Rank find largely **disjoint** site sets (9,732 vs 8,408 method-specific sites).
- Data and pLDDT handling: §2.6 and §4.

### 1.6 Cryptic-pocket methods

**CryptoSite** (Cimermancic et al. 2016, *JMB* 428:709, doi:10.1016/j.jmb.2016.01.029) [V]
- An SVM on sequence, structure, dynamics and MD features.
- Data: 84 cryptic sites in 79 proteins.
- Test: AUC 0.83; 0.74 without MD features.
- About 1–2 days per protein.

**PocketMiner** (Meller et al. 2023, *Nat Commun* 14:1177, doi:10.1038/s41467-023-36699-3) [V]
- A GVP-GNN on a single structure.
- Trained on MD-derived labels: a residue is positive if the LIGSITE pocket volume within 5 Å grows by > 40 Å³ in the next 40 ns.
- Test: 38 apo–holo pairs / 39 cryptic pockets. ROC-AUC 0.87 vs CryptoSite 0.85.
- Under 1 s per protein.
- Weaknesses:
  - It **cannot handle multi-chain input**.
  - Its independent per-protein macro AUC on CryptoBench is 0.499 (UniPocket's re-run). It is 0.76 pooled in CryptoBench's own evaluation, which shows how much metric convention matters.

**CryptoBench baselines** (Škrhák et al. 2025, *Bioinformatics* 41:btae745) [V]
- pLM-NN: frozen ESM-2 3B plus an MLP. AUC 0.86, AUPRC 0.36, MCC 0.39.
- On the PocketMiner-compatible subset: pLM-NN AUPRC 0.43 vs **PocketMiner 0.19**.
- On the P2Rank subset: pLM-NN 0.42 vs **P2Rank 0.21 on apo** (0.34 on holo).

**Newer cryptic work** [V, mostly abstracts or partial text]
- **Seq2Pocket / ESM-2 fine-tuning** (Škrhák et al., ACM BCB 2025). Merging with P2Rank reaches DCC-MAX above 80% on CryptoBench.
- **UniPocket** (ACM BCB 2026): CryptoBench ROC-AUC 0.816, AP 0.410. Its top-1 "hit" of 99.5% uses a generous definition.
- **Lacuna** (bioRxiv 2026, MIT): normal-mode conformer ensembles, then per-conformer pocket detection, then persistent-site clustering.
  - Top-5 cryptic recovery: 55.6% on the CryptoBench test fold; 73% on the PocketMiner set.
  - Median 2.6 s per chain on one CPU core.
  - **A cheap physics-based ensemble precedent, close to our design space.**
- **CrypToth** (MD-based; *JCIM* 2025).
- **Bowman lab 2026** (bioRxiv 10.64898/2026.01.21.700870): no method predicts absolute probabilities that a pocket is open.
- **AF2 MSA subsampling** (Meller et al. 2023, *JCTC* 19:4355) opens some cryptic pockets. Lazou et al. (*PNAS* 2024) show AF2 "remembers" PDB-frequent states (open and closed sampled in only 6 of 16 proteins).

---

## 2. Benchmarks and datasets

### 2.1 Legacy pocket benchmarks (P2Rank lineage)

All are in github.com/rdk/p2rank-datasets [V]. No license is stated for the dataset files; P2Rank itself is MIT.

| Set | Content | Role / caveats |
|---|---|---|
| **CHEN11** (Chen, Mizianty, Gao & Kurgan 2011, *Structure* 19:613, doi:10.1016/j.str.2011.02.015) | 251 proteins, 476 ligands; at most 1 protein per SCOP family | P2Rank training set. 0.5% site overlap with LIGYSIS [V]. |
| **JOINED** | B48/U48 (48 apo/holo pairs), B210, DT198, ASTEX(85) | P2Rank validation. Contains the U/B48 apo–holo pairs. |
| **COACH420** | 420 single-chain(-ish) structures, from the COACH test set (Yang, Roy & Zhang 2013, *Bioinformatics* 29:2588) minus CHEN11/JOINED proteins; drug and natural ligands | Unbiased for P2Rank. **Heavily overlaps sc-PDB / BioLiP / PLINDER training sets** (PUResNet removed 122; PUResNetV2 kept only 100; DeepPocket removed 2,418 sc-PDB structures; YuelPocket dropped it) [V]. About 70% single-ligand; 56% of entries have different chain counts in the asymmetric unit vs the biological assembly [V]. |
| **HOLO4K** | 4,009 entries (the folder has 4,543 files; 4,009 is the paper-correct list) [V]; multi-chain asymmetric units; based on the Schmidtke et al. 2010 (*JCIM* 50:2191) list | 62% multi-ligand; 40% with chain counts differing from the biological assembly. **Repeated identical interfaces in homo-oligomers inflate success** (e.g. 1JQY has 14 copies of one interface) [V]. |
| **mlig variants** (`coach420(mlig)`, `holo4k(mlig)`, `joined(mlig)`) | Relevant ligands given explicitly via Binding MOAD 2013; proteins unknown to MOAD removed [V] | Use these to avoid ad-hoc relevance filters. GrASP's "Mlig+" is a further variant [S]. Exact sizes not verified. |
| **ah4h** (`ah4h.holoraw(mlig).ds`) | Undocumented apo/holo set from the Hoksza group [V file exists; contents NV] | Inspect before use. |
| **speed5, fptrain, DT198, B210, U/B48** | Speed-test and legacy sets | — |

**SC6K** (DeepPocket; Aggarwal 2022) is about 6k sc-PDB-derived complexes; DeepPocket's test set.
**PDBbind2020 / PDBbind-refined** are used as tests by VN-EGNN and EquiPocket and overlap sc-PDB.

### 2.2 LIGYSIS (Utgés & Barton 2024) — detailed

**Citation.** Utgés JS, Barton GJ. "Comparative evaluation of methods for the prediction of protein–ligand binding sites." *J Cheminform* 16:126 (2024), doi:10.1186/s13321-024-00923-z, PMC11552181.

**Data and code**
- Results and replication files: Zenodo 10.5281/zenodo.13121414.
- Code: github.com/bartongroup/LBS-comparison (MIT; release on Zenodo 10.5281/zenodo.14033286).
- Dataset pipeline: github.com/bartongroup/LIGYSIS.
- Web resource: LIGYSIS-web, *NAR* 2025 (PMC12230740), with 64,782 sites from 435,038 ligands in 25,003 proteins [V, subagent].

All facts below are [V].

**Construction**
1. Start from human reviewed UniProt (20,423 entries): 7,640 have PDB structures and 5,455 have ligand-bound structures.
2. Keep only **BioLiP-relevant ligands, ions included**: 3,513 proteins and 4,037 PDBe-KB segments.
3. Use 64,498 chains from 33,715 structures, taking **PISA preferred biological assemblies** from PDBe (via ProIntVar).
4. Compute contacts with pdbe-arpeggio and map residues to UniProt with SIFTS.
5. **Site definition.** Each ligand's fingerprint is the set of UniProt residue numbers it contacts.
   - Relative intersection: I_rel = |A∩B| / max(|A|, |B|); distance D = 1 − I_rel.
   - Average-linkage clustering cut at D = 0.5.
   - Unique interfaces only: e.g. 1/14 interfaces kept for 1JQY, 12/36 for 1PPR.
   - The result is 8,244 sites, defined **as sets of UniProt residues**, independent of superposition.
6. About 40% of sites are ions. LIGYSIS_NI (no ions) has 4,572 sites.
7. Final benchmark: the **PDBe-KB representative chain** of each segment, ligand-stripped and cleaned with clean_pdb.py.
   - 3,448 chains were predicted; 2,775 remained after dropping chains with missing UniProt mappings.
   - The benchmark has 6,882 reference sites.
   - **Predictions use a single chain, while sites are pooled across many entries and assemblies.** Some sites come from apo/holo variation and some from interfaces.
   - 54% of chains have more than one site.

**Methods**
- 13 originals with defaults: VN-EGNN, IF-SitePred, GrASP, PUResNet (v2.0 web server), DeepPocket (SEG and RESC modes), P2Rank (v2.4.2), P2Rank-CONS (PrankWeb conservation), fpocket (v4.0), fpocket+PRANK, and PocketFinder+, Ligsite+ and Surfnet+ (ConCavity re-implementations).
- 15 variants: redundancy-filtered (_NR) and rescored.

**Metrics**
- **Hit criterion: DCC ≤ 12 Å.** DCC is the distance from the predicted pocket centroid to the observed site centroid.
  - For each observed site, the "best" prediction is the one with minimum DCC among the top-N (or N+2).
  - **Matching is not stated to be one-to-one.**
- The authors show 4 Å DCC is too strict. **DCC of 10–12 Å gives results comparable to DCA = 4 Å** (Supplementary Note 6).
- Recall:
  - top-N recall: N = number of observed sites in the protein.
  - top-N+2 recall: recommended as the **universal "success rate"**.
  - Max recall: all predictions.
- Precision1K: precision of the top-1,000 predictions pooled across the dataset, so **scores must be comparable across proteins**.
- #TP at 100 FP.
- RRO and RVO: relative residue and volume overlap, for correct predictions only.
- Recall with an I_rel ≥ 0.5 residue criterion as an alternative.
- Residue level: F1 and MCC from binary pocket residues; AUC and AP from residue scores where available.
- **Redundancy:** a prediction is redundant if its centroid is within 5 Å of, or its residue Jaccard is ≥ 0.75 with, a higher-ranked pocket. Error bars are 95% CIs of proportions.

**Results**, pocket level (default mode):

| Method | Top-N | Top-N+2 | Max | Precision1K | #TP@100FP | RRO | RVO |
|---|---|---|---|---|---|---|---|
| fpocket+PRANK | 48.8 | **60.4** | 91.3 | 81.7 | 526 | 52.6 | 38.2 |
| DeepPocket-RESC | 46.6 | 58.1 | 89.3 | 81.7 | 637 | 53.1 | 38.2 |
| P2Rank-CONS | 48.8 | 53.9 | 57.0 | 90.7 | 932 | 56.4 | 43.8 |
| P2Rank | 46.7 | 51.9 | 57.0 | 79.2 | 586 | 54.4 | 58.2 |
| GrASP | 48.0 | 49.9 | 50.0 | **92.5** | 1017 | 54.5 | 59.8 |
| Ligsite+ | 41.3 | 48.4 | 49.7 | 52.3 | 115 | **77.6** | **77.0** |
| PocketFinder+ | 39.2 | 47.8 | 50.5 | 42.0 | 64 | 72.3 | 75.9 |
| fpocket | 38.8 | 46.5 | **91.3** | 47.3 | 94 | 52.6 | 38.2 |
| Surfnet+ | 37.7 | 45.8 | 48.9 | 39.5 | 61 | 71.7 | 72.0 |
| DeepPocket-SEG | 35.4 | 43.8 | 56.5 | 82.6 | 670 | 57.5 | 60.3 |
| PUResNet | 40.6 | 41.1 | 41.1 | 81.6 | 534 | 61.0 | 63.9 |
| VN-EGNN | 27.5 | 40.9 | 49.3 | 92.5 | **1301** | 32.8 | 27.6 |
| IF-SitePred | 19.8 | 25.7 | 52.1 | 91.0 | 961 | 46.5 | 40.4 |

Residue level, F1 / MCC (AUC / AP where available):

| Method | F1 / MCC | AUC / AP |
|---|---|---|
| PUResNet | 0.41 / 0.39 | — |
| GrASP | 0.39 / 0.34 | 0.70 / 0.42 |
| P2Rank-CONS | 0.36 / 0.30 | 0.76 / 0.46 |
| P2Rank | 0.31 / 0.26 | 0.74 / 0.42 |
| IF-SitePred | 0.29 / 0.24 | 0.76 / 0.45 |
| fpocket | 0.23 / 0.12 | — |

**Redundancy and rescoring**
- Redundancy: VN-EGNN 67%, IF-SitePred 49%, DeepPocket-SEG 31%; all others under 1%.
- Removing redundancy raised recall by 5.2 (VN-EGNN), 13.4 (IF-SitePred) and 5.6 (DeepPocket-SEG) points.
- Rescoring raised Precision1K by +34.4 (fpocket), +29.1 (Surfnet+) and +23.3 (PocketFinder+).

**Overlap of training sets with LIGYSIS sites** (matched by PDB ID + ligand ID):

| Training set | Overlapping sites |
|---|---|
| sc-PDB | 801 (9.7%) |
| bMOAD subset | 606 (7.6%) |
| PDBbind-refined | 310 (3.8%) |
| SC6K | 259 (3.1%) |
| HOLO4K | 207 (2.5%) |
| JOINED | 110 (1.3%) |
| COACH420 | 41 (0.5%) |
| CHEN11 | 40 (0.5%) |

**Authors' conclusions**
1. Recall matters most; always read precision alongside recall.
2. Top-N+2 recall should be the standard.
3. Redundant predictions are harmful.
4. A strong scoring scheme lifts both recall and precision.
5. Pocket-level metrics are more informative than residue-level ones.
6. DCC 4 Å is too strict.
7. Methods that output few pockets cap their own recall.
8. Training and testing should use ligands pooled across multiple structures.
9. Authors should share benchmark code.

**Stated limitations**
- Human proteins only.
- Representative chains only.
- Methods without scores are tie-broken by pocket ID.
- VN-EGNN and IF-SitePred give no residues; their pockets were taken as residues within 6 Å of the centroid.
- Several recent methods were excluded for technical reasons.

**Why LIGYSIS should be our primary benchmark**
- It is independent of all the major training sets (overlap ≤ 10%).
- It has multiple sites per protein.
- It is built on biological assemblies.
- **Sites are UniProt-residue sets, so they transfer directly onto AlphaFold models by residue number** (§4).
- It has published numbers for 13 methods plus code to reproduce them.

**Caveats**
- Ions make up about 40% of sites and few methods are trained on them, so report both LIGYSIS and LIGYSIS_NI.
- The 12 Å DCC criterion is lenient for small sites.
- Matching is not one-to-one.
- The input is a single chain, so interface sites are judged without the partner chain.

### 2.3 Newer large-scale and leakage-aware datasets

**PocketWatch** (zPocket, arXiv 2610.17696; huggingface.co/datasets/genesisml/PocketWatch) [V]
- Ligands from Q-BioLiP/RCSB up to 2026-01-02, using the LIGYSIS/BioLiP2 relevance criteria, pooled per UniProt.
- Temporal splits: validation after 2021-09-30; test sites first appearing on or after 2024-01-01.
- 11,016 training chains; test set of 1,275 targets and 1,562 pockets.
- Test pockets are stratified by sequence identity (MMseqs2), by structural pocket coverage (max_qcov) and by crypticity. Interface-only pockets are masked.
- **The best available temporal, novelty-stratified test set.** Check its license before use (not verified).

**UniSite-DS** (arXiv 2506.03237) [V]
- UniProt-centric and built with AHoJ: 11,510 UniProt IDs, 3,670 of them with multiple sites.
- Metric: IoU-based AP.
- Public release was promised; status unverified.

**PLINDER** (Durairaj et al., bioRxiv 10.1101/2024.07.17.603955; plinder-org.github.io) [V]
- 1.34 M systems from 163k PDB entries (snapshot 2024-04-09). Systems are biounit-based; ligands within 4 Å are merged; the pocket is residues within 6 Å.
- 25% of holo systems involve more than one protein chain.
- **Apo and AF2 structures are linked to each holo system** (identity ≥ 0.9, coverage ≥ 0.9, pocket identity ≥ 95%), with ligand transplantation and quality scores.
- Leakage-aware splits.
- License: data CC BY 4.0, code Apache-2.0.
- Releases: gs://plinder/2024-06/v2 and cameo3d.org/plinder (2026-09 release listed).

**Runs N' Poses** (Škrinjar et al., bioRxiv 10.1101/2025.02.03.636309) [V]
- About 2,600 post-2021-09-30 systems; a co-folding pose benchmark.
- Shows success falls with training-set similarity. Pockets are often correct even when poses are wrong.

**HiQBind** (Wang et al. 2025, *Digital Discovery*; MIT) [V]
- More than 30k complexes from BioLiP2 and Binding MOAD.
- PDBbind-free, so licensing is clean. Holo only.

**PoseBusters** (Buttenschoen et al. 2024, *Chem Sci*; 308 complexes) and the **Astex Diverse set** (85) [V]
- Docking sets with single ligands. Used for pocket "precision" by HOTPocket and RAPID-Net.
- Small, and not designed for multi-site recall.

### 2.4 Cryptic-site datasets

**CryptoBench** (Škrhák, Novotný, Feidakis, Krivák & Hoksza; *Bioinformatics* 41:btae745, 2025; PMC11725321) [V]

*Construction*
- Source: AHoJ-DB apo–holo pairs. Holo binding residues are those within 4.5 Å of the ligand.
- Filters: resolution ≤ 2.5 Å; TM-score ≥ 0.5; centroid shift ≤ 4 Å; radius-of-gyration change ≤ 20%; ligands with ≥ 5 atoms; common additives removed.
- **A pocket is cryptic if its apo–holo pocket RMSD is ≥ 2 Å.**
- Redundancy: 40% identity clustering; splits at 10% identity.

*Size*
- 1,107 apo structures; 1,361 cryptic and 1,445 non-cryptic pockets.
- 197 multi-chain pockets; 371 promiscuous.

*Splits*
- Test: 222 apo structures. Train: 885 (4 CV folds).

*Baselines*: §1.6.

*Caveats*
- The 2 Å rule misses side-chain-gated sites.
- "Non-cryptic" labels are noisy.
- PocketMiner fails on 60 test structures (multi-chain or other errors).
- 3 PocketMiner training proteins are more than 95% similar to test proteins.

*Availability*
- Data: OSF osf.io/pz4a9. Code: github.com/skrhakv/CryptoBench.
- The paper is CC BY 4.0; the data license is not verified.

**PocketMiner test set** [V]
- 38 apo–holo pairs / 39 cryptic pockets: 563 positive and 1,283 negative residues.
- Includes hyper-rigid negative proteins. At most 60% similarity to training.
- Supplementary Data 1; Zenodo 10.5281/zenodo.7532504.

**CryptoSite set** [V]
- 84 cryptic sites in 79 proteins; 92 regular pockets; 705 concave-patch controls.
- 14-structure apo test set.

### 2.5 HOTPocket — what it actually is

Carpenter & Altman, *J Cheminform* 17:180 (2025), doi:10.1186/s13321-025-01125-x; github.com/Helix-Research-Lab/HOTPocket (MIT); Zenodo 16891050 (pocket tables) and 17653326 (inputs) [V].

**HOTPocket is not a curated ground-truth benchmark.** It is a **proteome-wide atlas of predicted pockets**, plus a scorer.

*How it was built*
- 7 finders were run on the human proteome: AutoSite, CASTp, CAVITY/CavitySpace, fpocket 4.1, LIGSITEcs, P2Rank 2.5 and PocketMiner.
- Inputs: 97,811 structures (77,420 PDB from Sept 2023; 20,391 AF2 from AFDB, July 2021) covering 20,402 UniProt proteins.
- These gave 15.6 M raw pockets. hotpocketNN (ESM-2 3B MLP, score threshold 0.4) filtered them to about 2.4 M.
- AF2 pockets with mean pLDDT < 70 were discarded.
- Ground truth for evaluation: 11,434 BioLiP2 pockets.
- Training: 778 BioLiP2 pockets on Therapeutic Target Database (TTD) targets, split 80/10/10 at 40% identity.

*How we can use it*
- (a) **As a comparison** for our proteome-wide AF-mode output, e.g. overlap and novelty vs consensus.
- (b) Its BioLiP2 / Astex / PoseBusters evaluation as a **secondary precision benchmark**, where P2Rank wins on human BioLiP (§1.5).
- (c) Its CSV of the 7 methods' pockets on human AF2 models, as **precomputed baseline outputs** that save running CASTp, CAVITY and others ourselves.

*Caveats*
- Its "precision at top-N" metric rewards emitting few pockets.
- Training and evaluation leakage were handled only at 40% identity.

### 2.6 AF2BIND data [V]

- Training labels are on Zenodo 10.5281/zenodo.17646944 and at github.com/sokrypton/af2bind:
  - 1,897 proteins and 2,708 ligands; PDB as of March 2023; single chains of length 40–500.
  - Ligands with 10–200 heavy atoms and > 100 Å² buried surface; RSCC > 0.85; no additives or covalent ligands.
  - Labels within 5 Å, augmented from homologs (TM > 0.8). Clustered by MMseqs2 (30%) plus Foldseek.
- **Test set: 67 proteins (21,322 residues), held out by sequence, structure and pocket similarity**, plus GPCR and bromodomain hold-outs.
  - This makes it a clean residue-level secondary benchmark (ROC-AUC / AP).
- Human-proteome predictions (20,302 sites in 13,686 proteins): Zenodo 10.5281/zenodo.17683379.
- License not stated.

### 2.7 Apo–holo resources

- **AHoJ** (Feidakis et al. 2022, *Bioinformatics* btac701) and **AHoJ-DB** (*JMB* 2024, doi:10.1016/j.jmb.2024.168545; apoholo.cz/db) [V]
  - AHoJ maps ligand sites to UniProt via SIFTS, superposes all same-UniProt chains with TM-align, and labels each as apo or holo.
  - AHoJ-DB: 468k interactions in 28k proteins, from BioLiP2-relevant ligands. Release v2c (2025-04-03) has 515k entries.
  - **Under 50% of sites have an apo form; about 24% of sites lie at interfaces of two or more chains.**
  - License not stated; the server is free for non-commercial use.
- **HAP / HAP-small** (IF-SitePred, Carbery et al. 2024) [V]
  - HOLO4K PDB structures paired with AFDB models: 688 pairs. HAP-small is the 280 pairs with < 25% identity to P2Rank's training set.
  - Results (top-1 DCA): P2Rank 0.81 on PDB and 0.81 on AF2; fpocket 0.75 → 0.50.
  - Caveat: **98.4% of HAP AF2 models are within 2 Å RMSD of the PDB structure**, so HAP is easy.
- **APObind** (Aggarwal et al. 2021, arXiv 2108.09926): apo matches for PDBbind v2019. It inherits PDBbind's restrictions on redistribution.
- **U/B48** (in JOINED; MetaPocket 2): 48 apo/holo pairs; old and small.
- **PLINDER apo/pred links**: see §2.3; the most scalable source.
- **LIGYSIS** pools ligands across holo entries. Its representative chain may be apo or holo, and no explicit apo stratification has been published.

### 2.8 Interface / multimer pocket data

**There is no dedicated homo- vs hetero-dimer ligand-site benchmark** (as of Oct 2026) [V, subagent search]. Relevant pieces:

- **Fraction of sites at interfaces**:
  - AHoJ-DB: about 24% of BioLiP2 sites contact two or more chains.
  - PLINDER: 25% of holo systems involve more than one chain.
  - CryptoBench: 197 of 2,806 pockets are multi-chain.
- **Asymmetric unit vs assembly**: HOLO4K and COACH420 use asymmetric units, and 40% / 56% of entries differ from the biological unit [V]. Homo-oligomer copies inflate HOLO4K success. Some later papers split HOLO4K into chains ("HOLO4K-sc"), which destroys interface sites.
- **Pliushcheuskaya & Künze 2025** (*JCIM* 65:6949, PMC12264969; Zenodo 10.5281/zenodo.14918827) [V]
  - Covers membrane-embedded **protein–lipid** interfaces: 39 GPCRs, 59 ion channels, 420 soluble controls.
  - Ion-channel sites are often shared between subunits.
  - Top-7 DCC: P2Rank is best on ion channels (34%). PUResNet fails on multimers.
- **PPI-modulator databases**: 2P2Idb v2 (27 PPI complexes, 274 protein–inhibitor complexes), iPPI-DB and TIMBAL.
  - These mostly cover orthosteric PPI inhibitors that bind **one** partner's interface surface. They do not cover pockets formed by two chains.
  - Davis & Sali 2010 (*PLoS Comput Biol*): 197 of 2,619 families show overlap between their small-molecule and protein-binding sites.

**Recommended construction for a dimer stratum** (§7):
- Classify LIGYSIS / BioLiP2 / PLINDER sites in PISA preferred assemblies as single-chain, homo-interface or hetero-interface.
- Require cross-entry support to exclude crystal contacts.
- Evaluate on (i) the isolated chain, (ii) the biological assembly and (iii) AF-Multimer/AF3 models.

### 2.9 Training-set datasets and licensing

- **sc-PDB** (Desaphy et al. 2015, *NAR* 43:D399, doi:10.1093/nar/gku928) [V]
  - 2017 release: 16,034 entries in 4,782 proteins. Downloadable at drugdesign.unistra.fr/scPDB. No license stated.
  - **One ligand and one site per entry**, so other true sites appear as negatives.
  - Overlaps COACH420, HOLO4K and LIGYSIS (9.7% of LIGYSIS sites).
- **PDBbind** [V]
  - v2020 general set: 19,443 entries. Versions after v2021 are commercial (PDBbind+).
  - The license forbids redistributing derivatives. **Do not build shipped benchmarks on it.**
- **BioLiP2** (Zhang et al. 2024, *NAR* 52:D404, PMC10767969) [V]
  - 781,684 interactions at publication, of which 407,951 are small molecules and 174,257 metal ions. Updated weekly. BSD license.
  - Relevance rules:
    - A contact is within the sum of vdW radii + 0.5 Å.
    - A binding residue needs at least 2 contacts; a site needs at least 2 binding residues.
    - 463 artefact ligand codes are discarded if they occur more than 15 times in a structure or bind only 2 consecutive residues; otherwise they are kept only if named in the PubMed title or abstract.
  - zhanggroup.org/BioLiP; figshare 10.6084/m9.figshare.23641701.
  - **The cleanest source of "relevant ligand" labels.**
- **Binding MOAD** [V]
  - Discontinued after 2020 (41,409 structures); a 7.1 GB Zenodo mirror exists.
  - Defines the "relevant ligands" behind the mlig sets.

---

## 3. Metrics: exact definitions and conventions

| Metric | Definition | Threshold / convention | Notes |
|---|---|---|---|
| **DCA** | Minimum distance from the predicted pocket centre to **any heavy atom of the ligand** | 4 Å (P2Rank; HAP; DeepPocket) | Called "DCC criterion" in the P2Rank paper and "DCA" elsewhere [V]. Lenient for large ligands; favours small pocket centres sitting on any part of the ligand. |
| **DCC** | Distance from the predicted pocket centre to the ligand centroid (or the site centroid, in LIGYSIS) | 4 Å (Kalasanty, VN-EGNN, DeepPocket segmentation); **12 Å (LIGYSIS)** | 4 Å DCC is much stricter than 4 Å DCA. LIGYSIS shows 10–12 Å DCC ≈ 4 Å DCA [V]. |
| **DVO** | Discretised volume overlap: \|V_pred ∩ V_true\| / \|V_pred ∪ V_true\| | Usually reported only for correctly located pockets | V_true is typically the sc-PDB/VolSite cavity or the voxels near the ligand (Kalasanty, DeepPocket) [NV exact voxelisation]. |
| **RRO / RVO** (LIGYSIS) | Overlap of residues or volume relative to the observed site, for correct predictions | — | Rewards large pockets (Ligsite+ is best) [V]. |
| **I_rel** | \|A∩B\| / max(\|A\|, \|B\|) on residue sets | ≥ 0.5 as a hit | Residue-overlap alternative to DCC [V]. |
| **Top-n / top-(n+2) success** (P2Rank lineage) | **Ligand-centric.** For each relevant ligand, success if any of the protein's top-n pockets (n = number of relevant ligands in that protein) satisfies DCA ≤ 4 Å. Rate = successes / ligands. | — | Each ligand weighs equally. **Uses an oracle n.** Not one-to-one: one pocket can satisfy several ligands. UniSite found such double counting in about 9–19% of proteins [V]. |
| **Top-N / N+2 recall** (LIGYSIS) | **Site-centric.** N = number of observed *sites* (clustered ligands). Success if the min-DCC prediction among the top-N(+2) is ≤ 12 Å. | — | Ligands are first clustered into sites, so redundant ligands do not dominate [V]. |
| **Max recall** | Same, using all predictions | — | Measures candidate generation alone. |
| **Precision1K / TP@100FP** | Pooled across the dataset: precision of the 1,000 highest-scoring predictions; TPs before the 100th FP | — | **Requires scores calibrated across proteins.** |
| **Pocket "precision"** (HOTPocket, GrASP) | Fraction of predicted (top-k) pockets that hit any relevant ligand | — | Rewards methods that emit few pockets. Always pair with recall. |
| **IoU-AP** (UniSite) | AP over predictions ranked by score, where a TP has residue IoU ≥ τ (0.3 / 0.5) with an unmatched true site | One-to-one | Penalises duplicates and wrong extent [V]. |
| **Residue level** | Binary pocket residues give F1 and MCC; residue scores give ROC-AUC and AUPRC | Positives: residues within 4–5 Å of a relevant ligand (CryptoBench 4.5 Å; PocketMiner and AF2BIND 5 Å; BioLiP2 vdW + 0.5 Å) | **Pooled vs per-protein macro averaging changes results dramatically.** PocketMiner on CryptoBench: 0.76 pooled vs 0.499 macro [V]. |
| **Redundancy %** (LIGYSIS) | A prediction within 5 Å centroid distance, or with residue Jaccard ≥ 0.75, of a higher-ranked one | — | Report it, and apply non-maximum suppression (NMS) ourselves. |

**Which ligands are "relevant"**
- P2Rank's evaluation uses configurable filters: minimum heavy atoms, contact distance to the protein, and an ignore list of het codes (water, common additives). Check `config/default.groovy` for the exact defaults [NV].
- The mlig sets use Binding MOAD's explicit "valid ligand" lists [V].
- LIGYSIS uses BioLiP relevance **including ions** [V].
- CryptoBench uses ≥ 5 atoms and removes glycerol, sulfate and phosphate [V].

**Recommendation.** Report both:
- ligand-centric DCA 4 Å top-n / top-(n+2), for legacy comparability;
- site-centric DCC 12 Å top-N / N+2 (LIGYSIS) plus max recall, Precision1K and redundancy.

Add a stricter **one-to-one** variant: Hungarian matching on DCA or DCC, or IoU-AP. Use bootstrap CIs (resampling proteins, or clusters) and paired tests against P2Rank.

---

## 4. AlphaFold-structure pocket detection

**Evidence on AF models**

- **HAP** (Carbery 2024) [V]: P2Rank top-1 DCA is unchanged (0.81 → 0.81), DeepPocket and IF-SitePred barely move, and **fpocket collapses from 0.75 to 0.50** (top-3: 0.83 → 0.67).
  - The likely cause is cavities in low-confidence regions and loose packing, which spoils fpocket's ranking.
  - But HAP models are within 2 Å of the PDB structure, so the test is easy.
- **Akdel et al. 2022** (*NSMB* 29:1056) [V]: AutoSite on 225 LBSp proteins.
  - Pockets with mean pLDDT > 90 behaved like experimental structures.
  - Low-confidence pockets were larger and mostly failed.
  - 25% of residues across 11 proteomes are newly and confidently modelled.
- **CavitySpace** (*Biomolecules* 2022): 237,872 cavities on 18,672 human AF models, but **only about 26% of "strong" cavities lie in reliable regions** [V]. This is direct evidence of hallucinated pockets.
- **Zillmer & Walther 2025** (*PLoS Comput Biol*, PMC12324681) [V]: P2Rank-CONS on 223k AFDB v4 models from 11 species.
  - Pockets were excluded if mean pLDDT < 70 or mean PAE > 10 Å.
  - Recovered 52.1% of 22,371 UniProt-annotated sites.
  - Failure modes: flexible loops occluding sites; pockets larger than true sites (median 24 vs 6 residues).
- **HOTPocket** drops AF2 pockets with mean pLDDT < 70. **AF2BIND** trims termini with pLDDT < 50 and removes internal stretches of 7 or more residues below 50 [V].
  - These are the main published precedents for pLDDT pruning. All thresholds are conventions, not optimised values.
- **AlphaFill** (Hekkelman et al. 2023, *Nat Methods* 20:205) [V]
  - Transplants ligands from PDB-REDO homologs (> 25% identity) into AFDB models. Quality is reported as local RMSD and a transplant clash score (TCS).
  - 586k of 995k models received ligands. No pLDDT filtering.
  - Useful as a label source for AF-mode tests, but biased towards sites with known homologs.
- **Holo-likeness and memorisation** [V]
  - AF2/AF3 tend to predict PDB-frequent states: Lazou et al. 2024, *PNAS* 121:e2412719121; Yu et al. 2026, *PNAS*.
  - In Yu et al., about 75% of ligand-free AF3 enzyme models are holo-like when the PDB is holo-rich, and **pLDDT does not separate binders from non-binders**.
  - Consequence: AF-mode results on well-studied proteins overstate apo performance.
- **Docking and virtual screening on AF models**
  - Karelina et al. 2023 (*eLife*): GPCR pockets are close, but docking poses are worse than on experimental structures [V].
  - Díaz-Rovira et al. 2023 (*JCIM* 63:1668): enrichment drops when templates are restricted [V].
  - Lyu et al. 2024 (*Science* 384:eadn6354): prospective hit rates on AF2 models are similar to those on crystal structures [V-partial].
- **Proteome atlases** [V]
  - HOTPocket (human; 7 methods).
  - CavitySpace (human).
  - AF2BIND (human; 20,302 sites).
  - PocketVec (*Nat Commun* 2024; more than 32k sites in about 20k human domains).
  - Zillmer & Walther (11 species).
  - PrankWeb serves P2Rank on AFDB models on demand.
- **Co-folding as a pocket finder**
  - zPocket (§1.5) uses frozen ESMFold2 / Protenix features and is the strongest recent result. It is heavy (6B-parameter trunk).
  - FoldBench and Runs N' Poses show pocket placement by co-folding is memorisation-driven.
  - Blind apo pocket detection by co-folding has not been benchmarked [V, subagent].

**Specific AF failure modes our `--af-mode` must handle**

| Failure mode | Evidence | Mitigation idea |
|---|---|---|
| Low-pLDDT "spaghetti" packed against the core creates pockets | CavitySpace (26% of strong cavities reliable); fpocket collapse on HAP | Down-weight atoms by pLDDT; trim low-pLDDT termini and internal stretches (AF2BIND rule); recompute buriedness without them. |
| Low-pLDDT regions occlude real pockets | Zillmer & Walther (loops occluding sites) | Make low-pLDDT atoms "soft" or transparent in buriedness and probe-accessibility calculations. |
| Fake inter-domain clefts under high PAE | Zillmer & Walther used a PAE > 10 Å exclusion | Down-weight pockets whose lining residues span high-PAE pairs. |
| Apo-like side chains narrow clefts | General apo knowledge; CryptoBench apo drop | Side-chain-tolerant buriedness: Cβ-based or with reduced side-chain radii. |
| The B-factor column holds pLDDT | P2Rank `alphafold` config drops the B-factor feature | Never use the B-factor feature as a mobility proxy in AF mode. |

**Recommended AF evaluation design**
1. **Use LIGYSIS proteins** with AFDB v6 models (the local copy at /cluster/work/beltrao/shared/alphafold).
   - LIGYSIS sites are **UniProt residue sets**, so the ground truth maps onto the AF model by residue number. No superposition is needed, though site centroids should be recomputed on the model.
   - Optionally also superpose all holo ligands (PLINDER-style) for DCA.
2. Run every method on the **same full-length AF model**: P2Rank `-c alphafold` and `default`, and fpocket.
   - Also run on a domain-cropped or pLDDT-trimmed model, so the AF-mode gain over naive input is measured separately from the gain over P2Rank.
3. **Stratify:**
   - by pocket mean pLDDT: < 50, 50–70, 70–90, > 90;
   - by inter-residue PAE within the pocket;
   - by whether the AF pocket is closer to the holo or the apo conformation (pocket RMSD to each);
   - by PDB template availability (proxy: release date relative to AF training, or number of PDB entries).
4. Report the number of pockets and the false-positive rate within low-pLDDT regions. Use the "hallucination rate": predicted pockets with mean pLDDT < 50 that match no known site.
5. Secondary checks:
   - HAP-small (280 pairs), for comparability with IF-SitePred and P2Rank numbers (P2Rank top-1 0.76; fpocket 0.48 on AF2) [V].
   - The AF2BIND 67-protein test set.
   - Overlap with HOTPocket and AF2BIND proteome predictions.

---

## 5. Analytical and physical features with evidence for ligandability

**Buriedness / protrusion / enclosure**
- The single strongest signal.
- P2Rank's top feature is protrusion (atom count within 10 Å); protrusion alone reaches 64.2% top-n on COACH420 [V].
- Related measures:
  - SiteMap enclosure (the largest coefficient in SiteScore and Dscore) [S];
  - LIGSITE PSP counts;
  - Ghecom morphological pocketness;
  - PocketPicker ray buriedness;
  - fpocket alpha-sphere density.
- Multi-scale buriedness (several probe or ray radii) distinguishes deep clefts from shallow grooves.
- Cheap: neighbour counts in one KD-tree pass, or ray-casting / convolution on a grid on GPU.

**Size and shape**
- Volume, depth, and number of site points with a √n saturation (SiteMap).
- Nayal & Honig 2006 (*Proteins* 63:892, SCREEN) found size and shape descriptors to be the most discriminative for drug-binding cavities [NV].
- Pocket compactness and roughness predict NMR fragment hit rates: Hajduk et al. 2005, *J Med Chem* 48:2518, doi:10.1021/jm049131r [NV details].

**Hydrophobicity / apolar surface**
- Cheng et al. 2007 (*Nat Biotechnol* 25:71, doi:10.1038/nbt1273): MAPPOD, a maximal-affinity model from curvature and hydrophobic surface area, predicts druggability [NV numbers].
- Schmidtke & Barril 2010 (*J Med Chem* 53:5858, doi:10.1021/jm100574m): the fpocket druggability score is a logistic regression on hydrophobicity and polarity descriptors. They also showed polar groups can be decisive, i.e. "druggable = enclosed hydrophobic" is too simple [S].
- Scales: Kyte–Doolittle, Eisenberg, or atom-type apolar SASA. pyKVFinder reports per-cavity hydropathy with the Eisenberg–Weiss scale [S].
- Use atom-level apolar fractions (C/S vs N/O) rather than residue scales where possible; this keeps the feature robust to side-chain rotamers.

**Druggability scores** (useful as ranking priors and as an output field, not as the ligandability label)
- SiteMap Dscore and DoGSiteScorer (SVM; 88% on NRDLD [S]).
- DrugPred (Krasowski et al. 2011, *JCIM* 51:2829; 91% on NRDLD [S]).
- fpocket drug score; FTMap hot-spot strength [S]; PockDrug (Borrel 2015 [NV]).
- NRDLD (Krasowski 2011) is the standard druggability dataset.
- Druggability is **not the same as ligandability**. Benchmarks like LIGYSIS count cofactors, ions and sugars too.

**Electrostatics and polarity**
- Matters for ion, phosphate and nucleotide sites, which are about 40% of LIGYSIS sites if ions are included.
- Simple Coulomb potential at a probe point, or local counts of charged atoms (Asp/Glu carboxylate O; Lys/Arg N), are cheap proxies.
- Metal sites need dedicated features: clusters of His Nε/Cys S/Asp/Glu O with fixed geometry. P2Rank's SAS-point approach misses many of them, judging by its max recall of 57%.

**Desolvation / hot spots**
- FTMap consensus clusters and mixed-solvent MD capture the same effect.
- A cheap proxy is the count of apolar atoms and of "displaceable" polar contacts around a probe, or a grid score with an implicit solvent term (AutoSite/AutoGrid maps, as in AutoSite and Q-SiteFinder).

**Residue propensities**
- Binding sites are enriched in aromatics (Trp, Tyr, Phe, His), Met, Gly and Arg (e.g. Soga et al. 2007, *JCIM* 47:400, PLB index [NV specifics]).
- P2Rank includes residue-level features.

**Conservation and conservation-free proxies**
- Conservation helps: on LIGYSIS, P2Rank-CONS improves top-N+2 by 2 points, Precision1K from 79.2 to 90.7, and MCC from 0.26 to 0.30 [V].
- ConCavity and LIGSITEcsc are older evidence.
- But MSAs are slow and need databases. PLM embeddings are the conservation-free surrogate: ESM-2 in VN-EGNN, HOTPocket and Seq2Pocket; ESM-IF in IF-SitePred.
  - Even frozen ESM-2 3B plus an MLP gives AUPRC 0.36–0.43 on CryptoBench, above every structure-only baseline [V].
- A small PLM, such as ESM-2 35M–150M per-residue embeddings reduced by PCA, could be an optional lightweight feature within our "no large neural nets" constraint. This is a design decision to make explicitly.

**Flexibility**
- B-factor (P2Rank default), normal modes (Lacuna), or pLDDT (AF).
- Lacuna's normal-mode ensemble plus persistent-pocket clustering reaches top-5 cryptic recovery of 55.6% on CryptoBench at 2.6 s per chain [V-abstract]. This is evidence that cheap physics-based ensembles help with cryptic sites.

---

## 6. Fair-comparison pitfalls and best practices

1. **Sequence and fold leakage.**
   - sc-PDB, PDBbind, BioLiP and PLINDER all overlap COACH420 and HOLO4K heavily: 2,418 and 7,951 sc-PDB structures had to be removed (DeepPocket); only 100 of COACH420 survived (PUResNetV2) [V].
   - DeepSurf lost about 5 points below 40% similarity [V].
   - zPocket shows per-pocket AUPRC falling from 0.80 to 0.33 for novel pockets [V].
   - **Practice:**
     - Hold out by UniProt and by MMseqs2 30% / Foldseek clusters against every training source, including data used to fit any learned scorer.
     - Stratify the test results by maximum identity to training.
     - Prefer temporal splits (PocketWatch).
2. **Our own scorer must not be trained on test proteins.**
   - If we train a lightweight scorer on sc-PDB, BioLiP2 or PLINDER, remove LIGYSIS, COACH420, HOLO4K, CryptoBench-test and AF-test UniProts (and their clusters) first.
   - P2Rank's advantage is partly a tiny, disjoint training set (CHEN11).
3. **Redundancy and duplicates.**
   - Apply NMS: centroid < 5 Å or Jaccard ≥ 0.75, as in LIGYSIS.
   - Report the redundancy %.
   - Use one-to-one matching in a strict variant, because DCA/DCC double counting happens in about 9–19% of proteins [V].
4. **Number of predictions.**
   - Top-n uses the oracle n (true ligand or site count), which inflates practical performance.
   - Also report top-1, top-3, top-5, max recall, and precision at score thresholds.
   - Methods that output few pockets look precise but cap their recall [V].
5. **Ligand relevance and cofactor bias.**
   - Fix one relevance definition (BioLiP2 rules, or the mlig/MOAD lists).
   - Report ions separately (LIGYSIS vs LIGYSIS_NI).
   - Report results by ligand class (drug-like, cofactor, sugar, ion, lipid). Cofactor sites (ATP, NAD, FAD, heme) are highly recurrent and easy to memorise.
6. **Chains vs assemblies.**
   - COACH420 and HOLO4K are asymmetric units with crystal-contact and duplicate interfaces [V].
   - LIGYSIS predicts on a single representative chain, but its sites come from assemblies.
   - **Practice:**
     - For the dimer stratum, use PISA/PDBe preferred assemblies.
     - Deduplicate symmetric sites.
     - Always state what input every method received.
     - Run all methods on identical inputs.
7. **Apo vs holo input.**
   - Ligand-stripped holo structures overestimate real-world performance. CryptoBench P2Rank AUPRC is 0.34 on holo vs 0.21 on apo [V].
   - Report apo results separately.
8. **AF evaluation bias.**
   - AF models of PDB-rich proteins are holo-like [V]. Stratify by template availability and by the pocket's conformational state.
   - pLDDT filters remove hard cases, so report results with and without filtering, plus the count of pockets dropped.
9. **Copying baseline numbers.**
   - Copied numbers propagate errors: VN-EGNN copied baselines from the now-withdrawn EquiPocket [V].
   - **Re-run every baseline ourselves** with pinned versions, defaults and configs. Use LIGYSIS's published numbers only as a check that we reproduce them, e.g. P2Rank 51.9 / fpocket+PRANK 60.4 top-N+2.
10. **Metric naming.**
    - "DCC" means DCA in P2Rank-lineage papers [V]; "DCA precision" vs "DCC criterion" is confused in HOTPocket.
    - Define every metric in the code and the report.
11. **Statistics.**
    - Use bootstrap 95% CIs over proteins (or clusters), and paired comparisons (McNemar per site, or paired bootstrap of the difference).
    - LIGYSIS has about 6,900 sites, so ±1.2 points is roughly the 95% CI half-width at 50% [estimate].
12. **Runtime reporting.**
    - Wall time per structure on stated hardware (CPU model, core count, GPU), warm and cold (P2Rank's JVM start-up is 6.8 s vs 0.9 s warm [V]).
    - Throughput in structures per hour for 1k / 10k structures. I/O-inclusive vs compute-only.
    - Exclude or report pre-computation (MSAs, PLM embeddings, conservation) separately.
13. **Licensing.**
    - Do not redistribute PDBbind-derived data.
    - Prefer BioLiP2 (BSD), PLINDER (CC BY), LIGYSIS code (MIT).
    - sc-PDB, AHoJ-DB, CryptoBench data and AlphaFill data have unclear licenses: keep them local and do not commit them.

---

## 7. Implications for our design (actionable takeaways)

1. **Primary benchmark: LIGYSIS (human, 2,775 chains / 6,882 sites), run with the exact LBS-comparison protocol.**
   - Metrics: DCC ≤ 12 Å top-N+2 recall (headline); also top-N, max recall, Precision1K, #TP@100FP, RRO, residue MCC/AUPRC.
   - Report both **LIGYSIS** and **LIGYSIS_NI**.
   - Reproduce P2Rank (51.9), P2Rank-CONS (53.9) and fpocket+PRANK (60.4) first, as a harness sanity check.
   - **The target to "clearly beat" is fpocket+PRANK's 60.4 top-N+2 together with P2Rank-CONS's Precision1K of about 91**, not P2Rank's 51.9 alone.
2. **Add strict variants:**
   - one-to-one Hungarian matching;
   - DCA 4 Å using all superposable ligands;
   - IoU-AP.
   These guard against gaming a lenient 12 Å DCC with many large or small pockets.
3. **Second primary benchmark for novelty: a temporal / low-coverage held-out set.**
   - Use PocketWatch's 2024+ test split if its license allows (license not verified); otherwise build it from BioLiP2 with a release-date cut.
   - Report results by pocket-coverage / identity bins. **This is where memorising methods drop** (0.80 → 0.33 AUPRC), and where an analytical method should shine.
4. **Legacy/secondary: COACH420(mlig) and HOLO4K(mlig).**
   - Ligand-centric DCA 4 Å top-n / top-(n+2), with P2Rank re-run by us (published 72.0 / 68.6 top-n).
   - Add a HOLO4K **biological-assembly** variant and a deduplicated variant.
   - Add HAP-small (AF) for comparability with the IF-SitePred table.
5. **Apo and cryptic stratification.**
   - Primary: the **CryptoBench test split** (222 apo structures). Report pocket-level top-k DCA/DCC and residue AUPRC/MCC, on all pockets and on the cryptic subset, and single-chain vs multi-chain separately.
   - The bars: P2Rank-apo AUPRC 0.21, pLM-NN 0.42.
   - Secondary: the PocketMiner 38-pair set (ROC-AUC 0.87 bar), and an apo counterpart set for LIGYSIS UniProts built from AHoJ-DB v2c or PLINDER apo links.
   - Report the holo→apo drop for each method.
6. **AF mode benchmark.**
   - LIGYSIS UniProts on AFDB v6 models, with sites transferred by UniProt residue number.
   - Stratify by pocket pLDDT band, by PAE, by holo-like vs apo-like pocket state, and by template richness.
   - Compare naive input vs `--af-mode` vs P2Rank `-c alphafold` vs fpocket.
   - Track hallucinated-pocket rates in low-pLDDT regions.
   - Secondary: HAP-small, the AF2BIND test set, and overlap with HOTPocket / AF2BIND proteome atlases.
7. **Dimer benchmark: we must build it** (no published one exists).
   - Classify sites in PISA preferred assemblies as single-chain, homo-interface or hetero-interface. Contacts on at least 2 chains, at least 3 residues on the minor chain, with cross-entry support.
   - Sources: LIGYSIS / BioLiP2 / PLINDER.
   - Inputs: isolated chain vs full assembly vs (optionally) AF-Multimer/AF3 models.
   - Deduplicate symmetric copies, so each homodimer site counts once.
   - Expect about 24–25% of sites to be interfacial (AHoJ-DB, PLINDER).
   - Secondary: the HOLO4K multi-chain subset, and the Pliushcheuskaya ion-channel set.
8. **Where the headroom over P2Rank lies (ranked)**
   - (a) **Candidate recall.** P2Rank's max recall is 57% vs fpocket's 91% on LIGYSIS. Add a geometric candidate generator in the fpocket or grid-buriedness style, plus dedicated shallow-site and metal-site detectors.
   - (b) **Ranking and calibration.** Rescoring alone takes fpocket from 46.5 to 60.4 top-N+2 and Precision1K from 47 to 82. Calibrated cross-protein scores are needed for Precision1K.
   - (c) **NMS and redundancy control.**
   - (d) **Apo and cryptic robustness.** P2Rank-apo AUPRC is 0.21. Cheap flexibility (normal modes, side-chain-tolerant buriedness) is the lever.
   - (e) **AF robustness via pLDDT/PAE-aware geometry**; fpocket-style geometry collapses on AF models.
   - (f) **Interface sites**, which current single-chain evaluations ignore.
   - Realistic goal: **≥ 62–65% top-N+2 on LIGYSIS with Precision1K ≥ 90**, without large PLMs.
   - Seq2Pocket's about 69% (3B PLM, author-reported) is the current ceiling claim.
9. **Analytical features with proven value**
   - Multi-scale buriedness / protrusion / enclosure (top-ranked everywhere).
   - Size with √n saturation.
   - Apolar fraction / hydrophobic contact density.
   - Polar and charged atom counts (for ion and nucleotide sites).
   - Residue propensities.
   - Local flexibility proxies.
   - All computable on grids or neighbour lists, so they map cleanly to NumPy/Numba and PyTorch/CuPy.
   - A light GBM / logistic scorer over point- and pocket-level features (PRANK-style) is the right "lightweight learned" component.
10. **Conservation surrogate.**
    - Conservation is worth about 2 points top-N+2 and about 11 points Precision1K for P2Rank [V].
    - Decide explicitly whether to allow:
      - (i) no evolutionary signal;
      - (ii) a small PLM embedding (e.g. ESM-2 35M/150M, GPU-optional);
      - (iii) a precomputed conservation track.
    - Report all variants separately, so the "pure analytical" claim stays clean.
11. **Training data for any learned scorer.**
    - BioLiP2-relevant ligands pooled per UniProt (LIGYSIS-style multi-site labels; multi-structure pooling helps per LIGYSIS), or PLINDER (CC BY).
    - Use cluster-level hold-out against **all** test sets.
    - Keep the scorer small and inspect feature importances. Ideally train on a CHEN11-sized, fold-diverse set as an ablation, to show the gain is not memorisation.
12. **Baselines to run ourselves**, with pinned versions:
    - P2Rank 2.5.x (`default`, `alphafold`, conservation variants if feasible), fpocket 4.x, fpocket+PRANK.
    - Optionally: GrASP, DeepPocket and PUResNetV2, using released weights and noting their training overlap; PocketMiner on CryptoBench.
    - Use published numbers only as a reproduction check.
13. **Runtime targets and reporting.**
    - P2Rank is about 0.5–0.9 s per protein warm per core; fpocket about 0.2 s.
    - "Ultra-fast" credibly means **≤ 50–100 ms per average protein per CPU core**, and ≥ 10³ structures per minute on one GPU.
    - Report warm and cold timings and I/O, with hardware stated.
14. **Statistics and integrity.**
    - Fix seeds; bootstrap CIs over proteins / clusters; paired tests vs P2Rank on identical inputs.
    - Freeze all metric code (with toy tests) before method development.
    - Never retune on the test sets; tune on a held-out validation split, e.g. JOINED, a LIGYSIS-disjoint BioLiP2 dev split, or CryptoBench train folds.

---

### Key URLs

**Benchmarks and data**
- LIGYSIS: PMC11552181 · github.com/bartongroup/LBS-comparison · Zenodo 10.5281/zenodo.13121414
- p2rank-datasets: github.com/rdk/p2rank-datasets
- CryptoBench: osf.io/pz4a9 · github.com/skrhakv/CryptoBench
- PocketWatch: huggingface.co/datasets/genesisml/PocketWatch · arXiv 2610.17696
- HOTPocket: github.com/Helix-Research-Lab/HOTPocket · Zenodo 16891050 / 17653326
- AF2BIND: github.com/sokrypton/af2bind · Zenodo 17646944 / 17683379
- AHoJ-DB: apoholo.cz/db
- PLINDER: plinder-org.github.io/plinder
- BioLiP2: zhanggroup.org/BioLiP
- sc-PDB: drugdesign.unistra.fr/scPDB
- Membrane-interface set: Zenodo 10.5281/zenodo.14918827

**Methods**
- P2Rank: github.com/rdk/p2rank (MIT) · PMC6091426
- IF-SitePred / HAP: PMC10941399 · github.com/oxpig/binding-sites
- Seq2Pocket: bioRxiv 10.64898/2026.01.28.702257 · github.com/skrhakv/seq2pocket
- UniSite: arXiv 2506.03237
- AlphaFill: PMC9911346 · CavitySpace: PMC9312471 · Zillmer & Walther: PMC12324681 · Akdel 2022: PMC9663297
