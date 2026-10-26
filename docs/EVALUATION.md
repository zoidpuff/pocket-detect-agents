# Evaluation harness — definitions, conventions, validation

Code: `src/pocketdetect/structure.py` (loading), `src/pocketdetect/ligands.py` (ground truth),
`src/pocketdetect/evaluate.py` (metrics, statistics), `scripts/evaluate.py` (CLI),
`scripts/compare_methods.py` (paired comparison), `scripts/validate_vs_p2rank.py`
(cross-check against P2Rank's own evaluator). Tests: `tests/test_{structure,ligands,evaluate,controls,real_pdb,cli}.py`.

**Rule: every method (ours and baselines) is scored by this harness from the common prediction
format, against the same cached ground truth. Never quote a tool's self-reported numbers next to
ours.** Changing a definition below requires a new rule-set / criterion name, not an edit of an
existing one.

## 1. Structure loading (`load_structure`)

| Aspect | Convention |
|---|---|
| Formats | PDB, mmCIF, optionally `.gz` (gemmi 0.7.5 `read_structure`, `setup_entities`) |
| Models | first model only |
| Altlocs | `first` (default; first altloc letter seen per residue *position* chain+seqnum+icode, = BioJava/gemmi) or `occupancy` (highest summed occupancy, ties → first); atoms without altloc always kept. `all` keeps everything (special use) |
| IDs | author chain ID, author residue number + insertion code; residue ID string `chain:resnum[icode]`, e.g. `A:52A`, `A-2:17` (assembly copy), `B:-3` |
| Hydrogens | element H or D; removed by default (`remove_hydrogens=True`) |
| Residue classes | `PROTEIN`: amino-acid-like residue (gemmi tabulated amino acid such as MSE/SEP/TPO, or has N, CA, C) whose peptide-bonded fragment (C(i)–N(i+1) ≤ 2.0 Å within a chain) contains ≥1 ATOM-record residue. `WATER`: HOH/DOD/WAT/H2O/…. `LIGAND`: any other HETATM or non-polymer/branched residue, including free amino acids and all-HETATM peptidic inhibitors. `OTHER_POLYMER`: nucleic acids etc. |
| Options | `chains=[...]`, `protein_only`, `remove_water`, `remove_hetero` (= protein only), `altloc` |
| B-factor | `b_factor` array (holds pLDDT for AlphaFold models) |
| Assemblies | `load_assembly(pdb_id, n)` reads the wwPDB pre-generated file `<mirror>/assemblies/mmCIF/divided/<id[1:3]>/<id>-assembly<n>.cif.gz`; symmetry copies are separate chains named `A-2`, `A-3`, … in model 1. Mirror root: `/cluster/work/beltrao/shared/rsyncPDB/data` (env `POCKETDETECT_PDB_MIRROR`) |

Speed: ~7 µs/atom (pure-Python loop over gemmi atoms; 2.5k-atom kinase 0.05 s, 60k-atom capsid 0.4 s).

## 2. Ground truth: ligand rule sets (`pocketdetect.ligands`)

A rule set is a named, frozen `LigandRules`; `get_rules(name)`, `register_rules(...)`. The rule set
name and its full parameter dict are recorded in every evaluation (`run_info.json`).

### 2.1 `p2rank` (default for headline metrics)

Re-implementation of P2Rank's ligand categorisation. Sources: rdk/p2rank GitHub, develop branch
commit `b9b54b7e49513b665c913cc90fbfa68ec4133521` — `Params.groovy` (defaults),
`domain/Ligands.groovy` (`loadForProtein`, `categorizeLigand`), `geom/Struct.groovy`
(`getLigandGroups`, `isLigandCandidateGroup`, `isAminoAcidResidueHeuristic`), `domain/Ligand.groovy`;
behaviour checked empirically against the P2Rank 2.5.1 binary (Section 6).

1. **Candidates**: residues of class `LIGAND` (non-water, non-protein, HETATM or non-polymer).
   Residue names BioJava types as amino acid / nucleotide without chem-comp download (20 standard
   amino acids + CSE, DAR, MSE, SEC, TYS; A, C, G, I, U, DA, DC, DG, DI, DT — the set bundled in
   P2Rank's biojava-structure jar) are never candidates (P2Rank requires GroupType HETATM). So free
   amino acids (TRP in 1zt9) and standard residues of peptide ligands are not ligands.
2. **Ignored** (removed before clustering): `HOH DOD WAT NAG MAN UNK GLC ABA MPD GOL SO4 PO4`
   (`ignore_het_groups`).
3. **Clustering**: remaining groups are merged by single linkage on atom–atom distance ≤ 1.7 Å
   (`ligand_clustering_distance`); a merged ligand is named `A&B|chain:num&chain:num`.
4. **Categorisation** (in order): *small* if < 5 atoms (`min_ligand_atoms`); *distant* if min
   distance ligand atom → protein heavy atom > 4.0 Å (`ligand_protein_contact_distance`); *distant*
   if distance of the ligand's mass-weighted centre → nearest protein heavy atom > 5.5 Å
   (`ligc_prot_dist`); else **relevant**. Only relevant ligands are evaluation targets.
5. **Explicit ligand spec** (manifest column `ligand_spec`; P2Rank `.ds` syntax `NAME`,
   `NAME[C_123A]`, `NAME[group_id:C_123A]`, `NAME[atom_id:1234]`, comma separated): matching
   candidates are clustered and are all relevant (no size/distance filters, as P2Rank); other
   candidates become *ignored*. Definitions matching nothing (or unparseable, e.g. `<CONFLICTS>` in
   `coach420(mlig).ds`) are recorded in `GroundTruth.meta['unmatched_spec']`, not fatal (P2Rank
   behaviour); `strict_spec=True` makes them errors.
6. **Ligand atoms**: heavy atoms only; first altloc.
7. **Ligand centre** (for DCC): geometric centroid of the ligand atoms.
8. **Binding residues** (for residue metrics): protein residues with any heavy atom ≤ 4.0 Å from any
   ligand atom.

### 2.2 `p2rank_strict` (reproduces P2Rank 2.5.1 exactly; use for literature comparability checks)

`p2rank` plus the P2Rank code's quirks: (a) hydrogens are kept in ligand atom sets (atom count,
DCA, centres); (b) `Struct.isHydrogenAtom` also drops protein atoms whose *name* starts with H or has
H as 2nd character (Arg NH1/NH2, Tyr OH, Trp CH2) from the protein atoms used by the
contact / centre-distance filters; (c) BioJava's residue heuristic: a non-standard amino acid in a
chain is a protein residue only if the previous residue is absent or standard and the next exists
and is standard — otherwise (C-terminal or consecutive modified residues, e.g. PTR pairs, CGU, STA)
it becomes a ligand candidate; (d) ligand altlocs "by name": first record of every atom name per
ligand residue (first conformer + atoms present only in later conformers).

### 2.3 `all_contacting`

Every non-water HETATM/non-polymer group (any size, standard names included) with min distance ≤ 4 Å
to the protein. For "any ligand" sanity analyses only.

### 2.4 Site-defined benchmarks (LIGYSIS, CryptoBench, PocketWatch, AF, dimers)

Ground truth is a *site table* (`pocketdetect.sites`, Section 10), not ligands extracted by rules.

## 3. Prediction format

One CSV per structure, `<pred_dir>/<structure_id>.csv`:

```
rank,score,center_x,center_y,center_z,residues
1,0.93,10.1,-3.2,22.0,A:45 A:46 A:52A B:12
```

`rank` is authoritative for ordering (must be unique integers; duplicates → error; gaps allowed);
`score` is used only for the pooled AP. `residues` space-separated `chain:resnum[icode]` (author
numbering, same chain IDs as the input structure), may be empty. A header-only file = zero
pockets. A missing file = zero pockets and is counted (`pred_missing`, `n_pred_missing`).

## 4. Metrics

Notation per structure: relevant ligands *j* = 1..n, pockets *i* = 1..m in rank order, pocket
centre *p_i*, ligand atoms *A_j*, ligand centre *c_j*.

* **DCA**(i,j) = min_{a∈A_j} ‖p_i − a‖ (heavy atoms). **DCC**(i,j) = ‖p_i − c_j‖.
* **Criterion** `DCAt` / `DCCt`: pocket *i* identifies ligand *j* iff distance ≤ t (inclusive).
  `rank_<crit>`(j) = smallest *i* that identifies *j* (any pocket may identify any number of ligands;
  no one-to-one assignment — P2Rank convention), −1 if none.
* **`LDCCt`** (LIGYSIS style): only the pocket closest to *c_j* by DCC counts (ties → better rank);
  rank = its rank if its DCC ≤ t else −1.
* **Success@k** (ligand-centric): `success_<crit>_<lab>` = 0 < rank ≤ k, for labels `top1, top3,
  top5, top10` (k fixed) and `topn` (k = n), `topn2` (k = n + 2), n = number of relevant ligands of
  that structure.
* **Averages**: `ligand` = mean over all ligands of the dataset (P2Rank's headline);
  `protein` = mean over structures of the per-structure mean. Structures with n = 0 are excluded
  from every summary statistic (reported as `n_structures_no_ligands`).
* **Distances**: `dca_top1`, `dcc_top1` (top-ranked pocket), `dca_min`, `dcc_min` (over all pockets).
* **Residue overlap** (only if the method outputs residues; NaN otherwise): for ligand *j* the
  *matched* pocket is the first pocket identifying it under the primary criterion within
  top-(n+2). With P = its residue set, T = binding residues: `res_jaccard_matched` = |P∩T|/|P∪T|
  (a residue-level analogue of DVO; 0 if unmatched), `res_irel_matched` = |P∩T|/|T| (LIGYSIS
  IRel), `res_precision_matched` = |P∩T|/|P|; `res_jaccard_best_topn2` = max Jaccard over the
  top-(n+2) pockets. Summary also reports `res_jaccard_matched_given_success` (mean over matched
  ligands only, as DeepPocket reports DVO).
* **Residue-level classification** (per structure, universe = protein residues of the structure):
  T = ∪ binding residues of relevant ligands, P = ∪ residues of the top-(n+2) pockets (`res_topn2_*`)
  or all pockets (`res_all_*`). TP = |P∩T|, FP = |P∖T| (predicted IDs that are not protein residues
  of the structure count as FP and are reported in `n_pred_res_unknown` — catches chain/numbering
  mismatches), FN = |T∖P|, TN = |U| − |(P∩U)∪T|. Precision, recall, F1, MCC (MCC = 0 if a margin is
  empty). Summary: `protein` (macro) and `micro` (pooled counts).
* **Pocket-level precision** `pocket_precision_<lab>`: fraction of the top-min(k, m) pockets that
  identify ≥ 1 relevant ligand (primary criterion); NaN if m = 0.
* **AP** (per structure, rank-based): pockets matched greedily one-to-one in rank order (each
  pocket takes the unmatched ligand it identifies with the smallest distance; duplicates of an
  already found ligand are false positives). AP = (1/n) Σ_i precision@i · TP_i (missed ligands
  contribute 0). Summary `ap/protein` = mean over structures.
* **Pooled AP** (`ap_pooled`): all pockets of all structures ranked by `score` (descending; tied
  scores form one threshold), same greedy TP labels, recall denominator = total relevant ligands.
  Measures cross-protein score calibration (relevant for proteome-wide ranking); skipped if any
  score is non-finite.
* **Primary criterion**: first entry of `--criteria` (default `DCA4,DCC4` → DCA 4 Å).

## 5. Statistics

* **Bootstrap CIs**: nonparametric bootstrap over structures (with ≥ 1 relevant ligand), B = 1000
  (`--n-boot`), percentile 95 % interval. Ligand-averaged rates are ratio estimators
  Σ_w successes / Σ_w ligands; micro residue metrics recomputed from resampled pooled counts; pooled
  AP recomputed with bootstrap multiplicities. Seed: `pocketdetect.seeds.BOOTSTRAP_SEED` = first
  entry of the fixed project list `SEEDS`.
* **Paired comparison** (`compare_methods`, `scripts/compare_methods.py`): requires identical ligand
  keys (same ground truth; the script also checks the ground-truth cache key). Reports both methods'
  rates, the difference A − B (ligand- and protein-averaged) with a paired cluster-bootstrap 95 % CI
  (proteins resampled, pairs kept; B = 10 000) and two-sided bootstrap p = 2·min(P(Δ*≤0), P(Δ*≥0)),
  plus the exact McNemar test on discordant ligands (binomial, p = 0.5; assumes ligand independence
  — prefer the bootstrap when structures carry several ligands).

## 6. Validation

* Unit tests with known answers (`pytest tests/`): thresholds 3.9/4.0/4.1 Å (inclusive), DCA vs
  DCC divergence, top-n/top-(n+2), no predictions, more ligands than pockets, rank vs file order and
  score ties, duplicate ranks (error), duplicate centres (AP FP), multi-chain + insertion-code
  residue IDs, unknown residue IDs, LDCC vs DCC, hydrogen exclusion (`p2rank` vs `p2rank_strict`),
  altlocs (`first` / `occupancy`), first model only, MSE in chain = protein, free amino acid and
  all-HETATM peptide = ligand, clustering at 1.7 Å, ignore list, small/distant categories,
  explicit specs, bootstrap determinism, pooled AP with ties, McNemar.
* **Controls** (`tests/test_controls.py`): 30 synthetic globular proteins with one carved pocket +
  ligand — positive control (ligand centroids as predictions) gives 1.000 for every criterion/k,
  AP and pooled AP = 1; negative control (random points in the 2.5–3.5 Å solvent shell, seeds
  SEEDS[:3]) gives DCA4 top-1 ≈ 0.06, top-(n+2) ≈ 0.11, top-10 ≈ 0.26. Real entry 1m17: centroid →
  success; a random surface point hits erlotinib 2 % of the time (200 seeds).
* **Real entries** (`tests/test_real_pdb.py`, PDB mirror): 1M17 → exactly `AQ4|A:999` (29 atoms,
  binding residues include hinge Met769); 3EML (ZMA relevant, SO4 ignored); 2W83 explicit specs
  reproduce P2Rank's own test expectations (GTP,MG → 6; GTP → 3; GTP[B_1173] → 1); MSE never a ligand.
* **Against P2Rank 2.5.1** (`prank eval-predict <ds> -log_cases 1`, P2Rank's predictions converted to
  our format and scored by our harness; reports in `results/validation/eval_vs_p2rank/`):

| dataset | rules | ligands ours / P2Rank | DCA4 top-n | DCA4 top-(n+2) | DCC4 top-n | DCC4 top-(n+2) |
|---|---|---|---|---|---|---|
| COACH420 | p2rank_strict | 510 / 510 | 72.35 / 72.4 | 77.45 / 77.5 | 43.92 / 43.9 | 47.06 / 47.1 |
| COACH420(mlig) | p2rank_strict | 379 / 379 | 69.39 / 69.4 | 73.09 / 73.1 | 41.95 / 42.0 | 44.59 / 44.6 |
| HOLO4K | p2rank_strict | 9248 / 9251 | 72.99 / 73.0 | 78.24 / 78.2 | 46.80 / 46.8 | 50.59 / 50.6 |
| HOLO4K(mlig) | p2rank_strict | 6874 / 6877 | 76.58 / 76.6 | 83.39 / 83.4 | 49.24 / 49.2 | 54.03 / 54.0 |
| COACH420 | p2rank | 510 / 510 | 72.16 / 72.4 | 77.25 / 77.5 | 43.92 / 43.9 | 47.06 / 47.1 |
| HOLO4K | p2rank | 9208 / 9251 | 72.98 / 73.0 | 78.28 / 78.2 | 46.93 / 46.8 | 50.68 / 50.6 |

  (P2Rank prints percentages rounded to 0.1.) Per-ligand DCA4 ranks agree for every paired ligand
  except 1 of 9 symmetric AI1 copies in 1fd7. Residual `p2rank_strict` differences: 1p52 (DAR in a
  non-polymer chain is a P2Rank ligand), 1vbt (ALT). `p2rank` differs from P2Rank by design on
  ~0.5 % of HOLO4K ligands (H-counted small ligands such as DMS, the Arg/Tyr name quirk, modified
  residues inside chains, altloc-only atoms).

## 7. Known differences from published protocols

* **P2Rank papers**: our default `p2rank` rules differ from the P2Rank code as listed in 2.2
  (deliberately: heavy atoms only, no name-based H bug, modified residues in chains are protein,
  first altloc). Use `p2rank_strict` when a number must be comparable to P2Rank's own tables.
  P2Rank's develop branch evaluates DCC to the mass-weighted centre; 2.5.1 (and we) use the
  geometric centroid.
* **(mlig) datasets**: `coach420(mlig)`/`holo4k(mlig)` restrict ground truth to MOAD-valid ligand
  codes (explicit specs); several papers report on the non-mlig versions. State which is used.
* **DVO**: DeepPocket's DVO is volumetric (predicted pocket grid vs ligand-occupied volume). The
  common prediction format has no volumes, so we report residue-set Jaccard instead. Not comparable
  in absolute value.
* **DCC with other centres**: some papers use the centroid of ligand heavy atoms (= ours) — others the
  centre of binding-site residues; LIGYSIS uses its site centre and a 12 Å threshold with
  closest-pocket matching (`LDCC12`).
* **Ligand counts**: other papers' versions of COACH420/HOLO4K ground truth (e.g. PUResNet,
  DeepPocket, VN-EGNN) re-derive ligands with their own filters; our counts follow P2Rank.
* **Hydrogens / altlocs / multi-model** handling is fixed as in Section 1 for all methods.

## 8. CLI

```
python scripts/evaluate.py --dataset-manifest <csv> --pred-dir <dir> --out <dir> \
    --ligand-rules p2rank [--criteria DCA4,DCC4,LDCC12] [--n-boot 1000] [--jobs 16] [--gt-cache f.pkl]
```

Manifest columns: `structure_id,structure_path[,ligand_spec][,chains]` (paths relative to the
manifest's directory allowed; `chains` comma-separated author IDs, empty/`*` = all). Outputs:
`ligands.csv` (one row per relevant ligand, all per-ligand columns above), `proteins.csv`,
`pockets.csv` (per pocket: `hit_any`, `tp_greedy`, `matched_ligand`), `summary.csv`
(`metric, average, value, ci_low, ci_high, n`), `ground_truth.csv` (all ligands with category,
centre, distances, binding residues), `run_info.json` (rules, seed, counts, ground-truth errors,
timings), `gt_cache.pkl` (reused when manifest + rules + loader/ligand code are unchanged).
Ground-truth extraction failures abort unless `--allow-gt-errors`. HOLO4K (4009 structures,
32 jobs, B = 1000): ~15 s wall time.

Paired comparison: `python scripts/compare_methods.py --a <eval_dir_A> --b <eval_dir_B> --metrics success_DCA4_topn,...`.

## 9. Open questions

* DCA ground truth is missing for 5 % of scored LIGYSIS sites (no ligand copy could be placed without
  clashes / off-site); those sites are excluded from the DCA (C1-strict) denominator for every method.
* The dimer benchmark is evaluated through Track D's `scripts/eval_dimer.py` (copies, J ≥ 0.5 grouping,
  label-free collapse), not through the site table; only its CIs are recomputed here with cluster units.

## 10. Site-table protocol (LIGYSIS LBS-comparison) — `pocketdetect.site_eval`

**Site table** (`pocketdetect.sites`, TSV): `structure_id, site_id, center_x/y/z, residues, n_sites, scored`
(+ optional `structure_path, input_chains, ligand_atoms, n_residues_structure, group`, metadata). `n_sites` = N of
top-N for the structure; `scored` = site is an evaluation target. Predictions: the common CSV (`<structure_id>.csv`).

**Definitions** (port of bartongroup/LBS-comparison v2.0.0 `benchmark_lib.py` + notebook 12; Utgés et al. 2024):
* DCC = distance pocket centre → site centre, **rounded to 0.01 Å** before thresholding (their code rounds).
* Closest-pocket matching: for each scored site the pocket with minimal DCC (ties → larger IRel → better rank);
  `recall_topn`/`recall_topn2` = that pocket has DCC ≤ 12 Å and rank ≤ N (+2); `recall_max` = no rank cap.
  IRel = |pocket ∩ site residues| / |site residues|.
* Stricter variants: `recall_anyp_*` (any pocket within rank ≤ N+k with DCC ≤ 12), `recall_1to1_*` (maximum
  one-to-one Hungarian matching of sites to top-(N+k) pockets), `recall_ldca_*` (pocket minimising DCA = min
  distance to the site's ligand heavy atoms; DCA ≤ 4 Å and rank ≤ N+k; needs `ligand_atoms`), `recall_anyp_dca_*`.
* Precision (pooled over the dataset, all pockets ranked by score desc; ties → structure_id, rank): a pocket is TP
  if DCC ≤ 12 Å to any scored site of its structure. `P1K` = precision of the first 1,000; `TP@100FP` = TPs before
  the 101st FP; `P1K_1to1` / `TP@100FP_1to1`: a pocket is TP only if it hits a site not yet claimed by a
  higher-ranked pocket (claims the closest unclaimed site).
* RRO = mean IRel of closest pockets with DCC ≤ 12 Å (any rank) and IRel > 0. RVO (volumetric) not implemented
  (needs pocket grids; their 3.3 GB shape dict is for their methods only).
* Residue level per structure: true = ∪ residues of all sites (scored or not), predicted = ∪ residues of all
  pockets; F1, MCC (sklearn conventions: 0 when undefined); NaN if the method reports no residues; MCC needs
  `n_residues_structure`. Mean over structures.
* Missing prediction file = zero pockets: all its sites are misses (denominators never shrink).

**LIGYSIS specifics** (`pocketdetect.ligysis`, restricted unpickler for their pickles): 2,775 representative chains,
6,882 sites (N counts all of them), 6,721 scored = their `filter_redundant_pockets2(LIGYSIS, Jaccard 0.75, 5 Å)`
(port verified identical to their function). Site centre = CA centroid of the site residues, stored by the authors
in a transformed frame x_T = R(x − c0); `data/processed/gt/ligysis_lbs/sites.tsv` (built by
`scripts/build_gt_ligysis.py`) maps centres back to the wwPDB asymmetric-unit frame (median deviation from a
re-computed CA centroid 0.0005 Å; 4 sites > 1 Å use the recomputed centroid) and converts the authors' residue ids
(`aas` = label_seq_id) to author ids `chain:num[icode]`. Input = mirror mmCIF restricted to `input_chains`.

**Reproduction gate** (`scripts/reproduce_ligysis.py`, report `results/validation/ligysis_reproduction.md`):
the authors' published per-chain predictions of all 33 method variants scored by our harness reproduce their
published top-N / top-N+2 / max-recall **exactly (identical counts) for 31/33 and within 0.4 points for all 33**
(differences only from tie-breaking among duplicate predictions); P1K and TP@100FP identical except for the 4
methods whose score ties they shuffled (PUResNet, PocketFinder, Ligsite, Surfnet); RRO identical. Their own
functions (`filter_redundant_pockets2`, `compare_ALL_predictions`, `success_rate_COMB`, executed via ast without
running the module) on raw P2Rank / GrASP predictions give the same counts as our harness. The same predictions
mapped to the original frame and scored against the new-method site table give identical counts.

**Other site tables:** CryptoBench test (`data/processed/gt/cryptobench_test`, 222 apo structures, 270 sites, centre =
CA centroid of site residues on apo, `input_chains` must be respected), PocketWatch test
(`data/processed/gt/pocketwatch_test`, 1,566 ESMFold structures, 1,573 scored / 728 masked sites, N = scored sites,
centre = their ligand-centroid transfer recipe). Builders: `scripts/build_gt_{cryptobench,pocketwatch,ligysis}.py`;
each output dir has a README with sources, sha256s and rules.

## 11. Cluster bootstrap and test gating

* Resampling units (`pocketdetect.clusters`, `scripts/build_test_clusters.py`): MMseqs2 easy-cluster
  `--min-seq-id 0.3 -c 0.5 --cov-mode 0` over all test-union sequences (`data/processed/clusters/test_clusters.tsv`;
  4,846 clusters; LIGYSIS 1,687, CryptoBench 203 of 222 structures, PocketWatch 1,088). Multi-chain structures map
  to all their chains' clusters, merged by union–find. Sensitivity: UniProt accession units. 1,000 resamples,
  seed `BOOTSTRAP_SEED`; percentile CIs; P1K recomputed in every resample with unit multiplicities.
* `evaluate.summarize`/`compare_methods` and `site_eval.summarize_sites`/`compare_site_methods` accept the unit
  mapping; paired comparisons treat failures/missing predictions as misses (never dropped) and report counts.
* `scripts/run_test_eval.py` is the only test entry point: refuses modified tracked files, records commit, model
  sha256, scoring-code / PREREG / GT / prediction-dir sha256s, caps non-final versions at 3 per benchmark × method,
  never overwrites, writes `results/test/<version>/<benchmark>/<method>/` (summary.csv = cluster bootstrap,
  summary_uniprot.csv = sensitivity) and appends `results/TEST_LEDGER.md`. Benchmarks: `pocketdetect.benchmarks.REGISTRY`
  (ligysis, coach420_mlig, holo4k_mlig [rules p2rank_strict; prediction files `<name>.pdb.csv`], pocketwatch,
  cryptobench, af_set, dimer). `scripts/run_dev_eval.py` = same evaluation for dev sets, no ledger, refuses test GT.
  Self-test with published baselines: LIGYSIS fpocket+PRANK 60.42 % top-N+2 [58.36, 62.32], P1K 81.7 % [77.7, 85.2];
  COACH420(mlig) P2Rank 2.5.1 DCA4 top-n 69.39 % [63.80, 75.26].

### 11b. LIGYSIS C1-strict ligand atoms

(`scripts/build_ligysis_ligands.py`, `pocketdetect.importers.ligysis_ligands`): every
ligand instance listed for a site (`LIGYSIS_human_chains_per_lig_MAY_2024.pkl`, 58,152 instances from 20,300
entries) is located in the asymmetric-unit mmCIF (crystal-symmetry image if it touches the chain only through
symmetry), the contacted chain is superposed onto the representative chain on UniProt-paired CA of the site core
(SIFTS; rigid-core Kabsch, core RMSD ≤ 2 Å), the ligand heavy atoms are transformed, copies without a heavy atom
≤ 4.5 Å from a rep-chain site-residue heavy atom are dropped (1,280 off-site copies), duplicate copies and non-ion
atoms < 2.2 Å from rep-chain protein atoms are removed (site dropped if > 50 % clash). Result: 6,387 / 6,721 scored
sites with ligand atoms (`ligand_atoms` in `data/processed/gt/ligysis_lbs/sites.tsv`). Checks: ligands of the rep
entry itself are reproduced exactly for 129/140 sampled cases (7 differ by one bond length = clash-removed atoms,
4 instances not placed on that chain); oracles: a ligand atom → DCA4 top-N+2 = 100 %, ligand-atom centroid → 95.2 %
(union centroids of spread multi-ligand sites can be > 4 Å from every atom), site CA centroid → 76.8 %.
Primary C1-strict metric `recall_ldca_topn2` (pocket with minimal DCA must have DCA ≤ 4 Å and rank ≤ N+2);
`recall_anyp_dca_topn2` reported alongside.

## 12. AF set (O5) and dimers (O6)

* **AF set** (`scripts/build_gt_af.py` from Track C `data/processed/af/gt/*.json.gz`): `data/processed/gt/af_set/sites.tsv`
  = the 2,526 full-length AFDB v6 models with included LBS-benchmark sites (6,731 sites); N = all of them,
  scored = non-redundant (LIGYSIS filter on the model: 6,502); centre = CA centroid on the model; ligand atoms =
  Track C transfers with local RMSD ≤ 2 Å (6,040 scored sites). Sensitivity table `af_set_allsites` (all LIGYSIS
  sites of those UniProts). Oracle (site centres): top-N+2 = 1.0, DCA4 = 0.77, hallucination 0.
* **AF utilities** (`pocketdetect.af_eval`): pocket pLDDT (mean residue CA pLDDT over pocket residues, or residues
  ≤ 6 Å of the centre); `drop_low_plddt` = the "naive + drop pockets with mean pLDDT < 70" baseline
  (`scripts/af_plddt_filter_baseline.py` turns any prediction dir into that baseline); hallucination =
  pocket with mean pLDDT < 50 matching no site (DCC ≤ 12 Å to any site), reported over all pockets and the top-(N+2),
  always using the pLDDT of the unpruned model; `run_benchmark` adds `hallucination_rate`, `hallucination_rate_topn2`,
  `low_plddt_pocket_rate` (cluster-bootstrap CIs) for `af_set*`.
* **Dimers** (`--benchmark dimer`): assembly-mode inputs, J ≥ 0.5 site grouping (PREREG A1), label-free prediction
  collapse, success if any copy is hit — computed by Track D's evaluator; headline `recall_DCC12_topn2@interface`
  (O6), `recall_DCA4_topn2@interface`, `precision_DCC12_topn2@all`, with 30 % cluster (and UniProt) bootstrap
  over complexes. Oracle (GT copies) = 1.0.

## 13. Pre-registered test pipeline

1. **Predictions** — `scripts/predict_benchmarks.py --version <v> --freeze <FREEZE.md>` (only after the freeze).
   Models (and AF rankers) are read from FREEZE.md and must match their listed sha256. PREREG mapping: ranker `all`
   for LIGYSIS full / AF set / dimers, `small` for LIGYSIS_NI / COACH420(mlig) / HOLO4K(mlig) / CryptoBench /
   PocketWatch; `--af-mode` (+ AF ranker) for PocketWatch and the AF set; presets default (primary) and accurate;
   seeds SEEDS[:3] (model files `_s0.._s2`). Inputs are the baselines' (`baselines/outputs/inputs/<bench>`, dimer
   assemblies), LIGYSIS predictions are mapped from the authors' TRANS frame to the evaluation frame with
   `pocketdetect.ligysis.trans_preds_to_original` (identical to the baseline conversion; verified file-by-file).
   Backend: fast CLI on GPUs; the reference CLI (sharded CPU processes) where AF mode needs an AF ranker that the
   fast CLI cannot take. Wall-clock per run → `results/test/<v>/preds/{index.json,TIMINGS.md}`; `--plan` prints the plan.
2. **Analysis** — `scripts/run_prereg_analysis.py --version <v>`: every BURROW variant is scored through the gated
   `run_test_eval.py` (ledger); comparators are re-scored from their baseline prediction directories with the same
   code/GT (`results/test/<v>/comparators`). Logic in `pocketdetect.prereg`:
   * Family 1 (fixed sequence, each seed, vs published fpocket+PRANK per-chain predictions **and** the fpocket
     4.2.3+PRANK 2.4.2 re-run): C1 = recall_topn2 Δ ≥ 2.0 and CI low > 0; C1-strict = recall_ldca_topn2 CI low > 0;
     C2 = P1K CI low > −3. Common-chain-subset sensitivity for the published comparator.
   * Family 2 (each seed): O2a/O2b/O3/O4/O5/O6 vs the best (highest point estimate) of the listed baselines;
     one-sided bootstrap p for H0 Δ ≤ −2, Holm step-down at α = 0.025; pass / fail (upper CI < −2) / inconclusive;
     superiority = CI low > 0. Endpoint verdict = pass in all three seeds (implementation choice; per-seed reported).
   * Paired comparisons: rows matched on the reference key, missing = miss, rows undefined for both dropped
     (C1-strict sites without ligand atoms); pooled rates; same unit weights for both arms; P1K recomputed per resample.
   * Descriptive: secondary metrics, LIGYSIS_NI, accurate preset, strata (fold novelty = max Foldseek TM band of the
     structure's chains, LIGYSIS ligand class ion / nucleotide-cofactor / other, AF site pLDDT band, dimer site class;
     strata < 100 rows flagged), controls (oracle, protein centroid, random surface, ≥ 12 Å surface tiling).
   * Output `results/test/<v>/SUMMARY.md` + `summary.json` with provenance.
3. **Dry run** — `--dry-run --stand-in p2rank_2.5.1` writes to `results/test/_dryrun/` (no ledger, no gating): a
   baseline stands in for BURROW; checks method-vs-itself (Δ = 0, CI = [0, 0]) and oracle-vs-comparator.
