# TRAINSET — leakage-controlled, LIGYSIS-style training and dev sets (Track A)

Spec: DESIGN.md §2.5, overridden by §9 (v3 amendments: frozen test union, complete leakage rule
incl. E-value, train-vs-dev rule, clash dropping). Owner: Track A. All numbers below are copied from
the JSON reports named next to them (single source of truth = those files; re-run the pipeline to
regenerate).

## Pipeline (run in this order; every step is a script, all seeds from `pocketdetect.seeds`)

| step | script | main outputs |
|---|---|---|
| all (in order, detached) | `scripts/data/build_trainset_all.sh` | logs in `data/processed/train/work/*.log` |
| 1 test union | `scripts/data/build_trainset_testunion.py` | `data/manifests/test_union.tsv` (+`.sha256`, `_meta.tsv`, `_pdb_ids.txt`, `_uniprots.txt`, `_counts.json`), `data/processed/test_sequences.fasta` |
| 2 candidates | `scripts/data/build_trainset_candidates.py` | `data/processed/train/work/candidate_{chains,ligands}.parquet`, `masked_ligands.parquet`, `candidates_counts.json` |
| 3+4 leakage, clusters, split | `scripts/data/build_trainset_split.py` | `data/processed/train/split_chains.{parquet,tsv}`, `train_chains.txt`, `dev_chains.txt`, `split_counts.json` |
| 5+6 pooling, sites, assemblies | `scripts/data/build_trainset_pool.py --max-per-cluster 3` | `data/processed/{train,dev}/manifest.csv`, `.../entries/<entry_id>.npz`, `data/processed/train/pool_counts.json` |
| 5b assembly-partner leak check | `scripts/data/build_trainset_asmcheck.py` | `asmcheck_counts.json` (drops leaking assembly inputs) |
| 5c final pass on deployed inputs | `scripts/data/build_trainset_finalcheck.py` | `final_dropped.tsv`, `finalcheck_counts.json` |
| 7 verification | `scripts/data/verify_trainset.py` | `data/processed/train/verify_report.json` |
| strata (later) | `scripts/data/build_trainset_foldseek_strata.py` | `data/manifests/test_fold_novelty.tsv`, `data/processed/train/fold_to_test.tsv` |

Library code: `src/pocketdetect/datasets/` — `pdbinfo.py` (mmCIF entity/UniProt/method reader),
`biolip.py` (BioLiP2 reader, ion/metal classification), `leakage.py` (the rule + MMseqs2 wrapper),
`pooling.py` (alignment, Kabsch, LIGYSIS site clustering, entry loader), `trainset.py` (**reader for
consumers**: `read_manifest`, `load_entry`, `load_input`, `TrainEntry.ligand_atoms/site_table/mask_residues`).

Inputs: BioLiP2 snapshot of 2025-11-12 (`data/raw/biolip2/current`, sha256 in DATA manifest and
`candidates_counts.json`), local wwPDB mirror (`/cluster/work/beltrao/shared/rsyncPDB/data`,
`derived_data/pdb_entry_type.txt` for the method), SIFTS `pdb_chain_uniprot.tsv.gz`
(2026-10-18; downloaded by Track C to `data/processed/af/external/`, read-only use here, sha256 in
`candidates_counts.json`).


## 1. Test union (frozen before the training build)

`data/manifests/test_union.tsv` (columns `id, sequence, source_set`), sha256 in
`data/manifests/test_union.sha256`; the same sha256 is recorded in `split_counts.json`,
`pool_counts.json` and in the `meta_json` of every entry npz (`test_union_sha256`). Later-added test
sets must be overlap-checked against the final lists (`verify_trainset.py`), never silently joined.

| source_set | what | sequence used |
|---|---|---|
| `ligysis` | the 2,775 evaluated LIGYSIS representative chains | canonical entity sequence (`_entity_poly.pdbx_seq_one_letter_code_can`) of the chain in the PDB mirror |
| `ligysis_extra` | the other 673 chains of `LIGYSIS_3448_chains.pkl` (conservative) | as above |
| `coach420`, `holo4k` | **all protein entities** of every PDB entry in `coach420.ds` / `holo4k.ds` | as above (fallback: observed sequence of the dataset file for obsolete IDs) |
| `cryptobench` | all splits (test + train-fold-0..3): apo chain(s) and holo chain(s) of every pair (`A-B` selections split) | as above |
| `pocketminer` | val + test apo chains (no-chain IDs -> all protein entities); PDB IDs of all `all-structures` files | as above |
| `pocketwatch_test` | 1,566 test-split proteins | UniProt canonical sequence (`sequences.parquet`); PDB IDs of all their sites |
| `af_set` | Track C AF set, `data/processed/af/af_sequences.fasta` (all records, incl. `status=excluded`) | AFDB/UniProt sequence |
| `dimer` | Track D `data/processed/dimer/manifest_chains.tsv` — every chain incl. hetero partners | manifest sequence |

CHEN11 is **not** a test set. HOTPocket, scPDB and AF2BIND are not test sets of this project and are
not in the union (if one is used later it must be overlap-checked and reported).
Also written: `test_union_meta.tsv` (pdb, chain, UniProt per row; UniProt = mmCIF `_struct_ref` UNP
+ SIFTS), `test_union_pdb_ids.txt`, `test_union_uniprots.txt`, `data/processed/test_sequences.fasta`.

## 2. Candidates (BioLiP2)

Rows of `BioLiP.txt.gz` with method X-ray (`diffraction` in wwPDB `pdb_entry_type.txt`) or EM,
0 < resolution <= 3.0 A, receptor chain >= 50 residues (BioLiP receptor sequence = observed
residues). Ligands = BioLiP's own relevant-ligand set **including metal ions**; BioLiP polymer
ligands (`peptide`, `rna`, `dna`) are excluded as labels (their binding residues are kept as
`mask_residues`, see §5). Ion flag = CCD formula with exactly one heavy atom (or an instance with one
heavy atom); metal flag = that atom is a metal (halides / NH4 are ions but not metals).
UniProt = BioLiP column 18, completed by SIFTS.

## 3. Leakage removal (DESIGN §9 rule)

MMseqs2 `easy-search` (-s 7.5, --max-seqs 5000, -e 1e-3) of all unique candidate sequences against
the unique test-union sequences. A chain is removed if any hit has **E < 1e-3 AND fident >= 0.30 AND
(qcov >= 0.5 OR tcov >= 0.5 OR alnlen >= 80)** (`pocketdetect.datasets.leakage.apply_rule`), or if
its PDB ID is in any test set (incl. holo partners / PocketWatch site entries / dimer entries), or
any of its UniProt accessions is a test UniProt.

## 4. Clustering and split

Surviving unique sequences: `mmseqs easy-cluster --min-seq-id 0.3 -c 0.5 --cov-mode 0 -s 7.5`.
30% clusters that share a UniProt accession are merged (union-find) into *split groups* so that no
UniProt straddles train/dev. Groups are permuted with `numpy.random.default_rng(SEEDS[0])`; the first
10% (by number of groups) are dev. Then dev unique sequences are searched against train unique
sequences and every dev chain with a rule hit is removed from dev (same rule as §3).

## 5. Representatives, pooling, sites

Per (split, UniProt) — chains without UniProt are grouped per 30% cluster:
* **Representative**: among chains with observed length >= 0.9 x the longest observed chain of that
  UniProt, the best resolution (ties: more BioLiP ligands, longer, PDB ID).
* **Pool**: up to 20 *other* PDB entries of the same UniProt (one chain per entry, the one with most
  ligands), greedily chosen to add new ligand CCD codes, then by resolution. Only chains that
  survived leakage removal and belong to the same split are used.
* **Sequence-anchored local superposition**: global alignment (BLOSUM62, gaps -10/-1, free end gaps)
  source -> representative; required identity >= 0.9 over aligned columns. Per ligand: source Calpha
  within 10 A of any ligand heavy atom, mapped through the alignment (>= 6 pairs), Kabsch onto the
  representative; kept if Calpha RMSD <= 2.0 A (stored per ligand, `lig_rmsd`, `lig_npairs`).
  (SIFTS residue-level numbering was not used; alignment is robust to renumbering.)
* **Plausibility / clashes** (vs. the representative chain's protein heavy atoms = deployed input):
  ligands whose nearest protein atom is > 4.5 A away are dropped; non-ion ligand atoms < 2.2 A from a
  protein heavy atom are dropped (DESIGN §9); a site whose ligands lose > 50% of their heavy atoms is
  dropped. Ions are never clash-trimmed (metal–ligand distances are ~2 A).
* **Sites**: LIGYSIS clustering — each ligand's fingerprint = its BioLiP binding residues (source
  numbering) mapped onto the representative; distance = 1 − |A∩B|/min(|A|,|B|) (relative
  intersection); average linkage; cut 0.5 (scipy `fcluster`, criterion `distance`). Exact
  per-ligand linkage up to 2,500 ligands (beyond: on unique fingerprints).
* **Mask residues**: binding residues of polymer ligands (peptide/RNA/DNA) of all chains of the
  UniProt (BioLiP renumbered residues mapped by alignment). Not labels — consumers may exclude them
  from negatives.
* **Assembly 1**: availability recorded for every entry (`has_assembly1`). For a seeded draw
  (`default_rng([SEEDS[0], 30])`, p = 0.30) of entries with an assembly-1 file (and <= 60 chains,
  <= 150k atoms): pooled ligands are mapped onto every assembly chain of the same UniProt (local
  superposition again, same RMSD rule), own BioLiP ligands of other-entity chains (same split only)
  are added (exact superposition), clash/near filters are re-applied against all assembly protein
  atoms, and ligands are re-clustered with contact fingerprints (protein atoms <= 4 A, residues
  `chain:num`). `asm_site_is_interface` = >= 3 residues on each of >= 2 chains. Assembly partner
  chains are leak-checked afterwards (`build_trainset_asmcheck.py`: rule vs test union and vs the
  other split); failing entries lose their assembly input.

## 6. Selection

Each 30% cluster contributes its UniProt with most PDB entries (`rank_in_cluster = 0`, the strict
"one per cluster" set); up to 2 more UniProts per cluster are added if they lie in different 50%
identity clusters (mmseqs, cov 0.5) — `--max-per-cluster 3`. Consumers wanting one entry per cluster
filter `rank_in_cluster == 0`; for cluster-balanced training weight entries by 1/(entries in cluster).

## 7. Output format

`data/processed/{train,dev}/manifest.csv`: `entry_id, split, status, uniprot, pdb, chain,
structure_path, resolution, method, n_res, cluster30, group, rank_in_cluster, n_uniprot_entries,
n_sites, n_sites_ion, n_ligands, n_ligands_ion, n_sources_used, max_lig_rmsd, n_mask_residues,
has_assembly1, assembly_status, n_asm_sites, n_asm_interface_sites, npz`.

`entries/<entry_id>.npz` (no pickles; `np.load(..., allow_pickle=False)`; read with
`pocketdetect.datasets.trainset.load_entry`):

| key | shape / type | meaning |
|---|---|---|
| `meta_json` | str | entry metadata incl. structure path, chain, split, cluster, test-union sha256, parameters |
| `rep_res_keys`, `rep_seq` | (R,) str, str | representative residues (author `num[icode]`) and sequence |
| `lig_coords`, `lig_elements` | (A,3) float32, (A,) str | pooled ligand heavy atoms (after clash trimming), representative frame |
| `lig_ptr` | (L+1,) int | atom offsets per ligand |
| `lig_site`, `lig_ccd`, `lig_id`, `lig_source` | (L,) | site index, CCD, `CCD|chain:resnum` in source, source `pdb_chain` |
| `lig_rmsd`, `lig_npairs`, `lig_min_dist`, `lig_n_atoms_orig` | (L,) | local Calpha RMSD, #pairs, min distance to protein (pre-trim), atoms before trimming |
| `lig_is_ion`, `lig_is_metal` | (L,) bool | ion / metal flags |
| `site_residues` | (S,) str | `;`-joined representative residue keys (union of fingerprints) |
| `site_is_ion`, `site_has_ion`, `site_n_ligands`, `site_n_sources`, `site_center` | (S,...) | all-ion site, any ion, counts, centroid of kept atoms |
| `mask_kinds`, `mask_residues` | (3,) str | peptide/rna/dna binding residues (not labels) |
| `asm_path`, `asm_chains`, `asm_lig_*`, `asm_site_*` | | assembly-1 input and its ligands/sites (frame of the assembly file; residues `chain:num`) |

Model input (DESIGN §2.1) = `load_input(entry)`: the representative chain, protein heavy atoms only
(or the whole assembly with `assembly=True`). Coordinates are in the deposited mmCIF frame.


## 8. Results (build of 2026-10-26, test union sha256 `27c3d793…6a45`)

Sources: `data/manifests/test_union_counts.json`, `data/processed/train/{candidates_counts→work/,split_counts,pool_counts,asmcheck_counts,finalcheck_counts,verify_report}.json`.

**Test union** — 21,227 rows / 13,337 unique sequences; 20,483 PDB IDs; 7,641 UniProts.
Rows per source: cryptobench 6,622 · holo4k 4,675 · af_set 3,473 · ligysis 2,775 · pocketwatch_test 1,566 ·
dimer 921 (400 entries, all chains incl. partners; manifest sha256 `0196c7e3…`) · ligysis_extra 673 ·
coach420 461 · pocketminer 61. 10 entries absent from the mirror (obsolete) → 11 rows from the
dataset file's observed sequence. Versions used: dimer = Track D `manifest_chains.tsv` (final),
AF = Track C `af_sequences.fasta` (final, 3,473 sequences).

**Candidates** — BioLiP2 989,058 rows / 511,644 chains → 268,470 chains (120,598 entries, 120,161 unique
observed sequences, 29,146 UniProts) with 548,473 ligand rows (171,612 ion, 170,062 metal);
124,461 polymer-ligand rows (rna 68k, peptide 29k, dna 26k) excluded as labels.

**Leakage removal** — 207,690 / 268,470 chains removed (**77.4%**): sequence rule 205,783
(observed seq 201,674; +4,604 only via canonical entity sequence), PDB-ID 31,770, UniProt 120,988
(overlapping reasons). 65.8% of candidate UniProts removed. Of 10,370 candidate 30% clusters,
**42.7% fully removed** (< 50% → no fallback needed; 52.7% touched). The AF set and HOLO4K/CryptoBench/LIGYSIS
each account for ~50k of the 90k leaking unique sequences (union, heavily overlapping).

**Split** — 5,972 surviving 30% clusters → 5,846 UniProt-merged groups; 10% → dev. Dev-vs-train rule
removed 1,242 of 5,432 dev chains (groups 585 → 517). Chain lists: train 55,348 / dev 4,190
(`train_chains.txt`, `dev_chains.txt`).

**Entries** (`--max-per-cluster 3`; final pass dropped 1 train entry (test hit with the
structure-derived sequence) and 6 dev entries (dev-vs-train incl. train source chains)):

| | train | dev |
|---|---|---|
| entries (representatives) | **6,644** | **596** |
| 30% clusters represented / `rank_in_cluster==0` entries | 5,303 | 513 (512 rank-0) |
| sites | 14,073 | 1,261 |
| sites/protein mean · median · p90 · max | 2.12 · 1 · 4 · 51 | 2.12 · 1 · 4 · 23 |
| entries without any site (all sites clash-dropped) | 13 | 0 |
| ion-only site fraction | 37.7% | 35.0% |
| ligands (pooled instances) · ion fraction · metal fraction | 36,580 · 41.8% · 41.2% | 3,037 · 33.5% · 32.8% |
| entries with >= 1 pooled source entry | 45% | 46% |
| assembly-1 file available | 100% | 100% |
| assembly inputs stored (seeded 30% draw) | 1,744 (26.2%) | 143 (24.0%) |
| assembly sites · interface sites | 6,731 · 648 | 444 · 31 |
| method X-ray / EM | 5,940 / 704 | 534 / 62 |

Pooling statistics (all 7,247 built entries): 40,683 ligand rows considered; RMSD > 2 Å rejected 453;
< 6 anchor pairs 121; source alignment identity < 0.9: 197 source chains; ligand instance not found
in mmCIF 211; > 4.5 Å from the representative 7. **Clash rule**: 12,051 of 679,349 non-ion+ion ligand
heavy atoms (1.8%) dropped (< 2.2 Å, non-ions); **61 sites dropped** for losing > 50% of atoms (170
ligands). Assemblies: 2,117 built, 229 dropped because a partner chain hits the test union (213
entries) or the other split (101), 59 skipped as too large.

Size: entries 71 MB (train) + 6.4 MB (dev); work files ~0.9 GB. Committable lists:
`data/manifests/trainset_{train,dev}_entries.tsv`.

## 9. Verification (`scripts/data/verify_trainset.py` → `verify_report.json`: **PASS**)

* Test union sha256 identical in every entry.
* Re-extracted observed sequences (pocketdetect loader) of 7,240 representatives, the pooled
  source chains (17,496 train / 1,414 dev) and all stored assembly partner chains vs the test union: **0 rule hits**;
  canonical entity sequences of representatives: **0 hits**; PDB-ID overlap **0**; UniProt overlap
  (mmCIF `_struct_ref` + SIFTS, reps and sources) **0**; dev reps vs train reps+sources: **0 hits**;
  UniProts shared train/dev: 0.
* Spot checks (seeded sample, 497 entries incl. 197 with assemblies; 2,524 ligands): 100% of pooled
  ligands within 4.5 Å of the representative chain; 100% of non-ion ligands with min distance
  >= 2.2 Å (min-distance quantiles 1% 1.78 Å [ions], 10% 2.02, 50% 2.48, 90% 2.91 Å); 961 own ligands
  coincide exactly (max deviation 0.0 Å) with the deposited HETATM atoms; 0 of 6,202 site residues
  missing from the chain; assembly ligands 100% near / non-clashing.
* Earlier verification runs (before the canonical-sequence search and the final pass were added)
  found 2 representatives with borderline observed-sequence hits (fident 0.30–0.31, ~160 aligned
  residues), 711 representatives hitting via their full canonical entity sequence (unobserved
  regions / fusion partners; e.g. a GFP fusion → UniProt P42212), and assembly partner hits —
  these led to the canonical search (step 3), mmCIF UniProt refs, the partner check (5b) and the
  final pass (5c). Recorded here so the reduction from ~7.9k to 7.2k entries is traceable.

## 10. Fold-novelty strata (`build_trainset_foldseek_strata.py`)

Foldseek (exhaustive, E < 1e-3) of every structure-backed test chain vs all train/dev representatives;
score = max(qTM, tTM). Outputs `data/manifests/test_fold_novelty.tsv` (strata < 0.5 / 0.5–0.7 / > 0.7)
and `data/processed/train/fold_to_test.tsv` (per entry max TM to any test chain; `fold_disjoint` =
< 0.5 for the fold-disjoint ablation). Run of 2026-10-26 (`foldseek_counts.json`): 21,211 test
structures (40 peptide-size chains < 10 residues not written → counted as `<0.5`), 7,240 train/dev
representatives. Test chains per stratum (max TM to train/dev):

| source | < 0.5 | 0.5–0.7 | > 0.7 |
|---|---|---|---|
| ligysis (2,775) | 1,839 | 394 | 542 |
| coach420 | 186 | 88 | 185 |
| holo4k | 1,541 | 907 | 2,217 |
| cryptobench | 3,204 | 1,173 | 2,242 |
| pocketwatch_test | 272 | 446 | 848 |
| af_set | 1,172 | 892 | 1,409 |
| dimer | 617 | 119 | 185 |

Fold-disjoint ablation: 4,016 / 6,644 train entries (60.4%) have max TM < 0.5 to every test chain
(> 20% → the TM ≥ 0.6 fallback is not needed; 34.3% of train entries have TM ≥ 0.6 to some test chain).
Sequence-level exclusion therefore still leaves substantial fold-level similarity — the strata above
are the place to see it.

## 11. Silent-assumption list

1. BioLiP2's relevance curation is trusted as is (it keeps some GOL/SO4/CL/NA where judged biological).
2. "X-ray" = wwPDB `diffraction` (also neutron/fibre); EM from `pdb_entry_type.txt`; resolution from the BioLiP column.
3. Chain length >= 50 and training-side observed sequence = BioLiP receptor sequence (observed residues).
4. Ion = exactly one heavy atom (CCD formula, or actual instance); metal = that atom is a metal; halides/NH4+ are ions, not metals.
5. Polymer ligands (peptide/RNA/DNA) are not labels; their residues are only stored as `mask_residues`. Their sites will look like negatives unless the consumer masks them.
6. Test-side sequences are canonical entity sequences (incl. tags, unobserved parts) — conservative; training side is checked with both observed and canonical sequences.
7. LIGYSIS: all 3,448 listed chains are protected, not only the 2,775 evaluated; PocketMiner val is protected too; CryptoBench all folds; COACH420/HOLO4K all protein entities.
8. MMseqs2 identity = `fident` (identical/alignment columns), one alignment per pair (no `--alt-ali`), -s 7.5, max 5,000 targets per query; hits outside MMseqs2's prefilter are invisible (verification re-search is the guard).
9. UniProt = BioLiP ∪ SIFTS (2026-10-18) ∪ mmCIF `_struct_ref` UNP; chimeric chains carry several accessions and are removed if any is a test UniProt.
10. Clustering is on unique observed sequences; UniProts spanning clusters merge them into one split group; a UniProt is assigned to its modal cluster for selection (so 5,303 of 5,380 train clusters are represented).
11. Dev-vs-train hits are resolved by removing the dev chain (never the train chain).
12. Representative "completeness" = observed length relative to the longest observed chain of the UniProt (not UniProt length).
13. Pooling uses one chain per other entry (most ligands), other chains of the representative's own entry are not pooled; up to 20 entries chosen greedily for new CCD codes. 55% of entries have no other usable entry (single structure).
14. Chains without UniProt are grouped per 30% cluster but only pooled when alignment identity >= 0.9.
15. Fingerprints use BioLiP's binding residues of the source chain mapped by alignment (LIGYSIS uses its own contact definition on UniProt numbering); site residues = union of member fingerprints.
16. Clash/near filters are computed against the representative chain alone (the deployed single-chain input), not the full entry; ions are never clash-trimmed.
17. First altloc only; ligand instance matched by (author chain, CCD, author residue number; insertion code ignored).
18. Assembly 1 of any type (author/software) is used for training inputs; > 60 chains or > 150k atoms skipped; partner chains must pass the leakage rule.
19. Coordinates stay in the deposited mmCIF (or assembly file) frame; consumers must load the same file (`structure_path` / `asm_path`).
20. The 50%-identity diversity clusters (`clu50`) only steer which extra UniProts are added per 30% cluster; `rank_in_cluster == 0` is the strict one-per-cluster set.
21. Sequences of obsolete test entries (10) come from the dataset files' observed residues.


## 12. Apo counterparts of train/dev entries (Track A2)

Purpose: apo-robust training (pocket labels on ligand-free conformations). Runner:
`scripts/data/build_apo_train_all.sh` (search → pool → leakcheck → verify; detached with nohup).
Scripts `scripts/data/build_apo_train_{search,pool,leakcheck}.py`, `scripts/data/verify_apo_train.py`;
reader `pocketdetect.datasets.apotrain` (`read_apo_manifest`, `load_apo_entry` = `trainset.load_entry`,
`load_input`). Output `data/processed/apo_train/{train,dev}/{manifest.csv,entries/*.npz}` (104 MB incl. work).
Parents = the audit-filtered manifests (train 6,475, dev 579 entries).

**Method.**
1. Candidates: MMseqs2 search of each parent's observed representative sequence against all
   1,069,687 protein chains of the mirror (`pdb_seqres.txt`, 176,858 unique sequences; DBs created
   explicitly as protein — easy-search mis-typed this set as nucleotide), fident ≥ 0.95, query
   coverage ≥ 0.90; X-ray/EM, 0 < resolution ≤ 3.0 Å (`resolu.idx`); excluded: any test PDB ID
   (builder union + audit idmaps), the parent's own entry, the parent's pooled holo source entries.
   → 20,513 chains for 1,668 train / 161 dev parents. (Most parents are single-structure proteins
   after leakage removal, hence only ~26% have any candidate.)
2. Per parent, up to 15 candidate entries (best resolution first; one chain per entry). **Apo test**:
   no BioLiP2-listed ligand of that entry (any receptor chain; ions and peptide/RNA/DNA ligand
   chains included) with a heavy atom ≤ 6 Å from any heavy atom of the parent's site residues mapped
   onto the candidate. Entries released after the BioLiP2 snapshot (≈1–2%) use a stricter rule: any
   non-water hetero group outside a fixed additive list, and any nucleic acid, counts. Non-relevant
   hetero groups / protein partners near the site and BioLiP ligands elsewhere on the chain are
   allowed and recorded (`n_biolip_ligands_on_chain` > 0 for 4.7% train inputs).
3. Transfer per parent site: local Cα superposition (parent residues within 10 Å of the site's pooled
   ligand atoms, ≥ 6 aligned pairs, ≥ 70% of site residues present), kept if local RMSD ≤ 3.0 Å.
   Recorded: `site_local_rmsd`, `site_ca_rmsd`, `site_pocket_rmsd` (heavy-atom RMSD of matched
   site-residue atoms), `site_cryptic_like` = pocket RMSD ≥ 2.0 Å, `site_clash_frac`. Ligand atoms are not
   clash-trimmed (`lig_atom_min_dist` stored); ligands > 4.5 Å from the apo chain are dropped (14).
   Untransferred parent sites are stored as `mask_residues['untransferred_site']`.
4. ≤ 2 apo chains per parent, different entries: largest max pocket RMSD first, then the one whose
   per-site pocket-RMSD vector differs most (L1).
5. Leakage check of every selected chain (observed AND SEQRES sequence): §9 rule vs builder test union
   and the audit observed-sequence test set (`results/audit/test_union.fasta`), and vs the other
   split's representatives + apo inputs; test PDB IDs re-checked. 135 of 1,951 inputs dropped
   (`leak_dropped.tsv`: 116 test union, 105 audit set (overlapping), 20 other split) — mostly
   longer SEQRES constructs.

Candidate outcomes (5,365 chains examined): holo at site 4,404 · accepted 2,892 · no site transferable
228 · alignment identity < 0.9 151 · site residues unresolved 126.

**Results** (`pool_counts.json`, `leakcheck_counts.json`, `verify_report.json`):

| | train | dev |
|---|---|---|
| apo inputs | **1,656** | **160** |
| parents with ≥ 1 apo chain (of all parents) | 1,223 (18.9%) | 114 (19.7%) |
| parents with 2 apo chains | 433 | 46 |
| transferred sites | 3,202 | 325 |
| cryptic-like sites (pocket RMSD ≥ 2 Å) | 297 (**9.3%**) | 24 (7.4%) |
| apo inputs with ≥ 1 cryptic-like site | 263 | 22 |
| pocket RMSD median / p90 / p99 (Å) | 0.73 / 1.92 / 3.82 | 0.73 / 1.82 / 4.31 |

**Verification: PASS** — parents in the same split, no apo chain in both splits, none equal to its parent
chain; structure-derived sequences: 0 rule hits vs builder test union, vs audit test set, and vs the
other split's representatives; no PDB overlap with any test set, CryptoBench (apo+holo) or PocketMiner
entries; seeded 300-input sample: 0 BioLiP-relevant ligands within 6 Å of site residues (296
BioLiP-covered), 100% of transferred ligands within 4.5 Å of the apo chain (min-distance median 2.3 Å,
5% < 1.0 Å = closed/cryptic pockets).

**Assumptions / caveats.** Apo = "no BioLiP-relevant ligand near the site" (crystallisation additives,
non-relevant ions and protein partners may be present); the cryptic-like flag uses heavy-atom pocket RMSD
(side chains included), so it also captures side-chain rotamer changes; the chain with the largest pocket
change is preferred (selection bias toward conformational diversity, by design); apo inputs inherit the
parent's UniProt/cluster, so cluster-level weighting should count parent + apo together.
