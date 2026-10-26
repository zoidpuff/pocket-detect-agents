# DATA — datasets, sources, versions, checksums, licenses

All raw data lives in `data/raw/<dataset>/` (git-ignored). Each entry gives the source URL, version or date retrieved, checksum (sha256 of the downloaded archive, or of a manifest), license, and notes on restrictions. Never commit restricted data.

**Reproduce:** every download is a script in `scripts/data/`. Scripts are idempotent: they skip files already present and re-verify checksums.
- `download_*.sh` fetch the datasets. `install_tools.sh` installs MMseqs2 and Foldseek.
- `collect_manifests.sh` copies each per-dataset `MANIFEST.tsv` (file, sha256, bytes, retrieved_utc, url) into the tracked `data/manifests/`.
- `verify_datasets.py` runs the sanity checks: `.venv/bin/python -I scripts/data/verify_datasets.py [dataset ...]`. It writes `data/raw/VERIFY_REPORT.json`. The last full run on 2026-10-26 reported **ALL CHECKS PASSED**.
- Third-party pickles are loaded only through `scripts/data/safe_pickle.py`, a restricted unpickler that allows numpy, pandas and builtins only. Never `pickle.load` them directly.
- Pinned shas: GitHub `codeload` tarball shas are pinned in the scripts. GitHub can regenerate archives, so the **commit SHA is authoritative**. If a pinned tarball sha ever mismatches, re-check that the commit is unchanged before trusting the new file.
- All retrievals were on **2026-10-26 (UTC)** unless noted otherwise.

Total footprint of `data/raw` is about 48 GB, plus 73 MB in `tools/`. scPDB accounts for 21 GB of this (4.1 GB archive + 17 GB extracted).

## Summary table

| Dataset | Source | Version / retrieved | Checksum (sha256) | License | Local path | #entries |
|---|---|---|---|---|---|---|
| PDB mirror | /cluster/work/beltrao/shared/rsyncPDB (wwPDB rsync) | **snapshot ≈ 2026-10-16**: newest mmCIF mtime 2026-10-16 14:27 UTC, `derived_data/index/*.idx` dated Oct 16 | n/a (read-only mirror) | CC0 (wwPDB) | read-only | 261,139 mmCIF entries (divided). Covers 2024+ releases: 8,035/8,036 PocketWatch test PDB IDs present; missing: 9cis |
| AFDB v6 | /cluster/work/beltrao/shared/alphafold/v6 | v6 proteome + SwissProt tars | n/a (read-only mirror) | CC-BY 4.0 | read-only | 157 GB tars |
| P2Rank datasets (COACH420, HOLO4K, CHEN11, JOINED, FPTRAIN, AH4H, + mlig, + 3rd-party predictions) | github.com/rdk/p2rank-datasets | commit `0236ecb38cbb60b89849a1ea36fbd9ee93f3e906` (master, 2024-11-08) | `b608bc941d12087f60c94095e822edac53a0a8ce5fd6b7cf960254a383949ad2` (codeload tar.gz, 1.93 GB) | **no license file in repo**; structures derive from the PDB (CC0) | `data/raw/p2rank_datasets/current/` | see §1 |
| LIGYSIS + LBS-comparison results | Zenodo 14645504 (v2, 2025-01-21) and 13121414 (v1, 2024-07-29; same concept 13121413) | v2; 79 files, 4.14 GB | per file in `data/manifests/ligysis_lbs_comparison_results.tsv`; md5s of both Zenodo versions verified | CC-BY-4.0 | `data/raw/ligysis/lbs_comparison_results/` | 3,448 chains / 8,244 sites; 2,775 chains benchmarked |
| LBS-comparison code + input structures | Zenodo 14033286 (bartongroup/LBS-comparison v2.0.0, commit 820714c) | 2024-11-04 | `abeccd8233277b1869847060204530ed58d6485ec777dbcc3b2d1c0b056a076d` | Zenodo: CC-BY-4.0; repo LICENSE: MIT | `data/raw/ligysis/lbs_comparison_code/` | 4,435 ligand-stripped rep-chain PDBs |
| LIGYSIS pipeline | Zenodo 14178252 (JavierSanchez-Utges/LIGYSIS 1.0.0) | 2024-11-18 | `8fa7476e08a4589d35c1860eb7cdf377630c3ddf3f076cbc73edd5e63cf69c50` | CC-BY-4.0 | `data/raw/ligysis/ligysis_pipeline/` | code only |
| CryptoBench | OSF osf.io/pz4a9 (+ github skrhakv/CryptoBench `9a3432f4…`) | OSF node modified 2026-01-17 | cif-files.zip `8d15f897bfdfdf61c7d97a29f5f6ca2c5e03d73d8fb89be7da5bbc245cf56ae4`; dataset.json `79519369a1d32b63efd86b907013cacb5e02a68e4d711cdde20a23cdfe16ba7c`; all files verified against OSF sha256 (`data/manifests/cryptobench_osf.tsv`) | MIT (OSF + GitHub) | `data/raw/cryptobench/` | 1,107 apo / 3,899 holo / 5,493 pairs |
| PocketMiner (PM) test/val set | github Mickdub/gvp branch `pocket_pred`, commit `187062df3c94127e991669768009141a08fd5d8b` (2023-07-21), subtree `data/pm-dataset` only | — | per-file sha256 + git blob sha1 verified (`data/manifests/pocketminer.tsv`, 424 files) | MIT | `data/raw/pocketminer/data/pm-dataset/` | 35 test + 26 val apo chains |
| BU48 (apo/holo) | inside P2Rank datasets `joined/bu48` + metaPocket2 originals | as P2Rank | as P2Rank | as P2Rank | `…/current/joined/bu48/` | 48 apo/holo pairs (96 PDB files) |
| scPDB "2017" | http://bioinfo-pharma.u-strasbg.fr/scPDB/ressources/2016/scPDB.tar.gz (HTTP only) | Last-Modified 2018-01-09 | `a7fd8e142db6868cc65937f88124750365b976d36a8ec264e523289c9a75c079` (4.08 GB) | **none stated**; public download, no registration | `data/raw/scpdb/extracted/scPDB/` | 17,594 entries (site says 16,034; see §6) |
| AF2BIND | github sokrypton/af2bind commit `6867239a99d40cb67f3d40c837550fb91b575337` (2026-10-14) | — | `92d5acdc30fe76ee8661972da6f19dc24c256cd52e84b3255dce51218c4a58e3` | MIT | `data/raw/af2bind/af2bind-<commit>/` | 1,897 labelled chains |
| HOTPocket | Zenodo 17653326 (data v2, 2025-11-19; v1 = 16891050 not fetched) + github Helix-Research-Lab/HOTPocket `1ca16ce0…` | v2 | `data/manifests/hotpocket_zenodo_v2.tsv` (md5 verified vs Zenodo); code `0575c1639cee0af6b487727e0ba9ccb41d4015505f429bfa34ed11226ca81c07` | data CC-BY-4.0; code MIT | `data/raw/hotpocket/` | 2.48 M pockets; 11,434 known pockets |
| PocketWatch | huggingface.co/datasets/genesisml/PocketWatch, revision `db7bd0611a14cb98e3b1d24b343565418b7193ef` (2026-10-14) | — | upstream `CHECKSUMS.txt` md5 verified except README.md (see §9); sha256 in `data/manifests/pocketwatch_<rev>.tsv` | MIT (HF card) | `data/raw/pocketwatch/current/` | 14,135 proteins / 18,797 sites |
| BioLiP2 annotations | https://zhanggroup.org/BioLiP/ (`download/BioLiP*.txt.gz`, `data/*.fasta.gz`, `data/ligand.tsv.gz`) | snapshot 2026-10-26. Server Last-Modified: 2025-11-12 for BioLiP.txt.gz and BioLiP_nr.txt.gz; fasta and ligand files 2025-11-12 01:28 UTC | BioLiP.txt.gz `c92229bbc8c55c3bd84a9813c3e278ba62f4cfa44e6315cc98d9bf63ed64b6ec`; BioLiP_nr.txt.gz `594830b62f0dc368b6642958edb1a16ac66b702176c81868cedd646c1e4588f4` | "freely available" (no formal license stated); cite Zhang et al. NAR 2024 | `data/raw/biolip2/current/` (→ `2026-10-26/`) | 989,058 sites / 153,608 PDB entries |
| MMseqs2 | github soedinglab/MMseqs2 release 18-8cc5c (linux-avx2) | 2025-07-27 | `bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce` (= GitHub asset digest) | MIT | `tools/bin/mmseqs` | — |
| Foldseek | github steineggerlab/foldseek release 10-941cd33 (linux-avx2) | 2025-01-19 | `e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7` | GPL-3.0 | `tools/bin/foldseek` | — |

---

## 1. P2Rank datasets (COACH420, HOLO4K, CHEN11, JOINED, …): `download_p2rank_datasets.sh`

**Files.**
- `*.ds` files list the dataset items: one PDB path per line, plus an optional `ligand_codes` column in `(mlig)` variants. The data are in sub-directories.
- Per the upstream README, **the `.ds` file defines the dataset, not the directory**. For example, `holo4k/` has 4,543 PDBs but `holo4k.ds` lists 4,009.
- The directories also contain third-party predictions: fpocket 1.0, SiteHound, MetaPocket2, DeepSite and P2Rank 2.0. The `-fpocket.ds`, `-deepsite.ds`, `-mpk2.ds`, `-sitehound.ds`, `-lise.ds` and `-p2rank.ds` files list them. The `*-XXsubset.ds` files list the proteins on which method XX produced output.
- `_lists/*.list` are plain ID lists.

**How ligands/sites are defined.** There is no explicit site file; ligands are read from HETATM records at evaluation time.
- Standard variant: P2Rank's "relevant ligand" rules from `Params.groovy`:
  - `ignore_het_groups = HOH, DOD, WAT, NAG, MAN, UNK, GLC, ABA, MPD, GOL, SO4, PO4`
  - `min_ligand_atoms = 5`
  - `ligand_protein_contact_distance = 4 Å`; ligand centre within `ligc_prot_dist = 5.5 Å` of the protein
  - HETATM groups closer than 1.7 Å are merged
- `(mlig)` variant: the relevant ligand codes are listed explicitly. They come from Binding MOAD 2013. Proteins unknown to MOAD, or with conflicting codes, were removed.
- `chen11.ds` uses `PARAM.LIGANDS_SEPARATED_BY_TER=true`.

**Counts (`.ds` entries).**

| Dataset | Entries | `(mlig)` entries |
|---|---|---|
| COACH420 | 420 | 300 |
| HOLO4K | 4,009 | 3,448 |
| CHEN11 | 251 | — |
| JOINED | 537 | 371 |
| FPTRAIN | 222 | — |
| AH4H (`ah4h.holoraw`) | 380 | 239 |

- JOINED = ASTEX + B210 + DT198 + BU48, all from metaPocket 2.0.
- File counts in the dirs: holo4k/ 183,747 files (prediction outputs included); coach420/ 18,900; chen11/ 5,580; joined/ 14,246.

**Sanity check** (all files parse with gemmi; per-dataset numbers in `VERIFY_REPORT.json`). My re-implementation of the P2Rank rules is approximate: it groups per residue and does not merge across residues. With it, the count of relevant ligands was:

| Dataset | Approx. relevant ligands | Proteins with no qualifying ligand |
|---|---|---|
| COACH420 | 544 | 21 |
| HOLO4K | 9,540 | 15 |
| CHEN11 | 439 (README: 476) | 14 |
| JOINED | 564 | 98 |
| FPTRAIN | 232 | 42 |

Most common HOLO4K ligands: NAD, FAD, HEM, PLP, ADP, FMN. These numbers are sanity checks only. **The authoritative ligand rules for evaluation are those in the eval harness.**

**Surprises / caveats.**
- **Modified residues are HETATM in these PDB files.** MSE alone has 1,220 HETATM groups in HOLO4K, and SEP, LLP and others also occur. A naive "HETATM = ligand" rule therefore inflates counts by about 15%. Exclude tabulated amino-acid and nucleotide residues, as BioJava does.
- **JOINED contains BU48 *apo* structures.** 33 of the 98 ligand-less JOINED entries are bu48 apo files; most of the rest are b210. P2Rank simply finds no ligand in them. Exclude them explicitly or treat them separately.
- **147 JOINED PDB files have junk in the charge columns (79–80)**, left by metaPocket2's fix-elements script. Strict parsers fail on them. The verifier retries with ATOM/HETATM lines truncated to 78 columns, which fixed all of them.
- Two HOLO4K entries (1gd1, 1hyh) have a non-standard SCALEn crystal frame. Use no crystal symmetry in neighbour searches; it should not be used anyway.
- `(mlig)` codes not found as HETATM in the file:
  - COACH420(mlig): 8 entries, including one literal `<CONFLICTS>` code (1goqA) that the README says was removed.
  - HOLO4K(mlig): 70 entries. Mostly peptide ligands whose MOAD "codes" are amino-acid names (LEU, GLU, …), plus a few real misses (e.g. 1ahi NAD).
  - JOINED(mlig): 80; AH4H(mlig): 9.
- `1xgf` was removed upstream (all UNK).
- AH4H is undocumented upstream. It contains only `holo-raw` gzipped PDBs, with no apo partners in the repo.
- License: none in the repo. Use for research with citation (Krivák & Hoksza 2018, J Cheminform 10:39). Do not redistribute.

## 2. LIGYSIS (Utgés & Barton 2024, J Cheminform 16:126): `download_ligysis.sh`

**What it is.** The human subset of LIGYSIS, used as the reference for the largest independent comparison of ligand binding site (LBS) predictors to date.
- For each human UniProt protein, biologically relevant ligands (by BioLiP criteria) are pooled across all **PISA biological assemblies** of all its PDBe entries. Ligands are clustered by protein-interaction fingerprint into *sites*, and the sites are defined in **UniProt residue numbering** (`up_aas`).
- Ions are included in this benchmark version.
- The predictors were run on **one ligand-stripped representative chain per protein**. These chains are transformed into a common frame (`*.trans.pdb`) and ship in the code release under `DATA/trans_rep_chains/`.

**Files** (`lbs_comparison_results/`, 79 pickles):
- `LIGYSIS_sites_DEF_TRANS.pkl`: 8,244 sites on 3,448 rep chains. Columns: ACC, SEG, SITE_NAME, rep_chain, `aas` (PDB numbering on the rep chain), `up_aas`, centre (transformed frame), RoG, SASA, VOL.
- `LIGYSIS_3448_chains.pkl`: the chain list.
- `LIGYSIS_human_sites_APRIL_2024.pkl`: 10,689 sites before the rep-chain mapping.
- **Per-method predictions**, all on the same chains, with centres, ranks, scores and residues:
  - `{P2Rank, P2Rank_CONS, fpocket, VNEGNN, IFSP, GRASP, PURESNET, DeepPocket, POCKETFINDER, LIGSITE, SURFNET}_pockets_DEF_TRANS*.pkl`
  - PRANK-rescored variants (`*_pranked_*`)
  - `*_no_sifts*`: chains lacking a SIFTS mapping
- `ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl`: 428,248 rows, LIGYSIS sites plus all 22 method variants.
- `LIGYSIS_ress_binary_labels.pkl`: per-residue 0/1 for LIGYSIS and 11 methods on 2,775 chains (831,572 residues). Present only in v2.
- `ALL_COMPARISONS_EXTENDED6.pkl` (350 MB) and `MASTER_POCKET_SHAPE_DICT_EXTENDED_TRANS.pkl` (3.3 GB): pairwise comparison and pocket-shape tables.
- Their re-parsed versions of other benchmarks: `COACH420_df`, `HOLO4K_df`, `CHEN11_df`, `JOINED_df`, `B210_df`, `BU48_df`, `DT198_df`, `ASTEX_df`, `SC6K_df`, `scPDB_*_df`, `bMOAD_filt_df`, `pdbbind_refined_df`.

**Counts.**
- 3,448 chains; 8,244 sites.
- The paper's evaluated set is **2,775 chains**, after dropping chains without UniProt mapping. That is 6,882 LIGYSIS sites in the combined table.
- Coverage of each method's predictions on those 2,775 chains:

| Method | Chains with predictions |
|---|---|
| fpocket | 2,759 |
| VN-EGNN | 2,764 |
| GrASP | 2,771 |
| P2Rank | 2,402 |
| P2Rank+Cons | 2,579 |
| PUResNet | 2,360 |
| DeepPocket-Seg | 2,349 |
| IF-SitePred | 2,075 |

- Methods that fail on a chain must be scored as misses, not dropped.

**Checks.**
- Every site chain has a structure file.
- 25 sampled chains: all site residues (`aas`) exist in the structure, and the chains contain zero HETATM (they really are ligand-stripped).
- The md5s of v1 (13121414; 78 files) are byte-identical to v2. **Zenodo 13121414, requested by the PI, is fully covered.**

**Caveats.**
- The data are pickles; load them via `safe_pickle.py`.
- The transformed frame differs from the PDB frame. Use `PDB_rot_matrices_ALL_CLUST.pkl` / `PDB_orig_centroids_ALL_CLUST.pkl` to map back.
- The 4,435 structure files are more than the 3,448 chains: the extras are from other segment and cluster reps.
- The **full** LIGYSIS (~30k proteins, multi-species) is not in these records. Only the human benchmark subset is here.

## 3. CryptoBench (Škrhák et al. 2025, Bioinformatics 41:btae745): `download_cryptobench.sh`

**What it is.** A cryptic-site benchmark built from AHoJ-DB apo–holo pairs.
- Each **apo** chain is paired with one or more holo structures, each with a ligand (CCD code, chain, auth index).
- Pocket residues are given for both apo and holo as `chain_authseq[icode]`, together with the pocket RMSD (pRMSD) and an `is_main_holo_structure` flag.
- `noncryptic-pockets.json` lists additional non-cryptic pockets for 665 apo entries.

**Counts.**
- 1,107 apo entries, 3,899 unique holo structures, 5,493 apo–holo pairs.
- 5,005 mmCIF files (`cif-files/cif-files/*.cif`).
- Folds: test 222; train-fold-0..3: 219 / 222 / 222 / 222 (sum 1,107).

**Checks (exhaustive).**
- Every apo and holo has its CIF.
- All apo pocket residues exist in the apo CIF, once insertion codes are respected (e.g. `B_149A`).
- Holo ligand found at the stated chain/index for all pairs except 2, both nucleotide "ligands": 1kdp C/B/6407 and 5v0a DA/B/102.

**Caveats.**
- The OSF `benchmark/` folder holds the authors' TF model. It was downloaded but is irrelevant to us.
- Several holo structures per apo: decide whether to evaluate against the union of holo pockets or only the main holo.

## 4. PocketMiner set (Meller et al. 2023, Nat Commun 14:1177): `download_pocketminer.sh`

**What it is.** Cryptic pockets defined by large apo→holo opening.
- Residue-level labels on **apo** structures: 1 = forms a cryptic pocket, 0 = negative (residues judged unlikely to form a pocket). Unlabelled residues are excluded.
- `supplementary-tables.xlsx` maps apo to holo and classifies each pocket.
- `apo-structures/` has 61 cleaned apo PDBs (missing loops modelled, H added). `all-structures/` has 357 files including holo and `*_ligand_h.pdb`.

**Counts.**
- test: 35 apo chains, 563 positive / 1,283 negative residues. This matches the paper's 563 / 1,283.
- val: 26 chains, 436 / 375.
- The paper describes the set as 38 apo–holo pairs with 39 cryptic pockets.

**Caveats.**
- Label dictionaries are object `.npy` files, i.e. pickles; they are read via the restricted unpickler.
- 8 of the 114 `*ligand*` files have no HETATM.
- The repo's MD trajectories (~300 MB) were deliberately not fetched.

## 5. Apo–holo sets

- **BU48** (Huang & Schroeder 2006, via metaPocket 2.0): 48 bound/unbound pairs in `joined/bu48/`. The original tarball with the apo↔holo ID mapping is at `joined/_metapocket2_datasets/[originals]/48-bound-unbound.tar.gz` (`48_pdb_ids.txt`). LIGYSIS also has `BU48_df.pkl`. PUResNet and others used BU48 as their apo/holo test.
- **CryptoBench** (§3) and **PocketMiner** (§4) are the modern apo–holo cryptic sets.
- **Not downloaded: AHoJ-DB** (Feidakis et al., JMB 2024; apoholo.cz/db). It is PDB-wide, with millions of apo–holo pairs and a REST API. If more apo–holo pairs are needed, sample it via the API. Our PDB mirror supplies the structures.

## 6. scPDB (Desaphy et al. 2015, NAR 43:D399): `download_scpdb.sh`

**What it is.**
- One directory per site (`<pdb>_<n>`) with `protein.mol2`, `ligand.mol2`/`.sdf`, `site.mol2` (residues around the ligand), `cavity6.mol2` and `cavityALL.mol2` (VolSite cavity points) and `IFP.txt`/`ints_M.mol2` (interaction fingerprint).
- The sites are druggable-ligand binding sites. Ligands are filtered by rules (MW, no ions/solvents, etc.) and the protein is prepared (protonated).

**Counts.**
- **17,594 entries, 16,612 unique PDB IDs.** The website advertises "release 2017: 16,034 entries, 4,782 proteins, 6,326 ligands". The archive (path `ressources/2016/`, Last-Modified 2018-01-09) therefore does not match the advertised count. Reconcile against LIGYSIS's `scPDB_full_df.pkl` / `scPDB_filt_df.pkl` if exact release membership matters.
- All entries contain the four core mol2 files.
- In a 50-entry sample, ligand atoms ranged 10–118.

**License / restrictions.**
- No registration gate; it is a plain public link.
- **No license is stated anywhere on the site.** Treat as academic use only and **do not redistribute or commit derived files**.
- HTTP only (no TLS). Integrity rests on the pinned sha256.

## 7. AF2BIND (Gazizov et al., Nat Methods 2026; bioRxiv 10.1101/2023.10.15.562410): `download_af2bind.sh`

**Files.**
- `training/new_labels_2k*.pkl`: 1,897 PDB chains. Per residue: `dist_all` (min heavy-atom distance to any ligand), `dist_sub` (confident ligands), `mask`, `bind_alt` (binding seen in homologs with TM-score > 0.8), `seq`. Per `train_script.ipynb`, binding = `dist_all < 5 Å`.
- `assignments_2k.pkl`: 10 split seeds. Each maps about 640–695 chains to 0/1/2 (train/val/test); there are about 65 test chains per seed.
- `tmscores_2k.txt.gz`: pairwise TM-scores, used for the splits.
- `attempt_*.zip`: model weights.
- `preds/`: 20,300 AF2BIND per-residue CSVs for human AFDB models (v4).

**Caveats.**
- It is a residue-level, single-chain benchmark.
- Labels pool *all* ligands (including those in homologous structures), not a curated relevant-ligand set.
- The test split is small and seed-dependent.

## 8. HOTPocket = "Human Omnibus of Targetable Pockets" (Carpenter & Altman, J Cheminform 2025, doi:10.1186/s13321-025-01125-x): `download_hotpocket.sh`

**What it is.** HOTPocket is **not a classic benchmark**. It is a proteome-wide *catalogue* of about 2.4 M candidate pockets on the canonical human proteome (PDB + AF2), from 7 finders: AutoSite, CASTp, CavitySpace, fpocket, LIGSITEcs, P2Rank and PocketMiner. The pockets are filtered by hotpocketNN, an MLP on ESM2 embeddings (± method features).

**Files (Zenodo v2).**
- `all_pocket_preds/`: per-method candidates (e.g. P2Rank 750,966; fpocket 2.90 M; PocketMiner 7.19 M rows) and **`known_pockets.csv`: 11,434 BioLiP2-derived known pockets** (4,437 UniProt, 5,353 PDB). Residues are given as chain-residue lists.
- `hotpocket_csvs/`: final HOTPocket sets, `embs` 2,482,870 rows and `comb` 3,475,632, plus clustered versions.
- `BioLiP.txt`: the redundant BioLiP version the authors used.
- `uniprot/`: UP000005640 human proteome.
- `model_inputs.tar.gz` (2.2 GB of feature arrays for train/val/test) and `model_checkpoints`: archived only, not extracted.

**Usefulness.**
- `known_pockets.csv` gives human-proteome AF2 and PDB ground truth.
- The per-method predictions give free baseline outputs at proteome scale.
- The paper's own evaluation used a BioLiP subset, Astex and PoseBusters. Those sets are not packaged separately.
- v1 (Zenodo 16891050) had a PDB-ID-as-float bug (e.g. 8e05 → 800000). v2 fixes it; only v2 was fetched.

## 9. PocketWatch (zPocket paper, arXiv 2610.17696): `download_pocketwatch.sh`

**What it is.** A temporally held-out ligand binding site benchmark on **predicted structures**.
- Pockets come from Q-BioLiP/PDB contacts at 5.0 Å and are clustered into sites across all entries.
- They are projected onto one ESMFold2 (sample 0) fold of the canonical UniProt sequence.
- Split by each site's first release date:

| Split | Cutoff (first release) | Proteins | Sites | Scored sites |
|---|---|---|---|---|
| train | — | 11,016 | 14,331 | 11,850 |
| validation | ≥ 2021-09-30 | 1,553 | 2,165 | 1,505 |
| test | ≥ **2024-01-01** | 1,566 | 2,301 | 1,573 |

- Stratifications of the test split (`strata/`): sequence identity to the reference (MMseqs `test_vs_reference.m8`), structural (Foldseek members), and crypticity / fold agreement.

**Files fetched.**
- `data/*.parquet`: labels+mask per protein, sites, sequences, `site_policy`, site centroids.
- `projection/`: entry-frame ligand centroids and contact CA coordinates, to re-derive centroids on our own structures.
- `strata/`.
- `structures/`: 3,119 val+test ESMFold2 mmCIF.
- `structures_train/`: 11,016, unzipped.
- `verify_bundle.py`.
- Skipped: `msas_test.tar` (2.7 GB of MSAs).

**Checks.**
- Protein counts match `labels_manifest.json`.
- Every label vector matches its sequence length.
- All **scored** test sites were first observed ≥ 2024-01-03.
- 8,035 of the 8,036 PDB IDs behind test sites are in our mirror; 9cis is missing.

**Caveats.**
- **Read `site_policy.parquet`.** Masked sites are interface-only (3,318), pre-cutoff (495), no expressible residue (49) or subsumed (7). They are real pockets and must be neither positives nor negatives.
- `site_centroids` are in the **ESMFold2 fold frame**, while site clustering used Protenix folds.
- Upstream `CHECKSUMS.txt` md5 fails only for `README.md`, which was edited after the checksums were generated. All other files verify.
- The temporal split allows homologs across the cutoff; use the sequence-identity strata.

## 10. BioLiP2 (Zhang et al. 2024, NAR 52:D404): `download_biolip2.sh`

**Files.**
- `BioLiP.txt.gz` (redundant) and `BioLiP_nr.txt.gz` (receptors clustered at 90% identity). Each row is one ligand–receptor-chain site: PDB, chain, resolution, CCD ligand, ligand chain/serial, binding residues in PDB and renumbered numbering, EC, GO, affinities, UniProt and sequence. There are 21 tab-separated columns; see `readme.txt`.
- `protein.fasta.gz` (512,101 chains), `protein_nr.fasta.gz` (42,746), and `ligand.tsv.gz` (45,660 ligands; `readme_ligand.txt`).
- Structures were **not** mirrored; use the local PDB mirror.

**Counts.**
- Redundant: 989,058 sites, 153,608 PDB entries, 42,616 ligand codes.
- nr: 86,458 sites, 34,557 PDB entries, 6,020 ligand codes.

**Checks.**
- All rows have 21 columns.
- In 30 random nr rows, every ligand was found in the local PDB mirror mmCIF.

**Caveats.**
- BioLiP is updated weekly and files are unversioned. The snapshot is identified by retrieval date plus server Last-Modified: 2025-11-12, which suggests updates may have paused. It is stored as a dated directory.
- Since January 2025, PDBbind-CN affinities have been removed from BioLiP for licensing reasons.
- License: "freely available", with no formal license; cite the paper.
- Use BioLiP_nr plus MMseqs2 and Foldseek to filter training labels against all test sets.

## 11. Tools: `install_tools.sh`

- `tools/bin/mmseqs` (MMseqs2 18-8cc5c) and `tools/bin/foldseek` (Foldseek 10-941cd33), static AVX2 builds. The host CPU supports AVX2.
- `tools/bin/` and `tools/downloads/` are git-ignored.
- Functional test in `verify_datasets.py tools`: `easy-search` of 3 PocketMiner apo chains against themselves finds all self-hits (MMseqs2 3/3 at ≥ 99% identity; Foldseek 3/3).
- The GPU builds were not installed.

## Not obtained / open items

- **AHoJ-DB full dump**: not downloaded. It is very large and has a REST API (§5).
- **PocketWatch MSAs** (`msas_test.tar`, 2.7 GB): skipped on purpose.
- **HOTPocket v1** (Zenodo 16891050): superseded by v2, so it was not fetched.
- The HOTPocket evaluation sets (Astex, PoseBusters) are not packaged as data. PoseBusters (Zenodo) could be added if needed.
