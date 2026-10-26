# AF-mode test set (Track C): LIGYSIS sites on AFDB v6 human models

Owner: Track C data agent. Spec: DESIGN.md §2.6 (AF mode) and §1 objective O5. Everything below is
produced by scripts; nothing is hand-edited.

```
.venv/bin/python scripts/data/build_af_extract.py   # one pass over the human tar (+ SwissProt tar fallback), ~1 min
.venv/bin/python scripts/data/build_af_set.py       # site mapping, ligand transfer, annotations, ~1 min on 56 cores
.venv/bin/python scripts/data/verify_af_set.py      # 25 checks + controls -> verify_report.json, ~1 min
```

The inputs `data/processed/af/external/{uniprot_human_reviewed.tsv.gz,pdb_chain_uniprot.tsv.gz}`
were downloaded once with the curl commands listed in §8. Their checksums are recorded in
`build_info.json`. The build is deterministic: two runs give byte-identical CSVs.

## 1. Counts (build of 2026-10-26)

| | proteins | sites |
|---|---|---|
| LIGYSIS human UniProts / sites (`LIGYSIS_human_sites_APRIL_2024.pkl`) | 3,518 | 10,689 |
| excluded: AF model fragmented (> 2,700 aa, F1..Fn) | **42** (34 in benchmark) | 204 sites on excluded proteins |
| excluded: no AF model in the human or SwissProt tar (P36969, Q16881, Q9NZV6) | 3 (1 in benchmark) | |
| excluded: no valid site after checks | 8 (3 in benchmark) | |
| **included (site set "all")** | **3,465** | **10,341** |
| LBS-comparison benchmark (2,775 chains, 6,882 sites, 2,565 UniProts) | 2,527 included | **6,731 included**, 41 flagged, 110 on excluded proteins |
| benchmark, no ions (`ion_prop != 1`, LIGYSIS_NI convention) | | 4,216 |

Flagged sites (`include=False`) by reason:

- `restype_mismatch`: 76 sites, residue-type identity below 0.75.
- `duplicate_of:<site>`: 68 sites. The same residue set appears in two LIGYSIS segments of one protein; the first is kept.
- `pos_out_of_model`: 1 site.

Informational flags that do not exclude a site:

- `restype_partial`: 294 sites, identity between 0.75 and 1.
- `restype_uniprot_only`: 49 sites. There is no PDB reference, so only the UniProt check applies.

The full list of the 42 fragmented accessions is in `verify_report.json` (`fragmented_accs`).

pLDDT bands of included benchmark sites (mean pLDDT of the site residues):

| band | sites |
|---|---|
| ≥ 90 | 4,997 |
| 70–90 | 1,639 |
| 50–70 | 81 |
| < 50 | 14 |

Most LIGYSIS sites are confidently modelled, so the low-confidence strata are tiny and must be merged (DESIGN §3: strata with fewer than 100 sites are merged).

Other figures for included benchmark sites:

- 19.6% lie within 8 Å of a prunable low-pLDDT segment. By band: ≥ 90: 15%; 70–90: 33%; 50–70: 46%; < 50: 86%.
- Proteins: median length 469 aa (maximum 2,671). Median fraction of residues with pLDDT < 50 is 0.11 (90th percentile 0.42).
- PDB entries per UniProt (SIFTS): median 7, IQR 3–19.

## 2. Pipeline and decisions

**Models (step 1).**

- The script streams `UP000005640_9606_HUMAN_v6.tar` once (47,172 members, 12 s) and keeps `AF-<ACC>-F1-model_v6.cif.gz` for all 3,518 LIGYSIS accessions.
- A protein is *fragmented* if any member has F > 1. Fragmented proteins are not extracted. They are counted and listed.
- The 8 accessions absent from the proteome tar are searched once in `swissprot_cif_v6.tar`, which finds 5.
- The `source` column in `extract_index.csv` records which tar each model came from. All models are AFDB v6.

**Sequence and numbering checks.**

- AF chain A residue number equals the UniProt position for all 3,465 models.
- The AF sequence equals the UniProt canonical sequence (release 2026_03, reviewed human) for 3,461 of 3,465. The four exceptions are P08195, P24928, Q14669 and Q6IQ22, where UniProt changed the sequence after AFDB v6.
- Per-site residue-type check against an independent reference, in this order of preference:
  1. The LIGYSIS representative PDB chain: residue types at `aas`, paired with `up_aas`. Used for 8,089 sites.
  2. Otherwise, the SIFTS-mapped PDB chains that the site's ligands were transferred from (best identity). Used for 2,344 sites.
  3. Otherwise, the UniProt sequence (52 sites).
- A site is kept if residue-type identity is ≥ 0.75. This tolerates engineered point mutations in PDB constructs, while numbering shifts give identities of about 0–0.3.
- 96.5% of sites match exactly and 99.3% have identity ≥ 0.75.
- Example: for P08195, LIGYSIS numbering agrees with AF but not with the current UniProt sequence. Comparing against the PDB reference, rather than the current UniProt sequence, keeps these sites.

**Site centres on the AF model.** All are stored; the loader chooses one.

- `center_ca`: mean Cα of the site residues. **This is LIGYSIS's own definition.** In `ANALYSIS/7.0_CONCAT_data_LIGYSIS.ipynb`, `centre` is `np.mean` of the site residues' CA on the representative chain; it is residue-based, not ligand-based. We recomputed it on the 8,089 representative chains and reproduced the stored LIGYSIS centres to ≤ 0.0009 Å, which also validates the residue mapping. **Default for the LDCC12 primary metric.**
- `center_heavy`: centroid of all heavy atoms of the site residues.
- `center_sc`: centroid of side-chain heavy atoms (Gly uses CA). On AF models, side-chain placement is apo-like, so this is reported, not used by default.
- `center_lig`: centroid of the accepted transferred ligand atoms (see below). Available for 6,763 of 10,341 sites when only instances with RMSD ≤ 2 Å are used. Median distance to `center_ca` is 3.3 Å; 90th percentile is 7.3 Å.

**Ligand transfer (for DCA).**

- The site's ligand instances come from `LIGYSIS_human_chains_per_lig_MAY_2024.pkl`, with keys `pdb_RES_labelasym_authseq` taken from PDBe assemblies.
- Up to 5 successful instances per site are transferred, at most 20 attempts. Order: the representative entry first if it has a resolution, then by resolution (NMR last), then by key.
- Each instance is located in the wwPDB asymmetric-unit mmCIF from the local mirror:
  - All copies of the residue name are scored by the number of site residues they contact (≤ 4.5 Å) on chains SIFTS-mapped to the UniProt.
  - Ties prefer exact identifiers. Requirements: an exact ID match needs ≥ 1 contacted site residue; a match on one ID needs ≥ 2; any other copy needs ≥ min(5, max(2, ⌈0.2·|site|⌉)). This handles ligand renumbering and assembly-only copies.
- Superposition is a Kabsch fit of the site residues' Cα (UniProt-numbered via SIFTS) onto the AF Cα. For sites with fewer than 3 Cα (mostly ions), the fit uses Cα of residues within 8 Å of the ligand (`sup_set`).
- The local RMSD is stored for every instance. The default acceptance is RMSD ≤ 2 Å; the loader can change it. This mirrors the DESIGN §2.5 pooling rule.
- Results: 27,675 instances transferred. Median RMSD 0.37 Å; 91.6% have RMSD ≤ 2 Å. 92.8% of benchmark sites have at least one accepted ligand.
- Failures are listed in `af_ligands.csv`: 727 no contact (mostly interface ligands that need a symmetry mate), 69 not found (entry re-versioned), 64 too few Cα, 21 missing from SIFTS.

**Annotations** (columns of `af_sites.csv` and `af_manifest.csv`):

- Site pLDDT: `plddt_mean`, `plddt_min`, `plddt_band` (< 50 / 50–70 / 70–90 / ≥ 90), and `frac_res_plddt_lt50`.
- `frac_res_prunable`: the fraction of site residues removed by the DESIGN §2.6 / AF2BIND pruning rule. The rule prunes terminal runs with pLDDT < 50 and internal runs of ≥ 7 residues with pLDDT < 50.
- `dist_to_prunable` and `near_low_plddt`: minimum heavy-atom distance from the site to a prunable residue, and whether that distance is ≤ 8 Å.
- `dist_to_low_plddt`: the same distance to any residue with pLDDT < 50.
- Protein-level (manifest): `frac_plddt_lt50`, `frac_plddt_lt70`, `frac_prunable`, `length`.
- Template richness: `n_pdb_entries` and `n_pdb_chains` from SIFTS for the UniProt; `n_ligand_entries` and `n_ligand_instances` from LIGYSIS for the site.
- Holo-likeness:
  - `rmsd_ca_af_vs_rep`: local Cα RMSD between AF and the LIGYSIS representative chain at the site, available for 6,320 benchmark sites. Median 0.38 Å, 90th percentile 1.8 Å.
  - `rmsd_heavy_af_vs_rep`: heavy-atom RMSD over identically named atoms of same-type residues, after the Cα fit.
  - `lig_rmsd_min` and `lig_rmsd_median`: Cα RMSD of the AF site to the holo chains carrying the transferred ligands.
- `rep_resolution`: 568 benchmark sites have a representative chain without a resolution, i.e. NMR, including solid-state NMR (e.g. 6qeb for CA2). Their `rmsd_ca_af_vs_rep` can be large, so prefer `lig_rmsd_*` as the holo-likeness proxy.
- `ion_prop` and `monomer_prop`: LIGYSIS fields. Ion-only sites have `ion_prop == 1`.

## 3. Files (`data/processed/af/`, git-ignored)

| file | content |
|---|---|
| `models/AF-<ACC>-F1-model_v6.cif.gz` | 3,473 AF models (405 MB). B-factor = pLDDT. |
| `extract_index.csv`, `extract_info.json` | per accession: status (extracted / fragmented / absent), number of fragments, member name, sha256, source tar |
| `af_manifest.csv` | one row per LIGYSIS UniProt (3,518): `structure_id` = `AF-<ACC>-F1`, `status` / `reason`, `model_path`, `gt_path`, `in_lbs_benchmark`, site counts, sequence checks, protein pLDDT stats, `n_pdb_entries` |
| `af_sites.csv` / `.parquet` | one row per site (10,485): `include`, `flags`, centres (JSON lists), all annotations |
| `af_ligands.csv` | one row per attempted ligand transfer: status, match tier, chain, number of Cα, RMSD, resolution, residue identity |
| `gt/AF-<ACC>-F1.json.gz` | per-protein ground truth (see below) |
| `af_sequences.fasta` | AF model sequences of all extracted proteins. Use as the AF test set in Track A leakage filtering. |
| `build_info.json`, `verify_report.json`, `*.log` | parameters, input sha256, counts, check results |

**Ground-truth JSON** (`gt/*.json.gz`) has these top-level keys:

- `structure_id`, `acc`, `model_path`, `chain`, `length`, `residue_numbers`, `plddt` (per residue), `af_seq_eq_uniprot`, `params`.
- `sites`: a list with one entry per site. Each entry holds:
  - `site_name`, `include`, `flags`, `in_lbs_benchmark`, `ion_prop`
  - `residues` (UniProt positions = model residue numbers)
  - `center_ca`, `center_heavy`, `center_sc`, `center_lig`, `plddt_mean`
  - `ligands`: a list of transferred instances, each with `key`, `resname`, `rmsd`, `n_ca`, `match`, `coords` (AF frame, Å) and `elements`.

**Harness compatibility.** `src/pocketdetect/evaluate.py` is unchanged. `scripts/data/build_af_io.py` turns a GT JSON into a `pocketdetect.ligands.GroundTruth`, which `evaluate()` consumes directly. Each site becomes one `Ligand`, with fields set as follows:

| `Ligand` field | value |
|---|---|
| `ligand_id` | site name, the join key to `af_sites.csv` for stratification |
| `center` | the chosen site centre |
| `coords` | accepted transferred ligand atoms, used by DCA |
| `binding_residues` | `A:<pos>` |

`protein_residues` is `A:1..L`. Prediction files are `<pred_dir>/AF-<ACC>-F1.csv` in the standard prediction format, with residues `A:<num>`.

```python
import sys; sys.path.insert(0, "scripts/data")
from build_af_io import load_af_dataset
from pocketdetect.evaluate import evaluate
gts = load_af_dataset(site_set="all", ions="include", center="ca", atoms="ligand",
                      lig_rmsd_max=2.0, missing_ligand="center")
res = evaluate(gts, "results/preds/<method>/af", criteria=["LDCC12", "DCA4"])
```

Loader options:

- `site_set`: `"all"` (every valid LIGYSIS site of the UniProt) or `"benchmark"` (the 6,882-site LBS set, for pairing with the PDB-chain metric on the same proteins).
- `ions`: `"exclude"` gives the LIGYSIS_NI analogue.
- `center`: `ca`, `heavy`, `sc` or `ligand`.
- `missing_ligand`: for sites without an accepted ligand, `"center"` uses the centre as the single DCA atom (such sites are listed in `gt.meta["dca_fallback"]`); `"drop"` removes them.
- `benchmark_proteins=True` (default): only the 2,527 UniProts that carry benchmark sites.

**Recommendation to the PI** (not a pre-registration; that decision is yours):

- Primary O5: `site_set="all"`, `center="ca"`, LDCC12 (DCC ≤ 12 Å, LIGYSIS-style). A full-length model deserves the complete site set; sites missing from the representative chain are real sites.
- Strict secondary: DCA4 with `missing_ligand="drop"`.
- Report `site_set="benchmark"` alongside, paired with the PDB-chain results of the same UniProts.

## 4. Verification (`scripts/data/verify_af_set.py`; all 25 checks pass)

1. **Counts and bookkeeping.** Every LIGYSIS UniProt and site is accounted for. The benchmark sites partition exactly: 6,731 included + 41 flagged + 110 on excluded proteins = 6,882.
2. **Model files.** All 3,465 included models are present with sha256 equal to the extract index. A random sample of 150 has a single chain, numbering 1..L, and pLDDT in [0, 100] identical to the GT JSON.
3. **Residue types.** Match rate is 96.5% exact and 99.3% at ≥ 0.75 identity. No included site is below the threshold.
4. **LIGYSIS centre reproduction.** On 8,089 representative chains the maximum deviation is 0.0009 Å.
5. **pLDDT distributions** are reported, with a band-consistency check.
6. **Spot checks.** For 12 random sites, the Cα centre was recomputed independently from the model file (deviation < 0.001 Å). AF residue types equal the UniProt residue types. Transferred ligands lie 0.3–3.2 Å from site atoms.
7. **Known answers.**
   - CA2 (P00918): the catalytic Zn site contains His94/96/119. The 4 transferred Zn copies sit 1.87–1.92 Å from a His N atom in the AF model.
   - ESR1 (P03372): the ligand pocket contains E353/R394/H524. Genistein is accepted at 0.43 Å local RMSD. Two estradiol instances have about 8 Å RMSD over the 61-residue pooled site, which spans helix 12, so they are rejected under the 2 Å default.
8. **Harness controls through `pocketdetect.evaluate`.**
   - Oracle centres: LDCC12 top-n = 1.000. DCA4 top-n+2 = 0.786, because the Cα centroid is not always within 4 Å of a ligand atom.
   - Predictions 1,000 Å away: 0.
   - Oracle shifted by 30 Å: 0.049. Some shifted centres land on another site of the same protein.
   - Protein centroid: 0.217.

## 5. Caveats

- **Holo/memorisation bias.** Every protein has PDB structures, a median of 7 entries, and AF2 was trained on the PDB. Most sites are high-pLDDT and holo-like: the median local Cα RMSD to the representative chain is 0.38 Å. AF-mode results here are an optimistic estimate of performance on apo or novel proteins. Stratify by `n_pdb_entries`, `rmsd_ca_af_vs_rep` or `lig_rmsd_min`, and pLDDT band (LITERATURE §4, Yu et al. 2026).
- **Low-pLDDT strata are small.** Only 95 benchmark sites have mean pLDDT < 70. Use `near_low_plddt` (1,319 benchmark sites) and `frac_plddt_lt50` for the low-confidence analysis. The hallucination rate is a property of predictions on the whole model, and the site set is not needed for it.
- **Fragmented proteins are not covered.** The 42 proteins over 2,700 aa (e.g. P78527 DNA-PKcs, Q13315 ATM) are excluded, so the AF set is biased towards shorter proteins than the PDB-chain benchmark.
- **The site set is pooled across entries and assemblies, but the AF input is a monomer.** As in LIGYSIS, interface sites are judged without the partner chain. `monomer_prop` gives the fraction of a site's ligands that bind a single chain. Interface ligands are also the main cause of `no_contact` transfer failures.
- **`center_lig` exists only for sites with accepted transfers**, so it is not a complete-coverage centre. DCA uses transferred ligand atoms, which pool up to 5 instances, so DCA is slightly more lenient for multi-ligand sites.
- **Sequence reference.** Four AF models differ from current UniProt. Their sites were validated against PDB residue types; 2 sites of Q14669 were excluded.
- **No PAE.** See §6. The DESIGN §2.6 inter-domain mechanism must use the structure-only surrogate.

## 6. PAE (step 5): not obtained

- The AFDB API (`https://alphafold.ebi.ac.uk/api/prediction/<ACC>`) reports `latestVersion: 6` and `paeDocUrl = https://alphafold.ebi.ac.uk/files/AF-<ACC>-F1-predicted_aligned_error_v6.json`.
- A direct download from this cluster returned **HTTP 403** (tested with P03372 on 2026-10-26).
- Model and API metadata requests worked, so the block is specific to the file endpoint for our client. We did not try to work around it.
- The local AFDB copy has no PAE (models only).
- Because DESIGN §2.6 requires PAE "only if available for both train and test", and neither is available, **PAE is not used**.
- If the PI obtains access (e.g. an approved bulk-download route), the size is an estimate, not a measurement: PAE is an L×L matrix, so Σ L² = 1.78 × 10⁹ entries over the 3,465 included proteins. That is about 10 GB as JSON (about 5.5 bytes per value), 3.6 GB as float16 `.npz`, or 1.8 GB as uint8. Training proteins would need the same route.

## 7. Parameters (frozen before any method evaluation; `build_info.json` → `params`)

| parameter | value | role |
|---|---|---|
| `MIN_RESTYPE_ID` | 0.75 | site residue-type identity threshold |
| `MAX_LIG_PER_SITE` | 5 | ligand instances transferred per site |
| `LIG_CONTACT` | 4.5 Å | contact distance used to locate a ligand copy |
| `MIN_SUPERPOSE_CA` | 3 | minimum Cα for the local fit |
| `ENV_SUPERPOSE` | 8 Å | fallback superposition set (residues near the ligand) |
| `LIG_RMSD_ACCEPT` | 2 Å | default acceptance of a transferred ligand |
| `PLDDT_LOW` | 50 | low-pLDDT cutoff |
| `PRUNE_INTERNAL_MIN` | 7 | minimum internal low-pLDDT run that is pruned |
| `NEAR_LOW` | 8 Å | "near a low-pLDDT segment" distance |
| `PLDDT_BANDS` | 50 / 70 / 90 | site pLDDT band edges |

During development, `MIN_RESTYPE_ID` was lowered from 0.9 to 0.75 after inspecting the mismatch list: single engineered mutations in 6–9-residue sites versus numbering shifts. This was decided before any pocket prediction was run on this set.

## 8. DATA.md section (draft for the PI to merge)

| Dataset | Source | Version / retrieved | Checksum | License | Local path | Notes |
|---|---|---|---|---|---|---|
| AF test set models | AFDB v6 local mirror `UP000005640_9606_HUMAN_v6.tar` (+ `swissprot_cif_v6.tar` for 5 accessions) | v6, tar mtime 2025-08-24; extracted 2026-10-26 | per-model sha256 in `data/processed/af/extract_index.csv` | CC-BY 4.0 | `data/processed/af/models/` | 3,473 F1 models; 42 fragmented and 3 absent accessions excluded |
| UniProt human reviewed sequences | `https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,sequence_version,length,sequence&query=(organism_id:9606) AND (reviewed:true)` | release 2026_03 (2026-09-02); retrieved 2026-10-26 | sha256 `9507f641daa0384227b7e78311a19fdf2f316eea2fa408c2faead699a415d290` | CC-BY 4.0 | `data/processed/af/external/uniprot_human_reviewed.tsv.gz` | 20,431 entries |
| SIFTS pdb_chain_uniprot | `https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz` | 2026-10-18 (PDB 42.26, UniProt 2026.03); retrieved 2026-10-26 | sha256 `3ff549da37104bdd950cd78a939be55e93d7eb0279877f57d2134f58762f0f39` | CC-BY 4.0 (EMBL-EBI) | `data/processed/af/external/pdb_chain_uniprot.tsv.gz` | chain → UniProt residue mapping; PDB entry counts |
| AF test-set ground truth | derived (scripts/data/build_af_set.py) from LIGYSIS (Zenodo 14645504, MIT) + PDB mirror | 2026-10-26 | input sha256 in `data/processed/af/build_info.json` | derived; MIT / CC0 inputs | `data/processed/af/` | 3,465 proteins, 10,341 sites (benchmark subset 6,731) |
| AFDB PAE | `https://alphafold.ebi.ac.uk/files/AF-<ACC>-F1-predicted_aligned_error_v6.json` | not obtained (HTTP 403 from the cluster) | — | CC-BY 4.0 | — | not used (DESIGN §2.6 requires train + test) |

---

# Track C2: AF-model training and dev labels for the AF-aware ranker (DESIGN §2.6 item 4)

```
nohup .venv/bin/python scripts/data/build_af_train_index.py --jobs 16 > data/processed/af_train/index.log 2>&1   # ~1.5 min
nohup .venv/bin/python scripts/data/build_af_train_set.py   --jobs 16 > data/processed/af_train/build.log 2>&1   # ~1 min
.venv/bin/python scripts/data/build_af_stress.py            # AF stress list (descriptive)
.venv/bin/python scripts/data/build_af_train_verify.py      # 19 checks -> data/processed/af_train/verify_report.json
```

The build uses Track A's **audit-filtered** manifests: train has 6,475 entries and dev has 579. The 186 entries in `data/manifests/train_dev_audit_exclusions.tsv` are absent. The sha256 of each manifest is recorded in `build_info.json`, and the verify script fails if either manifest changes. If Track A changes its manifests again, re-run `build_af_train_set.py` and the verify script. The index step only needs re-running if new UniProts appear. `build_af_train_set.py` deletes old entry files before writing, so entries from a previous manifest cannot survive.

## C2.1 Accession → tar index (single pass per tar)

- All 47 local tars were scanned in parallel with 16 processes: 46 proteome tars plus `swissprot_cif_v6.tar`. `swissprot_pdb_v6.tar` was skipped because it holds the same models as PDB.
- The index `afdb_index.parquet` has 1,661,814 members over 989,696 accessions. It stores tar offsets, so later lookups need no rescan.
- In the same pass, the F1 `cif.gz` was extracted for every train/dev accession; the 7,260 accessions are taken from the pre-audit manifests, a superset of the current ones.
  - Fragmented proteins (any F > 1) are skipped.
  - A proteome tar is preferred over SwissProt. Where both contain a model, the sha256 values are identical.
- Result: 3,323 models extracted (277 MB), 8 fragmented, 3,929 absent.

Most absent accessions are unreviewed (TrEMBL) entries from species outside the 46 proteomes. The local mirror holds only those proteomes plus SwissProt.

## C2.2 Transfer of site ligands

**Mapping.** Each Track A representative chain is mapped onto the AF model as follows:

- The representative's observed sequence (`rep_seq`, matched to `rep_res_keys`) is aligned to the AF sequence. The alignment is a numba Gotoh overlap alignment with free end gaps: match +3, mismatch −2, gap open −8, gap extend −1.
- For chimeric entries with several accessions, the accession that maps the most site residues to identical residue types is used.
- SIFTS was not used for the mapping, so it serves as an independent check: on a sample, 728 of 728 site residues agree with SIFTS.

**Superposition.** For each site:

- A Kabsch fit is computed on the Cα of the mapped site residues. If fewer than 3 Cα are available, the fit uses the Cα of mapped residues within 8 Å of the site's ligand atoms.
- All pooled ligand heavy atoms of the site are moved rigidly with that fit.

**Drop rules.** A site is dropped, and counted, for any of the following:

- fewer than 50% of its residues mapped (`unmapped`);
- fewer than 3 superposition pairs (`too_few_pairs`);
- residue-type identity below 0.75 (`identity`);
- local RMSD above 3 Å (`rmsd`).

| | train | dev |
|---|---|---|
| entries (Track A, filtered) | 6,475 | 579 |
| entries with an AF model | 2,902 (**44.8%**) | 283 (**48.9%**) |
| skipped: no AF model / fragmented | 3,566 / 7 | 296 / 0 |
| **entries with ≥ 1 transferred site (`ok`)** | **2,844 (43.9%)**; 2,474 30% clusters; 2,311 with `rank_in_cluster==0` | **278 (48.0%)**; 254 clusters; 240 rank-0 |
| entries with an AF model but no valid site | 58 | 5 |
| sites on entries with an AF model → kept | 6,394 → **6,238** | 624 → **607** |
| dropped sites: RMSD > 3 / identity < 0.75 / unmapped / too few pairs | 89 / 49 / 17 / 1 | 7 / 7 / 3 / 0 |

- Local RMSD of kept sites: median 0.23 Å; 90th percentile 0.72 Å; 99th percentile 2.25 Å.
- Mean pLDDT of kept sites: ≥ 90 for 5,989; 70–90 for 799; 50–70 for 50; < 50 for 7.
- Median fraction of residues with pLDDT < 50 per protein: 0.017 for train, 0.011 for dev.

**Caveats.**

- **Coverage.** Coverage is about 45% of entries, limited to proteins in the local AFDB copy. The covered subset is biased towards SwissProt and model organisms.
- **Holo bias.** Training sites are as holo-biased and high-confidence as the test sites.
- **Few low-confidence examples.** The ranker will see few low-pLDDT sites. Low-pLDDT *negatives* (hallucinated pockets) are plentiful, because every model carries its disordered regions.

**Split integrity.** The split is exactly Track A's. The verify script confirms:

- No accession is shared between train and dev.
- No train/dev accession appears in the AF test set.
- There are no stale files.

## C2.3 Files (`data/processed/af_train/`, git-ignored)

| file | content |
|---|---|
| `afdb_index.parquet` | accession, fragment, version, extension, tar, member, data offset, size (all local AFDB members) |
| `model_index.csv`, `index_info.json` | per train/dev accession: extracted / fragmented / absent, source tar, sha256 |
| `models/AF-<ACC>-F1-model_v6.cif.gz` | 3,323 models |
| `{train,dev}/manifest.csv` | one row per Track A entry: `status` (ok / no_af_model / fragmented / no_alignment / no_valid_site), `acc`, `model_path`, `npz`, alignment identity and coverage, protein pLDDT stats, site counts and drop counts, `cluster30` / `group` / `rank_in_cluster` (copied from Track A) |
| `{train,dev}/entries/<entry_id>.npz` | AF-frame labels; no pickles |
| `sites.csv` | one row per site considered: kept or drop reason, RMSD, identity, mapped fraction, pairs, superposition set, pLDDT |

The `.npz` entry files contain:

- `meta_json`
- `af_res_nums`, `af_seq`, `plddt` (per residue)
- `lig_coords`, `lig_elements`, `lig_ptr`, `lig_site`, `lig_is_ion`, `lig_ccd`, `lig_id`
- `site_residues` (`;`-joined AF residue numbers, which equal UniProt positions)
- `site_src_index` (index of the site in the Track A entry)
- `site_rmsd`, `site_identity`, `site_frac_mapped`, `site_n_pairs`, `site_sup_set`, `site_is_ion`, `site_plddt_mean`, `site_plddt_min`, `site_center`

**Loader.** `scripts/data/build_af_train_io.py` mirrors `pocketdetect.datasets.trainset`:

- `read_af_manifest(split)`
- `load_af_entry(npz)`, which returns an `AFTrainEntry` with `.ligand_atoms(ions)`, `.site_table()`, `.site_residues()`, `.plddt`, `.res_nums`
- `load_af_input(entry)`, which loads the AF model with protein heavy atoms only; the B-factor holds pLDDT

## C2.4 AF stress set (descriptive; no new labels)

`data/processed/af/af_stress_set.csv` lists 300 proteins: the included LBS-subset proteins of the AF test set with the largest fraction of residues at pLDDT < 50 (ties broken by accession). The threshold is a fraction of at least 0.375. Labels are the existing AF test-set sites; filter `load_af_dataset()` by `structure_id`. Statistics are in `af_stress_stats.json`:

| | stress (300) | rest (2,227) |
|---|---|---|
| fraction pLDDT < 50, median (10th–90th percentile) | 0.49 (0.39–0.70) | 0.086 (0.004–0.26) |
| length, median | 745 | 440 |
| sites (all LIGYSIS sites) / per protein | 876 / 2.9 | 6,444 / 2.9 |
| sites within 8 Å of a prunable segment | **42.8%** | 17.4% |
| site pLDDT bands (≥ 90 / 70–90 / 50–70 / < 50) | 553 / 309 / 8 / 6 | 4,763 / 1,575 / 97 / 9 |

The true sites are still mostly confident; they are structured domains embedded in disordered proteins. The stress lies in the large disordered mass next to the sites. This set is therefore suited to measuring hallucinated pockets, for example pockets with mean pLDDT < 50 that match no site, and occlusion of real sites by nearby low-confidence segments.

## C2.5 Verification (`build_af_train_verify.py`: 19/19 PASS)

- **Split integrity** (12 checks): AF manifest entries equal the filtered Track A manifests; split columns are correct; no audit-excluded entries; manifest sha256 matches; npz files equal the ok entries; no accession shared between train and dev or with the AF test set.
- **Thresholds**: every kept site has RMSD ≤ 3 Å, identity ≥ 0.75 and mapped fraction ≥ 0.5.
- **Models**: model sha256 equals the index for all 3,122 ok entries.
- **Seeded sample of 200 entries**:
  - pLDDT equals the Cα B-factor.
  - Rigid-body transfer: intra-ligand distances deviate from the source atoms by at most 6×10⁻⁶ Å.
  - Every transferred site ligand lies within 4.63 Å of its AF site residues (median 2.3 Å).
  - Alignment mapping agrees with SIFTS for 728 of 728 residues.
- **Stress set**: 300 included LBS proteins, sorted, with nothing above the threshold left out.
