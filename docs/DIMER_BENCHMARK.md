# Dimer benchmark (Track D): homo- and hetero-oligomer ligand sites

Spec: DESIGN.md §2.7 and §9 (v3), docs/PREREG.md (O6; amendment A1, which makes j50 the GT grouping).
Owner: Track D data engineer. Files: `scripts/data/build_dimer_{common,scan,select}.py`,
`scripts/data/verify_dimer_set.py`, `scripts/eval_dimer.py`, `tests/test_eval_dimer.py`, `data/processed/dimer/`.

## 1. Reproduce

```bash
.venv/bin/python scripts/data/build_dimer_scan.py --jobs 60     # ~2 min, 26,321 candidate entries
.venv/bin/python scripts/data/build_dimer_select.py --jobs 32   # selection + files, ~10 s
.venv/bin/python scripts/data/verify_dimer_set.py               # checks; writes VERIFY_REPORT.json
.venv/bin/python -m pytest -q tests/test_eval_dimer.py
```

The pipeline is deterministic. Selection uses seed `pocketdetect.seeds.SEEDS[0]` (20261026). Stage 2 rebuilds every selected entry and asserts that the result is identical to the scan.

## 2. Sources and construction

- **Protein scope.** UniProt ACCs of the LIGYSIS benchmark: the 3,048 ACCs of the 8,244 sites in `LIGYSIS_sites_DEF_TRANS.pkl`, i.e. the 3,448 LIGYSIS chains. These are excluded from training. Hetero partner chains may be any protein. All chains, partners included, are listed in `manifest_chains.tsv` for the training-set leakage filter.
- **Candidate entries.** PDB entries with a BioLiP2 (snapshot 2025-11-12, `data/raw/biolip2/2026-10-26`) non-polymer ligand row whose receptor UniProt is in scope, with resolution ≤ 3.0 Å. Peptide, DNA and RNA "ligands" are ignored. 26,321 entries.
- **Entry filters**, applied to the asymmetric-unit mmCIF from the local mirror:
  - `_exptl.method` ⊆ {X-ray, EM};
  - 0 < resolution ≤ 3.0 Å;
  - `_pdbx_struct_assembly.details` of assembly 1 = `author_and_software_defined_assembly`.
- **Assembly expansion.** Our own code (gemmi operators) expands assembly 1 of the first model. Hydrogens and waters are removed, and only the first altloc is kept. Each new chain is one (author chain, operator) pair.
- **Assembly filters.**
  - 2–8 protein chains, every one with ≥ 50 observed residues. Entries with a shorter (peptide) chain are rejected.
  - No nucleic-acid or other polymer.
  - ≤ 99,999 atoms (PDB format limit).
  - At least one entity mapped (`_struct_ref`, UNP) to an in-scope ACC.
  - Author numbering consistent within each entity. Entities such as chain B numbered +1000 are rejected: 61 entries.
  - ≤ 10 inter-chain protein heavy-atom pairs < 1.5 Å.
- **Relevant ligands.** BioLiP2 annotations of the same entry (not pooled across entries), keyed by ligand author chain, CCD and author resnum, ions included. Every assembly copy of such a residue is a *ligand copy*. Ground truth comes from `pocketdetect.ligands.build_ground_truth` (rules `p2rank`) with an explicit P2Rank-style spec on the written holo file:
  - all spec'd ligands are relevant;
  - groups within 1.7 Å are merged, so covalently linked multi-residue ligands become one copy, and the duplicated copies of ligands sitting on a symmetry axis are merged;
  - heavy atoms only.
  Copies with no protein heavy atom within 4 Å are dropped (ligand subchain in the assembly without its receptor). `is_ion` = monoatomic ligand.
- **Contacts and classes.** Contact = protein heavy atom ≤ 4 Å of a ligand heavy atom. For each copy we count contacted residues per chain.
  - `homo_interface`: ≥ 3 residues on each of ≥ 2 chains, all of the same entity.
  - `hetero_interface`: as above, but those chains include ≥ 2 entities.
  - `single_chain`: all contacts on one chain.
  - `minor_interface`: contacts on ≥ 2 chains without satisfying the interface rule, e.g. an ion bridging 2 + 1 residues. Kept as sites (they count in N) but not interface.
- **Entity-level residue set** of a copy = {`E<entity_id>:<author resnum><icode>`}. Author numbering is used so that predictions, which report `chain:resnum`, map identically.
- **Unique sites (symmetric copies).**
  - `j50` (primary, PREREG A1): connected components of entity-level Jaccard ≥ 0.5.
  - `strict` (sensitivity): identical entity-level sets.
  - The class of a unique site is the most interface-like class among its copies. Under j50, 31 sites have copies of different classes; this is flagged by `class_consistent`.
- **Isolated-chain input.** The *primary chain* is the chain with the most site contacts (residues summed over all copies). Every other chain is also written as a context-free single-chain input. A copy belongs to an isolated chain *c* if it has ≥ min(3, max residues on any chain) contacted residues on *c*.
- **Selection** (one entry per UniProt / hetero complex):
  1. Pass 1 visits the UniProts that have an entry with ≥ 1 interface site, in seeded random order. For each it picks a random such entry that contains no already-covered in-scope UniProt.
  2. Pass 2 fills up to 400 complexes from UniProts without interface entries.
  3. All in-scope UniProts of a chosen complex become covered.

## 3. Counts (`VERIFY_REPORT.json`)

Scan of 26,321 candidates gave 5,375 qualifying entries for 1,119 UniProts; 246 of these UniProts have an entry with an interface site. Main rejections: assembly details 9,118; protein chain count (mostly monomers) 8,618; short chain 1,978; nucleic acid 940.

**Selected set**
- 400 complexes covering 450 LIGYSIS UniProts:
  - 274 homo and 126 hetero;
  - 384 X-ray and 16 EM;
  - median resolution 2.25 Å.
- 214 complexes (151 homo, 63 hetero) contain ≥ 1 interface site.
- Chains per complex: 2 → 268, 3 → 37, 4 → 79, 5–8 → 16.
- 1,062 chains with 554 unique sequences.
- 1,540 ligand copies.

| unique sites | total | single_chain | homo_interface | hetero_interface | minor_interface | ions |
|---|---|---|---|---|---|---|
| **j50 (primary)** | 807 | 494 | 177 | 66 | 70 | 301 |
| strict | 1,063 | 620 | 258 | 72 | 113 | 333 |

- The O6 endpoint covers 243 interface sites (j50).
- Ions are almost never interface sites: 9 of the 243.
- Primary-chain mode has 752 sites, 239 of them interface.

## 4. Files (`data/processed/dimer/`)

| file | content |
|---|---|
| `holo/<pdb>.pdb` | Assembly 1: polymer plus all non-water hetero groups. GT source only; **never a method input**. |
| `inputs/assembly/<pdb>.pdb` | **Method input**, full assembly. Polymer only (`structure.py` PROTEIN residues: standard plus modified amino acids, heavy atoms); no CRYST1. |
| `inputs/chains/<pdb>_<C>.pdb` | **Method input**, isolated chain in the same coordinate frame (all chains). |
| `inputs_assembly.csv`, `inputs_chains.csv` | `structure_id, complex_id, [chain, is_primary,] input_path, n_atoms` |
| `chain_map.tsv` | Chain-id mapping: `chain` (new id) ← `orig_auth_chain` + `operator` + `label_asym_ids`; `entity_id`, `uniprot`, residues, site contacts, `is_primary`. A chain keeps its author id if that id is one character and this is the id's first operator; otherwise it gets the next free id from A–Z, a–z, 0–9. |
| `gt/<pdb>.json` | Full GT: chains, entities (SEQRES, UniProt), primary chain. `copies` holds, per copy: `ligand_id`, `ccd`, `is_ion`, `center`, `coords`, `elements`, `binding_residues`, `chain_counts`, `entity_residues`, `site_class`, `site_strict`, `site_j50`, `spec`, `primary_chain_relevant`. Unique-site tables for both groupings. |
| `site_copies.tsv`, `unique_sites.tsv`, `complexes.tsv` | Flat tables of the same information. |
| `gt_manifest_assembly.csv`, `gt_manifest_primary_chain.csv` | Manifests for `scripts/evaluate.py`: ligand-centric, **no** symmetric collapse; useful for legacy DCA/DCC numbers. |
| `manifest_chains.tsv` | `pdb_id, chain, orig_auth_chain, operator, entity_id, uniprot, sequence`. Every chain, hetero partners included, for the leakage filter (test union). |
| `scan/entries.jsonl.gz`, `scan/scan_summary.json` | Every candidate with its rejection reason. |
| `selection.json`, `VERIFY_REPORT.json` | Selection parameters and verification output. |

## 5. Evaluation adaptor (`scripts/eval_dimer.py`)

Predictions use the `pocketdetect.evaluate` CSV format (`rank,score,center_x,center_y,center_z,residues`). Files are `<pred-dir>/<pdb>.csv` in assembly mode and `<pred-dir>/<pdb>_<chain>.csv` in chain mode. Run every method on the **identical** input files listed above. Residues use the `chain:resnum[icode]` ids of the input file.

**Scoring, per unit** (a complex in assembly mode; an isolated chain in chain mode, primary chain by default or `--chain-set all`):

1. **GT.**
   - Copies are grouped into unique sites: `--gt-grouping j50` (default, PREREG A1) or `strict`.
   - In chain mode, only copies relevant to that chain are used.
   - **N = number of unique sites.**
2. **Label-free prediction collapse**, applied to every method.
   - Predictions are visited in rank order. A prediction is dropped if its entity-level residue set (via `chain_map` entity ids) has Jaccard ≥ 0.5 with an already *kept* prediction.
   - Predictions without residues get the residues with any heavy atom ≤ 6 Å of the centre (PREREG rule for centre-only methods), computed on the input file.
   - Empty sets never merge.
   - `--no-collapse` exists for diagnostics only.
3. **Hits.** A prediction hits a unique site if it hits **any copy**.
   - **DCC12** (primary): centre ≤ 12 Å from that copy's ligand centroid.
   - **DCA4**: centre ≤ 4 Å from any heavy atom of that copy.
   - Success at k ∈ {top-N, **top-N+2**, all} if any of the first k kept predictions hits the site (not one-to-one, as in LIGYSIS).
   - `LDCC12` (LIGYSIS "closest prediction" rule) is reported as a descriptive variant.
4. **Precision** in the top-(N+2) kept predictions, in rank order:
   - a hit on an unmatched unique site is TP (matched to the closest such site);
   - a prediction that hits only already-matched sites is **neutral, not FP**;
   - no hit is FP.
5. **Missing or empty prediction file**: zero predictions, so all of the unit's sites are misses.

**Summary.**
- Recall is pooled over sites per stratum: all, interface (homo + hetero, the O6 endpoint), homo, hetero, single_chain, minor_interface, ion, non-ion.
- Percentile bootstrap CI: 1,000 resamples over complexes, seed `BOOTSTRAP_SEED`. Complexes are already one per UniProt / hetero complex. 30% sequence clusters across complexes are not computed here (caveat below).
- Outputs: `sites.csv` (per unique site), `units.csv`, `summary.csv`, `run_info.json`.
- For a paired assembly-vs-chain comparison, join `sites.csv` of both runs on (`complex_id`, `copies`), or restrict the assembly-mode sites to those present in primary-chain mode.

**Oracle ceilings** (one prediction per GT copy at its centroid with its residues; `--oracle`), recall at DCC12 top-N+2:

| grouping | assembly mode | primary-chain mode |
|---|---|---|
| j50 (primary) | 1.000 | 1.000 |
| strict | 0.791 (interface 0.758) | 0.943 (interface 0.855) |

Under the strict GT, copies that differ by one contact residue stay separate sites, while the Jaccard ≥ 0.5 prediction collapse merges predictions on them. Recall is therefore capped below 1. This is why PREREG A1 adopted j50.

**Tests.** `tests/test_eval_dimer.py` covers:
- any-copy hits;
- the collapse;
- duplicates counted as neutral rather than FP;
- FP and the top-k cut-off;
- DCA vs DCC on an elongated ligand;
- strict vs j50;
- an interface site with two copies;
- hetero entities not collapsing;
- chain-mode copy filtering;
- missing predictions;
- empty residue sets;
- the 6 Å residue fallback;
- oracle ceilings.

## 6. Verification (`scripts/data/verify_dimer_set.py`, PASS)

Each check below passed:
- **Entry rules**: hold for all 400 complexes, and no LIGYSIS UniProt appears in two complexes.
- **Inputs**: polymer only; chain ids equal `chain_map`; the isolated-chain inputs concatenate exactly to the assembly input.
- **GT**: re-extracted with `build_ground_truth` from the manifests, it is identical to the JSON (ids and centroids).
- **Contacts**: recomputed independently with a cKDTree on the polymer-only input, they are identical for all 1,540 copies, as are the classes; every copy has ≥ 1 contact.
- **Clashes**:
  - inter-chain protein clashes < 1.5 Å: at most 5 pairs, in 4 complexes;
  - 86 copies have a non-metal atom within 2.2 Å of protein (covalent ligands);
  - one deposited PLP–Lys bond is at 0.94 Å (6v8d, WARN).
- **Symmetric copies**: for 418 single-chain copy pairs on equal-entity chains, superposing the chains maps one copy's centroid onto the other with a median deviation of 0.12 Å (95% < 2 Å).
- **Oracle**: the j50 ceiling is 1.0.
- **Manifest**: `manifest_chains.tsv` covers every chain.

## 7. Caveats

- `author_and_software_defined_assembly` drops entries whose assembly 1 is author-only or software-only (9,118 of 26,321). The kept set is biased toward well-agreed assemblies; crystal contacts are minimised but not excluded.
- Sites are per entry (not pooled). A protein's other known sites, from other entries, count as unlabelled, so precision is pessimistic.
- Hetero partners can be any protein (antibodies, nanobodies). Partner sequences are in the test union; LIGYSIS UniProts are excluded from training by ACC.
- No cross-complex redundancy filter beyond one complex per UniProt: paralogues may co-occur, and partner proteins can recur. A 30% sequence clustering of the 554 unique sequences should be used for the PREREG cluster bootstrap if required.
- Class depends on the 4 Å / 3-residue cut-off. 31 j50 sites mix classes across copies and are assigned the most interface-like class.
- Entries with entity-inconsistent author numbering are excluded (61). Peptide-bound complexes (chain < 50 residues) are excluded by design.
