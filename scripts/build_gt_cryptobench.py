#!/usr/bin/env python
"""Build the CryptoBench test-split ground-truth site table (PREREG O4).

Writes ``<out>/sites.tsv`` (pocketdetect.sites format), ``<out>/inputs.sha256`` (every input file
incl. each apo mmCIF used) and ``<out>/README.md``. Deterministic; prints counts.

    .venv/bin/python -I scripts/build_gt_cryptobench.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from pocketdetect.importers import sha256_file
from pocketdetect.importers.cryptobench import MAX_POCKET_SIMILARITY, build_sites
from pocketdetect.sites import write_sites

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cryptobench"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(RAW / "osf/cryptobench/cryptobench-dataset"))
    ap.add_argument("--cif-dir", default=str(RAW / "cif-files/cif-files"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "data/processed/gt/cryptobench_test"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, stats = build_sites(args.dataset_dir, args.cif_dir, args.split)
    write_sites(df, out / "sites.tsv")

    ds = Path(args.dataset_dir)
    fold_file = ds / "folds" / ("test.json" if args.split == "test" else f"{args.split}.json")
    meta_inputs = [fold_file, ds / "folds.json", ds / "README.md"]
    lines = [f"{sha256_file(p)}  {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}" for p in meta_inputs]
    struct_lines = []
    for p in sorted(set(df["structure_path"])):
        struct_lines.append(f"{sha256_file(p)}  {Path(p).relative_to(ROOT) if Path(p).is_relative_to(ROOT) else p}")
    agg = hashlib.sha256("\n".join(struct_lines).encode()).hexdigest()
    (out / "inputs.sha256").write_text("\n".join(lines + struct_lines) + "\n")
    tsv_sha = sha256_file(out / "sites.tsv")

    n_struct = df["structure_id"].nunique()
    counts = (
        f"- apo structures (inputs): **{n_struct}** ({stats['multichain_inputs']} multi-chain inputs)\n"
        f"- holo entries (apo–holo–ligand pairs): {stats['holo_entries']}\n"
        f"- cryptic sites (rows, all scored): **{stats['sites']}** ({stats['multichain_sites']} span >1 chain)\n"
        f"- sites per structure: " + ", ".join(f"{k}: {v}" for k, v in df.groupby('structure_id')['n_sites'].first().value_counts().sort_index().items()) + "\n"
        f"- site residues: {stats['site_residues']}; unmapped to the apo file: {stats['residues_unmapped']}; "
        f"without CA: {stats['residues_no_ca']}; non-protein: {stats['residues_non_protein']}\n"
    )
    readme = f"""# CryptoBench {args.split} ground truth (site table)

Built by `scripts/build_gt_cryptobench.py` with `pocketdetect.importers.cryptobench`
(PREREG endpoint **O4**: top-N+2 recall, DCC <= 12 A to the site-residue centroid on the apo
structure). Format: `pocketdetect.sites` (`read_sites`).

## Counts
{counts}
## Sources
- CryptoBench dataset (OSF 10.17605/OSF.IO/PZ4A9; provenance in `data/manifests/cryptobench_osf.tsv`),
  split file `{fold_file.relative_to(ROOT) if fold_file.is_relative_to(ROOT) else fold_file}`, checked against `folds.json`.
- Apo mmCIF files from the CryptoBench `cif-files.zip` (`{Path(args.cif_dir).relative_to(ROOT) if Path(args.cif_dir).is_relative_to(ROOT) else args.cif_dir}`).
- Rule sources (read, not executed): CryptoBench GitHub @9a3432f (`src/F-statistics/stats.ipynb` pocket merging,
  `src/G-.../create-annotations.py` chain rule, `cryptobench-dataset/README.md` auth numbering).
- sha256 of every input (json + each of the {n_struct} apo mmCIFs) in `inputs.sha256`;
  aggregate sha256 of the structure lines: `{agg}`. `sites.tsv` sha256: `{tsv_sha}`.

## Rules applied
1. **Input structure** (`structure_id` = `<apo_pdb>_<chains>`): the apo entry restricted to the
   CryptoBench apo chain(s) = union of `apo_chain` (dash-separated) over the apo's holo entries;
   identical to CryptoBench's own `the_most_chains` rule for every test apo. `structure_path` is
   the **full** apo mmCIF; methods must be run on the chains in column `input_chains` only (all
   chains of the file otherwise). `n_residues_structure` = protein residues of those chains.
2. **Residue numbering**: `apo_pocket_selection` tokens `<chain>_<num>` are auth_asym_id/auth_seq_id
   (dataset README). Verified: all 18,899 test residue tokens exist under author numbering; apo vs
   holo residue names agree for 97.9% under auth numbering vs 34% under label numbering.
   No insertion codes occur. Written as `chain:resnum`.
3. **Sites**: CryptoBench's "cryptic pocket" definition (stats.ipynb, Table 2): pocket selections of
   all holo entries of an apo are merged iteratively when the overlap exceeds
   {MAX_POCKET_SIMILARITY:.0%} of either selection; merged site = union of residues
   (faithful port incl. order dependence). `n_holo_entries`, `holo_pdb_ids`, `ligands`
   (`<comp>:<chain>:<resnum>@<holo>`), `ligand_names`, `max_prmsd`, `uniprot` describe the holo
   entries whose selection is contained in the site.
4. **Centre** (`center_x/y/z`): mean of the CA atoms of the site residues on the apo structure
   (first altloc, `pocketdetect.structure.load_structure`). `center_heavy_x/y/z`: mean of all
   heavy atoms of those residues.
5. `n_sites` = number of merged cryptic sites of the apo structure; `scored = 1` for all rows.
   CryptoBench's auxiliary *non-cryptic* pockets are **not** included (they are not part of the
   O4 target set and do not count toward N). `ligand_atoms` empty (ligands are in holo frames).
"""
    (out / "README.md").write_text(readme)
    print(f"CryptoBench {args.split}: {n_struct} structures, {len(df)} sites (scored {int(df['scored'].sum())}), "
          f"{stats['holo_entries']} holo entries, residues unmapped {stats['residues_unmapped']}, "
          f"no-CA {stats['residues_no_ca']} -> {out / 'sites.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
