#!/usr/bin/env python
"""Build the PocketWatch temporal-test ground-truth site table (PREREG O3).

Writes ``<out>/sites.tsv`` (pocketdetect.sites format), ``<out>/inputs.sha256`` (every input file
incl. each ESMFold2 fold used) and ``<out>/README.md``. Deterministic; prints counts.

    .venv/bin/python -I scripts/build_gt_pocketwatch.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from pocketdetect.importers import sha256_file
from pocketdetect.importers.pocketwatch import build_sites, resolve_release_dir
from pocketdetect.sites import write_sites

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = [
    "README.md",
    "labels_manifest.json",
    "data/site_policy.parquet",
    "data/pocket_label_sites.parquet",
    "data/site_centroids.parquet",
    "data/sequences.parquet",
    "data/paper_excluded_test_proteins.csv",
    "projection/ligand_centroids.parquet",
    "projection/contact_residues.parquet",
    "structures/manifest.parquet",
]


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(ROOT / "data/raw/pocketwatch/current"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "data/processed/gt/pocketwatch_test"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = resolve_release_dir(args.raw_dir)
    df, stats = build_sites(args.raw_dir, args.split)
    write_sites(df, out / "sites.tsv")

    lines = [f"{sha256_file(root / f)}  {_rel(root / f)}" for f in INPUT_FILES]
    struct_lines = [f"{sha256_file(p)}  {_rel(Path(p))}" for p in sorted(set(df["structure_path"]))]
    agg = hashlib.sha256("\n".join(struct_lines).encode()).hexdigest()
    (out / "inputs.sha256").write_text("\n".join(lines + struct_lines) + "\n")
    tsv_sha = sha256_file(out / "sites.tsv")

    reasons = df[df["scored"] == 0]["mask_reason"].value_counts().sort_index()
    excl = df[df["paper_excluded"] == 1]
    counts = (
        f"- structures (ESMFold2 folds): **{stats['structures']}**; with >= 1 scored site: **{stats['structures_with_scored']}**\n"
        f"- sites (rows): **{stats['sites']}**; scored: **{stats['scored_sites']}**; masked (scored=0): {stats['masked_sites']} "
        f"(" + ", ".join(f"{k} {v}" for k, v in reasons.items()) + ")\n"
        f"- README/paper: test = 1,566 proteins, 2,301 sites, 1,573 scored sites, 1,279 scored chains — "
        f"{'reproduced exactly' if (stats['structures'], stats['sites'], stats['scored_sites'], stats['structures_with_scored']) == (1566, 2301, 1573, 1279) else 'NOT reproduced'}.\n"
        f"- paper-excluded proteins (`paper_excluded = 1`, VN-EGNN cannot read UNK): {excl['structure_id'].nunique()} proteins, "
        f"{int(excl['scored'].sum())} scored sites (kept; drop only to reproduce the paper's comparison tables -> 1,275 chains)\n"
        f"- rebuilt ligand centroids vs shipped `site_centroids.parquet`: max |dev| = {stats['max_centre_dev_vs_shipped']:.2e} A; "
        f"sites without centre: {stats['sites_without_centre']}; site residues missing from folds: {stats['residues_unmapped']}\n"
    )
    readme = f"""# PocketWatch {args.split} ground truth (site table)

Built by `scripts/build_gt_pocketwatch.py` with `pocketdetect.importers.pocketwatch` (PREREG endpoint
**O3**: top-N+2 recall, DCC <= 12 A on scored (unmasked) sites; inputs are their predicted structures).
Format: `pocketdetect.sites` (`read_sites`).

## Counts
{counts}
## Sources
- PocketWatch HF dataset `genesisml/PocketWatch` @ `{root.name}` (MIT; provenance in
  `data/manifests/pocketwatch_{root.name}.tsv`), read from `{_rel(root)}`.
- sha256 of every input table and of each of the {stats['structures']} folds in `inputs.sha256`;
  aggregate sha256 of the structure lines: `{agg}`. `sites.tsv` sha256: `{tsv_sha}`.
- Rule sources (read, not executed): dataset `README.md` (Scoring, site policy, coordinate frame),
  `labels_manifest.json`, `verify_bundle.py` (centroid recipe).

## Rules applied
1. **Inputs**: `structure_path` = `structures/<uniprot>.cif.gz`, the ESMFold2 sample-0 fold of the
   canonical sequence (single chain `A`, residues 1..L). `structure_id` = UniProt accession.
   `n_residues_structure` = protein residues of the fold (= L).
2. **Sites**: all `split == "{args.split}"` rows of `data/pocket_label_sites.parquet` (Q-BioLiP contacts at
   5 A, clustered into sites across entries). `residues` = `canonical_positions` as `A:<pos>`;
   `label_residues` = `single_chain_positions` (their per-residue positives for scored sites; empty for
   interface-only sites). Ligands: every site has >= 10 heavy atoms (min `min_heavy_atom_count` = 10),
   i.e. ions are excluded -> ranker `small` per PREREG O3.
3. **Mask**: `data/site_policy.parquet` `decision`: `scored` -> `scored = 1`; `masked` -> `scored = 0`
   (reason in `mask_reason`: interface_only / pre_cutoff / subsumed_by_seen). Masked sites are kept as
   rows with valid centres: they are real pockets and "never a negative", so an evaluator may use them
   to avoid counting hits on them as false positives (precision), but they are not recall targets.
4. **N** (`n_sites`) = number of **scored** sites of the protein, per the PocketWatch README ("top-N,
   where N is that protein's number of scored pockets, and the more forgiving top-(N+2)"). Masked sites
   do **not** count toward N. The {stats['structures'] - stats['structures_with_scored']} proteins whose sites are all masked have `n_sites = 0` (no recall
   target; the paper's 1,279-chain basis).
5. **Centre** (`center_x/y/z`) = their localization ground truth: the ligand centroid in the fold
   frame, rebuilt with the `verify_bundle.py` recipe — per member observation, Kabsch (reflection-
   corrected) of its entry-frame contact-residue CAs onto the fold CAs at the same canonical positions
   (positions outside 1..L / without CA dropped; >= 3 pairs required), applied to the entry-frame ligand
   centroid; site centre = mean over members. Verified against the shipped `site_centroids.parquet`.
   Note PocketWatch's own hit criterion is DCC <= 4 A to this centroid; PREREG O3 uses 12 A.
   `center_ca_x/y/z` (metadata) = CA centroid of `residues` on the fold (LIGYSIS-style), for sensitivity.
6. `ligand_atoms` is empty: the release ships per-observation ligand centroids only, no heavy-atom
   coordinates, so DCA cannot be computed from this release alone.
"""
    (out / "README.md").write_text(readme)
    print(f"PocketWatch {args.split}: {stats['structures']} structures ({stats['structures_with_scored']} with scored sites), "
          f"{stats['sites']} sites, {stats['scored_sites']} scored, {stats['masked_sites']} masked; "
          f"max centroid dev vs shipped {stats['max_centre_dev_vs_shipped']:.2e} A -> {out / 'sites.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
