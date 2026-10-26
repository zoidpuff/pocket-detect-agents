#!/usr/bin/env python
"""AF-mode test set (PREREG O5) → site tables (pocketdetect.sites) for scripts/run_test_eval.py.

  python -I scripts/build_gt_af.py  ->  data/processed/gt/af_set/sites.tsv          (primary)
                                        data/processed/gt/af_set_allsites/sites.tsv (sensitivity)

Source: Track C per-protein ground truth data/processed/af/gt/AF-<ACC>-F1.json.gz (built by
scripts/data/build_af_set.py; docs/AF_BENCHMARK.md). One structure = one full-length AFDB v6
model (chain A, residue number = UniProt position).

* primary (``af_set``): the 2,527 UniProts carrying LBS-comparison benchmark sites; sites = their
  included LBS-benchmark sites (6,731). As in the LIGYSIS protocol, N = all these sites of the
  model and ``scored`` = non-redundant subset (pocketdetect.ligysis.filter_redundant: residue
  Jaccard ≥ 0.75 or centre distance ≤ 5 Å, larger site kept) — on a full-length model sites of
  different LIGYSIS segments / representative chains can coincide.
* sensitivity (``af_set_allsites``): every included LIGYSIS site of those UniProts.
* centre = CA centroid of the site residues on the model (LIGYSIS definition, Track C ``center_ca``).
* ligand_atoms = transferred ligand heavy atoms with local CA RMSD ≤ 2 Å (Track C); empty if none
  (the site is then excluded from DCA metrics only).
* residues ``A:<UniProt position>``; ``plddt_mean`` and ``plddt_band`` of the site kept as strata.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.ligysis import filter_redundant  # noqa: E402
from pocketdetect.sites import format_atoms, write_sites  # noqa: E402

AF = ROOT / "data/processed/af"
LIG_RMSD_MAX = 2.0


def rows_for(gt: dict, benchmark_only: bool) -> list[dict]:
    out = []
    for s in gt["sites"]:
        if not s["include"] or (benchmark_only and not s["in_lbs_benchmark"]):
            continue
        ligs = [l for l in s["ligands"] if l["rmsd"] <= LIG_RMSD_MAX]
        atoms = np.vstack([np.asarray(l["coords"], float).reshape(-1, 3) for l in ligs]) if ligs else np.zeros((0, 3))
        c = s["center_ca"]
        out.append({
            "structure_id": gt["structure_id"], "structure_path": str(AF / gt["model_path"]), "input_chains": gt["chain"],
            "site_id": s["site_name"], "center_x": c[0], "center_y": c[1], "center_z": c[2],
            "residues": " ".join(f"{gt['chain']}:{u}" for u in s["residues"]),
            "ligand_atoms": format_atoms(atoms) if len(atoms) else "", "n_residues_structure": len(gt["residue_numbers"]),
            "uniprot": gt["acc"], "in_lbs_benchmark": s["in_lbs_benchmark"], "ion_prop": s["ion_prop"],
            "plddt_mean": s.get("plddt_mean"), "n_res": len(s["residues"]),
            "n_ligands_transferred": len(ligs),
        })
    return out


def finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["plddt_band"] = pd.cut(df.plddt_mean.astype(float), [-1, 50, 70, 90, 101], labels=["<50", "50-70", "70-90", ">=90"]).astype(str)
    tmp = df.assign(origin="X", rep_chain=df.structure_id, ID=np.arange(len(df)),
                    up_aas=[[int(t.split(":")[1]) for t in r.split()] for r in df.residues],
                    centre=list(zip(df.center_x, df.center_y, df.center_z)))
    nr = filter_redundant(tmp, score_col="n_res")
    df["scored"] = df.index.isin(nr.index)
    df["n_sites"] = df.groupby("structure_id").site_id.transform("size").astype(int)
    return df


def main():
    man = pd.read_csv(AF / "af_manifest.csv")
    man = man[(man.status == "included") & (man.in_lbs_benchmark)]
    prim, alls = [], []
    for p in man.gt_path:
        with gzip.open(AF / p, "rt") as f:
            gt = json.load(f)
        prim += rows_for(gt, True)
        alls += rows_for(gt, False)
    for name, rows in (("af_set", prim), ("af_set_allsites", alls)):
        df = finish(rows)
        out = ROOT / "data/processed/gt" / name
        out.mkdir(parents=True, exist_ok=True)
        write_sites(df, out / "sites.tsv")
        summ = {"n_structures": int(df.structure_id.nunique()), "n_sites": len(df), "n_scored": int(df.scored.sum()),
                "n_scored_with_ligand_atoms": int(((df.ligand_atoms != "") & df.scored).sum()),
                "plddt_band_scored": df[df.scored].plddt_band.value_counts().to_dict()}
        (out / "README.md").write_text(f"# {name} site table (PREREG O5)\n\nBuilt by `scripts/build_gt_af.py` from Track C "
                                       f"`data/processed/af/gt/*.json.gz` (definitions in the script docstring).\n\n```\n{json.dumps(summ, indent=2)}\n```\n")
        print(name, summ)


if __name__ == "__main__":
    main()
