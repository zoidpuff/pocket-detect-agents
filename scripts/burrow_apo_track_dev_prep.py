#!/usr/bin/env python
"""Extra apo guard for the v2 cycle: Track A2 apo chains of DEV proteins (data/processed/apo_train/dev).
Writes results/dev/biolip/apo_track_dev/{inputs/<id>.cif, sites_all.tsv, sites_small.tsv}
(sites = the parent's pooled sites transferred onto the apo chain; group = parent cluster30)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.biolip import sites, write_chain_input  # noqa: E402
from pocketdetect.datasets.apotrain import load_apo_entry, read_apo_manifest  # noqa: E402
from pocketdetect.sites import format_atoms  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

OUT = ROOT / "results/dev/biolip/apo_track_dev"


def main():
    m = read_apo_manifest("dev")
    dev = pd.read_csv(ROOT / "data/processed/dev/manifest.csv")
    grp = dict(zip(dev.entry_id, dev.cluster30))
    m = m[m.parent_entry_id.isin(grp)]
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "inputs" / ".gitignore").write_text("*\n")
    rows = {"all": [], "small": []}
    for eid, npz, par in zip(m.entry_id, m.npz, m.parent_entry_id):
        e = load_apo_entry(npz)
        f = OUT / "inputs" / f"{eid}.cif"
        if not f.exists():
            write_chain_input(e, f)
        nres = load_structure(f, protein_only=True).n_residues
        for v in ("all", "small"):
            ss = sites(e, v)
            for s in ss:
                rows[v].append(dict(structure_id=eid, site_id=s["site_id"], center_x=s["center"][0], center_y=s["center"][1],
                                    center_z=s["center"][2], residues=" ".join(sorted(s["residues"])), n_sites=len(ss), scored=1,
                                    ligand_atoms=format_atoms(s["atoms"]), n_residues_structure=nres, group=grp[par],
                                    structure_path=str(f), is_ion=int(s["is_ion"])))
    for v, r in rows.items():
        pd.DataFrame(r).to_csv(OUT / f"sites_{v}.tsv", sep="\t", index=False)
        print(v, len(r), "sites", len({x['structure_id'] for x in r}), "chains")


if __name__ == "__main__":
    main()
