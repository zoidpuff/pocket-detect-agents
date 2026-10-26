#!/usr/bin/env python
"""AF-dev evaluation inputs (Track C2 AF models of Track A dev entries; never the AF test set).

Writes
  results/dev_af/af_dev_sites_all.tsv     site table in the AF frame (ranker `all`: every transferred
                                           site, centre = centroid of transferred ligand atoms, as
                                           Track A's dev ``site_center``), structure_path = AF model
  results/dev_af/pdb_dev_sites_all.tsv    the PDB-dev site table restricted to the same entries
                                           (same proteins, PDB chain input; pairing)
  results/dev_af/af_dev_entries.csv       per entry: pLDDT stats, pruned residues, sites lost to pruning
  results/dev_af/af_dev_site_annot.csv    per site: pLDDT, fraction of residues pruned / cropped
  (--split train: af_train_* files only; used for the supplementary out-of-fold geometry ablation)

Structure id = Track A entry_id (same ids as the PDB dev set).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from build_af_train_io import load_af_entry, read_af_manifest  # noqa: E402
from pocketdetect.burrow.af import residue_plddt  # noqa: E402
from pocketdetect.burrow.af_variants import CONFIGS, keep_mask  # noqa: E402
from pocketdetect.sites import format_atoms, read_sites  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

OUT = ROOT / "results/dev_af"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "train"], default="dev",
                    help="train: AF train site table for the supplementary out-of-fold geometry ablation")
    split = ap.parse_args().split
    OUT.mkdir(parents=True, exist_ok=True)
    m = read_af_manifest(split)
    rows, ent, ann = [], [], []
    for eid, npz, grp in zip(m.entry_id, m.npz, m.cluster30):
        e = load_af_entry(npz)
        path = e.meta["model_path"]
        s = load_structure(path, structure_id=eid, protein_only=True)
        s = s.select(s.protein_heavy_mask())
        rid = s.residue_ids
        pl = residue_plddt(s)
        keep_af = keep_mask(s, CONFIGS["prune"])
        keep70 = keep_mask(s, CONFIGS["crop70"])
        pruned = set(rid[~keep_af].tolist())
        cropped = set(rid[~keep70].tolist())
        xyz, site = e.ligand_atoms(ions=True)
        st = e.site_table()
        ch = e.chain
        lost = 0
        for i, r in st.iterrows():
            res = [f"{ch}:{x}" for x in r.residues]
            atoms = xyz[site == i]
            rows.append(dict(structure_id=eid, site_id=f"s{i}", center_x=r.center[0], center_y=r.center[1],
                             center_z=r.center[2], residues=" ".join(sorted(res)), n_sites=len(st), scored=1,
                             ligand_atoms=format_atoms(atoms), n_residues_structure=s.n_residues, group=grp,
                             structure_path=path, is_ion=int(r.is_ion), site_plddt=float(r.plddt_mean),
                             site_rmsd=float(r.rmsd)))
            fp = np.mean([x in pruned for x in res]) if res else 0.0
            fc = np.mean([x in cropped for x in res]) if res else 0.0
            lost += fp >= 0.5
            ann.append(dict(structure_id=eid, site_id=f"s{i}", plddt_mean=float(r.plddt_mean),
                            plddt_min=float(r.plddt_min), is_ion=int(r.is_ion), frac_pruned=fp, frac_cropped70=fc,
                            any_pruned=int(fp > 0), lost_pruned=int(fp >= 0.5), all_pruned=int(fp == 1.0),
                            lost_crop70=int(fc >= 0.5)))
        ent.append(dict(structure_id=eid, group=grp, model_path=path, n_res=s.n_residues, n_sites=len(st),
                        plddt_mean=float(pl.mean()), frac_lt50=float((pl < 50).mean()), frac_lt70=float((pl < 70).mean()),
                        frac_ge90=float((pl >= 90).mean()), n_pruned=len(pruned), n_cropped70=len(cropped),
                        sites_lost_pruned=int(lost)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"af_{split}_sites_all.tsv", sep="\t", index=False)
    pd.DataFrame(ent).to_csv(OUT / f"af_{split}_entries.csv", index=False)
    a = pd.DataFrame(ann)
    a.to_csv(OUT / f"af_{split}_site_annot.csv", index=False)
    if split == "train":
        print(f"AF train: {df.structure_id.nunique()} structures, {len(df)} sites; lost to pruning {a.lost_pruned.sum()}")
        return
    pdb = read_sites(ROOT / "results/dev/biolip/dev_sites_all.tsv")
    pdb = pdb[pdb.structure_id.isin(set(m.entry_id))]
    pdb.to_csv(OUT / "pdb_dev_sites_all.tsv", sep="\t", index=False)
    print(f"AF dev: {df.structure_id.nunique()} structures, {len(df)} sites ({int(df.is_ion.sum())} ion); "
          f"PDB-dev same entries: {pdb.structure_id.nunique()} structures, {len(pdb)} sites")
    print(f"sites with any residue pruned {a.any_pruned.sum()}, >=50% pruned {a.lost_pruned.sum()}, all pruned "
          f"{a.all_pruned.sum()}; >=50% cropped(pLDDT<70) {a.lost_crop70.sum()}")
    e = pd.DataFrame(ent)
    print(e[["n_res", "plddt_mean", "frac_lt50", "frac_lt70", "n_pruned"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
