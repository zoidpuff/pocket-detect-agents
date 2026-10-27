#!/usr/bin/env python
"""DESCRIPTIVE only: holo counterparts of the CryptoBench TEST apo sites (222 apo structures / 270 sites). Does not modify cryptobench_test GT.

Writes data/processed/gt/cryptobench_test_holo/:
  sites_holo.tsv  one row per paired site: structure_id = APO structure id (so apo and holo rows pair on (structure_id, site_id)), structure_path = holo mmCIF
                  (inputs restricted to the holo chain(s)), centre = CA centroid of holo_pocket_selection on the holo, ligand_atoms = real holo ligand atoms
                  of the member entries sharing that holo structure.
  sites_apo.tsv   the same paired sites on the apo structure (apo centre/residues as in cryptobench_test) with ligand_atoms = holo ligands transferred
                  into the apo frame (CA Kabsch on pocket residues, RMSD <= 4 A) -> apo DCA4.
  Pairing: per apo the main holo (is_main_holo_structure, else first entry); an apo site is paired if some member entry uses that holo structure.
  N (top-N) = number of paired sites of the structure, identically in both tables.
  inputs/{apo,holo}/<apo structure id>.pdb|cif chain-restricted structure files.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("tf", ROOT / "scripts/build_gt_cryptobench_trainfolds.py")
tf = importlib.util.module_from_spec(spec)
sys.modules["tf"] = tf
spec.loader.exec_module(tf)
from pocketdetect.importers import protein_residue_count, residue_centroids  # noqa: E402
from pocketdetect.importers.cryptobench import load_split, parse_selection_residue  # noqa: E402
from pocketdetect.sites import read_sites, write_sites  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

OUT = ROOT / "data/processed/gt/cryptobench_test_holo"


def write_chain_restricted(path, chains, dst_noext):
    st = gemmi.read_structure(str(path))
    keep = set(chains)
    m = st[0]
    for i in range(len(m) - 1, -1, -1):
        if m[i].name not in keep:
            del m[i]
    if all(len(c.name) == 1 for c in m):
        st.write_pdb(str(dst_noext) + ".pdb")
    else:
        st.make_mmcif_document().write_file(str(dst_noext) + ".cif")


def main():
    apo_sites = read_sites(ROOT / "data/processed/gt/cryptobench_test/sites.tsv")
    test = load_split(tf.DS, "test")
    tab = pd.read_csv(ROOT / "data/processed/gt/cryptobench_test/sites.tsv", sep="\t", dtype=str, keep_default_na=False)
    for d in ("apo", "holo"):
        (OUT / "inputs" / d).mkdir(parents=True, exist_ok=True)
    hrows, arows = [], []
    for sid, g in tab.groupby("structure_id", sort=False):
        apo = g.apo_pdb_id.iloc[0]
        holos = test[apo]
        main = next((h for h in holos if h.get("is_main_holo_structure")), holos[0])
        hid, hch = main["holo_pdb_id"], main["holo_chain"].split("-")
        paired = []
        for r in g.itertuples():
            pset = set(r.residues.split())
            mem = [h for h in holos if {parse_selection_residue(t) for t in h["apo_pocket_selection"]} <= pset
                   and h["holo_pdb_id"] == hid and h["holo_chain"] == main["holo_chain"]]
            if mem:
                paired.append((r, mem))
        if not paired:
            continue
        try:
            hs = load_structure(tf.CIF / f"{hid}.cif", structure_id=sid, chains=hch)
        except Exception:
            continue
        for fn in ("apo", "holo"):
            pass
        write_chain_restricted(tf.CIF / f"{apo}.cif", g.input_chains.iloc[0].split("-"), OUT / "inputs/apo" / sid)
        write_chain_restricted(tf.CIF / f"{hid}.cif", hch, OUT / "inputs/holo" / sid)
        nres_h = protein_residue_count(hs)
        for r, mem in paired:
            hres = sorted({parse_selection_residue(t) for h in mem for t in h["holo_pocket_selection"]})
            c = residue_centroids(hs, hres)
            if np.isnan(c["center_ca"]).any():
                continue
            real = [tf.ligand_xyz(h["holo_pdb_id"], h["ligand"], h["ligand_chain"], h["ligand_index"]) for h in mem]
            real = [x for x in real if len(x)]
            tr = []
            for h in mem:
                x, st_ = tf.transfer(dict(h, _apo=apo))
                if x is not None:
                    tr.append(x)
            hrows.append(dict(structure_id=sid, structure_path=str(tf.CIF / f"{hid}.cif"), site_id=r.site_id, center_x=c["center_ca"][0], center_y=c["center_ca"][1],
                              center_z=c["center_ca"][2], residues=" ".join(hres), scored=1, input_chains="-".join(hch), apo_site_residues=r.residues,
                              ligand_atoms=tf.fmt_atoms(np.concatenate(real)) if real else "", n_residues_structure=nres_h, holo_pdb_id=hid))
            arows.append(dict(structure_id=sid, structure_path=r.structure_path, site_id=r.site_id, center_x=r.center_x, center_y=r.center_y, center_z=r.center_z,
                              residues=r.residues, scored=1, input_chains=r.input_chains, ligand_atoms=tf.fmt_atoms(np.concatenate(tr)) if tr else "",
                              n_residues_structure=r.n_residues_structure, holo_pdb_id=hid))
    H, A = pd.DataFrame(hrows), pd.DataFrame(arows)
    for D in (H, A):
        D["n_sites"] = D.groupby("structure_id")["site_id"].transform("size")
    write_sites(H, OUT / "sites_holo.tsv")
    write_sites(A, OUT / "sites_apo.tsv")
    n_lig_a = int((A.ligand_atoms != "").sum())
    (OUT / "README.md").write_text(f"# CryptoBench TEST holo counterparts (descriptive only)\n\nBuilt by scripts/build_gt_cryptobench_test_holo.py. {H.structure_id.nunique()} apo structures / {len(H)} paired sites "
                                   f"(of 222 / 270 official). Apo-frame ligand transfer available for {n_lig_a}/{len(A)} sites. See the script docstring for rules.\n")
    print(len(H), H.structure_id.nunique(), n_lig_a)


if __name__ == "__main__":
    main()
