#!/usr/bin/env python
"""DEV-ONLY ground truth from the CryptoBench TRAIN folds (885 apo structures; never the test split).

Writes into data/processed/gt/cryptobench_trainfolds/:
  sites.tsv            cryptic sites on the apo structures, same construction as scripts/build_gt_cryptobench.py
                       (pocketdetect.importers.cryptobench.build_sites, folds train-fold-0..3 concatenated; + column `fold`),
                       plus `ligand_atoms` = heavy atoms of the member holo ligands transferred into the apo frame by a
                       Kabsch fit of the holo CA atoms onto the paired apo CA atoms of the pocket residues
                       (apo_pocket_selection[i] <-> holo_pocket_selection[i]); entries with <3 pairs or fit RMSD > 4 A are not transferred.
  sites_noncryptic.tsv CryptoBench's auxiliary non-cryptic pockets of the same apo structures (same merge rule), no ligand atoms.
  sites_holo.tsv       the holo counterpart of every apo (main holo entry; holo chain(s) only): sites = holo_pocket_selection of the
                       entries that share that holo structure (merged), ligand_atoms = the real ligand atoms in the holo file.
  README.md
    .venv/bin/python scripts/build_gt_cryptobench_trainfolds.py
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.importers import abspath_nofollow, protein_residue_count, residue_centroids  # noqa: E402
from pocketdetect.importers.cryptobench import build_sites, input_chains, load_split, merge_pockets, parse_selection_residue  # noqa: E402
from pocketdetect.sites import write_sites  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

DS = ROOT / "data/raw/cryptobench/osf/cryptobench/cryptobench-dataset"
CIF = ROOT / "data/raw/cryptobench/cif-files/cif-files"
OUT = ROOT / "data/processed/gt/cryptobench_trainfolds"
_cache: dict = {}


def gem(pdb):
    if pdb not in _cache:
        _cache[pdb] = gemmi.read_structure(str(CIF / f"{pdb}.cif"))
    return _cache[pdb]


def ca_map(pdb):
    out = {}
    for ch in gem(pdb)[0]:
        for r in ch:
            a = r.find_atom("CA", "*")
            if a is not None:
                out[f"{ch.name}_{r.seqid.num}"] = np.array([a.pos.x, a.pos.y, a.pos.z])
    return out


def ligand_xyz(pdb, name, chain, index):
    st = gem(pdb)
    cand = [r for ch in st[0] if ch.name == chain for r in ch if r.name == name and str(r.seqid.num) == str(index)]
    if not cand:
        cand = [r for ch in st[0] if ch.name == chain for r in ch if r.name == name]
    if not cand:
        return np.zeros((0, 3))
    r = cand[0]
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|; returns (R, t, rmsd)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = qc - R @ pc
    rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def fmt_atoms(x):
    return " ".join(f"{a:.3f},{b:.3f},{c:.3f}" for a, b, c in x)


def transfer(h):
    try:
        apo_ca = ca_map(h["_apo"])
        holo_ca = ca_map(h["holo_pdb_id"])
        pairs = [(a, b) for a, b in zip(h["apo_pocket_selection"], h["holo_pocket_selection"]) if a in apo_ca and b in holo_ca]
        if len(pairs) < 3:
            return None, "few_pairs"
        P = np.array([holo_ca[b] for _, b in pairs])
        Q = np.array([apo_ca[a] for a, _ in pairs])
        R, t, rmsd = kabsch(P, Q)
        if rmsd > 4.0:
            return None, f"rmsd{rmsd:.1f}"
        lig = ligand_xyz(h["holo_pdb_id"], h["ligand"], h["ligand_chain"], h["ligand_index"])
        if not len(lig):
            return None, "no_ligand"
        return lig @ R.T + t, "ok"
    except Exception as e:  # noqa
        return None, f"err:{e}"[:40]


def main():
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    data = OrderedDict()
    folds = {}
    for k in range(4):
        d = load_split(DS, f"train-fold-{k}")
        for a in d:
            assert a not in data
            data[a] = d[a]
            folds[a] = k
    test = load_split(DS, "test")
    assert not (set(data) & set(test)), "train/test overlap"
    # NOTE train folds contain apos whose holo entries use different apo chains (union of chains is used as input; test split obeys the strict rule)
    # ---- cryptic sites (apo)
    frames, stats = [], {}
    for k in range(4):
        df, st = build_sites(DS, CIF, f"train-fold-{k}", strict_chains=False)
        df["fold"] = k
        frames.append(df)
        for m, v in st.items():
            stats[m] = stats.get(m, 0) + v
    sites = pd.concat(frames, ignore_index=True)
    tr_status = {}
    lig = []
    for r in sites.itertuples():
        holos = data[r.apo_pdb_id]
        pset = set(r.residues.split())
        mem = [h for h in holos if {parse_selection_residue(t) for t in h["apo_pocket_selection"]} <= pset]
        atoms = []
        for h in mem:
            h = dict(h, _apo=r.apo_pdb_id)
            x, status = transfer(h)
            tr_status[status] = tr_status.get(status, 0) + 1
            if x is not None:
                atoms.append(x)
        lig.append(fmt_atoms(np.concatenate(atoms)) if atoms else "")
    sites["ligand_atoms"] = lig
    write_sites(sites, out / "sites.tsv")
    # ---- non-cryptic (apo)
    nc = json.load(open(DS / "auxiliary-data/non-cryptic-pockets/noncryptic-pockets.json"))
    rows = []
    for apo, holos in data.items():
        entries = nc.get(apo, [])
        if not entries:
            continue
        chains = set(input_chains(holos, strict=False))
        entries = [e for e in entries if set(e["apo_chain"].split("-")) <= chains]
        if not entries:
            continue
        sid = f"{apo}_{'-'.join(input_chains(holos, strict=False))}"
        s = load_structure(CIF / f"{apo}.cif", structure_id=sid, chains=sorted(chains))
        pockets = merge_pockets([e["apo_pocket_selection"] for e in entries])
        for j, p in enumerate(pockets, 1):
            rids = [parse_selection_residue(t) for t in p]
            c = residue_centroids(s, rids)
            if np.isnan(c["center_ca"]).any():
                continue
            rows.append(dict(structure_id=sid, structure_path=abspath_nofollow(CIF / f"{apo}.cif"), site_id=f"nc{j}",
                             center_x=c["center_ca"][0], center_y=c["center_ca"][1], center_z=c["center_ca"][2],
                             residues=" ".join(rids), n_sites=len(pockets), scored=1, input_chains="-".join(input_chains(holos, strict=False)),
                             uniprot=entries[0]["uniprot_id"], fold=folds[apo], n_residues_structure=protein_residue_count(s)))
    nc_df = pd.DataFrame(rows)
    write_sites(nc_df, out / "sites_noncryptic.tsv")
    # ---- holo counterpart
    hrows = []
    for apo, holos in data.items():
        main = next((h for h in holos if h.get("is_main_holo_structure")), holos[0])
        ents = [h for h in holos if h["holo_pdb_id"] == main["holo_pdb_id"] and h["holo_chain"] == main["holo_chain"]]
        hchains = main["holo_chain"].split("-")
        sid = f"{main['holo_pdb_id']}_{main['holo_chain']}"
        try:
            s = load_structure(CIF / f"{main['holo_pdb_id']}.cif", structure_id=sid, chains=hchains)
        except Exception:
            continue
        pockets = merge_pockets([e["holo_pocket_selection"] for e in ents])
        for j, p in enumerate(pockets, 1):
            rids = [parse_selection_residue(t) for t in p]
            c = residue_centroids(s, rids)
            if np.isnan(c["center_ca"]).any():
                continue
            pset = set(rids)
            atoms = [ligand_xyz(e["holo_pdb_id"], e["ligand"], e["ligand_chain"], e["ligand_index"]) for e in ents
                     if {parse_selection_residue(t) for t in e["holo_pocket_selection"]} <= pset]
            atoms = [a for a in atoms if len(a)]
            hrows.append(dict(structure_id=sid, structure_path=abspath_nofollow(CIF / f"{main['holo_pdb_id']}.cif"), site_id=f"site{j}",
                              center_x=c["center_ca"][0], center_y=c["center_ca"][1], center_z=c["center_ca"][2],
                              residues=" ".join(rids), n_sites=len(pockets), scored=1, input_chains="-".join(hchains),
                              ligand_atoms=fmt_atoms(np.concatenate(atoms)) if atoms else "", apo_pdb_id=apo,
                              uniprot=main["uniprot_id"], fold=folds[apo], n_residues_structure=protein_residue_count(s)))
    h_df = pd.DataFrame(hrows)
    h_df["n_sites"] = h_df.groupby("structure_id")["site_id"].transform("size")
    write_sites(h_df, out / "sites_holo.tsv")
    n_lig = int((sites.ligand_atoms != "").sum())
    readme = f"""# CryptoBench TRAIN-fold ground truth (DEV ONLY; the test split is never touched)

Built by `scripts/build_gt_cryptobench_trainfolds.py`.
- `sites.tsv`: {sites.structure_id.nunique()} apo structures, {len(sites)} cryptic sites (merge rule of scripts/build_gt_cryptobench.py), folds 0-3 (column `fold`).
  `ligand_atoms` (holo ligand heavy atoms transferred to the apo frame by a CA Kabsch fit on the pocket residues) available for {n_lig}/{len(sites)} sites; transfer status counts over holo entries: {tr_status}.
- `sites_noncryptic.tsv`: {nc_df.structure_id.nunique()} apo structures, {len(nc_df)} auxiliary non-cryptic pockets (CryptoBench `noncryptic-pockets.json`, same merge rule).
- `sites_holo.tsv`: {h_df.structure_id.nunique()} holo structures (main holo entry of each apo), {len(h_df)} sites, real ligand atoms.
Counts: {stats}. Test-split overlap asserted empty. Resampling units for dev bootstraps: UniProt (column `uniprot`).
"""
    (out / "README.md").write_text(readme)
    print(readme)


if __name__ == "__main__":
    main()
