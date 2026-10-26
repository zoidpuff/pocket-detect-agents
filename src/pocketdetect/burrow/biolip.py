"""BURROW adapters for Track A's LIGYSIS-style BioLiP2 train/dev sets (docs/TRAINSET.md).

Structure id = ``entry_id`` (chain input) or ``entry_id + "__asm1"`` (assembly-1 input).
Label variants (DESIGN §2.4.4):
  ``all``   every pooled site; point label: <= 4 Å of a non-ion ligand atom or <= 3 Å of an ion
  ``small`` sites that are not ion-only; ion atoms ignored (point label <= 4 Å of non-ion atoms)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.trainset import TrainEntry, load_entry, read_manifest
from ..sites import format_atoms
from ..structure import load_structure
from .prep import Protein, protein_from_structure

ASM = "__asm1"


def entries(split: str) -> pd.DataFrame:
    """Track A manifest; if env ``BURROW_TRAINSET`` points to a snapshot made by
    scripts/burrow_relabel_biolip.py, read manifest + entries from there instead."""
    import os
    snap = os.environ.get("BURROW_TRAINSET")
    if not snap:
        return read_manifest(split)
    m = read_manifest(split, root=Path(snap))
    m["npz"] = [str(Path(snap) / split / "entries" / Path(p).name) for p in m.npz]
    return m


def structure_ids(m: pd.DataFrame, assembly: bool) -> list[tuple[str, str, bool]]:
    """(structure_id, npz, is_assembly) for chain inputs (+ stored assembly inputs)."""
    out = [(e, n, False) for e, n in zip(m.entry_id, m.npz)]
    if assembly:
        out += [(e + ASM, n, True) for e, n, s in zip(m.entry_id, m.npz, m.assembly_status) if s == "stored"]
    return out


def load_protein_entry(e: TrainEntry, assembly: bool = False) -> Protein:
    if assembly:
        s = load_structure(str(e.arrays["asm_path"]), structure_id=e.entry_id + ASM, protein_only=True)
    else:
        s = load_structure(e.meta["structure_path"], structure_id=e.entry_id, chains=[e.chain], protein_only=True)
    return protein_from_structure(s)


def ligand_atoms(e: TrainEntry, assembly: bool = False):
    """(coords (N,3), site (N,), is_ion (N,)) of pooled ligand heavy atoms."""
    a = e.arrays
    p = "asm_" if assembly else ""
    ptr, site, ion = a[f"{p}lig_ptr"], a[f"{p}lig_site"], a[f"{p}lig_is_ion"]
    n = np.diff(ptr)
    return a[f"{p}lig_coords"].astype(np.float64), np.repeat(site, n), np.repeat(ion, n)


def sites(e: TrainEntry, variant: str = "all", assembly: bool = False):
    """List of dicts: site_id, center, residues (frozenset 'chain:num'), atoms (k,3), is_ion."""
    a = e.arrays
    p = "asm_" if assembly else ""
    xyz, site, ion = ligand_atoms(e, assembly)
    res = e.site_residues(p)
    out = []
    for i in range(len(res)):
        is_ion = bool(a[f"{p}site_is_ion"][i])
        if variant == "small" and is_ion:
            continue
        m = site == i
        if variant == "small":
            m &= ~ion
        if not m.any():
            continue
        if not assembly and "site_center" in a and variant == "all":
            c = a["site_center"][i].astype(np.float64)
        else:
            c = xyz[m].mean(0)
        rr = frozenset(r if assembly else f"{e.chain}:{r}" for r in res[i])
        out.append(dict(site_id=f"s{i}", center=c, residues=rr, atoms=xyz[m], is_ion=is_ion))
    return out


def write_site_table(m: pd.DataFrame, out: Path, variant: str, root_inputs: Path | None = None) -> pd.DataFrame:
    """pocketdetect.sites TSV for the dev set (chain inputs).  ``structure_path`` = the cleaned
    single-chain input file under ``root_inputs`` (same file for every method)."""
    rows = []
    for eid, npz, n_res, grp in zip(m.entry_id, m.npz, m.n_res, m.cluster30):
        e = load_entry(npz)
        if root_inputs is not None and (root_inputs / f"{eid}.cif").exists():  # residues actually in the input
            n_res = load_structure(root_inputs / f"{eid}.cif", protein_only=True).n_residues
        ss = sites(e, variant)
        for s in ss:
            rows.append(dict(structure_id=eid, site_id=s["site_id"], center_x=s["center"][0], center_y=s["center"][1],
                             center_z=s["center"][2], residues=" ".join(sorted(s["residues"])), n_sites=len(ss),
                             scored=1, ligand_atoms=format_atoms(s["atoms"]), n_residues_structure=int(n_res),
                             group=grp, structure_path=str(root_inputs / f"{eid}.cif") if root_inputs else "",
                             is_ion=int(s["is_ion"])))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return df


def write_chain_input(e: TrainEntry, out: Path) -> Path:
    """Polymer-only single-chain mmCIF file (identical input for BURROW and baselines)."""
    import gemmi
    st = gemmi.read_structure(e.meta["structure_path"])
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    model = st[0]
    for i in range(len(st) - 1, 0, -1):
        del st[i]
    for i in reversed(range(len(model))):   # keep only the representative chain (all its parts)
        if model[i].name != e.chain:
            del model[i]
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    out.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(out))
    return out
