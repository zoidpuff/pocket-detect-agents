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


# --------------------------------------------------------------------------- v2: extra training inputs
AF = "__af"     # AlphaFold model of a training protein (data/processed/af_train), spaghetti-pruned
SCD = "__scd"   # side-chain-dropout copy of the representative chain (apo-like augmentation)
SCD_NEAR, SCD_FAR, SCD_NEAR_DIST = 0.5, 0.2, 8.0
_BB_CB = {"N", "CA", "C", "O", "OXT", "CB"}


APO = "__apo_"  # apo PDB chain of a training protein (data/processed/apo_train; id = <parent>__apo_<pdb>_<chain>)


def base_id(sid: str) -> str:
    if APO in sid:
        return sid.split(APO)[0]
    for suf in (ASM, AF, SCD):
        if sid.endswith(suf):
            return sid[: -len(suf)]
    return sid


def af_manifest(split: str = "train") -> pd.DataFrame:
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "data"))
    from build_af_train_io import read_af_manifest
    return read_af_manifest(split)


def load_af(npz):
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "data"))
    from build_af_train_io import load_af_entry
    return load_af_entry(npz)


def load_protein_af(eaf) -> Protein:
    """AF model, polymer heavy atoms, low-pLDDT spaghetti pruned exactly as --af-mode (af.prune_mask);
    no soft-occlusion weights and pLDDT never used as a feature."""
    from .af import prune_mask
    s = load_structure(eaf.meta["model_path"], structure_id=eaf.entry_id + AF, protein_only=True)
    s = s.select(s.protein_heavy_mask())
    keep = prune_mask(s)
    return protein_from_structure(s.select(keep[s.res_index]))


def load_protein_scd(e: TrainEntry, seed: int) -> Protein:
    """Representative chain with side chains beyond CB deleted for a random subset of residues
    (prob SCD_NEAR within SCD_NEAR_DIST Å of a pooled ligand atom, SCD_FAR elsewhere)."""
    from scipy.spatial import cKDTree
    s = load_structure(e.meta["structure_path"], structure_id=e.entry_id + SCD, chains=[e.chain], protein_only=True)
    s = s.select(s.protein_heavy_mask())
    xyz, _, _ = ligand_atoms(e, False)
    near = np.zeros(s.n_residues, bool)
    if len(xyz):
        d, _ = cKDTree(xyz).query(s.coords, k=1)
        np.logical_or.at(near, s.res_index, d <= SCD_NEAR_DIST)
    import hashlib
    h = int(hashlib.sha256(e.entry_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([seed, h])
    drop = rng.random(s.n_residues) < np.where(near, SCD_NEAR, SCD_FAR)
    keep = ~(drop[s.res_index] & ~np.isin(s.atom_name.astype(str), list(_BB_CB)))
    return protein_from_structure(s.select(keep))


def af_sites(eaf, variant: str = "all"):
    """Site dicts (as :func:`sites`) for an AF training entry (residues 'A:<uniprot num>')."""
    a = eaf.arrays
    n = np.diff(a["lig_ptr"])
    xyz = a["lig_coords"].astype(np.float64)
    site = np.repeat(a["lig_site"], n)
    ion = np.repeat(a["lig_is_ion"], n)
    res = [r.split(";") if r else [] for r in a["site_residues"]]
    out = []
    for i in range(len(res)):
        is_ion = bool(a["site_is_ion"][i])
        if variant == "small" and is_ion:
            continue
        m = site == i
        if variant == "small":
            m &= ~ion
        if not m.any():
            continue
        c = a["site_center"][i].astype(np.float64) if variant == "all" else xyz[m].mean(0)
        out.append(dict(site_id=f"s{i}", center=c, residues=frozenset(f"A:{r}" for r in res[i]), atoms=xyz[m],
                        is_ion=is_ion))
    return out
