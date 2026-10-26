"""Reader for the leakage-controlled LIGYSIS-style training/dev sets (docs/TRAINSET.md).

Layout::

    data/processed/{train,dev}/manifest.csv          one row per entry (representative chain)
    data/processed/{train,dev}/entries/<entry_id>.npz  pooled sites (no pickles; allow_pickle=False)

Coordinates are in the frame of the representative's deposited mmCIF (``structure_path``) or,
for ``asm_*`` arrays, of the PDB assembly-1 file (``asm_path``).

Example::

    from pocketdetect.datasets.trainset import read_manifest, load_entry, load_input
    man = read_manifest("train")
    e = load_entry(man.npz.iloc[0])
    s = load_input(e)                     # representative chain, polymer heavy atoms only
    xyz, site = e.ligand_atoms(ions=True)  # pooled ligand heavy atoms + site index per atom
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"

__all__ = ["TrainEntry", "read_manifest", "load_entry", "load_input"]


@dataclass
class TrainEntry:
    meta: dict
    arrays: dict

    @property
    def entry_id(self) -> str:
        return self.meta["entry_id"]

    @property
    def chain(self) -> str:
        return self.meta["chain"]

    @property
    def n_sites(self) -> int:
        return len(self.arrays["site_residues"])

    @property
    def has_assembly(self) -> bool:
        return "asm_lig_coords" in self.arrays

    def site_residues(self, prefix: str = "") -> list[list[str]]:
        """Residue keys per site. Chain sites: author 'num[icode]' of the representative chain;
        assembly sites (prefix='asm_'): 'chain:num[icode]' in the assembly file."""
        return [s.split(";") if s else [] for s in self.arrays[f"{prefix}site_residues"]]

    def ligand_atoms(self, ions: bool = True, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
        """(coords (N,3) float32, site index (N,) int32) of all pooled ligand heavy atoms (after
        clash filtering); ``ions=False`` drops ion ligands (single heavy atom)."""
        a = self.arrays
        ptr, site, is_ion = a[f"{prefix}lig_ptr"], a[f"{prefix}lig_site"], a[f"{prefix}lig_is_ion"]
        n_per = np.diff(ptr)
        keep_lig = np.ones(len(site), bool) if ions else ~is_ion
        keep_atom = np.repeat(keep_lig, n_per)
        return a[f"{prefix}lig_coords"][keep_atom], np.repeat(site, n_per)[keep_atom]

    def site_table(self, prefix: str = "") -> pd.DataFrame:
        a = self.arrays
        d = dict(residues=self.site_residues(prefix), is_ion=a[f"{prefix}site_is_ion"])
        if not prefix:
            d.update(has_ion=a["site_has_ion"], n_ligands=a["site_n_ligands"], n_sources=a["site_n_sources"],
                     center=list(a["site_center"]))
        else:
            d.update(chains=a["asm_site_chains"], is_interface=a["asm_site_is_interface"])
        return pd.DataFrame(d)

    def mask_residues(self) -> dict:
        """Representative residues binding polymer ligands (peptide / rna / dna) in any entry of
        the UniProt — NOT labels; use to optionally exclude them from negatives."""
        return {str(k): (str(v).split(";") if v else []) for k, v in zip(self.arrays["mask_kinds"], self.arrays["mask_residues"])}


def read_manifest(split: str, root: Path = PROC, ok_only: bool = True) -> pd.DataFrame:
    m = pd.read_csv(Path(root) / split / "manifest.csv", dtype={"chain": str, "pdb": str, "uniprot": str}, keep_default_na=False)
    if ok_only and "status" in m:
        m = m[m.status == "ok"]
    m["npz"] = [str(ROOT / p) if not Path(p).is_absolute() else p for p in m.npz]
    return m.reset_index(drop=True)


def load_entry(path: str | Path) -> TrainEntry:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta_json")))
    return TrainEntry(meta=meta, arrays=arrays)


def load_input(entry: TrainEntry, assembly: bool = False):
    """Model input per DESIGN §2.1: polymer (protein) heavy atoms only of the representative chain,
    or of the whole assembly-1 when ``assembly=True``."""
    from ..structure import load_structure

    if assembly:
        if not entry.has_assembly:
            raise ValueError(f"{entry.entry_id}: no assembly input stored")
        return load_structure(str(entry.arrays["asm_path"]), protein_only=True)
    return load_structure(entry.meta["structure_path"], chains=[entry.chain], protein_only=True)
