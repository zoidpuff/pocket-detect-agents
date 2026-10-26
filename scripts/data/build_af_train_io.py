"""Reader for the AF-model train/dev labels (Track C2), mirroring
``pocketdetect.datasets.trainset`` (read_manifest / load_entry / load_input).

    import sys; sys.path.insert(0, "scripts/data")
    from build_af_train_io import read_af_manifest, load_af_entry, load_af_input
    man = read_af_manifest("train")              # status == ok only, same split as Track A
    e = load_af_entry(man.npz.iloc[0])
    s = load_af_input(e)                         # AF model, protein heavy atoms (B-factor = pLDDT)
    xyz, site = e.ligand_atoms(ions=True)        # transferred pooled ligand atoms, AF frame
    e.site_table()                               # residues (AF numbering = UniProt), RMSD, identity, pLDDT
    e.plddt                                      # per residue, aligned with e.res_nums

Residue keys are plain residue numbers of AF chain A (== UniProt positions), i.e. the same
"num" convention as Track A's chain sites (``chain:num`` = ``A:<num>``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AFT = ROOT / "data" / "processed" / "af_train"

__all__ = ["AFTrainEntry", "read_af_manifest", "load_af_entry", "load_af_input"]


@dataclass
class AFTrainEntry:
    meta: dict
    arrays: dict

    @property
    def entry_id(self) -> str:
        return self.meta["entry_id"]

    @property
    def chain(self) -> str:
        return self.meta["af_chain"]

    @property
    def n_sites(self) -> int:
        return len(self.arrays["site_residues"])

    @property
    def plddt(self) -> np.ndarray:
        return self.arrays["plddt"]

    @property
    def res_nums(self) -> np.ndarray:
        return self.arrays["af_res_nums"]

    def site_residues(self) -> list[list[str]]:
        return [s.split(";") if s else [] for s in self.arrays["site_residues"]]

    def ligand_atoms(self, ions: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(coords (N,3) float32 in the AF frame, site index (N,) int32); ``ions=False`` drops ions."""
        a = self.arrays
        n_per = np.diff(a["lig_ptr"])
        keep = np.repeat(np.ones(len(a["lig_site"]), bool) if ions else ~a["lig_is_ion"], n_per)
        return a["lig_coords"][keep], np.repeat(a["lig_site"], n_per)[keep]

    def site_table(self) -> pd.DataFrame:
        a = self.arrays
        return pd.DataFrame(dict(
            residues=self.site_residues(), src_site=a["site_src_index"], is_ion=a["site_is_ion"],
            rmsd=a["site_rmsd"], identity=a["site_identity"], frac_mapped=a["site_frac_mapped"],
            n_pairs=a["site_n_pairs"], sup_set=a["site_sup_set"], plddt_mean=a["site_plddt_mean"],
            plddt_min=a["site_plddt_min"], center=list(a["site_center"])))


def read_af_manifest(split: str, root: Path = AFT, ok_only: bool = True) -> pd.DataFrame:
    m = pd.read_csv(Path(root) / split / "manifest.csv", dtype={"chain": str, "pdb": str, "uniprot": str, "acc": str},
                    keep_default_na=False)
    if ok_only:
        m = m[m.status == "ok"]
    m = m.copy()
    m["npz"] = [str(ROOT / p) if p and not Path(p).is_absolute() else p for p in m.get("npz", [""] * len(m))]
    m["model_path_abs"] = [str(Path(root) / p) if p else "" for p in m.get("model_path", [""] * len(m))]
    return m.reset_index(drop=True)


def load_af_entry(path: str | Path) -> AFTrainEntry:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta_json")))
    return AFTrainEntry(meta=meta, arrays=arrays)


def load_af_input(entry: AFTrainEntry):
    """AF model input: protein heavy atoms of the (single-chain) AF model; B-factor holds pLDDT."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from pocketdetect.structure import load_structure

    return load_structure(entry.meta["model_path"], protein_only=True)
