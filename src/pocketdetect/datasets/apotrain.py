"""Reader for the apo counterparts of the training/dev entries (Track A2; docs/TRAINSET.md §12).

Layout::

    data/processed/apo_train/{train,dev}/manifest.csv
    data/processed/apo_train/{train,dev}/entries/<parent_entry_id>__apo_<pdb>_<chain>.npz

The npz keys mirror the train-set entries, so :class:`pocketdetect.datasets.trainset.TrainEntry`
methods (``ligand_atoms``, ``site_table``, ``site_residues``, ``mask_residues``) and
``trainset.load_input`` work unchanged. Coordinates are in the frame of the apo chain's deposited
mmCIF (``meta['structure_path']``). Extra keys: ``site_parent_index``, ``site_local_rmsd``,
``site_ca_rmsd``, ``site_pocket_rmsd``, ``site_cryptic_like`` (pocket RMSD >= 2 A), ``site_npairs``,
``site_coverage``, ``site_clash_frac``, ``lig_parent_index``, ``lig_parent_rmsd``,
``lig_atom_min_dist`` (per ligand atom, distance to the apo chain; atoms are NOT clash-trimmed).
``mask_kinds`` additionally contains ``untransferred_site`` (parent sites that could not be
transferred; their residues should not be used as negatives).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .trainset import ROOT, TrainEntry, load_entry, load_input

__all__ = ["read_apo_manifest", "load_apo_entry", "load_input"]

APO = ROOT / "data" / "processed" / "apo_train"


def read_apo_manifest(split: str, root: Path = APO) -> pd.DataFrame:
    m = pd.read_csv(Path(root) / split / "manifest.csv", dtype={"chain": str, "pdb": str}, keep_default_na=False)
    m["npz"] = [str(ROOT / p) if not Path(p).is_absolute() else p for p in m.npz]
    return m.reset_index(drop=True)


def load_apo_entry(path) -> TrainEntry:
    return load_entry(path)
