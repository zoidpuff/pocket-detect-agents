"""Ground-truth importers: convert external benchmark annotations into the common *site table*
(:mod:`pocketdetect.sites`).

* :mod:`pocketdetect.importers.cryptobench` — CryptoBench test split (cryptic sites on apo chains).
* :mod:`pocketdetect.importers.pocketwatch` — PocketWatch temporal test split (ESMFold2 folds).

Raw downloads are treated as untrusted data: only JSON / parquet / mmCIF are parsed, nothing from
the raw directories is imported or executed, and no pickles are read.

Shared helpers live here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from pocketdetect.structure import PROTEIN, Structure

__all__ = ["sha256_file", "residue_centroids", "protein_residue_count", "abspath_nofollow"]


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def abspath_nofollow(path: str | os.PathLike) -> str:
    """Absolute path without resolving symlinks (keeps the /cluster/project prefix stable)."""
    return os.path.abspath(os.fspath(path))


def protein_residue_count(s: Structure) -> int:
    return int((s.r_kind == PROTEIN).sum())


def residue_centroids(s: Structure, residue_ids: Iterable[str]) -> dict:
    """CA centroid and heavy-atom centroid of the given residues (``chain:resnum[icode]``).

    Residues absent from ``s`` are reported in ``missing``; residues present but without a CA atom
    in ``no_ca`` (they still contribute to the heavy-atom centroid). The CA centroid uses the first
    CA atom of each residue (after the structure's altloc selection). NaN centres if nothing
    matched."""
    rids = s.residue_ids
    index = {rid: i for i, rid in enumerate(rids)}
    wanted = list(dict.fromkeys(residue_ids))
    found = [index[r] for r in wanted if r in index]
    missing = [r for r in wanted if r not in index]
    ca_xyz, heavy_xyz, no_ca, non_protein = [], [], [], []
    heavy = ~s.is_hydrogen
    for ri in found:
        a0, a1 = int(s.r_start[ri]), int(s.r_end[ri])
        names = s.atom_name[a0:a1]
        ca = np.flatnonzero(names == "CA")
        if len(ca):
            ca_xyz.append(s.coords[a0 + ca[0]])
        else:
            no_ca.append(str(rids[ri]))
        hv = heavy[a0:a1]
        heavy_xyz.append(s.coords[a0:a1][hv])
        if s.r_kind[ri] != PROTEIN:
            non_protein.append(str(rids[ri]))
    nan3 = np.full(3, np.nan)
    ca_c = np.mean(np.asarray(ca_xyz, dtype=np.float64), axis=0) if ca_xyz else nan3
    hv_all = np.concatenate(heavy_xyz).astype(np.float64) if heavy_xyz else np.zeros((0, 3))
    hv_c = hv_all.mean(axis=0) if len(hv_all) else nan3
    return {
        "center_ca": ca_c,
        "center_heavy": hv_c,
        "found": [str(rids[i]) for i in found],
        "missing": missing,
        "no_ca": no_ca,
        "non_protein": non_protein,
    }
