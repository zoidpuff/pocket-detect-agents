"""Resampling units for the cluster bootstrap (PREREG: 30 % sequence clusters of test chains,
UniProt accession as sensitivity).

Cluster table (``scripts/build_test_clusters.py``): ``data/processed/clusters/test_clusters.tsv`` with
columns ``id`` (test-union id ``<source>|<pdb>_<chain>`` or ``<source>|<uniprot>``), ``source_set``,
``cluster30`` (MMseqs2 easy-cluster representative, min-seq-id 0.3, -c 0.5, cov-mode 0) and
``uniprot``. A benchmark structure made of several chains is mapped to all its chains' units; units
connected through such structures are merged (union–find), so one resampling unit never splits a
structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

__all__ = ["CLUSTER_TABLE", "load_clusters", "structure_units", "groups_for_structures"]

CLUSTER_TABLE = Path(__file__).resolve().parents[2] / "data" / "processed" / "clusters" / "test_clusters.tsv"


def load_clusters(path: str | Path = CLUSTER_TABLE) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def structure_units(source: str, structure_id: str, ids: pd.Index) -> list[str]:
    """Test-union ids belonging to a benchmark structure id (naming conventions of our GT tables)."""
    sid = structure_id.removesuffix(".pdb").removesuffix(".cif").removesuffix(".gz")
    cands: list[str] = []
    if f"{source}|{sid}" in ids:  # exact (ligysis 1a52_A, pocketwatch uniprot)
        return [f"{source}|{sid}"]
    if sid.startswith("AF-") and f"{source}|{sid.split('-')[1]}" in ids:  # AF-<ACC>-F1
        return [f"{source}|{sid.split('-')[1]}"]
    if "_" in sid:  # 1fd4_G-H  (CryptoBench multi-chain)
        pdb, chains = sid.split("_", 1)
        cands = [f"{source}|{pdb.lower()}_{c}" for c in chains.replace("-", ",").split(",")]
        cands = [c for c in cands if c in ids]
        if cands:
            return cands
    if len(sid) == 5:  # COACH420 style 148lE
        c = f"{source}|{sid[:4].lower()}_{sid[4]}"
        if c in ids:
            return [c]
    pref = f"{source}|{sid[:4].lower()}_"  # whole entry (HOLO4K)
    return [i for i in ids if i.startswith(pref)]


def groups_for_structures(structure_ids: Iterable[str], source: str | Iterable[str], level: str = "cluster30",
                          table: pd.DataFrame | None = None) -> Mapping[str, str]:
    """structure_id -> resampling-unit label at ``level`` ('cluster30' or 'uniprot').
    Structures that cannot be mapped get their own unit (``self:<structure_id>``)."""
    t = load_clusters() if table is None else table
    sources = [source] if isinstance(source, str) else list(source)
    t = t[t.source_set.isin(sources)]
    lab = dict(zip(t.id, t[level]))
    ids = pd.Index(t.id)
    sids = list(structure_ids)
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    units_of = {}
    for s in sids:
        units = []
        for src in sources:
            units += [lab[u] for u in structure_units(src, s, ids) if lab.get(u)]
        units = units or [f"self:{s}"]
        units_of[s] = units
        for u in units[1:]:
            parent[find(u)] = find(units[0])
    return {s: find(units_of[s][0]) for s in sids}
