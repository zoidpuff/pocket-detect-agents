"""Common *site-table* ground-truth format for site-defined benchmarks (LIGYSIS, CryptoBench,
PocketWatch, AF set, dimers). One TSV row per reference site; read/write helpers.

Columns (required unless marked optional):

* ``structure_id``   — key of the input structure; predictions are ``<pred_dir>/<structure_id>.csv``
* ``structure_path`` — input structure file given to every method (optional for reproduction-only
  tables, e.g. LIGYSIS published predictions)
* ``site_id``        — unique within structure
* ``center_x, center_y, center_z`` — reference site centre **in the frame of structure_path**
* ``residues``       — site residues, space-separated ``chain:resnum[icode]`` (author numbering of
  structure_path; for LIGYSIS reproduction UniProt numbers as ``U:<n>``)
* ``n_sites``        — N used by top-N / top-N+2 for this structure (LIGYSIS: all sites of the chain,
  *including* redundant ones, as in the LBS-comparison code)
* ``scored``         — 1 if the site is an evaluation target (LIGYSIS: non-redundant site;
  PocketWatch: unmasked), 0 if it only contributes to N / precision
* ``ligand_atoms``   — optional, heavy-atom coordinates of the ligands defining the site in the same
  frame: space-separated ``x,y,z`` triples (enables DCA criteria)
* ``n_residues_structure`` — optional, number of protein residues of the structure (residue TN)
* ``group``          — optional resampling unit for the cluster bootstrap (e.g. 30 % seq cluster)
* any further columns are carried through as metadata (``uniprot``, strata, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SITE_COLUMNS", "Site", "read_sites", "write_sites", "sites_by_structure", "format_atoms", "parse_atoms"]

SITE_COLUMNS = ["structure_id", "site_id", "center_x", "center_y", "center_z", "residues", "n_sites", "scored"]
OPTIONAL_COLUMNS = ["structure_path", "ligand_atoms", "n_residues_structure", "group"]


@dataclass
class Site:
    structure_id: str
    site_id: str
    center: np.ndarray
    residues: frozenset
    n_sites: int
    scored: bool
    ligand_atoms: np.ndarray | None = None  # (k,3)
    meta: dict = field(default_factory=dict)


def format_atoms(coords) -> str:
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    return " ".join(f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in coords)


def parse_atoms(s) -> np.ndarray | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return None
    return np.array([[float(v) for v in t.split(",")] for t in str(s).split()], dtype=np.float64)


def write_sites(df: pd.DataFrame, path: str | Path) -> None:
    missing = [c for c in SITE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"site table lacks {missing}")
    if df.duplicated(["structure_id", "site_id"]).any():
        raise ValueError("duplicate (structure_id, site_id)")
    out = df.copy()
    out["scored"] = out["scored"].astype(int)
    out.to_csv(path, sep="\t", index=False, float_format="%.3f")


def read_sites(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"structure_id": str, "site_id": str, "residues": str, "ligand_atoms": str}, keep_default_na=False, na_values={"center_x": [""], "center_y": [""], "center_z": [""]})
    missing = [c for c in SITE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: site table lacks {missing}")
    df["scored"] = df["scored"].astype(int).astype(bool)
    df["n_sites"] = df["n_sites"].astype(int)
    return df


def sites_by_structure(df: pd.DataFrame) -> dict[str, list[Site]]:
    out: dict[str, list[Site]] = {}
    has_atoms = "ligand_atoms" in df.columns
    meta_cols = [c for c in df.columns if c not in SITE_COLUMNS + ["ligand_atoms"]]
    for row in df.itertuples(index=False):
        r = row._asdict()
        res = r["residues"]
        out.setdefault(r["structure_id"], []).append(
            Site(
                structure_id=r["structure_id"],
                site_id=str(r["site_id"]),
                center=np.array([r["center_x"], r["center_y"], r["center_z"]], dtype=np.float64),
                residues=frozenset(str(res).split()) if isinstance(res, str) else frozenset(res or ()),
                n_sites=int(r["n_sites"]),
                scored=bool(r["scored"]),
                ligand_atoms=parse_atoms(r["ligand_atoms"]) if has_atoms else None,
                meta={c: r[c] for c in meta_cols},
            )
        )
    return out
