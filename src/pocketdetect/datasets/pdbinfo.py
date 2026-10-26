"""Entry-level metadata from wwPDB mmCIF files (local mirror): polymer entities, their canonical
sequences, author chain IDs, UniProt cross-references, experimental method and resolution.

Only the mmCIF *categories* are read (gemmi.cif), not the coordinates, so this is cheap.
"""

from __future__ import annotations

from pathlib import Path

import gemmi

from ..structure import pdb_entry_path

__all__ = ["entry_info", "clean_seq", "THREE_TO_ONE"]

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def clean_seq(s: str) -> str:
    """Strip whitespace/newlines from a one-letter sequence and upper-case it."""
    return "".join(str(s).split()).upper()


def _col(block: gemmi.cif.Block, cat: str, tag: str) -> list:
    t = block.find_values(f"{cat}.{tag}")
    return [gemmi.cif.as_string(v) for v in t] if t else []


def entry_info(pdb_id: str | None = None, path: str | Path | None = None) -> dict:
    """Return ``{pdb_id, method, resolution, entities: [{entity_id, type, seq, chains, uniprot}]}``.

    ``seq`` is ``_entity_poly.pdbx_seq_one_letter_code_can`` (full deposited sequence, incl.
    unobserved residues), ``chains`` are author chain IDs (``pdbx_strand_id``), ``uniprot`` the
    UNP accessions from ``_struct_ref`` of that entity. Raises FileNotFoundError if absent."""
    if path is None:
        path = pdb_entry_path(pdb_id)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    doc = gemmi.cif.read(str(path))
    b = doc.sole_block()
    pid = (pdb_id or b.name).lower()
    methods = _col(b, "_exptl", "method")
    res = None
    for cat, tag in (("_refine", "ls_d_res_high"), ("_em_3d_reconstruction", "resolution"), ("_reflns", "d_resolution_high")):
        vals = [v for v in _col(b, cat, tag) if v not in ("?", ".", "")]
        if vals:
            try:
                res = min(float(v) for v in vals)
                break
            except ValueError:
                pass
    unp: dict[str, set] = {}
    ent_ids = _col(b, "_struct_ref", "entity_id")
    dbs = _col(b, "_struct_ref", "db_name")
    accs = _col(b, "_struct_ref", "pdbx_db_accession")
    for e, db, a in zip(ent_ids, dbs, accs):
        if db.upper() in ("UNP", "UNIPROT") and a not in ("?", "."):
            unp.setdefault(e, set()).add(a)
    ents = []
    eids = _col(b, "_entity_poly", "entity_id")
    types = _col(b, "_entity_poly", "type")
    seqs = _col(b, "_entity_poly", "pdbx_seq_one_letter_code_can")
    strands = _col(b, "_entity_poly", "pdbx_strand_id")
    for i, e in enumerate(eids):
        ents.append(dict(
            entity_id=e,
            type=types[i] if i < len(types) else "",
            seq=clean_seq(seqs[i]) if i < len(seqs) else "",
            chains=[c.strip() for c in (strands[i] if i < len(strands) else "").split(",") if c.strip()],
            uniprot=sorted(unp.get(e, ())),
        ))
    return dict(pdb_id=pid, method=";".join(methods), resolution=res, entities=ents)
