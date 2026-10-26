"""Atom radii and per-atom chemical class matrix (heavy atoms, standard residues).

``atom_class_matrix`` turns element / atom-name / residue-name arrays into a
dense (N, C) float32 0/1 matrix used as the generic property input of
:func:`pocketdetect.geometry.neighbor_features`.  Unknown residues fall back
to element-only classes (backbone/side-chain/donor/... are 0).
"""
from __future__ import annotations

import numpy as np

# Bondi (1964) van der Waals radii, Å.  Metals/others: generic 1.8 Å.
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "NA": 2.27, "K": 2.75, "CA": 1.97, "ZN": 1.39, "FE": 1.56,
    "MN": 1.61, "CU": 1.40, "NI": 1.63, "CO": 1.50,
}
DEFAULT_RADIUS = 1.80


def vdw_radii(elements) -> np.ndarray:
    """Bondi vdW radii (float32) for an array of element symbols."""
    el = np.asarray(elements).astype(str)
    uniq, inv = np.unique(np.char.upper(np.char.strip(el)), return_inverse=True)
    r = np.array([VDW_RADII.get(u, DEFAULT_RADIUS) for u in uniq], np.float32)
    return r[inv.reshape(-1)]


BACKBONE = {"N", "CA", "C", "O", "OXT"}

AROMATIC = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "NE1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

# side-chain carbons bonded only to C/H/S (i.e. apolar carbons)
HYDROPHOBIC_SC_C = {
    "ALA": {"CB"}, "VAL": {"CB", "CG1", "CG2"}, "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"}, "MET": {"CB", "CG", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2"},
    "TRP": {"CB", "CG", "CD2", "CE3", "CZ2", "CZ3", "CH2"},
    "PRO": {"CB", "CG"}, "CYS": {"CB"}, "THR": {"CG2"}, "LYS": {"CB", "CG", "CD"},
    "ARG": {"CB", "CG"}, "GLU": {"CB", "CG"}, "GLN": {"CB", "CG"},
    "ASP": {"CB"}, "ASN": {"CB"}, "HIS": {"CB"}, "MSE": {"CB", "CG", "CE"},
}

DONORS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}, "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "CYS": {"SG"},
}
ACCEPTORS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "ASN": {"OD1"}, "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"}, "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "MET": {"SD"},
    "MSE": {"SE"},
}
POSITIVE = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}}
NEGATIVE = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL",
}

ATOM_CLASSES = (
    "all", "C", "N", "O", "S", "aromatic", "hydrophobic_sc_C", "donor", "acceptor",
    "backbone", "sidechain", "positive", "negative",
)


def _classify(el: str, name: str, res: str) -> list[float]:
    std = res in STANDARD_RESIDUES
    bb = std and name in BACKBONE
    v = [1.0, el == "C", el == "N", el == "O", el in ("S", "SE"),
         name in AROMATIC.get(res, ()),
         name in HYDROPHOBIC_SC_C.get(res, ()),
         (name in DONORS.get(res, ())) or (bb and name == "N" and res != "PRO"),
         (name in ACCEPTORS.get(res, ())) or (bb and name in ("O", "OXT")),
         bb, std and not bb,
         name in POSITIVE.get(res, ()),
         (name in NEGATIVE.get(res, ())) or (bb and name == "OXT")]
    return [float(x) for x in v]


def atom_class_matrix(elements, atom_names, res_names):
    """(N, len(ATOM_CLASSES)) float32 0/1 matrix and the list of column names.

    Columns: ``all`` (ones, gives plain atom counts / protrusion), element
    classes C/N/O/S, aromatic ring atoms, apolar side-chain carbons, H-bond
    donors / acceptors (heavy atoms), backbone / side chain, positively /
    negatively charged groups (Lys NZ, Arg NE/NH*, Asp/Glu carboxylate O, OXT).
    """
    el = np.char.upper(np.char.strip(np.asarray(elements).astype(str)))
    nm = np.char.upper(np.char.strip(np.asarray(atom_names).astype(str)))
    rs = np.char.upper(np.char.strip(np.asarray(res_names).astype(str)))
    keys = np.char.add(np.char.add(np.char.add(el, "|"), np.char.add(nm, "|")), rs)
    uniq, inv = np.unique(keys, return_inverse=True)
    table = np.array([_classify(*u.split("|")) for u in uniq], np.float32).reshape(-1, len(ATOM_CLASSES))
    return table[inv.reshape(-1)], list(ATOM_CLASSES)
