"""Input preparation: polymer-only heavy atoms + per-atom chemistry tables (DESIGN §2.1).

Everything that is not a protein residue (ligands, waters, ions, glycans, nucleic acids) and all
hydrogens are removed, so predictions are invariant to HETATM ligand records.  B-factors are kept
only as ``plddt`` for AF mode (never used as a feature in standard mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.atomtypes import AROMATIC, NEGATIVE, POSITIVE, atom_class_matrix, vdw_radii
from ..structure import Structure, load_structure

# 7 residue groups (§2.3)
RES_GROUPS = ("aliphatic", "aromatic", "polar", "positive", "negative", "his_cys", "gly_pro_other")
_RG = {
    **{r: 0 for r in ("ALA", "VAL", "LEU", "ILE", "MET", "MSE")},
    **{r: 1 for r in ("PHE", "TRP", "TYR")},
    **{r: 2 for r in ("SER", "THR", "ASN", "GLN")},
    **{r: 3 for r in ("LYS", "ARG")},
    **{r: 4 for r in ("ASP", "GLU")},
    **{r: 5 for r in ("HIS", "CYS")},
}

# first-hit chemistry classes
HIT_CLASSES = ("apolar", "polar", "charged", "aromatic")

# potential metal-coordinating atoms (§2.2); side-chain ones flagged separately
_COORD_SC = {"HIS": {"ND1", "NE2"}, "CYS": {"SG"}, "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
             "MET": {"SD"}, "MSE": {"SE"}}

BACKBONE_CB = {"N", "CA", "C", "O", "OXT", "CB"}


@dataclass
class Protein:
    id: str
    coords: np.ndarray        # (N,3) float32 input frame
    radii: np.ndarray         # (N,) float32 Bondi vdW
    element: np.ndarray
    atom_name: np.ndarray
    res_name: np.ndarray
    res_index: np.ndarray     # (N,) int32 -> residue table
    residue_ids: np.ndarray   # (R,) object "chain:num[icode]"
    chain: np.ndarray         # (N,) author chain
    plddt: np.ndarray         # (N,) float32 b-factor column (AF only)
    props: np.ndarray = None  # (N, F) float32 neighbour-sum properties
    prop_names: list = field(default_factory=list)
    hit_class: np.ndarray = None   # (N,) int8 first-hit class
    bb_mask: np.ndarray = None     # (N,) bool backbone + CB
    coord_mask: np.ndarray = None  # (N,) bool potential metal-coordinating atom
    coord_sc_mask: np.ndarray = None


def _wcn(coords, res_index, atom_name):
    """Per-residue weighted contact number over CA atoms (sum 1/d^2), mapped to atoms."""
    n_res = int(res_index.max()) + 1 if len(res_index) else 0
    ca = np.flatnonzero(atom_name == "CA")
    rc = np.full((n_res, 3), np.nan, np.float64)
    rc[res_index[ca]] = coords[ca]
    # residues without CA: mean of their atoms
    miss = np.isnan(rc[:, 0])
    if miss.any():
        sums = np.zeros((n_res, 3))
        np.add.at(sums, res_index, coords.astype(np.float64))
        cnt = np.bincount(res_index, minlength=n_res)[:, None]
        rc[miss] = sums[miss] / np.maximum(cnt[miss], 1)
    from scipy.spatial import cKDTree
    t = cKDTree(rc)
    pairs = t.query_pairs(30.0, output_type="ndarray")
    w = np.zeros(n_res)
    if len(pairs):
        d2 = ((rc[pairs[:, 0]] - rc[pairs[:, 1]]) ** 2).sum(1)
        inv = 1.0 / np.maximum(d2, 1.0)
        np.add.at(w, pairs[:, 0], inv)
        np.add.at(w, pairs[:, 1], inv)
    return w[res_index].astype(np.float32)


def protein_from_structure(s: Structure) -> Protein:
    """Polymer-only (PROTEIN residues, heavy atoms) view of a loaded structure."""
    mask = s.protein_heavy_mask()
    s = s.select(mask)
    el = np.char.upper(s.element.astype(str))
    nm = np.char.upper(s.atom_name.astype(str))
    rn = np.char.upper(s.res_name.astype(str))
    cls, names = atom_class_matrix(el, nm, rn)
    n = len(el)
    rg = np.zeros((n, len(RES_GROUPS)), np.float32)
    rg[np.arange(n), np.array([_RG.get(r, 6) for r in rn], np.int64) if n else np.zeros(0, np.int64)] = 1.0
    keys = list(zip(rn.tolist(), nm.tolist()))
    coord_sc = np.array([a in _COORD_SC.get(r, ()) for r, a in keys], bool)
    coord = coord_sc | (nm == "O")
    wcn = _wcn(s.coords, s.res_index, nm)
    props = np.concatenate([cls, rg, wcn[:, None], coord[:, None].astype(np.float32)], 1).astype(np.float32)
    prop_names = list(names) + [f"rg_{g}" for g in RES_GROUPS] + ["wcn", "coord"]
    # first-hit class: charged > aromatic > polar(N/O) > apolar (C, S, Se, other)
    charged = np.array([a in POSITIVE.get(r, ()) or a in NEGATIVE.get(r, ()) for r, a in keys], bool) | (nm == "OXT")
    arom = np.array([a in AROMATIC.get(r, ()) for r, a in keys], bool)
    polar = np.isin(el, ["N", "O"])
    hc = np.zeros(n, np.int8)
    hc[polar] = 1
    hc[arom] = 3
    hc[charged] = 2
    return Protein(
        id=s.id, coords=np.ascontiguousarray(s.coords, np.float32), radii=vdw_radii(el),
        element=el, atom_name=nm, res_name=rn, res_index=s.res_index.astype(np.int32),
        residue_ids=s.residue_ids, chain=s.chain.astype(str), plddt=s.b_factor.astype(np.float32),
        props=props, prop_names=prop_names, hit_class=hc, bb_mask=np.isin(nm, list(BACKBONE_CB)),
        coord_mask=coord, coord_sc_mask=coord_sc)


def load_protein(path, structure_id: str | None = None, chains=None) -> Protein:
    s = load_structure(path, structure_id=structure_id, chains=chains, protein_only=True)
    return protein_from_structure(s)


def protein_from_arrays(coords, element, atom_name, res_name, res_num, chain, structure_id="synthetic",
                        b_factor=None) -> Protein:
    """Build a :class:`Protein` from plain per-atom arrays (synthetic inputs / tests)."""
    from ..structure import PROTEIN
    coords = np.asarray(coords, np.float32)
    n = len(coords)
    chain = np.asarray(chain).astype(str)
    res_num = np.asarray(res_num, np.int32)
    key = np.char.add(np.char.add(chain, ":"), res_num.astype(str))
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    res_index = rank[inv].astype(np.int32)
    fr = first[order]
    s = Structure(
        id=structure_id, coords=coords, element=np.asarray(element).astype(str),
        atom_name=np.asarray(atom_name).astype(str), altloc=np.array([""] * n), occupancy=np.ones(n, np.float32),
        b_factor=np.full(n, 100.0, np.float32) if b_factor is None else np.asarray(b_factor, np.float32),
        serial=np.arange(1, n + 1, dtype=np.int32), res_index=res_index, chain=chain,
        res_name=np.asarray(res_name).astype(str), res_num=res_num, ins_code=np.array([""] * n),
        is_hetero=np.zeros(n, bool), r_chain=chain[fr], r_name=np.asarray(res_name).astype(str)[fr],
        r_num=res_num[fr], r_icode=np.array([""] * len(fr)), r_kind=np.full(len(fr), PROTEIN, np.int8),
        r_hetero=np.zeros(len(fr), bool), r_entity=np.array(["Polymer"] * len(fr)))
    return protein_from_structure(s)
