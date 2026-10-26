"""LIGYSIS-style ligand pooling onto a representative chain (DESIGN.md §2.5, §9).

Building blocks:
* :func:`load_entry`      — gemmi read of an mmCIF entry (first model, first altloc, no H) into
                            per-chain protein arrays and a ligand-instance lookup.
* :func:`align_map`       — global sequence alignment (BLOSUM62, affine gaps, free end gaps) that
                            maps residue indices of one chain onto another.
* :func:`kabsch`          — optimal superposition (rotation, translation, RMSD).
* :func:`cluster_sites`   — LIGYSIS site clustering: ligand residue fingerprints, distance =
                            1 - |A∩B| / min(|A|,|B|) (relative intersection), average linkage,
                            cut at 0.5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import gemmi
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

__all__ = ["ChainData", "EntryData", "load_entry", "prep_structure", "align_map", "kabsch", "cluster_sites", "parse_bres",
           "residue_key"]

_STD20 = set("ACDEFGHIKLMNPQRSTVWY")
_BRES = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")


def residue_key(num: int, icode: str) -> str:
    icode = icode.strip() if icode else ""
    return f"{int(num)}{icode}"


@dataclass
class ChainData:
    name: str
    res_keys: list  # "num[icode]" per protein residue (author numbering)
    res_names: list
    seq: str  # one-letter, non-standard -> X
    ca: np.ndarray  # (R,3) float64, NaN if no CA
    atoms: np.ndarray  # (A,3) float32 protein heavy atoms
    atom_res: np.ndarray  # (A,) int32 residue index
    atom_names: list | None = None  # (A,) atom names (for per-residue atom matching)

    @property
    def key_index(self) -> dict:
        return {k: i for i, k in enumerate(self.res_keys)}


@dataclass
class EntryData:
    pdb: str
    chains: dict = field(default_factory=dict)  # name -> ChainData (protein chains)
    ligands: dict = field(default_factory=dict)  # (chain, resname, num) -> list[(coords (k,3), elements)]
    assembly_details: str = ""


def _one_letter(name: str) -> str:
    info = gemmi.find_tabulated_residue(name)
    if info is None:
        return "X"
    c = info.one_letter_code.upper()
    return c if c in _STD20 else "X"


def prep_structure(path: str) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    return st


def load_entry(path: str, want_ligands: bool = True, structure: gemmi.Structure | None = None) -> EntryData:
    """``structure``: an already prepared gemmi.Structure (see :func:`prep_structure`)."""
    st = structure if structure is not None else prep_structure(path)
    model = st[0]
    ed = EntryData(pdb=st.name.lower())
    for ch in model:
        poly = ch.get_polymer()
        keys, names, seq, ca, atoms, ares, anames = [], [], [], [], [], [], []
        for r in poly:
            info = gemmi.find_tabulated_residue(r.name)
            is_aa = (info is not None and info.is_amino_acid()) or (r.find_atom("CA", "*") is not None and r.find_atom("N", "*") is not None)
            if not is_aa:
                continue
            ri = len(keys)
            keys.append(residue_key(r.seqid.num, r.seqid.icode))
            names.append(r.name)
            seq.append(_one_letter(r.name))
            a = r.find_atom("CA", "*")
            ca.append([a.pos.x, a.pos.y, a.pos.z] if a is not None else [np.nan] * 3)
            for at in r:
                atoms.append((at.pos.x, at.pos.y, at.pos.z))
                ares.append(ri)
                anames.append(at.name)
        if keys:
            if ch.name in ed.chains:  # same author chain split into several gemmi chains: append
                old = ed.chains[ch.name]
                off = len(old.res_keys)
                ed.chains[ch.name] = ChainData(ch.name, old.res_keys + keys, old.res_names + names, old.seq + "".join(seq),
                                               np.vstack([old.ca, np.array(ca, float)]),
                                               np.vstack([old.atoms, np.array(atoms, np.float32)]),
                                               np.concatenate([old.atom_res, np.array(ares, np.int32) + off]),
                                               (old.atom_names or []) + anames)
            else:
                ed.chains[ch.name] = ChainData(ch.name, keys, names, "".join(seq), np.array(ca, float),
                                               np.array(atoms, np.float32).reshape(-1, 3), np.array(ares, np.int32), anames)
        if want_ligands:
            for r in ch:
                if r.is_water():
                    continue
                if r.entity_type == gemmi.EntityType.Polymer:
                    continue
                xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r], float).reshape(-1, 3)
                el = [a.element.name.upper() for a in r]
                ed.ligands.setdefault((ch.name, r.name, int(r.seqid.num)), []).append((xyz, el))
    return ed


_MATRIX = None


def align_map(a: str, b: str) -> tuple[np.ndarray, float, int]:
    """Map positions of sequence ``a`` to ``b``. Returns (map_ab (len(a),) int with -1 for
    unaligned, identity over aligned columns, number of aligned columns)."""
    global _MATRIX
    m = np.full(len(a), -1, np.int64)
    if a == b:
        m[:] = np.arange(len(a))
        return m, 1.0, len(a)
    i = b.find(a) if len(a) <= len(b) else -1
    if i >= 0:
        m[:] = np.arange(len(a)) + i
        return m, 1.0, len(a)
    j = a.find(b) if len(b) < len(a) else -1
    if j >= 0:
        m[j:j + len(b)] = np.arange(len(b))
        return m, 1.0, len(b)
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    if _MATRIX is None:
        _MATRIX = SubstitutionMatrix.std_protein_matrix()
    ali = align_optimal(ProteinSequence(a), ProteinSequence(b), _MATRIX, gap_penalty=(-10, -1),
                        terminal_penalty=False, max_number=1)[0]
    tr = ali.trace
    ok = (tr[:, 0] >= 0) & (tr[:, 1] >= 0)
    tr = tr[ok]
    m[tr[:, 0]] = tr[:, 1]
    n = len(tr)
    ident = float(np.mean(np.array(list(a))[tr[:, 0]] == np.array(list(b))[tr[:, 1]])) if n else 0.0
    return m, ident, n


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R, t, rmsd such that (P @ R.T + t) best fits Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - pc @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return R, t, rmsd


def parse_bres(s: str) -> list[str]:
    """BioLiP binding-residue string 'F43 R45 A100A' -> residue keys ['43', '45', '100A']."""
    out = []
    for tok in str(s).split():
        m = _BRES.match(tok)
        if m:
            out.append(f"{int(m.group(2))}{m.group(3)}")
    return out


def cluster_sites(fingerprints: list[set], cut: float = 0.5) -> np.ndarray:
    """LIGYSIS ligand-site clustering. Returns 0-based site label per ligand (sites ordered by
    first occurrence). Fingerprints must be non-empty."""
    n = len(fingerprints)
    if n == 0:
        return np.zeros(0, np.int64)
    if n == 1:
        return np.zeros(1, np.int64)
    # dedupe identical fingerprints (common: same ligand in many entries) to keep n^2 small
    uniq, inv = {}, np.empty(n, np.int64)
    for i, f in enumerate(fingerprints):
        k = frozenset(f)
        inv[i] = uniq.setdefault(k, len(uniq))
    U = list(uniq)
    m = len(U)
    if m == 1:
        return np.zeros(n, np.int64)
    allres = sorted(set().union(*U))
    ridx = {r: i for i, r in enumerate(allres)}
    X = np.zeros((m, len(allres)), np.float32)
    for i, f in enumerate(U):
        X[i, [ridx[r] for r in f]] = 1
    inter = X @ X.T
    sz = X.sum(1)
    D = 1.0 - inter / np.minimum.outer(sz, sz)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, 1)
    if n <= 2500:  # exact per-ligand average linkage (identical fingerprints at distance 0)
        full = D[np.ix_(inv, inv)]
        lab = fcluster(linkage(full[np.triu_indices(n, 1)], method="average"), t=cut, criterion="distance")
    else:  # very large pools: average linkage over unique fingerprints (approximation, documented)
        lab = fcluster(linkage(D[np.triu_indices(m, 1)], method="average"), t=cut, criterion="distance")[inv]
    # relabel by first occurrence
    order, out = {}, np.empty(n, np.int64)
    for i, l in enumerate(lab):
        out[i] = order.setdefault(int(l), len(order))
    return out
