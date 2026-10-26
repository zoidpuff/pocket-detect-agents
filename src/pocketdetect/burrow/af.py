"""AF mode, phase-1 subset (DESIGN §2.6 mechanisms 1-2; the AF-trained ranker is not built yet).

(1) spaghetti pruning: drop terminal segments with pLDDT < 50 and internal stretches of >= 7
    consecutive residues with pLDDT < 50 (per chain, residue pLDDT = CA B-factor, else atom mean);
(2) soft occlusion: remaining atoms get opacity w = clip((pLDDT - 40) / 40, 0, 1); atoms with
    w < 0.5 do not exclude candidate points (see features.point_features ``weights``).
No-op guarantee: if every residue has pLDDT >= 80, nothing is pruned and all weights are 1, so the
geometry is identical to standard mode.
"""
from __future__ import annotations

import numpy as np

from ..structure import Structure
from .prep import Protein, protein_from_structure

LOW = 50.0
MIN_INTERNAL = 7


def residue_plddt(s: Structure) -> np.ndarray:
    n = s.n_residues
    val = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(val, s.res_index, s.b_factor)
    np.add.at(cnt, s.res_index, 1)
    out = val / np.maximum(cnt, 1)
    ca = np.flatnonzero(s.atom_name == "CA")
    out[s.res_index[ca]] = s.b_factor[ca]
    return out


def prune_mask(s: Structure) -> np.ndarray:
    """Per-residue keep mask."""
    pl = residue_plddt(s)
    keep = np.ones(s.n_residues, bool)
    for ch in dict.fromkeys(s.r_chain.tolist()):
        idx = np.flatnonzero(s.r_chain == ch)
        low = pl[idx] < LOW
        i = 0
        while i < len(idx):
            if not low[i]:
                i += 1
                continue
            j = i
            while j < len(idx) and low[j]:
                j += 1
            terminal = i == 0 or j == len(idx)
            if terminal or (j - i) >= MIN_INTERNAL:
                keep[idx[i:j]] = False
            i = j
    return keep


def af_protein(s: Structure) -> tuple[Protein, np.ndarray, dict]:
    """Pruned polymer-only Protein + per-atom opacity weights (None if all 1) + stats."""
    s = s.select(s.protein_heavy_mask())
    keep_res = prune_mask(s)
    s2 = s.select(keep_res[s.res_index])
    P = protein_from_structure(s2)
    w = np.clip((P.plddt - 40.0) / 40.0, 0.0, 1.0).astype(np.float32)
    stats = dict(n_res=int(s.n_residues), n_res_pruned=int((~keep_res).sum()), n_soft_atoms=int((w < 1).sum()))
    return P, (None if (w >= 1).all() else w), stats
