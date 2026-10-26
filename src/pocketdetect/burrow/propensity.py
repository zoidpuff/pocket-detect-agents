"""Knowledge-based ligand-contact propensity of protein atom types (training set only).

Atom type = (residue name, atom name) of standard heavy atoms (others pooled to element).
Among *exposed* atoms (heavy atom <= 4 Å of a BURROW candidate point), contact = within 4 Å of a
non-ion pooled ligand atom.  log-odds(type) = logit(p_type) - logit(p_all) with p_type shrunk
towards p_all by a pseudo-count of ``ALPHA`` exposed atoms (pools rare types).

Pocket features (ranker, phase 3): mean / sum of log-odds over the pocket's lining atoms (<= 4 Å
of its points) and the mean over side-chain lining atoms.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

ALPHA = 20.0
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def atom_types(P) -> np.ndarray:
    return np.char.add(np.char.add(P.res_name.astype(str), "."), P.atom_name.astype(str))


def type_counts(P, pts, lig_xyz):
    """(types, exposed counts, contact counts) for one structure."""
    t = atom_types(P)
    tree = cKDTree(P.coords)
    exp = np.zeros(len(t), bool)
    if len(pts):
        for l in tree.query_ball_point(pts, 4.0):
            exp[l] = True
    con = np.zeros(len(t), bool)
    if len(lig_xyz):
        for l in tree.query_ball_point(lig_xyz, 4.0):
            con[l] = True
    con &= exp
    u, inv = np.unique(t[exp], return_inverse=True)
    ce = np.bincount(inv, minlength=len(u))
    cc = np.bincount(inv, weights=con[exp].astype(float), minlength=len(u))
    return u, ce, cc


def build_table(counts_list):
    """counts_list: iterable of (types, exposed, contact) -> {type: log-odds}, plus '__default__'."""
    E, Cc = {}, {}
    for u, ce, cc in counts_list:
        for a, e, c in zip(u.tolist(), ce.tolist(), cc.tolist()):
            E[a] = E.get(a, 0.0) + e
            Cc[a] = Cc.get(a, 0.0) + c
    e_all, c_all = sum(E.values()), sum(Cc.values())
    p_all = c_all / max(e_all, 1.0)
    logit = lambda p: np.log(p / (1 - p))
    tab = {a: float(logit((Cc[a] + ALPHA * p_all) / (E[a] + ALPHA)) - logit(p_all)) for a in E}
    tab["__default__"] = 0.0
    return tab


def atom_logodds(P, table) -> np.ndarray:
    t = atom_types(P)
    d = table.get("__default__", 0.0)
    return np.array([table.get(x, d) for x in t.tolist()], np.float32)


def pocket_propensity(pockets_idx, pts, P, lo, tree: cKDTree | None = None):
    """(n_pockets, 3): mean, sum, side-chain mean of atom log-odds over lining atoms."""
    tree = tree or cKDTree(P.coords)
    n = len(pockets_idx)
    out = np.zeros((n, 3), np.float32)
    if n == 0:
        return out
    used = np.unique(np.concatenate([np.asarray(i) for i in pockets_idx]))
    per = {}
    for i, l in zip(used.tolist(), tree.query_ball_point(pts[used], 4.0)):
        per[i] = np.asarray(l, np.int64)
    sc = ~np.isin(P.atom_name.astype(str), list(BACKBONE))
    for k, idx in enumerate(pockets_idx):
        a = np.unique(np.concatenate([per[i] for i in np.asarray(idx).tolist()]))
        if len(a):
            v = lo[a]
            out[k, 0] = v.mean()
            out[k, 1] = v.sum()
            s = sc[a]
            out[k, 2] = v[s].mean() if s.any() else 0.0
    return out


PROP_NAMES = ["prop_mean", "prop_sum", "prop_sc_mean"]
