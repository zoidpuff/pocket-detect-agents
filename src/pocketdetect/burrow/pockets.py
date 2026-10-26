"""Pocket formation (DESIGN §2.4.2): threshold -> single linkage -> chaining control -> NMS.

A pocket is a set of point indices.  Pocket strength = sum over its points of ``vol * value``
(value = point score, or buriedness for the geometry-only generator).  Centre = (vol * value)-
weighted centroid.  NMS follows the LIGYSIS redundancy definition: a pocket is suppressed if its
centre is < 5 Å from, or its residue set has Jaccard >= 0.75 with, a higher-ranked pocket.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.cluster import single_linkage


@dataclass
class Pocket:
    idx: np.ndarray       # point indices
    strength: float
    center: np.ndarray    # (3,) input frame
    residues: frozenset = frozenset()
    level: int = 0        # threshold level that produced it


def _kmeans_split(Xs, w, k, n_iter=10):
    """Deterministic weighted k-means (farthest-point init from the heaviest point)."""
    c = [int(np.argmax(w))]
    d = ((Xs - Xs[c[0]]) ** 2).sum(1)
    for _ in range(1, k):
        c.append(int(np.argmax(d)))
        d = np.minimum(d, ((Xs - Xs[c[-1]]) ** 2).sum(1))
    C = Xs[c].astype(np.float64)
    lab = np.zeros(len(Xs), np.int64)
    for _ in range(n_iter):
        D = ((Xs[:, None, :] - C[None]) ** 2).sum(-1)
        lab = D.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                ww = w[m][:, None] + 1e-9
                C[j] = (Xs[m] * ww).sum(0) / ww.sum()
    return lab


def _extent(Xs):
    if len(Xs) < 2:
        return 0.0
    c = Xs - Xs.mean(0)
    ev, evec = np.linalg.eigh(c.T @ c)
    proj = c @ evec[:, -1]
    return float(proj.max() - proj.min())


def components(X, sel, link: float, w, max_extent: float = 20.0, split_size: float = 12.0):
    """Single-linkage components of points ``sel`` (indices), with chaining control: components
    longer than ``max_extent`` (principal-axis extent) are split by weighted k-means into
    ceil(extent / split_size) parts.  Returns a list of index arrays."""
    sel = np.asarray(sel)
    if len(sel) == 0:
        return []
    lab = single_linkage(X[sel], link)
    out = []
    order = np.argsort(lab, kind="stable")
    bounds = np.flatnonzero(np.diff(lab[order])) + 1
    for grp in np.split(order, bounds):
        idx = sel[grp]
        ext = _extent(X[idx]) if len(idx) > 10 else 0.0
        if ext > max_extent:
            k = int(np.ceil(ext / split_size))
            sub = _kmeans_split(X[idx].astype(np.float64), w[idx], k)
            for j in range(k):
                m = sub == j
                if m.any():
                    out.append(idx[m])
        else:
            out.append(idx)
    return out


def make_pocket(idx, pts, vol, value, level=0):
    ww = vol[idx] * value[idx]
    s = float(ww.sum())
    if s <= 0:
        ww = np.ones(len(idx))
    c = (pts[idx].astype(np.float64) * ww[:, None]).sum(0) / ww.sum()
    return Pocket(idx=idx, strength=s, center=c, level=level)


def generate(X, pts, vol, value, thresholds, link=2.05, max_extent=20.0, split_size=12.0,
             min_vol=10.0, extra_value=None, sub_size: float | None = None, sub_min_extent: float = 10.0):
    """Multi-threshold candidate pockets (union over thresholds on ``value``).

    ``sub_size``: additionally split every component with extent > ``sub_min_extent`` into
    ceil(extent / sub_size) k-means sub-pockets (multi-scale candidates; the parent is kept)."""
    pockets = []
    for lev, th in enumerate(thresholds):
        sel = np.flatnonzero(value >= th)
        for idx in components(X, sel, link, vol * value, max_extent, split_size):
            if vol[idx].sum() < min_vol and not (extra_value is not None and extra_value[idx].any()):
                continue
            pockets.append(make_pocket(idx, pts, vol, value, lev))
            if sub_size:
                ext = _extent(X[idx]) if len(idx) > 10 else 0.0
                if ext > sub_min_extent:
                    k = int(np.ceil(ext / sub_size))
                    sub = _kmeans_split(X[idx].astype(np.float64), (vol * value)[idx], k)
                    for j in range(k):
                        m = sub == j
                        if vol[idx[m]].sum() >= min_vol:
                            pockets.append(make_pocket(idx[m], pts, vol, value, lev + 100))
    return pockets


def assign_residues(pockets, pts, atom_tree: cKDTree, atom_res, residue_ids, r: float = 4.0):
    """Residues with any heavy atom <= r of any pocket point."""
    for p in pockets:
        lst = atom_tree.query_ball_point(pts[p.idx], r)
        atoms = np.unique(np.concatenate([np.asarray(l, np.int64) for l in lst])) if len(lst) else np.zeros(0, int)
        p.residues = frozenset(residue_ids[np.unique(atom_res[atoms])].tolist()) if len(atoms) else frozenset()
    return pockets


def nms(pockets, scores, dist: float = 5.0, jacc: float = 0.75, max_n: int | None = None,
        overlap: float | None = None, min_score: float | None = None, min_keep: int = 0):
    """Greedy NMS in descending score order (ties -> earlier pocket). Returns kept indices.

    Suppress a pocket if its centre is < ``dist`` from a kept pocket, or residue Jaccard >= ``jacc``
    (LIGYSIS redundancy), or (``overlap``) the point-overlap coefficient |A∩B| / min(|A|,|B|) with a
    kept pocket is >= ``overlap`` (nested multi-scale / multi-threshold duplicates).  Pockets with
    score < ``min_score`` are dropped once ``min_keep`` pockets are kept."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    sets = {}
    for i in order:
        p = pockets[i]
        if min_score is not None and len(kept) >= min_keep and scores[i] < min_score:
            break
        ok = True
        for j in kept:
            q = pockets[j]
            if np.linalg.norm(p.center - q.center) < dist:
                ok = False
                break
            if jacc is not None and p.residues and q.residues:
                inter = len(p.residues & q.residues)
                if inter / len(p.residues | q.residues) >= jacc:
                    ok = False
                    break
            if overlap is not None:
                a = sets.setdefault(i, set(np.asarray(p.idx).tolist()))
                b = sets.setdefault(j, set(np.asarray(q.idx).tolist()))
                if a and b and len(a & b) / min(len(a), len(b)) >= overlap:
                    ok = False
                    break
        if ok:
            kept.append(int(i))
            if max_n is not None and len(kept) >= max_n:
                break
    return kept
