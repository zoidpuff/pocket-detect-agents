"""AF-aware pocket ranker (DESIGN §2.6 mechanisms 4-5).

The standard pocket descriptors (:func:`model.pocket_features`) are extended by pocket confidence
descriptors computed on the (pruned) AF input, whose per-atom ``plddt`` is the B-factor column:

* ``af_pl_mean`` / ``af_pl_min`` / ``af_pl_q25`` — residue pLDDT (CA, else atom mean) over lining residues
* ``af_res_lt70`` / ``af_res_lt50``  — fraction of lining residues with pLDDT < 70 / < 50
* ``af_pts_lt70`` / ``af_pts_lt50``  — fraction of pocket points with an atom of pLDDT < 70 / < 50 within 4 Å
* ``af_prot_mean`` / ``af_prot_lt70`` — protein-level context (input after pruning)
* inter-domain surrogate (structure only; no PAE): spectral bisection of the Cα contact graph
  (8 Å) into two parts; ``af_dom_span`` = min(#lining residues per part) / #lining residues and
  ``af_dom_pack`` = inter-part contacts / min(part size) (weak packing = low); both 0 / protein
  value when the smaller part has < 40 residues.

:class:`AFRanker` is a LightGBM ranker over ``rank_features`` (standard ∪ AF names); the point model
and candidate generation are the standard model's.  :func:`predict_protein_af` mirrors
:func:`pipeline.predict_protein` with the AF ranker swapped in.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .features import BurrowConfig, point_features
from .model import BurrowModel, FormationConfig, candidates, pocket_features, score_points
from .pipeline import Prediction, finalize
from .pockets import assign_residues
from .prep import Protein

AF_FEATURES = ["af_pl_mean", "af_pl_min", "af_pl_q25", "af_res_lt70", "af_res_lt50", "af_pts_lt70", "af_pts_lt50",
               "af_prot_mean", "af_prot_lt70", "af_dom_span", "af_dom_pack"]
DOM_MIN = 40


def residue_plddt_protein(P: Protein) -> np.ndarray:
    n = int(P.res_index.max()) + 1 if len(P.res_index) else 0
    s = np.bincount(P.res_index, weights=P.plddt.astype(np.float64), minlength=n)
    c = np.bincount(P.res_index, minlength=n)
    out = s / np.maximum(c, 1)
    ca = np.flatnonzero(P.atom_name == "CA")
    out[P.res_index[ca]] = P.plddt[ca]
    return out


def domain_split(P: Protein):
    """(part (R,) int8 in {0,1} or None, packing density).  Spectral bisection (Fiedler vector of
    the normalised Laplacian) of the Cα contact graph (8 Å)."""
    ca = np.flatnonzero(P.atom_name == "CA")
    R = int(P.res_index.max()) + 1 if len(P.res_index) else 0
    if len(ca) < 2 * DOM_MIN:
        return None, 0.0
    X = P.coords[ca].astype(np.float64)
    pairs = cKDTree(X).query_pairs(8.0, output_type="ndarray")
    if len(pairs) == 0:
        return None, 0.0
    import scipy.sparse as sp
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    n = len(ca)
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    L = laplacian(A, normed=True)
    try:
        v0 = np.linspace(-1.0, 1.0, n)  # deterministic start vector
        w, V = eigsh(L, k=2, sigma=-1e-3, which="LM", v0=v0)
        f = V[:, np.argsort(w)[1]]
    except Exception:
        return None, 0.0
    lab = (f > np.median(f)).astype(np.int8)
    # sweep cut: best normalised cut along the Fiedler order
    o = np.argsort(f, kind="stable")
    pos = np.empty(n, np.int64)
    pos[o] = np.arange(n)
    deg = np.asarray(A.sum(1)).ravel()
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    # edges crossing cut k (between positions k-1 and k): lo < k <= hi
    cut = np.zeros(n + 1)
    np.add.at(cut, lo + 1, 1.0)
    np.add.at(cut, hi + 1, -1.0)
    cut = np.cumsum(cut)[:n]
    vol = np.cumsum(deg[o])
    tot = vol[-1]
    ks = np.arange(DOM_MIN, n - DOM_MIN + 1)
    if len(ks) == 0:
        return None, 0.0
    ncut = cut[ks] * (1.0 / np.maximum(vol[ks - 1], 1) + 1.0 / np.maximum(tot - vol[ks - 1], 1))
    k = int(ks[np.argmin(ncut)])
    lab = np.zeros(n, np.int8)
    lab[o[k:]] = 1
    pack = float(cut[k]) / min(k, n - k)
    part = np.full(R, -1, np.int8)
    part[P.res_index[ca]] = lab
    return part, pack


def af_pocket_features(pockets, pts, P: Protein, res_pl=None, dom=None) -> np.ndarray:
    """(n_pockets, len(AF_FEATURES)) float32; pocket.residues must be assigned."""
    n = len(pockets)
    out = np.zeros((n, len(AF_FEATURES)), np.float32)
    if n == 0:
        return out
    res_pl = residue_plddt_protein(P) if res_pl is None else res_pl
    part, pack = domain_split(P) if dom is None else dom
    rid = {r: i for i, r in enumerate(P.residue_ids.tolist())}
    low70 = P.plddt < 70
    low50 = P.plddt < 50
    used = np.unique(np.concatenate([p.idx for p in pockets]))
    near70 = np.zeros(len(pts), bool)
    near50 = np.zeros(len(pts), bool)
    if low70.any():
        d, _ = cKDTree(P.coords[low70]).query(pts[used], k=1, distance_upper_bound=4.0)
        near70[used] = np.isfinite(d)
    if low50.any():
        d, _ = cKDTree(P.coords[low50]).query(pts[used], k=1, distance_upper_bound=4.0)
        near50[used] = np.isfinite(d)
    prot_mean = float(res_pl.mean()) if len(res_pl) else 0.0
    prot_lt70 = float((res_pl < 70).mean()) if len(res_pl) else 0.0
    for j, p in enumerate(pockets):
        ri = np.array([rid[r] for r in p.residues if r in rid], np.int64)
        if len(ri):
            v = res_pl[ri]
            out[j, 0:5] = (v.mean(), v.min(), np.quantile(v, 0.25), (v < 70).mean(), (v < 50).mean())
        else:
            out[j, 0:5] = (prot_mean, prot_mean, prot_mean, prot_lt70, 0.0)
        out[j, 5] = near70[p.idx].mean()
        out[j, 6] = near50[p.idx].mean()
        out[j, 7] = prot_mean
        out[j, 8] = prot_lt70
        if part is not None and len(ri):
            pr = part[ri]
            n0, n1 = int((pr == 0).sum()), int((pr == 1).sum())
            out[j, 9] = min(n0, n1) / max(len(ri), 1)
            out[j, 10] = pack
        else:
            out[j, 9] = 0.0
            out[j, 10] = pack
    return out


@dataclass
class AFRanker:
    rank_model: str
    rank_features: list
    meta: dict = field(default_factory=dict)
    _rb = None

    def booster(self):
        import lightgbm as lgb
        if self._rb is None:
            self._rb = lgb.Booster(model_str=self.rank_model)
        return self._rb

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(dict(rank_model=self.rank_model, rank_features=self.rank_features, meta=self.meta), fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


def candidate_table(P: Protein, model: BurrowModel, weights=None, score_override=None):
    """Point features -> point scores -> candidates (+residues) -> standard + AF pocket features.
    Returns (pockets, ps, score, F (n, k) float32, names)."""
    fd = dict(model.meta.get("feature_cfg") or {})
    cfg = BurrowConfig(**{**fd, "lengths": tuple(fd["lengths"]), "nb_radii": tuple(fd["nb_radii"])}) if fd else BurrowConfig()
    ps = point_features(P, cfg, weights)
    fc = FormationConfig(**model.formation)
    if score_override is not None:
        score = score_override(ps)
    else:
        pb, _ = model.boosters()
        score = score_points(pb, ps.feats, ps.names, model.point_features, model.meta.get("score_floor"))
    pk = candidates(ps.X, ps.points, ps.vol, ps.feats, ps.names, score, fc)
    assign_residues(pk, ps.points, cKDTree(P.coords), P.res_index, P.residue_ids)
    Fp, fn = pocket_features(pk, ps.X, ps.vol, ps.feats, ps.names, score, len(P.coords))
    if len(pk):
        Fa = af_pocket_features(pk, ps.points, P)
        F = np.concatenate([Fp, Fa], 1)
    else:
        F = np.zeros((0, 0), np.float32)
    return pk, ps, score, F, list(fn) + AF_FEATURES


def predict_protein_af(P: Protein, model: BurrowModel, ranker: AFRanker, weights=None) -> Prediction:
    t0 = time.perf_counter()
    pk, ps, score, F, names = candidate_table(P, model, weights)
    fc = FormationConfig(**model.formation)
    if len(pk):
        j = [names.index(n) for n in ranker.rank_features]
        rs = ranker.booster().predict(F[:, j], num_threads=1)
    else:
        rs = np.zeros(0)
    keep = finalize(pk, rs, fc)
    tm = dict(total=(time.perf_counter() - t0) * 1e3)
    return Prediction(P.id, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                      [pk[i].residues for i in keep], [ps.points[pk[i].idx] for i in keep], tm)


__all__ = ["AF_FEATURES", "AFRanker", "af_pocket_features", "candidate_table", "domain_split", "predict_protein_af",
           "residue_plddt_protein"]
