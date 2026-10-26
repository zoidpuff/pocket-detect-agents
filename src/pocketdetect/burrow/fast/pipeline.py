"""Fast BURROW prediction for one structure (CPU): same outputs as ``burrow.pipeline.predict_protein``.

Stages: point features (``fast.features``; the model's own feature design from
``meta['feature_cfg']``) -> point scores (FlatForest, bit-identical to LightGBM; optional
``score_floor``) -> candidate pockets + residues (``fast.formation``) -> pocket descriptors
(``fast.pocketfeat``; optional context / propensity extras via the reference functions) -> ranker
(FlatForest, optional ensemble) -> NMS (``fast.formation.nms``) -> reported residues (with the
reference's empty-set fallback).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..features import BurrowConfig
from ..model import BurrowModel, FormationConfig, longest_bur
from ..pipeline import Prediction
from . import formation as FF
from .pocketfeat import pocket_features, residue_chains
from .trees import FlatForest


def model_cfg(model: BurrowModel | None, cfg: BurrowConfig | None = None) -> BurrowConfig:
    """The feature design a model was trained with (as ``pipeline.predict_protein`` does)."""
    if model is not None and model.meta.get("feature_cfg"):
        fd = dict(model.meta["feature_cfg"])
        fd["lengths"] = tuple(fd["lengths"])
        fd["nb_radii"] = tuple(fd["nb_radii"])
        return BurrowConfig(**fd)
    return cfg if cfg is not None else BurrowConfig()


@dataclass
class FastModel:
    """A :class:`BurrowModel` compiled to flat forests (+ lazily created GPU forests)."""
    model: BurrowModel | None
    point: FlatForest | None
    rank: FlatForest | None
    fc: FormationConfig
    rank_extra: list = field(default_factory=list)
    cfg: BurrowConfig = field(default_factory=BurrowConfig)

    @classmethod
    def from_model(cls, model: BurrowModel | None) -> "FastModel":
        if model is None:
            return cls(None, None, None, FormationConfig())
        pb, rb = model.boosters()
        extra = [FlatForest.from_model_string(s) for s in (model.meta.get("rank_ensemble") or [])]
        return cls(model, FlatForest.from_booster(pb), FlatForest.from_booster(rb) if rb is not None else None,
                   FormationConfig(**model.formation), extra, model_cfg(model))

    @classmethod
    def load(cls, path) -> "FastModel":
        return cls.from_model(BurrowModel.load(path))

    def point_index(self, names):
        return [names.index(n) for n in self.model.point_features]

    def rank_predict(self, X):
        """Same as ``model.rank_predict`` (mean over the ranker ensemble, if any)."""
        s = self.rank.predict(X)
        if self.rank_extra:
            s = (s + sum(f.predict(X) for f in self.rank_extra)) / (1 + len(self.rank_extra))
        return s


def geo_values(F, names):
    """``model.geo_value``: mean of bur10, bur16 and the longest-range buriedness column."""
    return (F[:, names.index("bur10")] + F[:, names.index("bur16")] + F[:, names.index(longest_bur(names))]) / 3.0


def score_mask(fm: FastModel, F, names):
    """Points that the model scores (``model.meta['score_floor']``: b >= floor or metal candidate)."""
    floor = fm.model.meta.get("score_floor") if fm.model is not None else None
    if floor is None:
        return None
    return (geo_values(F, names) >= floor) | (F[:, names.index("metal_cand")] > 0)


def score_points(fm: FastModel, F, names):
    """Same as ``model.score_points`` (float32 scores; optional buriedness floor)."""
    if len(F) == 0:
        return np.zeros(0, np.float32)
    idx = fm.point_index(names)
    m = score_mask(fm, F, names)
    if m is None:
        return fm.point.predict(F[:, idx]).astype(np.float32)
    out = np.zeros(len(F), np.float32)
    if m.any():
        out[m] = fm.point.predict(F[m][:, idx])
    return out


def geo_scores(c, vol, gv, fc: FormationConfig):
    """Geometry-only pocket score, identical to ``model.geo_score`` (float32 pairwise sums)."""
    w = vol * np.clip((gv - fc.geo_floor) / (1.0 - fc.geo_floor), 0.0, 1.0) ** fc.geo_power
    return np.array([float(w[c.members(i)].sum()) for i in range(len(c))])


def featurize(P, cfg: BurrowConfig = BurrowConfig(), weights=None):
    """Point features: fast implementation unless the reference featurizer changed (guard)."""
    from . import features as FFE
    if FFE.supports(cfg):
        return FFE.point_features(P, cfg, weights)
    from ..features import point_features
    return point_features(P, cfg, weights)


def _reference_post(P, ps, fm: FastModel, ranker, tm, weights=None):
    """Reference implementation of everything after the features (drift fallback)."""
    from scipy.spatial import cKDTree
    from ..model import candidates, geo_score, pocket_features as ref_pf, rank_predict, score_points as ref_score
    from ..pipeline import finalize, output_residues
    from ..pockets import assign_residues
    t1 = time.perf_counter()
    geo = fm.model is None or ranker == "geometry"
    score = None
    if not geo:
        pb, rb = fm.model.boosters()
        score = ref_score(pb, ps.feats, ps.names, fm.model.point_features, fm.model.meta.get("score_floor"))
    t2 = time.perf_counter()
    pk = candidates(ps.X, ps.points, ps.vol, ps.feats, ps.names, score, fm.fc)
    tree = cKDTree(P.coords)
    assign_residues(pk, ps.points, tree, P.res_index, P.residue_ids)
    t3 = time.perf_counter()
    if geo:
        rs = geo_score(pk, ps.vol, ps.feats, ps.names, fm.fc)
    elif len(pk):
        Fp, fn = ref_pf(pk, ps.X, ps.vol, ps.feats, ps.names, score, len(P.coords))
        Fp, fn = _extras(P, ps, fm, Fp, fn, [p.idx for p in pk], np.array([p.center for p in pk]), tree)
        rs = rank_predict(fm.model, Fp[:, [fn.index(n) for n in fm.model.rank_features]])
    else:
        rs = np.zeros(0)
    keep = finalize(pk, rs, fm.fc)
    out_res = [output_residues(pk[i], tree, P) for i in keep]
    t4 = time.perf_counter()
    tm.update(point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3, rank=(t4 - t3) * 1e3)
    return Prediction(P.id, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                      out_res, [ps.points[pk[i].idx] for i in keep], tm)


def _extras(P, ps, fm, Fp, fn, idx_list, centers, tree=None):
    """Phase-3 ranker extras (context / propensity), computed by the reference functions."""
    meta = fm.model.meta
    ctx = meta.get("context")
    if ctx:
        from ..model import context_features
        Fx, nx = context_features(Fp, fn, centers, ctx)
        Fp, fn = np.concatenate([Fp, Fx], 1), fn + nx
    if meta.get("propensity"):
        from scipy.spatial import cKDTree
        from ..propensity import PROP_NAMES, atom_logodds, pocket_propensity
        lo = atom_logodds(P, meta["propensity"])
        Fx = pocket_propensity(idx_list, ps.points, P, lo, tree if tree is not None else cKDTree(P.coords))
        Fp, fn = np.concatenate([Fp, Fx], 1), fn + PROP_NAMES
    return Fp, fn


def _output_residues(P, c, keep):
    """Reported residue sets with the reference's fallback for empty sets (``output_residues``)."""
    rid = P.residue_ids
    out, tree = [], None
    for i in keep:
        r = c.residues(i)
        if len(r):
            out.append(frozenset(rid[r].tolist()))
            continue
        from types import SimpleNamespace
        from ..pipeline import output_residues
        if tree is None:
            from scipy.spatial import cKDTree
            tree = cKDTree(P.coords)
        out.append(output_residues(SimpleNamespace(residues=frozenset(), center=c.center[i]), tree, P))
    return out


def predict_from_points(P, ps, fm: FastModel, ranker: str = "learned", score=None, timings=None) -> Prediction:
    """Everything after the point features (CPU).  ``score`` may be precomputed (e.g. on GPU)."""
    from .guard import stage_ok
    tm = {} if timings is None else timings
    if not all(stage_ok(s) for s in ("formation", "pocket_features", "scoring")):
        return _reference_post(P, ps, fm, ranker, tm)
    t1 = time.perf_counter()
    fc = fm.fc
    geo = fm.model is None or ranker == "geometry"
    if not geo and score is None:
        score = score_points(fm, ps.feats, ps.names)
    t2 = time.perf_counter()
    names = ps.names
    gv = geo_values(ps.feats, names)
    is_metal = ps.feats[:, names.index("metal_cand")] > 0
    c = FF.candidates(ps.X, ps.points, ps.vol, gv, None if geo else score, is_metal, fc)
    FF.assign_residues(c, ps.points, P.coords, P.res_index, len(P.residue_ids))
    t3 = time.perf_counter()
    if geo:
        rs = geo_scores(c, ps.vol, gv, fc)
    elif len(c):
        rch, nch = residue_chains(P.residue_ids)
        Fp, fn = pocket_features(c, ps.X, ps.vol, ps.feats, names, score, len(P.coords), rch, nch)
        if fm.model.meta.get("context") or fm.model.meta.get("propensity"):
            Fp, fn = _extras(P, ps, fm, Fp, fn, [c.members(i) for i in range(len(c))], c.center)
        rs = fm.rank_predict(Fp[:, [fn.index(n) for n in fm.model.rank_features]])
    else:
        rs = np.zeros(0)
    keep = FF.nms(c, rs, fc.nms_dist, fc.nms_jacc, fc.max_out, fc.nms_overlap, fc.min_score, fc.min_keep)
    res = _output_residues(P, c, keep)
    t4 = time.perf_counter()
    tm.update(point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3, rank=(t4 - t3) * 1e3)
    return Prediction(P.id, c.center[keep].reshape(-1, 3), np.asarray(rs)[keep], res,
                      [ps.points[c.members(i)] for i in keep], tm)


def predict_protein(P, fm: FastModel, cfg: BurrowConfig | None = None, weights=None,
                    ranker: str = "learned") -> Prediction:
    """Drop-in fast replacement for ``burrow.pipeline.predict_protein`` (takes a :class:`FastModel`).
    Like the reference, a model carrying ``meta['feature_cfg']`` uses its own feature design."""
    cfg = model_cfg(fm.model, cfg)
    t0 = time.perf_counter()
    ps = featurize(P, cfg, weights)
    t1 = time.perf_counter()
    tm = {f"feat_{k}": v for k, v in ps.timings.items()}
    tm["features"] = (t1 - t0) * 1e3
    pred = predict_from_points(P, ps, fm, ranker, timings=tm)
    tm["total"] = (time.perf_counter() - t0) * 1e3
    return pred
